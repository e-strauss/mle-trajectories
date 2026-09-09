
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Configuration
INPUT_DIR = Path("./input")
if not INPUT_DIR.exists():
    INPUT_DIR = Path(".")

TRAIN_CSV = INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = INPUT_DIR / "train_images"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.image_ids = self.df["id_code"].values
        self.labels = self.df["diagnosis"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# Loss Functions
class OrdinalClassificationLoss(nn.Module):
    """Extended Binary Cross Entropy loss for ordinal targets: y > k for k in [0, K-2]"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, num_classes=5):
        levels = torch.arange(num_classes - 1, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        return self.bce(logits, binary_targets)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p).flatten(1)


class MultiSampleDropout(nn.Module):
    def __init__(self, in_features, out_features, num_samples=5, drop_rate=0.2):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.mean(torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0), dim=0)


class ResidualProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop_rate=0.2, num_samples=5):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, in_features)
        )
        self.head = MultiSampleDropout(in_features, out_features, num_samples=num_samples, drop_rate=drop_rate)

    def forward(self, x):
        normed = self.norm(x)
        x = x + self.mlp(normed)
        return self.head(x)


# Model Definitions
class ConvNeXtV2RegressionModel(nn.Module):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True, drop_rate=0.2, bottleneck_dim=384):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-3, -2, -1),
            drop_rate=drop_rate
        )
        feature_channels = self.backbone.feature_info.channels()
        
        # 1x1 conv projection bottlenecks for multi-stage feature aggregation
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, bottleneck_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.GELU()
            )
            for ch in feature_channels
        ])
        
        # Hybrid pooling: Generalized Mean (GeM) + Adaptive Max Pooling
        self.gem_pool = GeM()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        pooled_dim = bottleneck_dim * 2
        self.head = ResidualProjectionHead(
            in_features=pooled_dim,
            hidden_features=bottleneck_dim,
            out_features=1,
            drop_rate=drop_rate
        )

    def forward(self, x):
        feats = self.backbone(x)
        target_size = feats[0].shape[2:]
        
        fused_feat = 0
        for feat, proj in zip(feats, self.projections):
            proj_feat = proj(feat)
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(proj_feat, size=target_size, mode='bilinear', align_corners=False)
            fused_feat = fused_feat + proj_feat
            
        fused_feat = fused_feat / len(feats)
        
        gem_out = self.gem_pool(fused_feat)
        max_out = self.max_pool(fused_feat).flatten(1)
        pooled = torch.cat([gem_out, max_out], dim=1)
        
        return self.head(pooled).squeeze(-1)


class EVA02OrdinalModel(nn.Module):
    def __init__(self, model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True, drop_rate=0.2, bottleneck_dim=384):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.backbone.num_features
        self.num_stages = 3
        
        # 1x1 conv projection bottlenecks for multi-block spatial representations
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.GELU()
            )
            for _ in range(self.num_stages)
        ])
        
        self.gem_pool = GeM()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Concatenate CLS token + GeM pooled spatial + Max pooled spatial
        pooled_dim = embed_dim + bottleneck_dim * 2
        self.classifier = ResidualProjectionHead(
            in_features=pooled_dim,
            hidden_features=bottleneck_dim,
            out_features=num_classes - 1,
            drop_rate=drop_rate
        )

    def forward(self, x):
        # Patch embedding and position encoding
        x = self.backbone.patch_embed(x)
        pos_out = self.backbone._pos_embed(x)
        if isinstance(pos_out, tuple):
            x, rot_pos_embed = pos_out
        else:
            x = pos_out
            rot_pos_embed = None

        if getattr(self.backbone, 'patch_drop', None) is not None:
            x = self.backbone.patch_drop(x)
        if getattr(self.backbone, 'norm_pre', None) is not None:
            x = self.backbone.norm_pre(x)
        
        num_blocks = len(self.backbone.blocks)
        stage_indices = set(range(num_blocks - self.num_stages, num_blocks))
        collected_blocks = []
        
        for i, blk in enumerate(self.backbone.blocks):
            if rot_pos_embed is not None:
                x = blk(x, rope=rot_pos_embed)
            else:
                x = blk(x)
            if i in stage_indices:
                collected_blocks.append(x)
                
        final_tokens = self.backbone.norm(collected_blocks[-1])
        cls_token = final_tokens[:, 0]
        
        # Aggregate spatial patch tokens from intermediate blocks
        fused_spatial = 0
        for block_tokens, proj in zip(collected_blocks, self.projections):
            patch_tokens = block_tokens[:, 1:]
            B, L, C = patch_tokens.shape
            H = W = int(math.isqrt(L))
            spatial_map = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            fused_spatial = fused_spatial + proj(spatial_map)
            
        fused_spatial = fused_spatial / self.num_stages
        
        gem_out = self.gem_pool(fused_spatial)
        max_out = self.max_pool(fused_spatial).flatten(1)
        
        hybrid_representation = torch.cat([cls_token, gem_out, max_out], dim=1)
        return self.classifier(hybrid_representation)

    def predict_continuous(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs.sum(dim=1)


# Optimized Rounder for QWK
class OptimizedRounder:
    def __init__(self):
        self.coef_ = np.array([0.5, 1.5, 2.5, 3.5])

    def _loss(self, coef, X, y):
        sorted_coef = np.sort(coef)
        preds = np.digitize(X, sorted_coef)
        return -cohen_kappa_score(y, preds, weights="quadratic")

    def fit(self, X, y):
        initial_guesses = [
            np.array([0.5, 1.5, 2.5, 3.5]),
            np.array([0.55, 1.55, 2.55, 3.55]),
            np.array([0.45, 1.45, 2.45, 3.45]),
        ]

        # Quantile-based threshold initialization matching empirical distribution of y
        try:
            counts = np.bincount(y.astype(int), minlength=5)
            cum_dist = np.cumsum(counts / len(y))[:-1]
            cum_dist = np.clip(cum_dist, 1e-4, 1.0 - 1e-4)
            quantile_init = np.quantile(X, cum_dist)
            initial_guesses.append(quantile_init)
            initial_guesses.append(quantile_init - 0.1)
            initial_guesses.append(quantile_init + 0.1)
        except Exception:
            pass

        best_loss = float("inf")
        best_coef = np.copy(self.coef_)

        methods = ["Powell", "Nelder-Mead"]
        for init in initial_guesses:
            for method in methods:
                try:
                    res = minimize(
                        self._loss,
                        init,
                        args=(X, y),
                        method=method,
                        options={"maxiter": 500},
                    )
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_coef = res.x
                except Exception:
                    continue

        self.coef_ = np.sort(best_coef)

    def predict(self, X):
        return np.digitize(X, np.sort(self.coef_))


def main():
    seed_everything(42)

    # Load Metadata and create Train / Validation Split
    df = pd.read_csv(TRAIN_CSV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df, df["diagnosis"]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        break

    # ==========================
    # Train Model 1: ConvNeXtV2 (Regression)
    # ==========================
    img_size_1 = 384
    epochs_1 = 4
    lr_1 = 2e-4
    wd_1 = 1e-2

    train_transform_1 = transforms.Compose([
        transforms.Resize((img_size_1, img_size_1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform_1 = transforms.Compose([
        transforms.Resize((img_size_1, img_size_1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset_1 = RetinopathyDataset(train_df, TRAIN_IMAGES_DIR, transform=train_transform_1)
    val_dataset_1 = RetinopathyDataset(val_df, TRAIN_IMAGES_DIR, transform=val_transform_1)

    train_loader_1 = DataLoader(train_dataset_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model_1 = ConvNeXtV2RegressionModel(model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True).to(DEVICE)
    criterion_1 = nn.SmoothL1Loss()
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=lr_1, weight_decay=wd_1)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=epochs_1, eta_min=1e-6)
    scaler_1 = torch.cuda.amp.GradScaler()

    best_val_preds_1 = None
    best_qwk_1 = -1.0

    for epoch in range(epochs_1):
        model_1.train()
        for images, targets in train_loader_1:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer_1.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model_1(images)
                loss = criterion_1(outputs, targets)

            scaler_1.scale(loss).backward()
            scaler_1.step(optimizer_1)
            scaler_1.update()

        scheduler_1.step()

        model_1.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, targets in val_loader_1:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model_1(images)
                val_preds.extend(outputs.cpu().numpy().tolist())
                val_targets.extend(targets.numpy().tolist())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets, dtype=int)

        rounder = OptimizedRounder()
        rounder.fit(val_preds, val_targets)
        discrete_preds = rounder.predict(val_preds)
        qwk = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if qwk > best_qwk_1:
            best_qwk_1 = qwk
            best_val_preds_1 = val_preds

    # ==========================
    # Train Model 2: EVA-02 (Ordinal Classification)
    # ==========================
    img_size_2 = 448
    epochs_2 = 4
    lr_2 = 5e-5
    wd_2 = 0.05

    train_transform_2 = transforms.Compose([
        transforms.Resize((img_size_2, img_size_2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform_2 = transforms.Compose([
        transforms.Resize((img_size_2, img_size_2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset_2 = RetinopathyDataset(train_df, TRAIN_IMAGES_DIR, transform=train_transform_2)
    val_dataset_2 = RetinopathyDataset(val_df, TRAIN_IMAGES_DIR, transform=val_transform_2)

    train_loader_2 = DataLoader(train_dataset_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_2 = DataLoader(val_dataset_2, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model_2 = EVA02OrdinalModel(model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True).to(DEVICE)
    criterion_2 = OrdinalClassificationLoss()
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=lr_2, weight_decay=wd_2)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=epochs_2, eta_min=1e-6)
    scaler_2 = torch.cuda.amp.GradScaler()

    best_val_preds_2 = None
    best_qwk_2 = -1.0

    for epoch in range(epochs_2):
        model_2.train()
        for images, targets in train_loader_2:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer_2.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model_2(images)
                loss = criterion_2(logits, targets)

            scaler_2.scale(loss).backward()
            scaler_2.step(optimizer_2)
            scaler_2.update()

        scheduler_2.step()

        model_2.eval()
        val_preds = []
        with torch.no_grad():
            for images, targets in val_loader_2:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    continuous_preds = model_2.predict_continuous(images)
                val_preds.extend(continuous_preds.cpu().numpy().tolist())

        val_preds = np.array(val_preds)
        rounder = OptimizedRounder()
        rounder.fit(val_preds, val_targets)
        discrete_preds = rounder.predict(val_preds)
        qwk = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if qwk > best_qwk_2:
            best_qwk_2 = qwk
            best_val_preds_2 = val_preds

    # ==========================
    # Ensemble & Threshold Optimization
    # ==========================
    ensemble_val_preds = 0.5 * best_val_preds_1 + 0.5 * best_val_preds_2

    final_rounder = OptimizedRounder()
    final_rounder.fit(ensemble_val_preds, val_targets)
    final_discrete_preds = final_rounder.predict(ensemble_val_preds)
    final_validation_score = cohen_kappa_score(val_targets, final_discrete_preds, weights="quadratic")

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
