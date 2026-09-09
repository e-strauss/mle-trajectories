
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


# Model Definitions
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        if x.dim() == 4:
            return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)
        elif x.dim() == 3:
            return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return x


class ConvNeXtV2RegressionModel(nn.Module):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True, drop_rate=0.2, num_dropout_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        in_features = self.backbone.num_features
        self.gem = GeM()
        self.norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_dropout_samples)])
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gem(feat)
        feat = self.norm(feat)
        logits = torch.mean(torch.stack([self.head(drop(feat)) for drop in self.dropouts], dim=0), dim=0)
        return logits.squeeze(-1)


class EVA02OrdinalModel(nn.Module):
    def __init__(self, model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True, drop_rate=0.2, num_dropout_samples=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        in_features = self.backbone.num_features
        self.gem = GeM()
        self.norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_dropout_samples)])
        self.classifier = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gem(feat)
        feat = self.norm(feat)
        logits = torch.mean(torch.stack([self.classifier(drop(feat)) for drop in self.dropouts], dim=0), dim=0)
        return logits

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



import copy


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.module.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


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
    ema_model_1 = ModelEMA(model_1, decay=0.999)
    criterion_1 = nn.SmoothL1Loss()
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=lr_1, weight_decay=wd_1)
    scheduler_1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_1,
        max_lr=lr_1,
        total_steps=epochs_1 * len(train_loader_1),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=1e3,
    )
    scaler_1 = torch.cuda.amp.GradScaler()

    best_val_preds_1 = None
    best_qwk_1 = -1.0

    for epoch in range(epochs_1):
        model_1.train()
        for images, targets in train_loader_1:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE).float()

            optimizer_1.zero_grad()
            with torch.cuda.amp.autocast():
                if np.random.rand() < 0.5:
                    lam = np.random.beta(0.4, 0.4)
                    rand_index = torch.randperm(images.size(0)).to(DEVICE)
                    mixed_images = lam * images + (1 - lam) * images[rand_index]
                    mixed_targets = lam * targets + (1 - lam) * targets[rand_index]
                    outputs = model_1(mixed_images)
                    loss = criterion_1(outputs, mixed_targets)
                else:
                    outputs = model_1(images)
                    loss = criterion_1(outputs, targets)

            scaler_1.scale(loss).backward()
            scaler_1.step(optimizer_1)
            scaler_1.update()
            scheduler_1.step()
            ema_model_1.update(model_1)

        ema_model_1.module.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, targets in val_loader_1:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = ema_model_1.module(images)
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
    ema_model_2 = ModelEMA(model_2, decay=0.999)
    criterion_2 = OrdinalClassificationLoss()
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=lr_2, weight_decay=wd_2)
    scheduler_2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_2,
        max_lr=lr_2,
        total_steps=epochs_2 * len(train_loader_2),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=1e3,
    )
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
                if np.random.rand() < 0.5:
                    lam = np.random.beta(0.4, 0.4)
                    rand_index = torch.randperm(images.size(0)).to(DEVICE)
                    mixed_images = lam * images + (1 - lam) * images[rand_index]
                    targets_a, targets_b = targets, targets[rand_index]
                    logits = model_2(mixed_images)
                    loss = lam * criterion_2(logits, targets_a) + (1 - lam) * criterion_2(logits, targets_b)
                else:
                    logits = model_2(images)
                    loss = criterion_2(logits, targets)

            scaler_2.scale(loss).backward()
            scaler_2.step(optimizer_2)
            scaler_2.update()
            scheduler_2.step()
            ema_model_2.update(model_2)

        ema_model_2.module.eval()
        val_preds = []
        with torch.no_grad():
            for images, targets in val_loader_2:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    continuous_preds = ema_model_2.module.predict_continuous(images)
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
