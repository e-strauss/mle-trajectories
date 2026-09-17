
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
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

INPUT_DIR = Path("./input")
if not INPUT_DIR.exists():
    INPUT_DIR = Path(".")

TRAIN_CSV = INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = INPUT_DIR / "train_images"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        try:
            counts = np.bincount(y.astype(int), minlength=5)
            cum_dist = np.cumsum(counts / len(y))[:-1]
            cum_dist = np.clip(cum_dist, 1e-4, 1.0 - 1e-4)
            quantile_init = np.quantile(X, cum_dist)
            initial_guesses.append(quantile_init)
        except Exception:
            pass

        best_loss = float("inf")
        best_coef = np.copy(self.coef_)

        methods = ["Powell", "Nelder-Mead"]
        for init in initial_guesses:
            for method in methods:
                try:
                    res = minimize(self._loss, init, args=(X, y), method=method, options={"maxiter": 500})
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_coef = res.x
                except Exception:
                    continue

        self.coef_ = np.sort(best_coef)

    def predict(self, X):
        return np.digitize(X, np.sort(self.coef_))

def apply_mixup_cutmix(images, targets, alpha=0.4, p_cutmix=0.5):
    if np.random.rand() > 0.5:
        return images, targets
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    lam = np.random.beta(alpha, alpha)

    if np.random.rand() < p_cutmix:
        W = images.size(2)
        H = images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_targets = lam * targets + (1.0 - lam) * targets[indices]
        return mixed_images, mixed_targets
    else:
        mixed_images = lam * images + (1.0 - lam) * images[indices]
        mixed_targets = lam * targets + (1.0 - lam) * targets[indices]
        return mixed_images, mixed_targets

def get_parameter_groups(model, base_lr, backbone_mult=0.1):
    head_keywords = ["head", "fc", "classifier", "linear"]
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)

    if not backbone_params or backbone_mult == 1.0:
        return [{"params": model.parameters(), "lr": base_lr}]

    return [
        {"params": backbone_params, "lr": base_lr * backbone_mult},
        {"params": head_params, "lr": base_lr},
    ]

def train_and_eval(train_df, val_df, use_diff_lr=True, use_tta=True, use_mixup=False, epochs=4, lr=2e-4, seed=42):
    seed_everything(seed)
    img_size = 384
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RetinopathyDataset(train_df, TRAIN_IMAGES_DIR, transform=train_transform)
    val_dataset = RetinopathyDataset(val_df, TRAIN_IMAGES_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ConvNeXtV2RegressionModel(model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    
    backbone_mult = 0.1 if use_diff_lr else 1.0
    param_groups = get_parameter_groups(model, lr, backbone_mult=backbone_mult)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_qwk = -1.0

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE).view(-1).float()

            if use_mixup:
                images, targets = apply_mixup_cutmix(images, targets)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images).view(-1)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    if use_tta:
                        p0 = model(images)
                        p1 = model(torch.flip(images, dims=[-1]))
                        p2 = model(torch.flip(images, dims=[-2]))
                        p3 = model(torch.flip(images, dims=[-2, -1]))
                        preds = (p0 + p1 + p2 + p3) / 4.0
                    else:
                        preds = model(images)

                val_preds.extend(preds.view(-1).cpu().numpy().tolist())
                val_targets.extend(targets.view(-1).numpy().tolist())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets, dtype=int)

        rounder = OptimizedRounder()
        rounder.fit(val_preds, val_targets)
        discrete_preds = rounder.predict(val_preds)
        qwk = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if qwk > best_qwk:
            best_qwk = qwk

    return best_qwk

def main():
    seed_everything(42)
    df = pd.read_csv(TRAIN_CSV)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(df, df["diagnosis"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print("--- Starting Ablation Study on Training & Inference Pipeline Components ---")

    # Baseline: Differential LR (0.1x backbone), 4-view TTA, No MixUp/CutMix
    score_baseline = train_and_eval(train_df, val_df, use_diff_lr=True, use_tta=True, use_mixup=False)
    print(f"Baseline (Diff LR=True, TTA=True, Mixup=False) Validation QWK: {score_baseline:.5f}")

    # Ablation 1: No Test-Time Augmentation (Single-view forward pass)
    score_no_tta = train_and_eval(train_df, val_df, use_diff_lr=True, use_tta=False, use_mixup=False)
    delta_tta = score_no_tta - score_baseline
    print(f"Ablation 1 (Without 4-View TTA) Validation QWK: {score_no_tta:.5f} (Delta: {delta_tta:+.5f})")

    # Ablation 2: No Differential Learning Rate (Uniform LR for Backbone & Head)
    score_uniform_lr = train_and_eval(train_df, val_df, use_diff_lr=False, use_tta=True, use_mixup=False)
    delta_lr = score_uniform_lr - score_baseline
    print(f"Ablation 2 (Uniform Learning Rate / No Diff LR) Validation QWK: {score_uniform_lr:.5f} (Delta: {delta_lr:+.5f})")

    # Ablation 3: Adding Mixup / CutMix Data Augmentation
    score_mixup = train_and_eval(train_df, val_df, use_diff_lr=True, use_tta=True, use_mixup=True)
    delta_mixup = score_mixup - score_baseline
    print(f"Ablation 3 (With Mixup & CutMix Augmentation) Validation QWK: {score_mixup:.5f} (Delta: {delta_mixup:+.5f})")

    results = {
        "4-View Test-Time Augmentation (TTA)": abs(delta_tta),
        "Differential Learning Rate (0.1x Backbone LR)": abs(delta_lr),
        "Mixup / CutMix Regularization": abs(delta_mixup),
    }

    most_impactful_component = max(results, key=results.get)
    print("\n=======================================================")
    print(f"Component contributing the most to overall performance variance: {most_impactful_component} (|Delta| = {results[most_impactful_component]:.5f})")
    print("=======================================================")

if __name__ == "__main__":
    main()
