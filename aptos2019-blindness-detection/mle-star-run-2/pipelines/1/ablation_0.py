
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
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, num_classes=5):
        levels = torch.arange(num_classes - 1, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        return self.bce(logits, binary_targets)

# Model Definitions
class ConvNeXtV2RegressionModel(nn.Module):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)

class EVA02OrdinalModel(nn.Module):
    def __init__(self, model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes - 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

    def predict_continuous(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs.sum(dim=1)

# Rounders
class FixedRounder:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def predict(self, X):
        X_p = np.copy(X)
        res = np.zeros_like(X_p, dtype=int)
        res[X_p >= self.coef_[0]] = 1
        res[X_p >= self.coef_[1]] = 2
        res[X_p >= self.coef_[2]] = 3
        res[X_p >= self.coef_[3]] = 4
        return res

class OptimizedRounder:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        res = minimize(self._loss, self.coef_, args=(X, y), method="Powell")
        self.coef_ = res.x

    def predict(self, X):
        X_p = np.copy(X)
        res = np.zeros_like(X_p, dtype=int)
        res[X_p >= self.coef_[0]] = 1
        res[X_p >= self.coef_[1]] = 2
        res[X_p >= self.coef_[2]] = 3
        res[X_p >= self.coef_[3]] = 4
        return res

def train_convnext(train_df, val_df, use_augmentation=True, epochs=4):
    seed_everything(42)
    img_size = 384
    lr = 2e-4
    wd = 1e-2

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
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

    model = ConvNeXtV2RegressionModel().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_preds = None
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        val_preds = []
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                val_preds.extend(outputs.cpu().numpy().tolist())

        val_loss /= len(val_dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_preds = np.array(val_preds)

    return best_val_preds

def train_eva02(train_df, val_df, epochs=4):
    seed_everything(42)
    img_size = 448
    lr = 5e-5
    wd = 0.05

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
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

    model = EVA02OrdinalModel().to(DEVICE)
    criterion = OrdinalClassificationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_preds = None
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        val_preds = []
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
                    continuous_preds = torch.sigmoid(logits).sum(dim=1)
                val_loss += loss.item() * len(targets)
                val_preds.extend(continuous_preds.cpu().numpy().tolist())

        val_loss /= len(val_dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_preds = np.array(val_preds)

    return best_val_preds

def main():
    seed_everything(42)

    df = pd.read_csv(TRAIN_CSV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df, df["diagnosis"]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        break

    val_targets = val_df["diagnosis"].values.astype(int)

    # 1. Train Full Pipeline Models
    print("Training ConvNeXtV2 (with Augmentation)...")
    val_preds_convnext = train_convnext(train_df, val_df, use_augmentation=True, epochs=4)

    print("Training EVA-02 (Ordinal)...")
    val_preds_eva02 = train_eva02(train_df, val_df, epochs=4)

    # 2. Train Ablation Model: ConvNeXtV2 without Data Augmentation
    print("Training ConvNeXtV2 (without Augmentation for Ablation)...")
    val_preds_convnext_no_aug = train_convnext(train_df, val_df, use_augmentation=False, epochs=4)

    # ==========================
    # Evaluate Baseline & Ablations
    # ==========================
    results = {}

    # Baseline (Full Pipeline): ConvNeXtV2 + EVA-02 Ensemble + Optimized Rounder
    ensemble_preds = 0.5 * val_preds_convnext + 0.5 * val_preds_eva02
    opt_rounder = OptimizedRounder()
    opt_rounder.fit(ensemble_preds, val_targets)
    baseline_qwk = cohen_kappa_score(val_targets, opt_rounder.predict(ensemble_preds), weights="quadratic")
    results["Full Baseline Pipeline (Ensemble + Augmentations + Optimized Rounder)"] = baseline_qwk

    # Ablation 1: Disable Threshold Optimization (Fixed Standard Rounding)
    fixed_rounder = FixedRounder()
    ablation_no_opt_rounder_qwk = cohen_kappa_score(val_targets, fixed_rounder.predict(ensemble_preds), weights="quadratic")
    results["Ablation 1: Without Threshold Optimization (Fixed Rounding)"] = ablation_no_opt_rounder_qwk

    # Ablation 2: Disable Ensembling (Single Model ConvNeXtV2 with Optimized Rounder)
    opt_rounder_conv = OptimizedRounder()
    opt_rounder_conv.fit(val_preds_convnext, val_targets)
    ablation_single_conv_qwk = cohen_kappa_score(val_targets, opt_rounder_conv.predict(val_preds_convnext), weights="quadratic")
    results["Ablation 2: Without Ensembling (ConvNeXtV2 Only)"] = ablation_single_conv_qwk

    # Ablation 3: Disable Data Augmentation (ConvNeXtV2 No Augmentations with Optimized Rounder)
    opt_rounder_no_aug = OptimizedRounder()
    opt_rounder_no_aug.fit(val_preds_convnext_no_aug, val_targets)
    ablation_no_aug_qwk = cohen_kappa_score(val_targets, opt_rounder_no_aug.predict(val_preds_convnext_no_aug), weights="quadratic")
    results["Ablation 3: Without Data Augmentation (ConvNeXtV2 No-Aug)"] = ablation_no_aug_qwk

    # Print Ablation Performance Summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS (Validation Quadratic Weighted Kappa)")
    print("="*70)
    for experiment, score in results.items():
        delta = score - baseline_qwk
        print(f"{experiment:<60}: QWK = {score:.4f} (Delta: {delta:+.4f})")
    print("="*70)

    # Determine component with maximum performance drop when removed
    drops = {
        "Optimized Threshold Rounding (Powell post-processing)": baseline_qwk - ablation_no_opt_rounder_qwk,
        "Model Ensembling (ConvNeXtV2 + EVA02)": baseline_qwk - ablation_single_conv_qwk,
        "Data Augmentation Pipeline": ablation_single_conv_qwk - ablation_no_aug_qwk,
    }

    most_critical_component = max(drops, key=drops.get)
    print(f"\nMost impactful component: '{most_critical_component}' contributes the most to overall performance (Performance drop when removed: {drops[most_critical_component]:.4f}).")

if __name__ == "__main__":
    main()
