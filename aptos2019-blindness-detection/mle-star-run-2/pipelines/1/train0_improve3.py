
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


import numpy as np
from scipy.optimize import minimize
from scipy.stats import trim_mean
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

# Distribution-Matching Quantile Calibration Rounder with K-Fold Regularized Threshold Blending
class OptimizedRounder:
    def __init__(self, n_splits=5, l2_reg=0.05, trim_proportion=0.1):
        self.n_splits = n_splits
        self.l2_reg = l2_reg
        self.trim_proportion = trim_proportion
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    @staticmethod
    def _softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def _get_thresholds_from_logits(self, logits, X):
        probs = self._softmax(logits)
        cum_probs = np.clip(np.cumsum(probs)[:-1], 1e-6, 1.0 - 1e-6)
        thresholds = np.quantile(X, cum_probs)
        return thresholds, cum_probs

    def _loss(self, logits, X, y, prior_cum_probs):
        thresholds, cum_probs = self._get_thresholds_from_logits(logits, X)
        preds = np.digitize(X, thresholds)
        qwk = cohen_kappa_score(y, preds, weights="quadratic")
        reg_penalty = self.l2_reg * np.sum((cum_probs - prior_cum_probs) ** 2)
        return -qwk + reg_penalty

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=int).ravel()
        
        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes <= 1:
            self.coef_ = np.array([0.5])
            return self

        # Global empirical class probability mass prior and cumulative distribution
        class_counts = np.bincount(y - np.min(y), minlength=n_classes)
        prior_probs = (class_counts + 1e-5) / np.sum(class_counts + 1e-5)
        prior_cum_probs = np.clip(np.cumsum(prior_probs)[:-1], 1e-6, 1.0 - 1e-6)
        initial_logits = np.log(prior_probs)

        fold_thresholds = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            
            res = minimize(
                fun=self._loss,
                x0=initial_logits,
                args=(X_train, y_train, prior_cum_probs),
                method="Nelder-Mead",
                options={"maxiter": 500, "adaptive": True}
            )
            
            val_thresh, _ = self._get_thresholds_from_logits(res.x, X[val_idx])
            fold_thresholds.append(val_thresh)

        fold_thresholds = np.array(fold_thresholds)
        if len(fold_thresholds) > 2:
            self.coef_ = trim_mean(fold_thresholds, proportiontocut=self.trim_proportion, axis=0)
        else:
            self.coef_ = np.mean(fold_thresholds, axis=0)

        # Ensure monotonically increasing thresholds
        self.coef_ = np.sort(self.coef_)
        return self

    def predict(self, X):
        X_p = np.asarray(X, dtype=float).ravel()
        return np.digitize(X_p, self.coef_)


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
