
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

# Model Definition
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

# Optimized Rounder
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

def train_and_eval_model(
    train_loader,
    val_loader,
    val_targets,
    loss_fn_type="smooth_l1",
    use_cosine_scheduler=True,
    weight_decay=1e-2,
    epochs=4,
    lr=2e-4,
    seed=42
):
    seed_everything(seed)
    model = ConvNeXtV2RegressionModel(model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True).to(DEVICE)
    
    if loss_fn_type == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    elif loss_fn_type == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) if use_cosine_scheduler else None
    scaler = torch.cuda.amp.GradScaler()

    best_qwk = -1.0
    best_val_preds = None

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

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                val_preds.extend(outputs.cpu().numpy().tolist())

        val_preds = np.array(val_preds)
        rounder = OptimizedRounder()
        rounder.fit(val_preds, val_targets)
        discrete_preds = rounder.predict(val_preds)
        qwk = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if qwk > best_qwk:
            best_qwk = qwk
            best_val_preds = val_preds

    return best_qwk, best_val_preds

def main():
    seed_everything(42)

    df = pd.read_csv(TRAIN_CSV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df, df["diagnosis"]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        break

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

    val_targets = val_df["diagnosis"].values.astype(int)

    results = {}

    print("--- Running Full Baseline (SmoothL1 + Cosine LR + AdamW Decay 0.01) ---")
    score_baseline, _ = train_and_eval_model(
        train_loader, val_loader, val_targets,
        loss_fn_type="smooth_l1", use_cosine_scheduler=True, weight_decay=1e-2
    )
    results["Full Baseline (SmoothL1, Cosine LR, Weight Decay)"] = score_baseline
    print(f"Validation QWK: {score_baseline:.5f}\n")

    print("--- Ablation 1: Replacing SmoothL1 Loss with Standard MSE Loss ---")
    score_mse, _ = train_and_eval_model(
        train_loader, val_loader, val_targets,
        loss_fn_type="mse", use_cosine_scheduler=True, weight_decay=1e-2
    )
    results["Ablation 1: MSE Loss instead of SmoothL1"] = score_mse
    print(f"Validation QWK: {score_mse:.5f}\n")

    print("--- Ablation 2: Removing Cosine Annealing Learning Rate Scheduler ---")
    score_no_sched, _ = train_and_eval_model(
        train_loader, val_loader, val_targets,
        loss_fn_type="smooth_l1", use_cosine_scheduler=False, weight_decay=1e-2
    )
    results["Ablation 2: Constant LR (No Cosine Annealing)"] = score_no_sched
    print(f"Validation QWK: {score_no_sched:.5f}\n")

    print("--- Ablation 3: Removing Weight Decay (weight_decay = 0.0) ---")
    score_no_wd, _ = train_and_eval_model(
        train_loader, val_loader, val_targets,
        loss_fn_type="smooth_l1", use_cosine_scheduler=True, weight_decay=0.0
    )
    results["Ablation 3: Zero Weight Decay"] = score_no_wd
    print(f"Validation QWK: {score_no_wd:.5f}\n")

    print("=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    drops = {}
    for name, score in results.items():
        delta = score - score_baseline
        if name != "Full Baseline (SmoothL1, Cosine LR, Weight Decay)":
            drops[name] = abs(delta)
        print(f"{name:55s} | QWK: {score:.5f} | Delta: {delta:+.5f}")

    most_critical = max(drops, key=drops.get)
    print("=" * 60)
    print(f"Conclusion: '{most_critical}' caused the largest performance degradation ({drops[most_critical]:.5f}),")
    print("indicating that this component contributes the most to the overall model performance among tested factors.")
    print("=" * 60)

    print(f"Final Validation Performance: {score_baseline}")

if __name__ == "__main__":
    main()
