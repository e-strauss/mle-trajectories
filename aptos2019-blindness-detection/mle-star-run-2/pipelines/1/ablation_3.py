
import os
import random
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
    random.seed(seed)
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

# Pooling Modules
class GeMPooling(nn.Module):
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

class DualBranchPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.gem = GeMPooling(p=p, eps=eps)

    def forward(self, x):
        if x.dim() == 4:
            gem_feat = self.gem(x)
            max_feat = x.amax(dim=(-2, -1))
            return torch.cat([gem_feat, max_feat], dim=-1)
        elif x.dim() == 3:
            gem_feat = self.gem(x)
            max_feat = x.amax(dim=1)
            return torch.cat([gem_feat, max_feat], dim=-1)
        return torch.cat([x, x], dim=-1)

# Multi-Sample Dropout & Ablation Model
class ConvNeXtV2AblationModel(nn.Module):
    def __init__(
        self,
        model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        pretrained=True,
        pooling_type="gem",            # 'gem', 'gap', or 'dual_branch'
        use_multisample_dropout=True,  # True for multi-sample dropout, False for standard single dropout
        drop_rate=0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        in_features = self.backbone.num_features
        self.pooling_type = pooling_type
        self.use_multisample_dropout = use_multisample_dropout

        if pooling_type == "gem":
            self.pool = GeMPooling()
            head_in_features = in_features
        elif pooling_type == "dual_branch":
            self.pool = DualBranchPooling()
            head_in_features = in_features * 2
        elif pooling_type == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            head_in_features = in_features
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        self.norm = nn.LayerNorm(head_in_features)

        if self.use_multisample_dropout:
            self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
        else:
            self.single_dropout = nn.Dropout(drop_rate)

        self.head = nn.Linear(head_in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        if self.pooling_type == "gap" and feat.dim() == 4:
            feat = self.pool(feat).flatten(1)
        else:
            feat = self.pool(feat)
        feat = self.norm(feat)

        if self.use_multisample_dropout:
            logits = torch.mean(torch.stack([self.head(drop(feat)) for drop in self.dropouts], dim=0), dim=0)
        else:
            logits = self.head(self.single_dropout(feat))

        return logits.squeeze(-1)

# Threshold Optimization
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

        for init in initial_guesses:
            for method in ["Powell", "Nelder-Mead"]:
                try:
                    res = minimize(self._loss, init, args=(X, y), method=method, options={"maxiter": 400})
                    if res.fun < best_loss:
                        best_loss = res.fun
                        best_coef = res.x
                except Exception:
                    continue

        self.coef_ = np.sort(best_coef)

    def predict(self, X):
        return np.digitize(X, np.sort(self.coef_))

def train_and_eval(train_loader, val_loader, val_targets, config, epochs=4, lr=2e-4):
    seed_everything(42)
    model = ConvNeXtV2AblationModel(
        model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        pretrained=True,
        pooling_type=config["pooling_type"],
        use_multisample_dropout=config["use_multisample_dropout"],
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_qwk = -1.0

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

    return best_qwk

def main():
    seed_everything(42)

    df = pd.read_csv(TRAIN_CSV)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(df, df["diagnosis"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

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

    # Define Ablation Experiments
    experiments = {
        "Baseline (GeM Pooling + Multi-Sample Dropout)": {
            "pooling_type": "gem",
            "use_multisample_dropout": True,
        },
        "Ablation 1 (Global Average Pooling GAP instead of GeM)": {
            "pooling_type": "gap",
            "use_multisample_dropout": True,
        },
        "Ablation 2 (Single Dropout instead of Multi-Sample Dropout)": {
            "pooling_type": "gem",
            "use_multisample_dropout": False,
        },
        "Ablation 3 (Dual-Branch Pooling: GeM + Max Pooling)": {
            "pooling_type": "dual_branch",
            "use_multisample_dropout": True,
        },
    }

    results = {}
    print("Running Ablation Study on Head Architecture and Pooling Mechanisms...\n" + "=" * 70)

    for exp_name, config in experiments.items():
        print(f"Evaluating: {exp_name}...")
        score = train_and_eval(train_loader, val_loader, val_targets, config, epochs=4, lr=2e-4)
        results[exp_name] = score
        print(f"-> Validation QWK: {score:.5f}\n")

    baseline_score = results["Baseline (GeM Pooling + Multi-Sample Dropout)"]

    print("=" * 70)
    print("Ablation Study Summary:")
    print("-" * 70)
    for exp_name, score in results.items():
        delta = score - baseline_score
        print(f"{exp_name:<60} | QWK: {score:.5f} | Delta: {delta:+.5f}")

    # Determine largest contributor
    pooling_impact = baseline_score - results["Ablation 1 (Global Average Pooling GAP instead of GeM)"]
    dropout_impact = baseline_score - results["Ablation 2 (Single Dropout instead of Multi-Sample Dropout)"]
    dual_branch_gain = results["Ablation 3 (Dual-Branch Pooling: GeM + Max Pooling)"] - baseline_score

    print("=" * 70)
    if pooling_impact >= dropout_impact and pooling_impact > 0:
        print(f"Conclusion: GeM Pooling contributes the most to model performance with a positive delta of {pooling_impact:+.5f} over standard GAP.")
    elif dropout_impact > pooling_impact and dropout_impact > 0:
        print(f"Conclusion: Multi-Sample Dropout contributes the most to model performance with a positive delta of {dropout_impact:+.5f} over single dropout.")
    else:
        best_exp = max(results, key=results.get)
        print(f"Conclusion: Best performing configuration is '{best_exp}' with QWK: {results[best_exp]:.5f}.")

if __name__ == "__main__":
    main()
