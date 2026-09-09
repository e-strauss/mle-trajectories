
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


INPUT_DIR = Path("./input")
if not INPUT_DIR.exists():
    INPUT_DIR = Path(".")

TRAIN_CSV = INPUT_DIR / "train.csv"
TRAIN_IMAGES_DIR = INPUT_DIR / "train_images"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_black_margins(image, tol=10):
    """Crop non-informative dead black border margins via channel-sum thresholding."""
    img_np = np.array(image)
    if img_np.ndim == 3:
        mask = img_np.sum(axis=2) > (tol * 3)
    else:
        mask = img_np > tol

    if not np.any(mask):
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img_np[rmin : rmax + 1, cmin : cmax + 1]
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return image

    return Image.fromarray(cropped)


class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, use_crop=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.use_crop = use_crop
        self.image_ids = self.df["id_code"].values
        self.labels = self.df["diagnosis"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")

        if self.use_crop:
            image = crop_black_margins(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class ConvNeXtV2RegressionModel(nn.Module):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True, drop_rate=0.2, use_layernorm=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
        if use_layernorm:
            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 1)
            )
        else:
            self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)


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


def train_and_eval_experiment(train_df, val_df, val_targets, config, seed=42):
    seed_everything(seed)

    img_size = config.get("img_size", 384)
    use_crop = config.get("use_crop", True)
    use_layernorm = config.get("use_layernorm", True)
    epochs = config.get("epochs", 4)
    lr = config.get("lr", 2e-4)
    wd = config.get("wd", 1e-2)

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

    train_dataset = RetinopathyDataset(train_df, TRAIN_IMAGES_DIR, transform=train_transform, use_crop=use_crop)
    val_dataset = RetinopathyDataset(val_df, TRAIN_IMAGES_DIR, transform=val_transform, use_crop=use_crop)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ConvNeXtV2RegressionModel(
        model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        pretrained=True,
        drop_rate=0.2,
        use_layernorm=use_layernorm
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df, df["diagnosis"]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        break

    val_targets = val_df["diagnosis"].values.astype(int)

    experiments = {
        "Full Baseline (Crop Margin + 384x384 Res + Head LayerNorm)": {
            "use_crop": True,
            "img_size": 384,
            "use_layernorm": True,
        },
        "Ablation 1 (Without Black Margin Cropping)": {
            "use_crop": False,
            "img_size": 384,
            "use_layernorm": True,
        },
        "Ablation 2 (Lower Image Resolution: 224x224 instead of 384x384)": {
            "use_crop": True,
            "img_size": 224,
            "use_layernorm": True,
        },
        "Ablation 3 (Without Head LayerNorm Feature Normalization)": {
            "use_crop": True,
            "img_size": 384,
            "use_layernorm": False,
        },
    }

    results = {}
    print("=== Commencing Ablation Study ===")
    for exp_name, config in experiments.items():
        print(f"Running: {exp_name}...")
        score = train_and_eval_experiment(train_df, val_df, val_targets, config, seed=42)
        results[exp_name] = score
        print(f"-> {exp_name}: QWK = {score:.5f}\n")

    baseline_score = results["Full Baseline (Crop Margin + 384x384 Res + Head LayerNorm)"]
    print("=" * 60)
    print("Ablation Study Results Summary:")
    print("=" * 60)
    print(f"{'Configuration':<65} | {'Validation QWK':<15} | {'Delta vs Baseline'}")
    print("-" * 95)

    deltas = {}
    for exp_name, score in results.items():
        delta = score - baseline_score
        deltas[exp_name] = delta
        print(f"{exp_name:<65} | {score:<15.5f} | {delta:+.5f}")

    # Determine which ablation led to the largest degradation
    ablations_only = {k: v for k, v in deltas.items() if "Baseline" not in k}
    most_impactful_component = min(ablations_only, key=ablations_only.get)
    max_drop = abs(ablations_only[most_impactful_component])

    print("=" * 60)
    print(f"Most critical contributing component: '{most_impactful_component}' (performance drop of {max_drop:.5f} QWK when removed/modified).")
    print("=" * 60)


if __name__ == "__main__":
    main()
