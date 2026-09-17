
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

# Dataset Definition
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
class SquaredEarthMoversDistanceLoss(nn.Module):
    """Squared Earth Mover's Distance with continuous Expected Value Regression."""
    def __init__(self, num_classes=5, alpha=0.5, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pred_cdf = torch.cumsum(probs, dim=-1)
        
        target_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        target_cdf = torch.cumsum(target_one_hot, dim=-1)
        
        emd_loss = torch.sum((pred_cdf - target_cdf) ** 2, dim=-1)
        if self.reduction == "mean":
            emd_loss = emd_loss.mean()
        elif self.reduction == "sum":
            emd_loss = emd_loss.sum()
            
        if self.alpha > 0:
            class_indices = torch.arange(self.num_classes, device=logits.device, dtype=probs.dtype)
            expected_value = torch.sum(probs * class_indices, dim=-1)
            reg_loss = F.smooth_l1_loss(expected_value, targets.float(), reduction=self.reduction)
            return emd_loss + self.alpha * reg_loss
        return emd_loss


# Model Definition
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


class RetinopathyOrdinalModel(nn.Module):
    def __init__(self, model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", num_classes=5, pretrained=True, drop_rate=0.2, num_dropout_samples=5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        in_features = self.backbone.num_features
        self.gem = GeM()
        self.norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_dropout_samples)])
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gem(feat)
        feat = self.norm(feat)
        logits = torch.mean(torch.stack([self.classifier(drop(feat)) for drop in self.dropouts], dim=0), dim=0)
        return logits

    def predict_continuous(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        class_indices = torch.arange(self.num_classes, device=logits.device, dtype=probs.dtype)
        return torch.sum(probs * class_indices, dim=-1)


# Threshold Optimizer with Option for Empirical Quantile Warm-Start
class OptimizedRounder:
    def __init__(self, use_quantile_init=True):
        self.coef_ = np.array([0.5, 1.5, 2.5, 3.5])
        self.use_quantile_init = use_quantile_init

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

        if self.use_quantile_init:
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


def get_parameter_groups(model, base_lr=2e-4, backbone_mult=0.1):
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

    return [
        {"params": backbone_params, "lr": base_lr * backbone_mult},
        {"params": head_params, "lr": base_lr},
    ]


def run_experiment(
    train_loader,
    val_loader,
    val_targets,
    loss_type="emd_with_reg",
    clip_grad=True,
    use_quantile_init=True,
    epochs=4,
    seed=42
):
    seed_everything(seed)
    model = RetinopathyOrdinalModel(pretrained=True).to(DEVICE)

    if loss_type == "emd_with_reg":
        criterion = SquaredEarthMoversDistanceLoss(num_classes=5, alpha=0.5)
    elif loss_type == "pure_emd":
        criterion = SquaredEarthMoversDistanceLoss(num_classes=5, alpha=0.0)
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    optimizer = torch.optim.AdamW(get_parameter_groups(model, base_lr=2e-4, backbone_mult=0.1), lr=2e-4, weight_decay=1e-2)
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
                logits = model(images)
                if loss_type == "cross_entropy":
                    loss = criterion(logits, targets.long())
                else:
                    loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    p0 = model.predict_continuous(images)
                    p1 = model.predict_continuous(torch.flip(images, dims=[-1]))
                    p2 = model.predict_continuous(torch.flip(images, dims=[-2]))
                    p3 = model.predict_continuous(torch.flip(images, dims=[-2, -1]))
                    tta_preds = (p0 + p1 + p2 + p3) / 4.0

                val_preds.extend(tta_preds.cpu().numpy().tolist())

        val_preds = np.array(val_preds)
        rounder = OptimizedRounder(use_quantile_init=use_quantile_init)
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

    print("=== Starting Ablation Study on Loss Formulation, Optimization, and Post-Processing ===")

    # 1. Full Baseline Configuration
    print("\n[1/4] Evaluating Full Baseline (Squared EMD Loss + Aux Reg, Grad Clip=True, Quantile Init=True)...")
    baseline_qwk = run_experiment(
        train_loader, val_loader, val_targets,
        loss_type="emd_with_reg", clip_grad=True, use_quantile_init=True
    )
    print(f"Full Baseline Validation QWK: {baseline_qwk:.5f}")

    # 2. Ablation 1: Standard Cross-Entropy Loss (Replacing Ordinal Squared EMD Loss)
    print("\n[2/4] Evaluating Ablation 1: Standard Cross-Entropy Loss (No Ordinal/EMD Geometry)...")
    ablation_ce_qwk = run_experiment(
        train_loader, val_loader, val_targets,
        loss_type="cross_entropy", clip_grad=True, use_quantile_init=True
    )
    print(f"Ablation 1 (Standard Cross-Entropy) Validation QWK: {ablation_ce_qwk:.5f} (Delta: {ablation_ce_qwk - baseline_qwk:+.5f})")

    # 3. Ablation 2: No Gradient Norm Clipping
    print("\n[3/4] Evaluating Ablation 2: Without Gradient Norm Clipping (clip_grad=False)...")
    ablation_nogradclip_qwk = run_experiment(
        train_loader, val_loader, val_targets,
        loss_type="emd_with_reg", clip_grad=False, use_quantile_init=True
    )
    print(f"Ablation 2 (No Grad Norm Clipping) Validation QWK: {ablation_nogradclip_qwk:.5f} (Delta: {ablation_nogradclip_qwk - baseline_qwk:+.5f})")

    # 4. Ablation 3: Fixed Uniform Threshold Initialization (No Empirical Quantile Multi-start)
    print("\n[4/4] Evaluating Ablation 3: Fixed Threshold Initialization (use_quantile_init=False)...")
    ablation_noquantiles_qwk = run_experiment(
        train_loader, val_loader, val_targets,
        loss_type="emd_with_reg", clip_grad=True, use_quantile_init=False
    )
    print(f"Ablation 3 (No Quantile Init) Validation QWK: {ablation_noquantiles_qwk:.5f} (Delta: {ablation_noquantiles_qwk - baseline_qwk:+.5f})")

    # Summary of Ablation Results
    results = {
        "Full Baseline (Squared EMD + Aux Reg + GradClip + QuantileInit)": (baseline_qwk, 0.0),
        "Ablation 1: Standard Cross-Entropy Loss (Replacing Squared EMD)": (ablation_ce_qwk, ablation_ce_qwk - baseline_qwk),
        "Ablation 2: Without Gradient Norm Clipping": (ablation_nogradclip_qwk, ablation_nogradclip_qwk - baseline_qwk),
        "Ablation 3: Without Quantile Threshold Warm-Start": (ablation_noquantiles_qwk, ablation_noquantiles_qwk - baseline_qwk),
    }

    print("\n======================= ABLATION STUDY RESULTS =======================")
    print(f"{'Experiment':<65} | {'Val QWK':<10} | {'Delta':<10}")
    print("-" * 90)
    for name, (score, delta) in results.items():
        print(f"{name:<65} | {score:<10.5f} | {delta:+.5f}")
    print("=" * 90)

    # Determine which ablated component caused the largest performance degradation
    drops = {
        "Squared Earth Mover's Distance Loss Formulation": baseline_qwk - ablation_ce_qwk,
        "Gradient Norm Clipping Regularization": baseline_qwk - ablation_nogradclip_qwk,
        "Empirical Quantile-Based Multi-Start Threshold Search": baseline_qwk - ablation_noquantiles_qwk,
    }
    
    most_impactful_component = max(drops, key=drops.get)
    max_drop = drops[most_impactful_component]

    print(f"\nConclusion: The component that contributes the most to overall performance is '{most_impactful_component}' (Degradation when removed: {max_drop:.5f} QWK).")


if __name__ == "__main__":
    main()
