
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
        self.labels = self.df["diagnosis"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class ClassBalancedSoftQWKLoss(nn.Module):
    def __init__(self, num_classes=5, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        i, j = torch.meshgrid(
            torch.arange(num_classes, dtype=torch.float32),
            torch.arange(num_classes, dtype=torch.float32),
            indexing="ij",
        )
        weights = ((i - j) / (num_classes - 1)) ** 2
        self.register_buffer("weights", weights)

    def forward(self, probs, targets):
        batch_size = targets.size(0)
        device = targets.device

        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()

        class_counts = targets_one_hot.sum(dim=0)
        class_weights = (batch_size / (self.num_classes * (class_counts + 1.0))).detach()
        sample_weights = (targets_one_hot * class_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        sample_weights = sample_weights / (sample_weights.sum() + self.eps)

        weighted_probs = probs * sample_weights
        weighted_targets = targets_one_hot * sample_weights

        hist_pred = weighted_probs.sum(dim=0)
        hist_target = weighted_targets.sum(dim=0)

        observed = torch.matmul(probs.t(), weighted_targets)
        expected = torch.outer(hist_pred, hist_target)

        observed = observed / (observed.sum() + self.eps)
        expected = expected / (expected.sum() + self.eps)

        penalty_weights = self.weights.to(device)
        numerator = torch.sum(penalty_weights * observed)
        denominator = torch.sum(penalty_weights * expected)

        return numerator / (denominator + self.eps)


class OrdinalEMDLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, probs, targets):
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_target = torch.cumsum(targets_one_hot, dim=-1)
        emd_loss = torch.abs(cdf_pred[:, :-1] - cdf_target[:, :-1]).sum(dim=-1).mean()
        return emd_loss


class LogCoshExpectationLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("class_values", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, probs, targets):
        device = targets.device
        class_vals = self.class_values.to(device)
        expected_grade = torch.sum(probs * class_vals, dim=-1)
        diff = expected_grade - targets.float()

        abs_diff = torch.abs(diff)
        log_cosh = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - 0.6931471805599453
        return log_cosh.mean()


class AblatableOrdinalLoss(nn.Module):
    def __init__(
        self,
        num_classes=5,
        emd_weight=1.0,
        log_cosh_weight=0.5,
        qwk_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.emd_weight = emd_weight
        self.log_cosh_weight = log_cosh_weight
        self.qwk_weight = qwk_weight

        self.emd_loss_fn = OrdinalEMDLoss(num_classes=num_classes)
        self.log_cosh_loss_fn = LogCoshExpectationLoss(num_classes=num_classes)
        self.soft_qwk_loss_fn = ClassBalancedSoftQWKLoss(num_classes=num_classes, eps=eps)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        loss = torch.tensor(0.0, device=targets.device)

        if self.emd_weight > 0.0:
            loss = loss + self.emd_weight * self.emd_loss_fn(probs, targets)
        if self.log_cosh_weight > 0.0:
            loss = loss + self.log_cosh_weight * self.log_cosh_loss_fn(probs, targets)
        if self.qwk_weight > 0.0:
            loss = loss + self.qwk_weight * self.soft_qwk_loss_fn(probs, targets)

        return loss


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


class ConvNeXtV2OrdinalClassifier(nn.Module):
    def __init__(
        self,
        model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        num_classes=5,
        pretrained=True,
        drop_rate=0.2,
        num_dropout_samples=5,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        in_features = self.backbone.num_features
        self.gem = GeM()
        self.norm = nn.LayerNorm(in_features)
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_dropout_samples)])
        self.classifier = nn.Linear(in_features, num_classes)
        self.register_buffer("class_values", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gem(feat)
        feat = self.norm(feat)
        logits = torch.mean(torch.stack([self.classifier(drop(feat)) for drop in self.dropouts], dim=0), dim=0)
        return logits

    def predict_expected(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        expected = torch.sum(probs * self.class_values.to(x.device), dim=-1)
        return expected


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
                        options={"maxiter": 400},
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


def run_experiment(train_loader, val_loader, val_targets, loss_config, epochs=4):
    seed_everything(42)
    model = ConvNeXtV2OrdinalClassifier(num_classes=5, pretrained=True).to(DEVICE)
    criterion = AblatableOrdinalLoss(
        num_classes=5,
        emd_weight=loss_config.get("emd_weight", 1.0),
        log_cosh_weight=loss_config.get("log_cosh_weight", 0.5),
        qwk_weight=loss_config.get("qwk_weight", 1.0),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(get_parameter_groups(model, base_lr=2e-4, backbone_mult=0.1), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_qwk = -1.0

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    p0 = model.predict_expected(images)
                    p1 = model.predict_expected(torch.flip(images, dims=[-1]))
                    p2 = model.predict_expected(torch.flip(images, dims=[-2]))
                    p3 = model.predict_expected(torch.flip(images, dims=[-2, -1]))
                    expected_preds = (p0 + p1 + p2 + p3) / 4.0

                val_preds.extend(expected_preds.cpu().numpy().tolist())

        val_preds = np.array(val_preds)
        rounder = OptimizedRounder()
        rounder.fit(val_preds, val_targets)
        discrete_preds = rounder.predict(val_preds)
        qwk = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if qwk > best_val_qwk:
            best_val_qwk = qwk

    return best_val_qwk


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

    experiments = {
        "Full Composite Loss Baseline (EMD + Log-Cosh + Class-Balanced QWK)": {
            "emd_weight": 1.0,
            "log_cosh_weight": 0.5,
            "qwk_weight": 1.0,
        },
        "Ablation 1 (Without Class-Balanced Soft QWK Loss)": {
            "emd_weight": 1.0,
            "log_cosh_weight": 0.5,
            "qwk_weight": 0.0,
        },
        "Ablation 2 (Without Log-Cosh Expectation Loss)": {
            "emd_weight": 1.0,
            "log_cosh_weight": 0.0,
            "qwk_weight": 1.0,
        },
        "Ablation 3 (Without Ordinal EMD Loss)": {
            "emd_weight": 0.0,
            "log_cosh_weight": 0.5,
            "qwk_weight": 1.0,
        },
    }

    results = {}
    print("--- Starting Ablation Study on Ordinal Multi-Loss Objectives ---")
    for exp_name, loss_cfg in experiments.items():
        score = run_experiment(train_loader, val_loader, val_targets, loss_cfg, epochs=4)
        results[exp_name] = score
        print(f"{exp_name}: Validation QWK = {score:.5f}")

    baseline_score = results["Full Composite Loss Baseline (EMD + Log-Cosh + Class-Balanced QWK)"]
    print("\n--- Ablation Study Summary ---")
    print(f"Full Composite Baseline Score: {baseline_score:.5f}")

    max_drop = -float("inf")
    most_critical_component = ""

    for exp_name, score in results.items():
        if exp_name.startswith("Ablation"):
            delta = baseline_score - score
            print(f"{exp_name} Performance Drop: {delta:+.5f} (Val QWK: {score:.5f})")
            if delta > max_drop:
                max_drop = delta
                most_critical_component = exp_name

    print(f"\nMost critical component contributing to overall performance: {most_critical_component}")


if __name__ == "__main__":
    main()
