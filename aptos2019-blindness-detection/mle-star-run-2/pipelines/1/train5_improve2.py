
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
class SoftQWKLoss(nn.Module):
    """
    Differentiable Soft Quadratic Weighted Kappa (QWK) loss surrogate with
    Exponential Moving Average (EMA) marginal histogram tracking to eliminate
    gradient variance caused by small or class-imbalanced mini-batches.
    """
    def __init__(self, num_classes=5, ema_decay=0.9, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ema_decay = ema_decay
        self.eps = eps

        # Precompute quadratic weight matrix: W_ij = ((i - j) / (K - 1))^2
        i, j = torch.meshgrid(
            torch.arange(num_classes, dtype=torch.float32),
            torch.arange(num_classes, dtype=torch.float32),
            indexing='ij'
        )
        weights = ((i - j) / (num_classes - 1)) ** 2
        self.register_buffer('weights', weights)

        # Running EMA histograms initialized uniformly
        self.register_buffer('hist_pred_ema', torch.ones(num_classes, dtype=torch.float32) / num_classes)
        self.register_buffer('hist_target_ema', torch.ones(num_classes, dtype=torch.float32) / num_classes)

    def forward(self, probs, targets):
        """
        probs: (batch_size, num_classes) predicted class probabilities
        targets: (batch_size,) integer or float class targets
        """
        batch_size = targets.size(0)
        device = targets.device

        # One-hot encode ground truth targets
        targets_long = targets.long()
        targets_one_hot = F.one_hot(targets_long, num_classes=self.num_classes).float()

        # Batch-level marginals
        batch_hist_pred = probs.mean(dim=0)
        batch_hist_target = targets_one_hot.mean(dim=0)

        hist_pred_ema = self.hist_pred_ema.to(device)
        hist_target_ema = self.hist_target_ema.to(device)

        if self.training:
            # Update EMA marginals with batch statistics
            hist_pred = self.ema_decay * hist_pred_ema + (1.0 - self.ema_decay) * batch_hist_pred.detach()
            hist_target = self.ema_decay * hist_target_ema + (1.0 - self.ema_decay) * batch_hist_target.detach()
            self.hist_pred_ema.copy_(hist_pred)
            self.hist_target_ema.copy_(hist_target)
        else:
            hist_pred = hist_pred_ema
            hist_target = hist_target_ema

        # Observed confusion matrix
        observed = torch.matmul(probs.t(), targets_one_hot) / batch_size
        # Expected confusion matrix derived from stabilized marginals
        expected = torch.outer(hist_pred, hist_target)

        weights = self.weights.to(device)
        num = torch.sum(weights * observed)
        den = torch.sum(weights * expected)

        # Return 1 - QWK surrogate
        return num / (den + self.eps)


class DistanceWeightedOrdinalFocalLoss(nn.Module):
    """
    Asymmetric Distance-Weighted Ordinal Focal Loss.
    Scales cross-entropy loss with focal modulating factor (1 - p_t)^gamma and
    penalizes probability mass placed on distant classes via quadratic class distance |k - y|^2.
    """
    def __init__(self, num_classes=5, gamma=2.0, distance_weight=1.0, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.distance_weight = distance_weight
        self.eps = eps

        # Precompute normalized class coordinates
        classes = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer('classes', classes)

    def forward(self, probs, targets):
        batch_size = targets.size(0)
        device = targets.device
        targets_long = targets.long()

        # Extract ground truth class probability p_t for focal modulation
        p_t = probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # Standard focal cross-entropy component
        focal_ce = -focal_weight * torch.log(p_t)

        # Distance-weighted ordinal penalty: sum_k ((k - y) / (K - 1))^2 * P(Y = k)
        class_coords = self.classes.to(device).unsqueeze(0)  # (1, num_classes)
        target_coords = targets.float().unsqueeze(1)         # (batch_size, 1)
        quad_dist = ((class_coords - target_coords) / (self.num_classes - 1)) ** 2
        distance_penalty = torch.sum(quad_dist * probs, dim=1)

        total_loss = focal_ce + self.distance_weight * focal_weight * distance_penalty
        return total_loss.mean()


class ExpectedValueHuberLoss(nn.Module):
    """
    Expected-Value Huber Regression Loss: E[Y] = sum(k * P(Y = k)).
    Ensures smooth continuous expectation calibration for downstream optimal thresholding.
    """
    def __init__(self, num_classes=5, beta=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        classes = torch.arange(num_classes, dtype=torch.float32)
        self.register_buffer('classes', classes)

    def forward(self, probs, targets):
        device = targets.device
        class_values = self.classes.to(device).unsqueeze(0)  # (1, num_classes)
        expected_values = torch.sum(probs * class_values, dim=1)
        return F.smooth_l1_loss(expected_values, targets.float(), beta=self.beta)


class OrdinalClassificationLoss(nn.Module):
    """
    Robust Multi-Task Ordinal Loss combining:
    1. Rank-Consistent CORAL Binary Cross-Entropy on cumulative cutpoints
    2. EMA-Stabilized Differentiable Soft QWK Loss
    3. Asymmetric Distance-Weighted Ordinal Focal Loss
    4. Continuous Expected-Value Huber Regression Loss
    """
    def __init__(
        self,
        num_classes=5,
        coral_weight=1.0,
        qwk_weight=1.0,
        focal_weight=0.5,
        huber_weight=0.5,
        focal_gamma=2.0,
        ema_decay=0.9,
        eps=1e-6
    ):
        super().__init__()
        self.num_classes = num_classes
        self.coral_weight = coral_weight
        self.qwk_weight = qwk_weight
        self.focal_weight = focal_weight
        self.huber_weight = huber_weight

        self.soft_qwk = SoftQWKLoss(num_classes=num_classes, ema_decay=ema_decay, eps=eps)
        self.ordinal_focal = DistanceWeightedOrdinalFocalLoss(num_classes=num_classes, gamma=focal_gamma, eps=eps)
        self.expected_huber = ExpectedValueHuberLoss(num_classes=num_classes)

    def forward(self, logits, targets, num_classes=None):
        num_classes = num_classes if num_classes is not None else self.num_classes
        device = targets.device

        if logits.size(-1) == num_classes - 1:
            # CORAL cumulative threshold logits: P(y > k) for k in [0, K-2]
            levels = torch.arange(num_classes - 1, device=device).unsqueeze(0)
            binary_targets = (targets.unsqueeze(1) > levels).float()
            coral_loss = F.binary_cross_entropy_with_logits(logits, binary_targets)

            # Reconstruct normalized class probability distribution
            cum_probs = torch.sigmoid(logits)
            p0 = 1.0 - cum_probs[:, 0:1]
            pk = cum_probs[:, :-1] - cum_probs[:, 1:]
            pK = cum_probs[:, -1:]

            # Ensure valid non-negative probabilities with numerical stability
            pk = torch.clamp(pk, min=0.0)
            p0 = torch.clamp(p0, min=0.0)
            pK = torch.clamp(pK, min=0.0)
            probs = torch.cat([p0, pk, pK], dim=1)
            probs = probs + 1e-7
            probs = probs / probs.sum(dim=-1, keepdim=True)
        else:
            # Direct class logits
            coral_loss = torch.tensor(0.0, device=device)
            probs = F.softmax(logits, dim=-1)

        # 1. Stabilized Soft QWK Loss
        qwk_loss = self.soft_qwk(probs, targets)

        # 2. Distance-Weighted Ordinal Focal Loss
        focal_loss = self.ordinal_focal(probs, targets)

        # 3. Expected-Value Huber Regression Loss
        huber_loss = self.expected_huber(probs, targets)

        total_loss = (
            self.coral_weight * coral_loss
            + self.qwk_weight * qwk_loss
            + self.focal_weight * focal_loss
            + self.huber_weight * huber_loss
        )
        return total_loss


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


def main():
    seed_everything(42)

    # Load Metadata and create Train / Validation Split
    df = pd.read_csv(TRAIN_CSV)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(df, df["diagnosis"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

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

        if not backbone_params:
            return [{"params": head_params, "lr": base_lr}]
        if not head_params:
            return [{"params": backbone_params, "lr": base_lr}]

        return [
            {"params": backbone_params, "lr": base_lr * backbone_mult},
            {"params": head_params, "lr": base_lr},
        ]

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
    criterion_1 = nn.SmoothL1Loss().to(DEVICE)
    optimizer_1 = torch.optim.AdamW(get_parameter_groups(model_1, lr_1, backbone_mult=0.1), lr=lr_1, weight_decay=wd_1)
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
            scaler_1.unscale_(optimizer_1)
            torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)
            scaler_1.step(optimizer_1)
            scaler_1.update()

        scheduler_1.step()

        model_1.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, targets in val_loader_1:
                images = images.to(DEVICE)
                # 4-view TTA: original, horizontal flip, vertical flip, diagonal flip
                with torch.cuda.amp.autocast():
                    p0 = model_1(images)
                    p1 = model_1(torch.flip(images, dims=[-1]))
                    p2 = model_1(torch.flip(images, dims=[-2]))
                    p3 = model_1(torch.flip(images, dims=[-2, -1]))
                    tta_preds = (p0 + p1 + p2 + p3) / 4.0

                val_preds.extend(tta_preds.cpu().numpy().tolist())
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
    criterion_2 = OrdinalClassificationLoss(num_classes=5).to(DEVICE)
    optimizer_2 = torch.optim.AdamW(get_parameter_groups(model_2, lr_2, backbone_mult=0.1), lr=lr_2, weight_decay=wd_2)
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
            scaler_2.unscale_(optimizer_2)
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), max_norm=1.0)
            scaler_2.step(optimizer_2)
            scaler_2.update()

        scheduler_2.step()

        model_2.eval()
        val_preds = []
        with torch.no_grad():
            for images, targets in val_loader_2:
                images = images.to(DEVICE)
                # 4-view TTA: original, horizontal flip, vertical flip, diagonal flip
                with torch.cuda.amp.autocast():
                    p0 = model_2.predict_continuous(images)
                    p1 = model_2.predict_continuous(torch.flip(images, dims=[-1]))
                    p2 = model_2.predict_continuous(torch.flip(images, dims=[-2]))
                    p3 = model_2.predict_continuous(torch.flip(images, dims=[-2, -1]))
                    tta_preds = (p0 + p1 + p2 + p3) / 4.0

                val_preds.extend(tta_preds.cpu().numpy().tolist())

        val_preds = np.array(val_preds)
        rounder = OptimizedRounder()
        rounder.fit(val_preds, val_targets)
        discrete_preds = rounder.predict(val_preds)
        qwk = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if qwk > best_qwk_2:
            best_qwk_2 = qwk
            best_val_preds_2 = val_preds

    # ==========================
    # Ensemble Weight Optimization & Threshold Tuning
    # ==========================
    best_w = 0.5
    final_validation_score = -1.0
    best_rounder = None

    for w in np.linspace(0.0, 1.0, 101):
        blend_val_preds = w * best_val_preds_1 + (1.0 - w) * best_val_preds_2
        rounder = OptimizedRounder()
        rounder.fit(blend_val_preds, val_targets)
        discrete_preds = rounder.predict(blend_val_preds)
        score = cohen_kappa_score(val_targets, discrete_preds, weights="quadratic")

        if score > final_validation_score:
            final_validation_score = score
            best_w = w
            best_rounder = rounder

    print(f"Optimal Model 1 Weight: {best_w:.2f}")
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
