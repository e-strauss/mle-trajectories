
import copy
import math
import os
import random
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
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(row["diagnosis"], dtype=torch.float32)
        return image, target


class ConvNeXtV2ForDR(nn.Module):
    def __init__(
        self,
        model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=True,
        dropout_rate=0.3,
        use_mlp_head=True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features
        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(256, 1),
            )
        else:
            self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)


class OptimizedQWKRounder:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _loss(self, coef, x, y):
        sorted_coef = np.sort(coef)
        x_discrete = np.digitize(x, sorted_coef)
        return -cohen_kappa_score(y, x_discrete, weights="quadratic")

    def fit(self, x, y):
        res = minimize(self._loss, self.coef_, args=(x, y), method="Nelder-Mead")
        self.coef_ = list(np.sort(res.x))

    def predict(self, x):
        return np.digitize(x, self.coef_)


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device_type, enabled=(device_type == "cuda")
            ):
                preds = model(images)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(targets.view(-1).numpy())

    return np.array(all_preds), np.array(all_targets)


def run_experiment(
    train_loader,
    val_loader,
    device,
    use_ema=True,
    use_scheduler=True,
    use_mlp_head=True,
    epochs=6,
    lr=1e-4,
):
    seed_everything(42)

    model = ConvNeXtV2ForDR(
        model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=True,
        dropout_rate=0.3,
        use_mlp_head=use_mlp_head,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    if use_scheduler:
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=0.995) if use_ema else None

    best_qwk = -1.0
    best_val_preds = None
    best_val_targets = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(inputs)
                loss = criterion(preds.view(-1), targets.view(-1).float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            if ema is not None:
                ema.update(model)

        val_preds_raw, val_targets = evaluate(model, val_loader, device)

        if ema is not None:
            val_preds_ema, _ = evaluate(ema.ema_model, val_loader, device)
            rounder_ema = OptimizedQWKRounder()
            rounder_ema.fit(val_preds_ema, val_targets)
            val_qwk_ema = cohen_kappa_score(
                val_targets, rounder_ema.predict(val_preds_ema), weights="quadratic"
            )

            rounder_raw = OptimizedQWKRounder()
            rounder_raw.fit(val_preds_raw, val_targets)
            val_qwk_raw = cohen_kappa_score(
                val_targets, rounder_raw.predict(val_preds_raw), weights="quadratic"
            )

            if val_qwk_ema >= val_qwk_raw:
                epoch_val_qwk = val_qwk_ema
                epoch_val_preds = val_preds_ema
            else:
                epoch_val_qwk = val_qwk_raw
                epoch_val_preds = val_preds_raw
        else:
            rounder_raw = OptimizedQWKRounder()
            rounder_raw.fit(val_preds_raw, val_targets)
            epoch_val_qwk = cohen_kappa_score(
                val_targets, rounder_raw.predict(val_preds_raw), weights="quadratic"
            )
            epoch_val_preds = val_preds_raw

        if epoch_val_qwk > best_qwk:
            best_qwk = epoch_val_qwk
            best_val_preds = epoch_val_preds
            best_val_targets = val_targets

    final_rounder = OptimizedQWKRounder()
    final_rounder.fit(best_val_preds, best_val_targets)
    final_score = cohen_kappa_score(
        best_val_targets,
        final_rounder.predict(best_val_preds),
        weights="quadratic",
    )
    return final_score


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = "./input"
    csv_path = os.path.join(input_dir, "train.csv")
    img_dir = os.path.join(input_dir, "train_images")

    df = pd.read_csv(csv_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df["diagnosis"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    img_size = 384
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = RetinopathyDataset(
        train_df, img_dir, transform=train_transform
    )
    val_dataset = RetinopathyDataset(val_df, img_dir, transform=val_transform)

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    print("=== Running Baseline Model ===")
    baseline_score = run_experiment(
        train_loader,
        val_loader,
        device,
        use_ema=True,
        use_scheduler=True,
        use_mlp_head=True,
    )
    print(f"Baseline Validation QWK: {baseline_score:.4f}\n")

    print("=== Running Ablation 1: Disabled Model EMA ===")
    ablation_ema_score = run_experiment(
        train_loader,
        val_loader,
        device,
        use_ema=False,
        use_scheduler=True,
        use_mlp_head=True,
    )
    delta_ema = ablation_ema_score - baseline_score
    print(
        f"Ablation 1 (No EMA) Validation QWK: {ablation_ema_score:.4f} (Delta: {delta_ema:+.4f})\n"
    )

    print("=== Running Ablation 2: Disabled Warmup & Cosine LR Scheduler (Constant LR) ===")
    ablation_sched_score = run_experiment(
        train_loader,
        val_loader,
        device,
        use_ema=True,
        use_scheduler=False,
        use_mlp_head=True,
    )
    delta_sched = ablation_sched_score - baseline_score
    print(
        f"Ablation 2 (Constant LR) Validation QWK: {ablation_sched_score:.4f} (Delta: {delta_sched:+.4f})\n"
    )

    print("=== Running Ablation 3: Replaced MLP Head with Single Linear Layer ===")
    ablation_head_score = run_experiment(
        train_loader,
        val_loader,
        device,
        use_ema=True,
        use_scheduler=True,
        use_mlp_head=False,
    )
    delta_head = ablation_head_score - baseline_score
    print(
        f"Ablation 3 (Linear Head) Validation QWK: {ablation_head_score:.4f} (Delta: {delta_head:+.4f})\n"
    )

    drops = {
        "Model EMA (Exponential Moving Average)": abs(delta_ema),
        "Warmup & Cosine Annealing LR Scheduler": abs(delta_sched),
        "Multi-Layer MLP Classification Head with Regularization": abs(delta_head),
    }

    most_impactful = max(drops, key=drops.get)
    max_drop = drops[most_impactful]

    print("=== Summary of Ablation Results ===")
    print(f"Baseline QWK: {baseline_score:.4f}")
    print(f"Without Model EMA: {ablation_ema_score:.4f} (Impact: -{drops['Model EMA (Exponential Moving Average)']:.4f})")
    print(f"Without LR Scheduler: {ablation_sched_score:.4f} (Impact: -{drops['Warmup & Cosine Annealing LR Scheduler']:.4f})")
    print(f"Without MLP Head (Single Linear): {ablation_head_score:.4f} (Impact: -{drops['Multi-Layer MLP Classification Head with Regularization']:.4f})")
    print(
        f"\nConclusion: '{most_impactful}' contributes the most to the overall performance with a performance decrease of {max_drop:.4f} QWK when removed."
    )


if __name__ == "__main__":
    main()
