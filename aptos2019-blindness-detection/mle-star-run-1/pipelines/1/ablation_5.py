
import copy
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
      head_type="gelu_layernorm",
  ):
    super().__init__()
    self.backbone = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0
    )
    in_features = self.backbone.num_features

    if head_type == "gelu_layernorm":
      self.head = nn.Sequential(
          nn.Dropout(dropout_rate),
          nn.Linear(in_features, 256),
          nn.GELU(),
          nn.LayerNorm(256),
          nn.Dropout(dropout_rate / 2),
          nn.Linear(256, 1),
      )
    elif head_type == "relu_nonorm":
      self.head = nn.Sequential(
          nn.Dropout(dropout_rate),
          nn.Linear(in_features, 256),
          nn.ReLU(),
          nn.Dropout(dropout_rate / 2),
          nn.Linear(256, 1),
      )
    else:
      raise ValueError(f"Unsupported head_type: {head_type}")

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
      targets = targets.to(device, non_blocking=True)

      tta_images = [
          images,
          torch.flip(images, dims=[-1]),
          torch.flip(images, dims=[-2]),
          torch.flip(images, dims=[-2, -1]),
      ]

      batch_preds = []
      for aug_img in tta_images:
        with torch.amp.autocast(
            device_type=device_type, enabled=(device_type == "cuda")
        ):
          pred = model(aug_img)
        batch_preds.append(pred.view(-1))

      avg_preds = torch.stack(batch_preds, dim=0).mean(dim=0)
      all_preds.append(avg_preds)
      all_targets.append(targets.view(-1))

  all_preds = torch.cat(all_preds, dim=0)
  all_targets = torch.cat(all_targets, dim=0)

  min_target = all_targets.min()
  max_target = all_targets.max()
  all_preds = torch.clamp(all_preds, min=min_target, max=max_target)

  return all_preds.cpu().numpy(), all_targets.cpu().numpy()


def run_experiment(
    train_df,
    val_df,
    img_dir,
    device,
    exp_name="Baseline",
    rotation_degrees=180,
    head_type="gelu_layernorm",
    differential_lr=False,
    epochs=5,
    batch_size=16,
):
  seed_everything(42)
  img_size = 384

  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(
          (img_size, img_size), scale=(0.85, 1.0), ratio=(0.95, 1.05)
      ),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(degrees=rotation_degrees),
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

  model = ConvNeXtV2ForDR(
      model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
      pretrained=True,
      dropout_rate=0.3,
      head_type=head_type,
  )
  model = model.to(device)

  criterion = nn.MSELoss()

  if differential_lr:
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": 5e-5},
        {"params": model.head.parameters(), "lr": 5e-4},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, weight_decay=1e-2
    )
  else:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-2
    )

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
  scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
  ema = ModelEMA(model, decay=0.995)

  best_qwk = -1.0
  best_val_preds = None
  best_val_targets = None

  for epoch in range(epochs):
    model.train()
    for batch in train_loader:
      optimizer.zero_grad()
      inputs, targets = batch[0].to(device), batch[1].to(device)
      with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs = model(inputs)
        preds = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(preds.view(-1), targets.view(-1).float())

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      ema.update(model)

    val_preds_raw, val_targets = evaluate(model, val_loader, device)
    val_preds_ema, _ = evaluate(ema.ema_model, val_loader, device)

    rounder_raw = OptimizedQWKRounder()
    rounder_raw.fit(val_preds_raw, val_targets)
    val_discrete_raw = rounder_raw.predict(val_preds_raw)
    val_qwk_raw = cohen_kappa_score(
        val_targets, val_discrete_raw, weights="quadratic"
    )

    rounder_ema = OptimizedQWKRounder()
    rounder_ema.fit(val_preds_ema, val_targets)
    val_discrete_ema = rounder_ema.predict(val_preds_ema)
    val_qwk_ema = cohen_kappa_score(
        val_targets, val_discrete_ema, weights="quadratic"
    )

    if val_qwk_ema >= val_qwk_raw:
      epoch_val_qwk = val_qwk_ema
      epoch_val_preds = val_preds_ema
    else:
      epoch_val_qwk = val_qwk_raw
      epoch_val_preds = val_preds_raw

    if epoch_val_qwk > best_qwk:
      best_qwk = epoch_val_qwk
      best_val_preds = epoch_val_preds
      best_val_targets = val_targets

  final_rounder = OptimizedQWKRounder()
  final_rounder.fit(best_val_preds, best_val_targets)
  final_preds_discrete = final_rounder.predict(best_val_preds)
  final_score = cohen_kappa_score(
      best_val_targets, final_preds_discrete, weights="quadratic"
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

  print("=" * 70)
  print("STARTING ABLATION STUDY ON DIABETIC RETINOPATHY PIPELINE")
  print("=" * 70)

  # 1. Baseline Experiment
  print("\nRunning Baseline (Full Pipeline)...")
  score_baseline = run_experiment(
      train_df,
      val_df,
      img_dir,
      device,
      exp_name="Baseline",
      rotation_degrees=180,
      head_type="gelu_layernorm",
      differential_lr=False,
  )
  print(f"Baseline Validation QWK: {score_baseline:.5f}")

  # 2. Ablation 1: Minor Rotation (15°) instead of Full 180° Invariance
  print("\nRunning Ablation 1: Minor Rotation (15°) instead of 180°...")
  score_ablation1 = run_experiment(
      train_df,
      val_df,
      img_dir,
      device,
      exp_name="Minor Rotation (15°)",
      rotation_degrees=15,
      head_type="gelu_layernorm",
      differential_lr=False,
  )
  print(f"Ablation 1 Validation QWK: {score_ablation1:.5f}")

  # 3. Ablation 2: Simplified MLP Head (ReLU without LayerNorm)
  print(
      "\nRunning Ablation 2: Simplified MLP Head (ReLU without LayerNorm)..."
  )
  score_ablation2 = run_experiment(
      train_df,
      val_df,
      img_dir,
      device,
      exp_name="ReLU without LayerNorm Head",
      rotation_degrees=180,
      head_type="relu_nonorm",
      differential_lr=False,
  )
  print(f"Ablation 2 Validation QWK: {score_ablation2:.5f}")

  # 4. Ablation 3: Differential Learning Rate (Backbone 5e-5 / Head 5e-4)
  print(
      "\nRunning Ablation 3: Differential LR (Backbone 5e-5, Head 5e-4) vs"
      " Uniform LR..."
  )
  score_ablation3 = run_experiment(
      train_df,
      val_df,
      img_dir,
      device,
      exp_name="Differential LR",
      rotation_degrees=180,
      head_type="gelu_layernorm",
      differential_lr=True,
  )
  print(f"Ablation 3 Validation QWK: {score_ablation3:.5f}")

  print("\n" + "=" * 70)
  print("ABLATION STUDY SUMMARY RESULTS")
  print("=" * 70)
  print(f"{'Experiment':<50} | {'Val QWK':<10} | {'Delta':<10}")
  print("-" * 75)
  print(f"{'Baseline (Full Pipeline)':<50} | {score_baseline:<10.5f} | {'-':<10}")
  print(
      f"{'Ablation 1 (15° Rotation instead of 180°)':<50} |"
      f" {score_ablation1:<10.5f} | {score_ablation1 - score_baseline:<+10.5f}"
  )
  print(
      f"{'Ablation 2 (ReLU Head without LayerNorm)':<50} |"
      f" {score_ablation2:<10.5f} | {score_ablation2 - score_baseline:<+10.5f}"
  )
  print(
      f"{'Ablation 3 (Differential LR: Backbone 5e-5, Head 5e-4)':<50} |"
      f" {score_ablation3:<10.5f} | {score_ablation3 - score_baseline:<+10.5f}"
  )
  print("-" * 75)

  drops = {
      "Full 180-Degree Rotational Invariance": score_baseline - score_ablation1,
      "GELU + LayerNorm Head Normalization": score_baseline - score_ablation2,
      "Uniform vs Differential Learning Rate Tuning": abs(
          score_ablation3 - score_baseline
      ),
  }

  most_critical_part = max(drops, key=drops.get)
  print(
      f"\nMost influential pipeline component evaluated: {most_critical_part}"
      f" with an impact magnitude of {drops[most_critical_part]:.5f} QWK."
  )


if __name__ == "__main__":
  main()
