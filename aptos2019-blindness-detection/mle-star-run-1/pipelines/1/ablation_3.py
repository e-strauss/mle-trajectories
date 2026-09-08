
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
      use_multi_stage=True,
      stage_indices=(1, 2, 3),
  ):
    super().__init__()
    self.use_multi_stage = use_multi_stage
    self.stage_indices = stage_indices

    self.backbone = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0
    )

    if self.use_multi_stage:
      if hasattr(self.backbone, "feature_info"):
        fi = self.backbone.feature_info
        if hasattr(fi, "channels") and callable(fi.channels):
          feature_channels = fi.channels()
        elif isinstance(fi, list):
          feature_channels = [
              f["num_chs"]
              if isinstance(f, dict)
              else getattr(f, "num_chs", f)
              for f in fi
          ]
        else:
          feature_channels = None
      else:
        feature_channels = None

      if feature_channels is not None:
        self.stage_channels = [feature_channels[i] for i in self.stage_indices]
      else:
        with torch.no_grad():
          _, intermediates = self.backbone.forward_intermediates(
              torch.zeros(1, 3, 224, 224), indices=self.stage_indices
          )
          self.stage_channels = [feat.shape[1] for feat in intermediates]

      self.pool = nn.AdaptiveAvgPool2d(1)
      self.norms = nn.ModuleList(
          [nn.LayerNorm(c) for c in self.stage_channels]
      )
      total_in_features = sum(self.stage_channels)
    else:
      total_in_features = self.backbone.num_features

    self.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(total_in_features, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(dropout_rate / 2 if dropout_rate > 0 else 0.0),
        nn.Linear(256, 1),
    )

  def forward(self, x):
    if self.use_multi_stage:
      _, intermediates = self.backbone.forward_intermediates(
          x, indices=self.stage_indices
      )
      pooled_feats = [
          norm(self.pool(feat).flatten(1))
          for feat, norm in zip(intermediates, self.norms)
      ]
      fused = torch.cat(pooled_feats, dim=1)
      out = self.head(fused)
    else:
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
    exp_name,
    train_df,
    val_df,
    img_dir,
    device,
    use_multi_stage=True,
    dropout_rate=0.3,
    weight_decay=1e-2,
    epochs=6,
    batch_size=16,
    img_size=384,
):
  seed_everything(42)

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
      dropout_rate=dropout_rate,
      use_multi_stage=use_multi_stage,
  )
  model = model.to(device)

  criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=1e-4, weight_decay=weight_decay
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
        preds = model(inputs)
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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  input_dir = "./input"
  csv_path = os.path.join(input_dir, "train.csv")
  img_dir = os.path.join(input_dir, "train_images")

  df = pd.read_csv(csv_path)

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  train_idx, val_idx = next(skf.split(df, df["diagnosis"]))

  train_df = df.iloc[train_idx].reset_index(drop=True)
  val_df = df.iloc[val_idx].reset_index(drop=True)

  results = {}

  print("Running Baseline (Multi-Stage Feature Fusion + Dropout 0.3 + Weight Decay 1e-2)...")
  baseline_score = run_experiment(
      "Baseline",
      train_df,
      val_df,
      img_dir,
      device,
      use_multi_stage=True,
      dropout_rate=0.3,
      weight_decay=1e-2,
  )
  results["Baseline (Full Pipeline)"] = baseline_score
  print(f"Baseline Validation QWK: {baseline_score:.5f}")

  print("\nRunning Ablation 1: Single-Stage Features (No Multi-Stage Intermediate Feature Fusion)...")
  score_abl1 = run_experiment(
      "Ablation 1 (No Multi-Stage Fusion)",
      train_df,
      val_df,
      img_dir,
      device,
      use_multi_stage=False,
      dropout_rate=0.3,
      weight_decay=1e-2,
  )
  results["Ablation 1 (No Multi-Stage Feature Fusion)"] = score_abl1
  print(f"Ablation 1 Validation QWK: {score_abl1:.5f} (Delta: {score_abl1 - baseline_score:+.5f})")

  print("\nRunning Ablation 2: No Dropout Regularization (Dropout Rate = 0.0)...")
  score_abl2 = run_experiment(
      "Ablation 2 (No Dropout)",
      train_df,
      val_df,
      img_dir,
      device,
      use_multi_stage=True,
      dropout_rate=0.0,
      weight_decay=1e-2,
  )
  results["Ablation 2 (No Head Dropout)"] = score_abl2
  print(f"Ablation 2 Validation QWK: {score_abl2:.5f} (Delta: {score_abl2 - baseline_score:+.5f})")

  print("\nRunning Ablation 3: No Optimizer Weight Decay (Weight Decay = 0.0)...")
  score_abl3 = run_experiment(
      "Ablation 3 (No Weight Decay)",
      train_df,
      val_df,
      img_dir,
      device,
      use_multi_stage=True,
      dropout_rate=0.3,
      weight_decay=0.0,
  )
  results["Ablation 3 (No Weight Decay)"] = score_abl3
  print(f"Ablation 3 Validation QWK: {score_abl3:.5f} (Delta: {score_abl3 - baseline_score:+.5f})")

  print("\n========================================================")
  print("                   ABLATION SUMMARY                     ")
  print("========================================================")
  for exp, score in results.items():
    delta = score - baseline_score
    print(f"{exp:48s} : QWK = {score:.5f} (Delta = {delta:+.5f})")

  ablation_drops = {
      "Multi-Stage Feature Fusion": baseline_score - score_abl1,
      "Head Dropout Regularization": baseline_score - score_abl2,
      "AdamW Weight Decay Regularization": baseline_score - score_abl3,
  }

  most_impactful = max(ablation_drops, key=ablation_drops.get)
  print("========================================================")
  print(f"Conclusion: '{most_impactful}' contributes the most to overall performance with a drop of {ablation_drops[most_impactful]:.5f} QWK when disabled.")
  print("========================================================")


if __name__ == "__main__":
  main()
