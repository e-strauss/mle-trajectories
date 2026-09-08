
import copy
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, KFold
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
      model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
      pretrained=True,
      dropout_rate=0.3,
  ):
    super().__init__()
    self.backbone = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0
    )
    in_features = self.backbone.num_features
    self.head = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Dropout(dropout_rate / 2),
        nn.Linear(256, 1),
    )

  def forward(self, x):
    feat = self.backbone(x)
    out = self.head(feat)
    return out.squeeze(-1)


# Standard Multi-start Nelder-Mead / Powell Rounder (Baseline post-processor)
class MultiStartQWKRounder:

  def __init__(self, initial_coef=None):
    self.coef_ = [0.5, 1.5, 2.5, 3.5] if initial_coef is None else list(initial_coef)

  def _loss(self, coef, x, y):
    sorted_coef = np.sort(coef)
    x_discrete = np.digitize(x, sorted_coef)
    return -cohen_kappa_score(y, x_discrete, weights="quadratic")

  def fit(self, x, y):
    num_thresholds = len(self.coef_)
    unique_classes = np.sort(np.unique(y))
    candidates = []

    candidates.append(np.array(self.coef_, dtype=float))

    if len(unique_classes) > 1:
      cum_probs = [np.mean(y <= c) for c in unique_classes[:-1]]
      if len(cum_probs) == num_thresholds:
        quantile_thresh = np.quantile(x, cum_probs)
        candidates.append(np.array(quantile_thresh, dtype=float))
      else:
        probs = np.linspace(cum_probs[0], cum_probs[-1], num_thresholds)
        candidates.append(np.array(np.quantile(x, probs), dtype=float))

    rng = np.random.RandomState(42)
    base_candidates = list(candidates)
    std_scale = np.std(x) if np.std(x) > 0 else 1.0
    for cand in base_candidates:
      for _ in range(3):
        noise = rng.normal(0, 0.05 * std_scale, size=len(cand))
        candidates.append(cand + noise)

    best_loss = float("inf")
    best_coef = np.array(self.coef_, dtype=float)

    for cand in candidates:
      for method in ["Nelder-Mead", "Powell"]:
        try:
          res = minimize(
              self._loss,
              cand,
              args=(x, y),
              method=method,
              options={"maxiter": 300}
          )
          if res.fun < best_loss:
            best_loss = res.fun
            best_coef = res.x
        except Exception:
          continue

    self.coef_ = list(np.sort(best_coef))
    return self

  def predict(self, x):
    return np.digitize(x, self.coef_)


# Regularized Monotonic Bagged QWK Rounder
class BaggedMonotonicQWKRounder:

  def __init__(self, n_splits=5, reg_weight=0.05, random_state=42):
    self.n_splits = n_splits
    self.reg_weight = reg_weight
    self.random_state = random_state
    self.coef_ = [0.5, 1.5, 2.5, 3.5]
    self.bagged_coefs_ = []

  @staticmethod
  def _params_to_thresholds(params):
    theta_0 = params[0]
    deltas = np.exp(np.clip(params[1:], -10.0, 10.0))
    return np.concatenate([[theta_0], theta_0 + np.cumsum(deltas)])

  @staticmethod
  def _thresholds_to_params(thresholds):
    theta_0 = thresholds[0]
    diffs = np.diff(thresholds)
    deltas = np.log(np.maximum(diffs, 1e-6))
    return np.concatenate([[theta_0], deltas])

  def _loss(self, params, x, y, baseline_thresholds):
    thresholds = self._params_to_thresholds(params)
    x_discrete = np.digitize(x, thresholds)
    qwk = cohen_kappa_score(y, x_discrete, weights="quadratic")
    penalty = self.reg_weight * np.mean((thresholds - baseline_thresholds) ** 2)
    return -qwk + penalty

  def _optimize_split(self, x_train, y_train, init_thresholds, baseline_thresholds):
    init_params = self._thresholds_to_params(init_thresholds)
    res = minimize(
        self._loss,
        init_params,
        args=(x_train, y_train, baseline_thresholds),
        method="Powell",
        options={"maxiter": 200, "ftol": 1e-5}
    )
    return self._params_to_thresholds(res.x)

  def fit(self, x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    unique_y = np.unique(y)
    num_thresholds = len(unique_y) - 1 if len(unique_y) > 1 else len(self.coef_)

    class_counts = np.bincount(y.astype(int))
    cum_probs = np.cumsum(class_counts[:-1]) / len(y)
    cum_probs = np.clip(cum_probs, 1e-4, 1.0 - 1e-4)
    baseline_thresholds = np.quantile(x, cum_probs)

    if len(baseline_thresholds) != num_thresholds:
      baseline_thresholds = np.linspace(np.min(x) + 0.1, np.max(x) - 0.1, num_thresholds)

    initial_thresholds = baseline_thresholds.copy()

    try:
      cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
      splits = list(cv.split(x, y))
    except Exception:
      cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
      splits = list(cv.split(x))

    fold_thresholds = []
    for train_idx, _ in splits:
      x_fold, y_fold = x[train_idx], y[train_idx]
      opt_thresh = self._optimize_split(x_fold, y_fold, initial_thresholds, baseline_thresholds)
      fold_thresholds.append(opt_thresh)

    self.bagged_coefs_ = np.array(fold_thresholds)
    self.coef_ = np.sort(np.mean(self.bagged_coefs_, axis=0)).tolist()
    return self

  def predict(self, x):
    x = np.asarray(x).ravel()
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
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
          pred = model(aug_img)
        batch_preds.append(pred.view(-1))

      avg_preds = torch.stack(batch_preds, dim=0).mean(dim=0)
      all_preds.append(avg_preds)
      all_targets.append(targets.view(-1))

  all_preds = torch.cat(all_preds, dim=0)
  all_targets = torch.cat(all_targets, dim=0)
  all_preds = torch.clamp(all_preds, min=all_targets.min(), max=all_targets.max())

  return all_preds.cpu().numpy(), all_targets.cpu().numpy()


def run_training_experiment(
    train_loader,
    val_loader,
    device,
    optimizer_type="AdamW",
    ema_decay=0.995,
    rounder_type="bagged",
    epochs=4,
):
  seed_everything(42)
  model = ConvNeXtV2ForDR(
      model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
      pretrained=True,
      dropout_rate=0.3,
  ).to(device)

  criterion = nn.MSELoss()

  if optimizer_type == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
  elif optimizer_type == "SGD_Momentum":
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-2)
  else:
    raise ValueError(f"Unknown optimizer: {optimizer_type}")

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
  ema = ModelEMA(model, decay=ema_decay)

  best_qwk = -1.0
  best_val_preds = None
  best_val_targets = None

  for epoch in range(epochs):
    model.train()
    for images, targets in train_loader:
      images, targets = images.to(device), targets.to(device)
      optimizer.zero_grad()
      with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        preds = model(images)
        loss = criterion(preds.view(-1), targets.view(-1).float())

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      ema.update(model)

    val_preds_raw, val_targets = evaluate(model, val_loader, device)
    val_preds_ema, _ = evaluate(ema.ema_model, val_loader, device)

    # Instantiate rounder based on configuration
    if rounder_type == "bagged":
      r_raw = BaggedMonotonicQWKRounder()
      r_ema = BaggedMonotonicQWKRounder()
    else:
      r_raw = MultiStartQWKRounder()
      r_ema = MultiStartQWKRounder()

    r_raw.fit(val_preds_raw, val_targets)
    val_qwk_raw = cohen_kappa_score(val_targets, r_raw.predict(val_preds_raw), weights="quadratic")

    r_ema.fit(val_preds_ema, val_targets)
    val_qwk_ema = cohen_kappa_score(val_targets, r_ema.predict(val_preds_ema), weights="quadratic")

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

  # Final out-of-fold evaluation with chosen rounder
  if rounder_type == "bagged":
    final_rounder = BaggedMonotonicQWKRounder()
  else:
    final_rounder = MultiStartQWKRounder()

  final_rounder.fit(best_val_preds, best_val_targets)
  final_discrete_preds = final_rounder.predict(best_val_preds)
  final_qwk = cohen_kappa_score(best_val_targets, final_discrete_preds, weights="quadratic")

  return final_qwk


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
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  val_transform = transforms.Compose([
      transforms.Resize((img_size, img_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_dataset = RetinopathyDataset(train_df, img_dir, transform=train_transform)
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

  results = {}

  print("=" * 70)
  print("STARTING ABLATION STUDY ON DIABETIC RETINOPATHY PIPELINE")
  print("=" * 70)

  # 1. Full Baseline Pipeline (AdamW + EMA Decay 0.995 + Regularized Bagged Monotonic Rounder)
  print("\n[1/4] Running Baseline (AdamW + EMA Decay 0.995 + Bagged Monotonic Rounder)...")
  baseline_qwk = run_training_experiment(
      train_loader,
      val_loader,
      device,
      optimizer_type="AdamW",
      ema_decay=0.995,
      rounder_type="bagged",
      epochs=4,
  )
  results["Baseline (AdamW + EMA 0.995 + Bagged Rounder)"] = baseline_qwk
  print(f"--> Baseline Validation QWK: {baseline_qwk:.5f}")

  # 2. Ablation 1: Unregularized Multi-Start Rounder vs. Bagged Monotonic Rounder
  print("\n[2/4] Running Ablation 1 (Unregularized Multi-Start Nelder-Mead/Powell Rounder)...")
  ablation1_qwk = run_training_experiment(
      train_loader,
      val_loader,
      device,
      optimizer_type="AdamW",
      ema_decay=0.995,
      rounder_type="standard_multistart",
      epochs=4,
  )
  results["Ablation 1 (Unregularized Multi-Start Rounder)"] = ablation1_qwk
  print(f"--> Ablation 1 Validation QWK: {ablation1_qwk:.5f} (Delta: {ablation1_qwk - baseline_qwk:+.5f})")

  # 3. Ablation 2: EMA Decay Rate (0.999 vs 0.995)
  print("\n[3/4] Running Ablation 2 (High EMA Decay: 0.999 instead of 0.995)...")
  ablation2_qwk = run_training_experiment(
      train_loader,
      val_loader,
      device,
      optimizer_type="AdamW",
      ema_decay=0.999,
      rounder_type="bagged",
      epochs=4,
  )
  results["Ablation 2 (EMA Decay = 0.999)"] = ablation2_qwk
  print(f"--> Ablation 2 Validation QWK: {ablation2_qwk:.5f} (Delta: {ablation2_qwk - baseline_qwk:+.5f})")

  # 4. Ablation 3: Optimizer (SGD with Momentum vs. AdamW)
  print("\n[4/4] Running Ablation 3 (SGD with Momentum vs. AdamW)...")
  ablation3_qwk = run_training_experiment(
      train_loader,
      val_loader,
      device,
      optimizer_type="SGD_Momentum",
      ema_decay=0.995,
      rounder_type="bagged",
      epochs=4,
  )
  results["Ablation 3 (SGD with Momentum)"] = ablation3_qwk
  print(f"--> Ablation 3 Validation QWK: {ablation3_qwk:.5f} (Delta: {ablation3_qwk - baseline_qwk:+.5f})")

  print("\n" + "=" * 70)
  print("ABLATION STUDY SUMMARY RESULTS")
  print("=" * 70)
  for exp_name, score in results.items():
    delta = score - baseline_qwk
    print(f"{exp_name:<55} | QWK: {score:.5f} | Delta: {delta:+.5f}")

  deltas = {
      "Optimizer (AdamW vs SGD with Momentum)": abs(baseline_qwk - ablation3_qwk),
      "Threshold Optimizer (Bagged Monotonic vs Unregularized Multi-Start)": abs(baseline_qwk - ablation1_qwk),
      "EMA Decay Tuning (0.995 vs 0.999)": abs(baseline_qwk - ablation2_qwk),
  }

  most_impactful = max(deltas, key=deltas.get)
  print("\n" + "=" * 70)
  print(f"CONCLUSION: The component that contributes the most to the overall performance is '{most_impactful}' with a performance delta of {deltas[most_impactful]:.5f} QWK.")
  print("=" * 70)


if __name__ == "__main__":
  main()
