
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



import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, KFold

class OptimizedQWKRounder:
    """
    Regularized, monotonic threshold optimizer with internal K-fold bagging
    to prevent cutoff overfitting on out-of-fold validation sets.
    """
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
        
        # Unconstrained optimization of monotonic parameterization
        res = minimize(
            self._loss,
            init_params,
            args=(x_train, y_train, baseline_thresholds),
            method="Powell",
            options={"maxiter": 200, "ftol": 1e-5}
        )
        
        opt_thresholds = self._params_to_thresholds(res.x)
        
        # Fast coordinate refinement
        refined_thresholds = opt_thresholds.copy()
        for idx in range(len(refined_thresholds)):
            best_val = refined_thresholds[idx]
            best_score = self._loss(self._thresholds_to_params(refined_thresholds), x_train, y_train, baseline_thresholds)
            
            lower_bound = refined_thresholds[idx - 1] + 1e-4 if idx > 0 else refined_thresholds[idx] - 1.0
            upper_bound = refined_thresholds[idx + 1] - 1e-4 if idx < len(refined_thresholds) - 1 else refined_thresholds[idx] + 1.0
            
            grid = np.linspace(lower_bound, upper_bound, 30)
            for val in grid:
                cand = refined_thresholds.copy()
                cand[idx] = val
                cand_score = self._loss(self._thresholds_to_params(cand), x_train, y_train, baseline_thresholds)
                if cand_score < best_score:
                    best_score = cand_score
                    best_val = val
            refined_thresholds[idx] = best_val
            
        return refined_thresholds

    def fit(self, x, y):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        
        unique_y = np.unique(y)
        n_classes = len(unique_y)
        num_thresholds = n_classes - 1 if n_classes > 1 else len(self.coef_)

        # Compute empirical target quantile baseline cutoffs
        class_counts = np.bincount(y.astype(int))
        cum_probs = np.cumsum(class_counts[:-1]) / len(y)
        cum_probs = np.clip(cum_probs, 1e-4, 1.0 - 1e-4)
        baseline_thresholds = np.quantile(x, cum_probs)

        if len(baseline_thresholds) != num_thresholds:
            baseline_thresholds = np.linspace(np.min(x) + 0.1, np.max(x) - 0.1, num_thresholds)

        initial_thresholds = baseline_thresholds.copy()

        # Internal K-fold threshold bagging
        try:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(x, y))
        except Exception:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(x))

        fold_thresholds = []
        for train_idx, val_idx in splits:
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

      # 4-fold Test-Time Augmentation (TTA) transforms
      tta_images = [
          images,  # Identity
          torch.flip(images, dims=[-1]),  # Horizontal flip
          torch.flip(images, dims=[-2]),  # Vertical flip
          torch.flip(images, dims=[-2, -1]),  # Horizontal-Vertical flip
      ]

      batch_preds = []
      for aug_img in tta_images:
        with torch.amp.autocast(
            device_type=device_type, enabled=(device_type == "cuda")
        ):
          pred = model(aug_img)
        batch_preds.append(pred.view(-1))

      # Average predictions across all augmented views in output space
      avg_preds = torch.stack(batch_preds, dim=0).mean(dim=0)

      # Accumulate natively on GPU
      all_preds.append(avg_preds)
      all_targets.append(targets.view(-1))

  # Concatenate tensors on GPU
  all_preds = torch.cat(all_preds, dim=0)
  all_targets = torch.cat(all_targets, dim=0)

  # Post-process: clamp predictions strictly within known target range
  min_target = all_targets.min()
  max_target = all_targets.max()
  all_preds = torch.clamp(all_preds, min=min_target, max=max_target)

  # Single unified transfer to host memory
  return all_preds.cpu().numpy(), all_targets.cpu().numpy()



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

  model = ConvNeXtV2ForDR(
      model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
      pretrained=True,
      dropout_rate=0.3,
  )
  model = model.to(device)

  criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
  epochs = 6
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
      if isinstance(batch, (tuple, list)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
          outputs = model(inputs)
          preds = outputs.logits if hasattr(outputs, "logits") else outputs
          loss = criterion(preds.view(-1), targets.view(-1).float())
      elif isinstance(batch, dict):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        targets = batch.get(
            "labels", batch.get("target", batch.get("targets"))
        )
        model_inputs = {
            k: v
            for k, v in batch.items()
            if k not in ["labels", "target", "targets"]
        }
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
          outputs = model(**model_inputs)
          preds = outputs.logits if hasattr(outputs, "logits") else outputs
          loss = criterion(preds.view(-1), targets.view(-1).float())
      else:
        inputs, targets = batch.to(device), batch.to(device)
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
  final_validation_score = cohen_kappa_score(
      best_val_targets, final_preds_discrete, weights="quadratic"
  )

  print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
  main()
