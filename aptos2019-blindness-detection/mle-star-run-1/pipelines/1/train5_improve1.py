
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
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, KFold


class OptimizedQWKRounder:

    def __init__(
        self,
        initial_coef=None,
        n_splits=5,
        max_iter=25,
        min_margin=0.05,
        n_bracket_points=200,
        random_state=42,
    ):
        self.initial_coef = (
            list(initial_coef)
            if initial_coef is not None
            else [0.5, 1.5, 2.5, 3.5]
        )
        self.coef_ = list(self.initial_coef)
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.min_margin = min_margin
        self.n_bracket_points = n_bracket_points
        self.random_state = random_state

    def _score(self, coef, x, y):
        x_discrete = np.digitize(x, coef)
        return cohen_kappa_score(y, x_discrete, weights="quadratic")

    def _optimize_thresholds(self, x, y, init_coef):
        coef = np.array(sorted(init_coef), dtype=float)
        n_cuts = len(coef)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        best_score = self._score(coef, x, y)

        for _ in range(self.max_iter):
            improved = False
            for i in range(n_cuts):
                lower_bound = (
                    coef[i - 1] + self.min_margin
                    if i > 0
                    else x_min - self.min_margin
                )
                upper_bound = (
                    coef[i + 1] - self.min_margin
                    if i < n_cuts - 1
                    else x_max + self.min_margin
                )

                if lower_bound >= upper_bound:
                    continue

                candidates = np.linspace(
                    lower_bound, upper_bound, self.n_bracket_points
                )
                best_cut = coef[i]
                best_cut_score = best_score

                temp_coef = coef.copy()
                for cand in candidates:
                    temp_coef[i] = cand
                    score = self._score(temp_coef, x, y)
                    if score > best_cut_score:
                        best_cut_score = score
                        best_cut = cand

                if best_cut_score > best_score:
                    coef[i] = best_cut
                    best_score = best_cut_score
                    improved = True

            if not improved:
                break

        return np.sort(coef)

    def fit(self, x, y):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if self.n_splits is not None and self.n_splits > 1:
            try:
                cv = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                split_generator = cv.split(x, y)
            except ValueError:
                cv = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                split_generator = cv.split(x)

            fold_coefs = []
            for train_idx, _ in split_generator:
                x_train, y_train = x[train_idx], y[train_idx]
                fold_coef = self._optimize_thresholds(
                    x_train, y_train, self.initial_coef
                )
                fold_coefs.append(fold_coef)

            self.coef_ = list(np.mean(fold_coefs, axis=0))
        else:
            self.coef_ = list(
                self._optimize_thresholds(x, y, self.initial_coef)
            )

        # Enforce minimum margin between cuts post-averaging
        self.coef_ = list(np.sort(self.coef_))
        for i in range(1, len(self.coef_)):
            if self.coef_[i] < self.coef_[i - 1] + self.min_margin:
                self.coef_[i] = self.coef_[i - 1] + self.min_margin

        return self

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
