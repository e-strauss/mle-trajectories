
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

  def __init__(self, df, img_dir, transform=None, crop_borders=True):
    self.df = df.reset_index(drop=True)
    self.img_dir = img_dir
    self.transform = transform
    self.crop_borders = crop_borders

  def _crop_image_from_gray(self, img, tol=7):
    img_np = np.array(img)
    if img_np.ndim == 2:
      mask = img_np > tol
      if not mask.any():
        return img
      return Image.fromarray(img_np[np.ix_(mask.any(1), mask.any(0))])
    elif img_np.ndim == 3:
      gray = np.mean(img_np, axis=2)
      mask = gray > tol
      if not mask.any():
        return img
      y_indices = np.where(mask.any(axis=1))[0]
      x_indices = np.where(mask.any(axis=0))[0]
      if len(y_indices) == 0 or len(x_indices) == 0:
        return img
      ymin, ymax = y_indices[0], y_indices[-1] + 1
      xmin, xmax = x_indices[0], x_indices[-1] + 1
      return Image.fromarray(img_np[ymin:ymax, xmin:xmax])
    return img

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img_name = f"{row['id_code']}.png"
    img_path = os.path.join(self.img_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    if self.crop_borders:
      image = self._crop_image_from_gray(image)

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
