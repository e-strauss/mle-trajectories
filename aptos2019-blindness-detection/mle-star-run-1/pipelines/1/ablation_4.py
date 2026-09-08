
import copy
from concurrent.futures import ThreadPoolExecutor
import os
import random
import cv2
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


def crop_image_from_gray(img, tol=7):
  if img.ndim == 2:
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]
  elif img.ndim == 3:
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
      return img
    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


def ben_graham_preprocessing(img_rgb, sigma_x=30):
  img = crop_image_from_gray(img_rgb)
  h, w, _ = img.shape
  blur = cv2.GaussianBlur(img, (0, 0), sigma_x)
  enhanced = cv2.addWeighted(img, 4, blur, -4, 128)
  mask = np.zeros((h, w), dtype=np.uint8)
  center = (int(w / 2), int(h / 2))
  radius = int(min(h, w) * 0.48)
  cv2.circle(mask, center, radius, 1, -1)
  enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
  return enhanced


def preprocess_single_image(args):
  img_id, img_path = args
  image = cv2.imread(img_path)
  if image is not None:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  else:
    image = np.array(Image.open(img_path).convert("RGB"))

  raw_cropped = crop_image_from_gray(image)
  raw_resized = cv2.resize(
      raw_cropped, (384, 384), interpolation=cv2.INTER_AREA
  )

  ben_img = ben_graham_preprocessing(image, sigma_x=30)
  ben_resized = cv2.resize(ben_img, (384, 384), interpolation=cv2.INTER_AREA)

  return img_id, raw_resized, ben_resized


class RetinopathyPreloadedDataset(Dataset):

  def __init__(self, df, preprocessed_cache, transform=None, ben_graham=True):
    self.df = df.reset_index(drop=True)
    self.cache = preprocessed_cache
    self.transform = transform
    self.ben_graham = ben_graham

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img_id = row["id_code"]
    raw_img, ben_img = self.cache[img_id]

    image_arr = ben_img if self.ben_graham else raw_img
    image = Image.fromarray(image_arr)
    if self.transform:
      image = self.transform(image)

    target = torch.tensor(row["diagnosis"], dtype=torch.float32)
    return image, target


class ConvNeXtDR(nn.Module):

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


def evaluate(model, dataloader, device, use_tta=True):
  model.eval()
  all_preds = []
  all_targets = []
  device_type = "cuda" if device.type == "cuda" else "cpu"

  with torch.no_grad():
    for images, targets in dataloader:
      images = images.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)

      if use_tta:
        tta_images = [
            images,
            torch.flip(images, dims=[-1]),
            torch.flip(images, dims=[-2]),
        ]
        batch_preds = []
        for aug_img in tta_images:
          with torch.amp.autocast(
              device_type=device_type, enabled=(device_type == "cuda")
          ):
            pred = model(aug_img)
          batch_preds.append(pred.view(-1))
        avg_preds = torch.stack(batch_preds, dim=0).mean(dim=0)
      else:
        with torch.amp.autocast(
            device_type=device_type, enabled=(device_type == "cuda")
        ):
          avg_preds = model(images).view(-1)

      all_preds.append(avg_preds)
      all_targets.append(targets.view(-1))

  all_preds = torch.cat(all_preds, dim=0)
  all_targets = torch.cat(all_targets, dim=0)
  all_preds = torch.clamp(
      all_preds, min=all_targets.min(), max=all_targets.max()
  )
  return all_preds.cpu().numpy(), all_targets.cpu().numpy()


def run_training_experiment(
    train_df,
    val_df,
    preprocessed_cache,
    device,
    use_ben_graham=True,
    use_color_jitter=True,
    use_grad_clip=True,
    epochs=5,
    batch_size=32,
):
  seed_everything(42)

  transform_list = [
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(15),
  ]
  if use_color_jitter:
    transform_list.append(
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
    )

  transform_list.extend([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])
  train_transform = transforms.Compose(transform_list)

  val_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  train_dataset = RetinopathyPreloadedDataset(
      train_df,
      preprocessed_cache,
      transform=train_transform,
      ben_graham=use_ben_graham,
  )
  val_dataset = RetinopathyPreloadedDataset(
      val_df,
      preprocessed_cache,
      transform=val_transform,
      ben_graham=use_ben_graham,
  )

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

  model = ConvNeXtDR(
      model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
      pretrained=True,
      dropout_rate=0.3,
  ).to(device)

  criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
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
  best_preds, best_targets = None, None

  for epoch in range(epochs):
    model.train()
    for images, targets in train_loader:
      images = images.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)
      optimizer.zero_grad()

      with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        preds = model(images)
        loss = criterion(preds.view(-1), targets.view(-1).float())

      scaler.scale(loss).backward()
      if use_grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      ema.update(model)

    val_preds_ema, val_targets = evaluate(
        ema.ema_model, val_loader, device, use_tta=(epoch == epochs - 1)
    )

    rounder = OptimizedQWKRounder()
    rounder.fit(val_preds_ema, val_targets)
    val_qwk = cohen_kappa_score(
        val_targets, rounder.predict(val_preds_ema), weights="quadratic"
    )

    if val_qwk > best_qwk:
      best_qwk = val_qwk
      best_preds = val_preds_ema
      best_targets = val_targets

  final_rounder = OptimizedQWKRounder()
  final_rounder.fit(best_preds, best_targets)
  final_qwk = cohen_kappa_score(
      best_targets, final_rounder.predict(best_preds), weights="quadratic"
  )
  return final_qwk


def main():
  seed_everything(42)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  input_dir = "./input"
  csv_path = os.path.join(input_dir, "train.csv")
  img_dir = os.path.join(input_dir, "train_images")
  df = pd.read_csv(csv_path)

  print(f"Pre-caching and resizing {len(df)} images into RAM memory...")
  tasks = []
  for _, row in df.iterrows():
    img_id = row["id_code"]
    img_path = os.path.join(img_dir, f"{img_id}.png")
    tasks.append((img_id, img_path))

  preprocessed_cache = {}
  with ThreadPoolExecutor(max_workers=32) as executor:
    results = executor.map(preprocess_single_image, tasks)
    for img_id, raw_img, ben_img in results:
      preprocessed_cache[img_id] = (raw_img, ben_img)
  print("Pre-caching completed successfully.")

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  train_idx, val_idx = next(skf.split(df, df["diagnosis"]))
  train_df = df.iloc[train_idx].reset_index(drop=True)
  val_df = df.iloc[val_idx].reset_index(drop=True)

  print("=== Starting Ablation Study on Training & Preprocessing Pipeline ===")

  print("\nRunning Baseline (Full Pipeline: Ben Graham + ColorJitter + GradClip)...")
  baseline_score = run_training_experiment(
      train_df,
      val_df,
      preprocessed_cache,
      device,
      use_ben_graham=True,
      use_color_jitter=True,
      use_grad_clip=True,
  )
  print(f"Baseline Validation QWK: {baseline_score:.5f}")

  print("\nRunning Ablation 1: Disabled Ben Graham Preprocessing (Raw Images)...")
  ablation_1_score = run_training_experiment(
      train_df,
      val_df,
      preprocessed_cache,
      device,
      use_ben_graham=False,
      use_color_jitter=True,
      use_grad_clip=True,
  )
  print(f"Ablation 1 (No Ben Graham) Validation QWK: {ablation_1_score:.5f}")

  print("\nRunning Ablation 2: Disabled ColorJitter Photometric Augmentation...")
  ablation_2_score = run_training_experiment(
      train_df,
      val_df,
      preprocessed_cache,
      device,
      use_ben_graham=True,
      use_color_jitter=False,
      use_grad_clip=True,
  )
  print(f"Ablation 2 (No ColorJitter) Validation QWK: {ablation_2_score:.5f}")

  print("\nRunning Ablation 3: Disabled Gradient Norm Clipping...")
  ablation_3_score = run_training_experiment(
      train_df,
      val_df,
      preprocessed_cache,
      device,
      use_ben_graham=True,
      use_color_jitter=True,
      use_grad_clip=False,
  )
  print(f"Ablation 3 (No GradClip) Validation QWK: {ablation_3_score:.5f}")

  print("\n" + "=" * 60)
  print("ABLATION STUDY SUMMARY RESULTS")
  print("=" * 60)
  results = {
      "Baseline (Full Pipeline)": baseline_score,
      "Ablation 1 (No Ben Graham Preprocessing)": ablation_1_score,
      "Ablation 2 (No ColorJitter Augmentation)": ablation_2_score,
      "Ablation 3 (No Gradient Clipping)": ablation_3_score,
  }

  drops = {}
  for name, score in results.items():
    if name == "Baseline (Full Pipeline)":
      print(f"{name:<45}: {score:.5f} (Baseline)")
    else:
      delta = score - baseline_score
      drop = baseline_score - score
      drops[name] = drop
      print(f"{name:<45}: {score:.5f} (Delta: {delta:+.5f})")

  most_critical_ablation = max(drops, key=drops.get)
  max_drop = drops[most_critical_ablation]

  print("-" * 60)
  print(
      f"Most impactful component: {most_critical_ablation} (Removal caused a performance drop of {max_drop:.5f} QWK)"
  )
  print("=" * 60)

  print(f"Final Validation Performance: {baseline_score:.5f}")


if __name__ == "__main__":
  main()
