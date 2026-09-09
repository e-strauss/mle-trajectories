
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm
import torch
import torch.nn as nn
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

IMG_SIZE = 448
BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5
WEIGHT_DECAY = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
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

# Loss Function
class OrdinalClassificationLoss(nn.Module):
    """Extended Binary Cross Entropy loss for ordinal targets: y > k for k in [0, K-2]"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, num_classes=5):
        levels = torch.arange(num_classes - 1, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        return self.bce(logits, binary_targets)

# Model Definition
class EVA02OrdinalModel(nn.Module):
    def __init__(self, model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes - 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

    def predict_class(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)

def main():
    seed_everything(42)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(TRAIN_CSV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df, df["diagnosis"]):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        break

    train_dataset = RetinopathyDataset(train_df, TRAIN_IMAGES_DIR, transform=train_transform)
    val_dataset = RetinopathyDataset(val_df, TRAIN_IMAGES_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = EVA02OrdinalModel(model_name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=5, pretrained=True).to(DEVICE)
    criterion = OrdinalClassificationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_qwk = -1.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                with torch.cuda.amp.autocast():
                    preds = model.predict_class(images)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(targets.numpy().tolist())

        val_preds = np.array(val_preds, dtype=int)
        val_targets = np.array(val_targets, dtype=int)

        qwk = cohen_kappa_score(val_targets, val_preds, weights="quadratic")

        if qwk > best_qwk:
            best_qwk = qwk

    final_validation_score = best_qwk
    print(f"Final Validation Performance: {final_validation_score}")

if __name__ == "__main__":
    main()
