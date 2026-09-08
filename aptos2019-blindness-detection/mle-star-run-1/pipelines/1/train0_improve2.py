
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        target = torch.tensor(row['diagnosis'], dtype=torch.float32)
        return image, target

class ConvNeXtV2ForDR(nn.Module):
    """
    ConvNeXt-V2 Large configured for ordinal DR grading (4 binary threshold outputs).
    """
    def __init__(self, model_name='convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, dropout_rate=0.3, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

class OptimizedQWKRounder:
    """Optimizes decision boundaries [t1, t2, t3, t4] to maximize QWK on validation predictions."""
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _loss(self, coef, x, y):
        x_discrete = np.digitize(x, coef)
        return -cohen_kappa_score(y, x_discrete, weights='quadratic')

    def fit(self, x, y):
        res = minimize(self._loss, self.coef_, args=(x, y), method='Nelder-Mead')
        self.coef_ = sorted(res.x)

    def predict(self, x):
        return np.digitize(x, self.coef_)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        thresholds = torch.arange(4, device=device).unsqueeze(0)
        ordinal_targets = (targets.unsqueeze(1) > thresholds).float()

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, ordinal_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                preds = torch.sigmoid(logits).sum(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets))
    return np.concatenate(all_preds), np.concatenate(all_targets)

def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dir = './input'
    csv_path = os.path.join(input_dir, 'train.csv')
    img_dir = os.path.join(input_dir, 'train_images')

    df = pd.read_csv(csv_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df['diagnosis']))

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RetinopathyDataset(train_df, img_dir, transform=train_transform)
    val_dataset = RetinopathyDataset(val_df, img_dir, transform=val_transform)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = ConvNeXtV2ForDR(model_name='convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, dropout_rate=0.3, num_classes=4)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_qwk = -1.0
    best_val_preds = None
    best_val_targets = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        val_preds, val_targets = evaluate(model, val_loader, device)

        rounder = OptimizedQWKRounder()
        rounder.fit(val_preds, val_targets)
        val_discrete = rounder.predict(val_preds)
        val_qwk = cohen_kappa_score(val_targets, val_discrete, weights='quadratic')

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_val_preds = val_preds
            best_val_targets = val_targets

    final_rounder = OptimizedQWKRounder()
    final_rounder.fit(best_val_preds, best_val_targets)
    final_preds_discrete = final_rounder.predict(best_val_preds)
    final_validation_score = cohen_kappa_score(best_val_targets, final_preds_discrete, weights='quadratic')

    print(f'Final Validation Performance: {final_validation_score}')

if __name__ == '__main__':
    main()
