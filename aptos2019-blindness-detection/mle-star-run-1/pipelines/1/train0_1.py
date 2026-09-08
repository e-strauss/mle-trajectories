
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
    def __init__(self, model_name='convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)

class CoralLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        levels = torch.zeros((targets.size(0), self.num_classes - 1), device=targets.device)
        for i in range(self.num_classes - 1):
            levels[:, i] = (targets > i).float()

        loss = nn.functional.binary_cross_entropy_with_logits(logits, levels)
        return loss

class EVA02CoralModel(nn.Module):
    def __init__(self, model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        self.fc = nn.Linear(in_features, 1, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        feat = self.backbone(x)
        score = self.fc(feat)
        logits = score + self.thresholds
        return logits

class OptimizedQWKRounder:
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

def train_convnext_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            preds = model(images)
            loss = criterion(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate_convnext(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                preds = model(images)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)

def train_eva_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate_eva(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = probs.sum(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_preds), np.array(all_targets)

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

    img_size_cnext = 384
    train_transform_cnext = transforms.Compose([
        transforms.Resize((img_size_cnext, img_size_cnext)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform_cnext = transforms.Compose([
        transforms.Resize((img_size_cnext, img_size_cnext)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_size_eva = 448
    train_transform_eva = transforms.Compose([
        transforms.Resize((img_size_eva, img_size_eva)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform_eva = transforms.Compose([
        transforms.Resize((img_size_eva, img_size_eva)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 16
    train_loader_cnext = DataLoader(RetinopathyDataset(train_df, img_dir, transform=train_transform_cnext), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader_cnext = DataLoader(RetinopathyDataset(val_df, img_dir, transform=val_transform_cnext), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    train_loader_eva = DataLoader(RetinopathyDataset(train_df, img_dir, transform=train_transform_eva), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader_eva = DataLoader(RetinopathyDataset(val_df, img_dir, transform=val_transform_eva), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    # Train Model 1: ConvNeXtV2 Regression
    convnext_model = ConvNeXtV2ForDR(model_name='convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, dropout_rate=0.3).to(device)
    criterion_cnext = nn.SmoothL1Loss(beta=0.5)
    optimizer_cnext = torch.optim.AdamW(convnext_model.parameters(), lr=1e-4, weight_decay=1e-2)
    epochs_cnext = 6
    scheduler_cnext = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnext, T_max=epochs_cnext, eta_min=1e-6)
    scaler_cnext = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_preds_cnext = None
    best_qwk_cnext = -1.0
    val_targets = None

    for epoch in range(epochs_cnext):
        train_loss = train_convnext_epoch(convnext_model, train_loader_cnext, criterion_cnext, optimizer_cnext, scaler_cnext, device)
        scheduler_cnext.step()
        val_preds, targets = evaluate_convnext(convnext_model, val_loader_cnext, device)
        val_targets = targets

        rounder = OptimizedQWKRounder()
        rounder.fit(val_preds, val_targets)
        val_discrete = rounder.predict(val_preds)
        val_qwk = cohen_kappa_score(val_targets, val_discrete, weights='quadratic')

        if val_qwk > best_qwk_cnext:
            best_qwk_cnext = val_qwk
            best_val_preds_cnext = val_preds

    # Train Model 2: EVA02 Coral Model
    eva_model = EVA02CoralModel(model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=5).to(device)
    criterion_eva = CoralLoss(num_classes=5)
    optimizer_eva = torch.optim.AdamW(eva_model.parameters(), lr=5e-5, weight_decay=1e-4)
    epochs_eva = 5
    scheduler_eva = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_eva, T_max=epochs_eva, eta_min=1e-6)
    scaler_eva = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_preds_eva = None
    best_qwk_eva = -1.0

    for epoch in range(epochs_eva):
        train_loss = train_eva_epoch(eva_model, train_loader_eva, criterion_eva, optimizer_eva, scaler_eva, device)
        scheduler_eva.step()
        val_preds, _ = evaluate_eva(eva_model, val_loader_eva, device)

        rounder = OptimizedQWKRounder()
        rounder.fit(val_preds, val_targets)
        val_discrete = rounder.predict(val_preds)
        val_qwk = cohen_kappa_score(val_targets, val_discrete, weights='quadratic')

        if val_qwk > best_qwk_eva:
            best_qwk_eva = val_qwk
            best_val_preds_eva = val_preds

    # Ensemble continuous predictions
    ensemble_val_preds = 0.5 * best_val_preds_cnext + 0.5 * best_val_preds_eva

    final_rounder = OptimizedQWKRounder()
    final_rounder.fit(ensemble_val_preds, val_targets)
    final_preds_discrete = final_rounder.predict(ensemble_val_preds)
    final_validation_score = cohen_kappa_score(val_targets, final_preds_discrete, weights='quadratic')

    print(f'Final Validation Performance: {final_validation_score}')

if __name__ == '__main__':
    main()
