
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
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

        target = torch.tensor(row['diagnosis'], dtype=torch.long)
        return image, target

class CoralLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) Loss for Ordinal Classification.
    Converts a K-class ordinal problem into K-1 binary classification subtasks
    with shared weights and independent class thresholds to guarantee monotonicity.
    """
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
    """
    EVA-02 Large Patch14 448 vision transformer with an ordinal CORAL head.
    Captures multi-scale diabetic retinopathy micro-lesions via high-res attention.
    """
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

    def predict_class(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > 0.5).sum(dim=1)

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
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

def evaluate(model, dataloader, device):
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
                preds = (probs > 0.5).sum(dim=1)

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

    img_size = 448
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

    model = EVA02CoralModel(model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=5)
    model = model.to(device)

    criterion = CoralLoss(num_classes=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_qwk = -1.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        val_preds, val_targets = evaluate(model, val_loader, device)
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights='quadratic')

        if val_qwk > best_qwk:
            best_qwk = val_qwk

    final_validation_score = best_qwk
    print(f'Final Validation Performance: {final_validation_score}')

if __name__ == '__main__':
    main()
