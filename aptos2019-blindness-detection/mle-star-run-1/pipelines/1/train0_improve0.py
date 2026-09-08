
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
    ConvNeXt-V2 Large configured as a continuous regression model for DR grading.
    Treating 5 ordinal grades as continuous targets [0, 4] with MSE/SmoothL1 loss
    consistently outperforms multi-class cross-entropy for Quadratic Weighted Kappa.
    """
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

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
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

def evaluate(model, dataloader, device):
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

    model = ConvNeXtV2ForDR(model_name='convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, dropout_rate=0.3)
    model = model.to(device)


    import numpy as np

    criterion = nn.SmoothL1Loss(beta=0.5)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name.lower() for h in ['head', 'fc', 'classifier']):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 1e-2},
        {'params': head_params, 'lr': 2e-4, 'weight_decay': 1e-2}
    ])

    epochs = 6
    warmup_epochs = 1
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    def evaluate_with_tta(eval_model, loader, dev):
        eval_model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(dev)
                with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                    p_orig = eval_model(images)
                    p_hflip = eval_model(torch.flip(images, dims=[-1]))
                    p_vflip = eval_model(torch.flip(images, dims=[-2]))
                    p_hvflip = eval_model(torch.flip(images, dims=[-2, -1]))
                    pred = (p_orig + p_hflip + p_vflip + p_hvflip) / 4.0
                preds_all.append(pred.view(-1).cpu().numpy())
                targets_all.append(targets.view(-1).cpu().numpy())
        return np.concatenate(preds_all), np.concatenate(targets_all)

    best_qwk = -1.0
    best_val_preds = None
    best_val_targets = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        val_preds, val_targets = evaluate_with_tta(model, val_loader, device)

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
