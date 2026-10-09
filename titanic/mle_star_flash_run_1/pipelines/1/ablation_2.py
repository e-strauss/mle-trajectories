import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load data
train_df = pd.read_csv("./input/train.csv")

def prepare_features(df):
    df = df.copy()

    # Standardized Title categories (Mr, Mrs, Miss, Master, Rare)
    raw_title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    df["Title"] = raw_title.map(
        lambda x: title_mapping.get(x, "Rare") if pd.notna(x) else "Missing"
    )

    # Impute missing Age using median grouped by Pclass and Sex
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Sociological interaction feature
    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)

    # Family dynamics and categorization
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    def assign_family_tier(size):
        if size == 1:
            return "IsAlone"
        elif size <= 4:
            return "SmallFamily"
        else:
            return "LargeFamily"

    df["FamilyTier"] = df["FamilySize"].apply(assign_family_tier)

    # Simplified Cabin indicator
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Ticket frequency to identify shared ticket groups
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1)

    # Missing value handling for Fare and Embarked
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]

    return df, cat_cols, num_cols

df_processed, cat_cols, num_cols = prepare_features(train_df)
y = train_df["Survived"].values

# Train/Validation Split
indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost Training
X_cb = df_processed[cat_cols + num_cols].copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)

X_cb_train, X_cb_val = X_cb.iloc[idx_train], X_cb.iloc[idx_val]
y_cb_train, y_cb_val = y[idx_train], y[idx_val]

cb_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=5,
    cat_features=cat_cols,
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
)
cb_model.fit(
    X_cb_train, y_cb_train, eval_set=(X_cb_val, y_cb_val), early_stopping_rounds=50
)
cb_val_probs = cb_model.predict_proba(X_cb_val)[:, 1]

# Neural Network Data Preparation
cat_arrays_train, cat_arrays_val, cat_dims = [], [], []
for col in cat_cols:
    le = LabelEncoder()
    cat_train_encoded = le.fit_transform(df_processed.iloc[idx_train][col].astype(str))
    val_cats = df_processed.iloc[idx_val][col].astype(str).values
    mapping = {val: i for i, val in enumerate(le.classes_)}
    cat_val_encoded = np.array([mapping.get(v, 0) for v in val_cats])

    cat_arrays_train.append(cat_train_encoded)
    cat_arrays_val.append(cat_val_encoded)
    cat_dims.append(len(le.classes_))

X_nn_cat_train = np.stack(cat_arrays_train, axis=1).astype(np.int64)
X_nn_cat_val = np.stack(cat_arrays_val, axis=1).astype(np.int64)

scaler = StandardScaler()
X_nn_num_train = scaler.fit_transform(
    df_processed.iloc[idx_train][num_cols].values.astype(np.float32)
)
X_nn_num_val = scaler.transform(
    df_processed.iloc[idx_val][num_cols].values.astype(np.float32)
)

emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

class TabularNeuralNet(nn.Module):
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=[64, 32]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        total_emb_dim = sum(emb_dims)
        in_dim = total_emb_dim + num_features

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.network(x)

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

def train_neural_net(use_mixup=True, loss_type="focal", use_ema=True, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_train.shape[1]
    ).to(device)

    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_decay = 0.995

    if loss_type == "focal":
        criterion = FocalCrossEntropyLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    t_X_cat_train = torch.tensor(X_nn_cat_train, dtype=torch.long, device=device)
    t_X_num_train = torch.tensor(X_nn_num_train, dtype=torch.float32, device=device)
    t_y_train = torch.tensor(y_train, dtype=torch.long, device=device)

    t_X_cat_val = torch.tensor(X_nn_cat_val, dtype=torch.long, device=device)
    t_X_num_val = torch.tensor(X_nn_num_val, dtype=torch.float32, device=device)

    batch_size = 32
    num_epochs = 150
    num_samples = t_X_cat_train.size(0)
    steps_per_epoch = math.ceil(num_samples / batch_size)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    mixup_alpha = 0.2 if use_mixup else 0.0

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(num_samples)

        for i in range(0, num_samples, batch_size):
            indices_batch = perm[i : i + batch_size]
            batch_cat = t_X_cat_train[indices_batch]
            batch_num = t_X_num_train[indices_batch]
            batch_y = t_y_train[indices_batch]
            cur_bs = batch_cat.size(0)

            if mixup_alpha > 0 and cur_bs > 1:
                lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
                mix_perm = torch.randperm(cur_bs)

                batch_num_mixed = lam * batch_num + (1.0 - lam) * batch_num[mix_perm]
                cat_mask = torch.rand(batch_cat.shape, device=device) < lam
                batch_cat_mixed = torch.where(cat_mask, batch_cat, batch_cat[mix_perm])

                optimizer.zero_grad()
                outputs = model(batch_cat_mixed, batch_num_mixed)
                loss = lam * criterion(outputs, batch_y) + (1.0 - lam) * criterion(outputs, batch_y[mix_perm])
            else:
                optimizer.zero_grad()
                outputs = model(batch_cat, batch_num)
                loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if use_ema:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

    eval_model = ema_model if use_ema else model
    eval_model.eval()
    with torch.no_grad():
        val_outputs = eval_model(t_X_cat_val, t_X_num_val)
        val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

    return val_probs

# Running Ablation Studies
experiments = {
    "Baseline (MixUp + Focal Loss + EMA)": {"use_mixup": True, "loss_type": "focal", "use_ema": True},
    "Ablation 1 (No Tabular MixUp)": {"use_mixup": False, "loss_type": "focal", "use_ema": True},
    "Ablation 2 (CrossEntropy Loss vs Focal)": {"use_mixup": True, "loss_type": "ce", "use_ema": True},
    "Ablation 3 (No EMA Shadow Model)": {"use_mixup": True, "loss_type": "focal", "use_ema": False},
}

results = {}
deltas = {}

print("=== Running Ablation Study on Neural Training Components ===")

baseline_probs = None
for exp_name, params in experiments.items():
    nn_val_probs = train_neural_net(**params)
    ensemble_val_probs = 0.5 * cb_val_probs + 0.5 * nn_val_probs
    ensemble_preds = (ensemble_val_probs >= 0.5).astype(int)
    acc = accuracy_score(y_val, ensemble_preds)
    results[exp_name] = acc

    if exp_name == "Baseline (MixUp + Focal Loss + EMA)":
        baseline_acc = acc
        baseline_probs = ensemble_val_probs

for exp_name, acc in results.items():
    delta = acc - baseline_acc
    deltas[exp_name] = delta
    print(f"{exp_name}: Validation Accuracy = {acc:.5f} (Delta: {delta:+.5f})")

# Determine the most influential component
ablation_impacts = {k: abs(v) for k, v in deltas.items() if k != "Baseline (MixUp + Focal Loss + EMA)"}
most_impactful = max(ablation_impacts, key=ablation_impacts.get)

print("\n=== Ablation Study Conclusion ===")
print(f"The component with the largest performance impact when modified is '{most_impactful}' with an accuracy impact of {deltas[most_impactful]:+.5f}.")