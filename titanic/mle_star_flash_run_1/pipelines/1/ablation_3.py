import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for exact reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Feature Engineering and Preprocessing
train_df = pd.read_csv("./input/train.csv")

def prepare_features(df):
    df = df.copy()

    # Title extraction and grouping
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

    # Impute missing Age conditionally
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Sociological interaction and family dynamics
    df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    def assign_family_tier(size):
        if size == 1:
            return "IsAlone"
        elif size <= 4:
            return "SmallFamily"
        else:
            return "LargeFamily"

    df["FamilyTier"] = df["FamilySize"].apply(assign_family_tier)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1)
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]

    return df, cat_cols, num_cols

df_processed, cat_cols, num_cols = prepare_features(train_df)
y = train_df["Survived"].values

# Train/Validation Split (Stratified 80/20)
indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost Preprocessing
X_cb = df_processed[cat_cols + num_cols].copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)

X_cb_train, X_cb_val = X_cb.iloc[idx_train], X_cb.iloc[idx_val]
y_cb_train, y_cb_val = y[idx_train], y[idx_val]

# PyTorch Preprocessing
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_X_cat_train = torch.tensor(X_nn_cat_train, dtype=torch.long, device=device)
t_X_num_train = torch.tensor(X_nn_num_train, dtype=torch.float32, device=device)
t_y_train = torch.tensor(y_train, dtype=torch.long, device=device)

t_X_cat_val = torch.tensor(X_nn_cat_val, dtype=torch.long, device=device)
t_X_num_val = torch.tensor(X_nn_num_val, dtype=torch.float32, device=device)
t_y_val = torch.tensor(y_val, dtype=torch.long, device=device)


# 2. Neural Architecture Definitions
class NumericalTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_features, d_token) * (1.0 / math.sqrt(d_token))
        )
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(
        self,
        cat_dims,
        num_features,
        d_token=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        use_num_skip=True,
    ):
        super().__init__()
        self.use_num_skip = use_num_skip
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_c, d_token) for num_c in cat_dims]
        )
        self.num_tokenizer = NumericalTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 2),
        )
        if self.use_num_skip:
            self.num_skip = nn.Linear(num_features, 2)

    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        cat_tokens = [
            emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)
        ]
        tokens.append(torch.cat(cat_tokens, dim=1))
        tokens.append(self.num_tokenizer(x_num))

        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        logits = self.head(x[:, 0, :])

        if self.use_num_skip:
            logits = logits + self.num_skip(x_num)
        return logits


class ClassicTabularMLP(nn.Module):
    def __init__(self, cat_dims, num_features, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        in_dim = sum(emb_dims) + num_features

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.network(x)


# 3. Model Training Functions
def train_catboost():
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
    return cb_model.predict_proba(X_cb_val)[:, 1]


def train_neural_net(model, epochs=120, lr=3e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    batch_size = 32
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(t_X_cat_train.size(0))
        for i in range(0, t_X_cat_train.size(0), batch_size):
            indices_batch = permutation[i : i + batch_size]
            batch_cat = t_X_cat_train[indices_batch]
            batch_num = t_X_num_train[indices_batch]
            batch_y = t_y_train[indices_batch]

            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(t_X_cat_val, t_X_num_val)
        probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
    return probs


# 4. Running Ablation Study
cb_val_probs = train_catboost()

# Baseline: CatBoost + FT-Transformer (with Feature Tokenizer & Residual Numerical Skip)
set_seed(42)
ft_model_base = FTTransformer(
    cat_dims, num_features=X_nn_num_train.shape[1], use_num_skip=True
)
ft_probs_base = train_neural_net(ft_model_base)
baseline_probs = 0.5 * cb_val_probs + 0.5 * ft_probs_base
baseline_acc = accuracy_score(y_val, (baseline_probs >= 0.5).astype(int))

# Ablation 1: No Residual Numerical Skip Connection in FT-Transformer
set_seed(42)
ft_model_no_skip = FTTransformer(
    cat_dims, num_features=X_nn_num_train.shape[1], use_num_skip=False
)
ft_probs_no_skip = train_neural_net(ft_model_no_skip)
abl1_probs = 0.5 * cb_val_probs + 0.5 * ft_probs_no_skip
abl1_acc = accuracy_score(y_val, (abl1_probs >= 0.5).astype(int))

# Ablation 2: Replace FT-Transformer Backbone with Classical Tabular MLP
set_seed(42)
mlp_model = ClassicTabularMLP(cat_dims, num_features=X_nn_num_train.shape[1])
mlp_probs = train_neural_net(mlp_model, lr=1e-2)
abl2_probs = 0.5 * cb_val_probs + 0.5 * mlp_probs
abl2_acc = accuracy_score(y_val, (abl2_probs >= 0.5).astype(int))

# Ablation 3: Standalone FT-Transformer (Remove Tree Ensemble Blending)
abl3_acc = accuracy_score(y_val, (ft_probs_base >= 0.5).astype(int))

# 5. Summary & Key Findings
results = {
    "Baseline (CatBoost + FT-Transformer w/ Skip)": baseline_acc,
    "Ablation 1 (FT-Transformer without Numerical Skip)": abl1_acc,
    "Ablation 2 (Classic Tabular MLP instead of FT-Transformer)": abl2_acc,
    "Ablation 3 (Standalone FT-Transformer - No CatBoost)": abl3_acc,
}

print("=== Ablation Study Results ===")
for config, score in results.items():
    delta = score - baseline_acc
    print(f"{config}: Accuracy = {score:.5f} (Delta: {delta:+.5f})")

# Determine the most critical component
drops = {config: baseline_acc - score for config, score in results.items() if config != "Baseline (CatBoost + FT-Transformer w/ Skip)"}
most_impactful = max(drops, key=drops.get)
max_drop = drops[most_impactful]

print(f"\nMost impactful component to performance: '{most_impactful}' with a performance drop of {max_drop:.5f} when removed/altered.")