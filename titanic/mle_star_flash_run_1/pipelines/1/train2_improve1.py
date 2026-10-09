import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load data
train_df = pd.read_csv("./input/train.csv")


import pandas as pd


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

# 1. CatBoost Data Preparation & Training
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

# 2. Neural Network Data Preparation & Training
cat_arrays_train, cat_arrays_val, cat_dims = [], [], []
for col in cat_cols:
    le = LabelEncoder()
    cat_train_encoded = le.fit_transform(df_processed.iloc[idx_train][col].astype(str))
    # Handle any unseen categories in validation set safely
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


import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=True)

    def forward(self, x0, xl):
        return x0 * self.linear(xl) + xl


class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc_gate = nn.Linear(in_dim, out_dim)
        self.fc_val = nn.Linear(in_dim, out_dim)
        self.fc_out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        normed = self.norm(x)
        gate = F.silu(self.fc_gate(normed))
        val = self.fc_val(normed)
        out = self.fc_out(gate * val)
        out = self.dropout(out)
        return residual + out


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims,
        emb_dims,
        num_features,
        hidden_dims=[128, 64],
        num_cross_layers=2,
        emb_dropout=0.1,
        deep_dropout=0.1,
        out_dim=2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_c, e_dim) for num_c, e_dim in zip(cat_dims, emb_dims)]
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        total_emb_dim = sum(emb_dims)
        self.in_dim = total_emb_dim + num_features
        self.input_norm = nn.LayerNorm(self.in_dim)

        # Cross Network
        self.cross_layers = nn.ModuleList(
            [CrossLayer(self.in_dim) for _ in range(num_cross_layers)]
        )

        # Deep Network with SwiGLU
        deep_layers = []
        prev_dim = self.in_dim
        for h_dim in hidden_dims:
            deep_layers.append(SwiGLUBlock(prev_dim, h_dim, dropout=deep_dropout))
            prev_dim = h_dim
        self.deep_network = nn.Sequential(*deep_layers)

        # Direct linear skip from input to output
        self.skip_head = nn.Linear(self.in_dim, out_dim, bias=False)

        # Final head combining Cross and Deep streams
        combined_dim = self.in_dim + prev_dim
        self.out_norm = nn.LayerNorm(combined_dim)
        self.head = nn.Linear(combined_dim, out_dim)

    def forward(self, x_cat, x_num):
        if len(self.embeddings) > 0 and x_cat is not None and x_cat.size(1) > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(embs, dim=1)
            x_cat_emb = self.emb_dropout(x_cat_emb)
            if x_num is not None and x_num.size(1) > 0:
                x0 = torch.cat([x_cat_emb, x_num], dim=1)
            else:
                x0 = x_cat_emb
        else:
            x0 = x_num

        x0_norm = self.input_norm(x0)

        # Cross Stream
        xl = x0_norm
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0_norm, xl)

        # Deep Stream with SwiGLU
        xd = self.deep_network(x0_norm)

        # Combine streams and add direct linear skip
        combined = torch.cat([xl, xd], dim=1)
        out = self.head(self.out_norm(combined)) + self.skip_head(x0_norm)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = TabularNeuralNet(
    cat_dims, emb_dims, num_features=X_nn_num_train.shape[1]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10
)

t_X_cat_train = torch.tensor(X_nn_cat_train, dtype=torch.long, device=device)
t_X_num_train = torch.tensor(X_nn_num_train, dtype=torch.float32, device=device)
t_y_train = torch.tensor(y_train, dtype=torch.long, device=device)

t_X_cat_val = torch.tensor(X_nn_cat_val, dtype=torch.long, device=device)
t_X_num_val = torch.tensor(X_nn_num_val, dtype=torch.float32, device=device)
t_y_val = torch.tensor(y_val, dtype=torch.long, device=device)

batch_size = 32
num_epochs = 150
best_val_acc = 0.0

for epoch in range(num_epochs):
    nn_model.train()
    permutation = torch.randperm(t_X_cat_train.size(0))

    for i in range(0, t_X_cat_train.size(0), batch_size):
        indices_batch = permutation[i : i + batch_size]
        batch_cat = t_X_cat_train[indices_batch]
        batch_num = t_X_num_train[indices_batch]
        batch_y = t_y_train[indices_batch]

        optimizer.zero_grad()
        outputs = nn_model(batch_cat, batch_num)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    nn_model.eval()
    with torch.no_grad():
        val_outputs = nn_model(t_X_cat_val, t_X_num_val)
        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)

    scheduler.step(val_acc)

nn_model.eval()
with torch.no_grad():
    val_outputs = nn_model(t_X_cat_val, t_X_num_val)
    nn_val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()

# 3. Ensemble Predictions
ensemble_val_probs = 0.5 * cb_val_probs + 0.5 * nn_val_probs
ensemble_preds = (ensemble_val_probs >= 0.5).astype(int)

final_validation_score = accuracy_score(y_val, ensemble_preds)
print(f"Final Validation Performance: {final_validation_score}")