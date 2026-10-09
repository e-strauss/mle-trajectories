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


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_factor=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ffn_factor, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class TabularNeuralNet(nn.Module):
    def __init__(
        self,
        cat_dims=None,
        num_features=0,
        d_token=32,
        n_blocks=3,
        n_heads=4,
        ffn_factor=2,
        dropout=0.1,
        out_dim=2,
        **kwargs,
    ):
        super().__init__()
        cat_dims = cat_dims or []
        # Ensure d_token is divisible by n_heads
        if d_token % n_heads != 0:
            d_token = ((d_token // n_heads) + 1) * n_heads

        self.d_token = d_token
        self.cat_embeddings = (
            nn.ModuleList([nn.Embedding(num_c, d_token) for num_c in cat_dims])
            if cat_dims
            else nn.ModuleList()
        )

        self.num_embeddings = (
            nn.ModuleList([nn.Linear(1, d_token) for _ in range(num_features)])
            if num_features > 0
            else nn.ModuleList()
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.01)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_factor=ffn_factor,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, out_dim),
        )

    def forward(self, x_cat=None, x_num=None):
        batch_size = x_cat.size(0) if x_cat is not None else x_num.size(0)
        tokens = [self.cls_token.expand(batch_size, -1, -1)]

        if x_cat is not None and len(self.cat_embeddings) > 0:
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
                dim=1,
            )
            tokens.append(cat_tokens)

        if x_num is not None and len(self.num_embeddings) > 0:
            num_tokens = torch.stack(
                [emb(x_num[:, i : i + 1]) for i, emb in enumerate(self.num_embeddings)],
                dim=1,
            )
            tokens.append(num_tokens)

        x = torch.cat(tokens, dim=1)

        for block in self.blocks:
            x = block(x)

        cls_rep = x[:, 0]
        return self.head(cls_rep)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = TabularNeuralNet(
    cat_dims=cat_dims,
    num_features=X_nn_num_train.shape[1],
    d_token=32,
    n_heads=4,
    n_blocks=2,
    dropout=0.1,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
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
num_epochs = 100

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