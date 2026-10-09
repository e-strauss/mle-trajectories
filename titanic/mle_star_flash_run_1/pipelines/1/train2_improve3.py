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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicNumericalEmbedding(nn.Module):
    """
    Learnable Periodic (Fourier) Numerical Embedding for continuous features.
    Transforms each scalar feature into a rich multi-scale periodic representation.
    """
    def __init__(self, num_features, token_dim, sigma=0.5):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        half_dim = token_dim // 2
        self.frequencies = nn.Parameter(torch.randn(num_features, half_dim) * sigma)
        self.linear = nn.Linear(half_dim * 2, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)

    def forward(self, x_num):
        # x_num: (batch_size, num_features)
        # x_proj: (batch_size, num_features, half_dim)
        x_proj = 2 * math.pi * x_num.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        fourier = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        out = self.linear(fourier)
        return self.layer_norm(out)


class FeatureSqueezeAndExcitation(nn.Module):
    """
    Feature-level Squeeze-and-Excitation module to dynamically recalibrate
    feature token importances per instance.
    """
    def __init__(self, num_tokens, reduction=4):
        super().__init__()
        reduced_dim = max(4, num_tokens // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_tokens, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, num_tokens),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        # Squeeze along embedding dimension
        squeeze = x.mean(dim=-1)  # (batch_size, num_tokens)
        excitation = self.fc(squeeze).unsqueeze(-1)  # (batch_size, num_tokens, 1)
        return x * excitation


class FeatureGLU(nn.Module):
    """
    Feature-wise Gated Linear Unit for non-linear token transformation.
    """
    def __init__(self, token_dim):
        super().__init__()
        self.linear = nn.Linear(token_dim, token_dim * 2)

    def forward(self, x):
        h, gate = self.linear(x).chunk(2, dim=-1)
        return h * torch.sigmoid(gate)


class PairwiseInteractionPooling(nn.Module):
    """
    Computes pairwise dot-product interactions between all feature tokens.
    """
    def __init__(self, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        # Indices for the upper triangular pairwise interactions
        triu_indices = torch.triu_indices(num_tokens, num_tokens, offset=1)
        self.register_buffer("row_idx", triu_indices[0])
        self.register_buffer("col_idx", triu_indices[1])

    def forward(self, tokens):
        # tokens: (batch_size, num_tokens, token_dim)
        # Normalized dot products
        tokens_norm = F.normalize(tokens, p=2, dim=-1)
        interaction_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))
        interactions = interaction_matrix[:, self.row_idx, self.col_idx]
        return interactions


class TabularNeuralNet(nn.Module):
    """
    Tabular Gated Feature Network with periodic numerical embeddings,
    feature-wise gating (GLU + Squeeze-and-Excitation), and pairwise interaction pooling.
    """
    def __init__(self, cat_dims, emb_dims, num_features, hidden_dims=[128, 64], token_dim=32, dropout=0.2):
        super().__init__()
        self.token_dim = token_dim
        self.num_cat = len(cat_dims)
        self.num_num = num_features
        self.total_tokens = self.num_cat + self.num_num

        # Categorical feature embeddings projected to token_dim
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(num_c, e_dim),
                    nn.Linear(e_dim, token_dim) if e_dim != token_dim else nn.Identity(),
                )
                for num_c, e_dim in zip(cat_dims, emb_dims)
            ]
        )

        # Numerical periodic Fourier embeddings
        if self.num_num > 0:
            self.num_embedding = PeriodicNumericalEmbedding(self.num_num, token_dim)
        else:
            self.num_embedding = None

        # Feature-wise Gated Linear Unit
        self.token_glu = FeatureGLU(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)

        # Squeeze-and-Excitation across feature tokens
        self.se_module = FeatureSqueezeAndExcitation(self.total_tokens)

        # Pairwise interaction pooling
        if self.total_tokens > 1:
            self.interaction_pool = PairwiseInteractionPooling(self.total_tokens)
            num_interactions = self.total_tokens * (self.total_tokens - 1) // 2
        else:
            self.interaction_pool = None
            num_interactions = 0

        # Input dimension for final MLP: flattened tokens + pairwise interactions
        mlp_in_dim = (self.total_tokens * token_dim) + num_interactions

        # Classifier MLP head
        layers = []
        prev_dim = mlp_in_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        token_list = []

        # Process categorical features into tokens: (batch_size, 1, token_dim)
        if self.num_cat > 0:
            for i, emb_layer in enumerate(self.cat_embeddings):
                cat_tok = emb_layer(x_cat[:, i]).unsqueeze(1)
                token_list.append(cat_tok)

        # Process numerical features into tokens: (batch_size, num_num, token_dim)
        if self.num_num > 0 and self.num_embedding is not None:
            num_tok = self.num_embedding(x_num)
            token_list.append(num_tok)

        # Unified feature tokens: (batch_size, total_tokens, token_dim)
        tokens = torch.cat(token_list, dim=1) if len(token_list) > 1 else token_list[0]

        # Apply GLU gating with residual connection and LayerNorm
        tokens = self.token_norm(tokens + self.token_glu(tokens))

        # Dynamic feature importance recalibration
        tokens = self.se_module(tokens)

        # Extract pairwise interactions
        flat_tokens = tokens.flatten(start_dim=1)
        if self.interaction_pool is not None:
            interactions = self.interaction_pool(tokens)
            combined = torch.cat([flat_tokens, interactions], dim=1)
        else:
            combined = flat_tokens

        return self.network(combined)


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