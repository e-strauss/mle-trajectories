import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Set global random seed
np.random.seed(42)
torch.manual_seed(42)

# 1. Load Data and Feature Engineering
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

all_features = cat_cols + num_cols
seeds = [42, 100, 2024]

# -------------------------------------------------------------
# 2. CatBoost Multi-Seed Training
# -------------------------------------------------------------
X_cb = df_processed[all_features].copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)

X_cb_train, X_cb_val = X_cb.iloc[idx_train], X_cb.iloc[idx_val]
y_cb_train, y_cb_val = y[idx_train], y[idx_val]

cb_val_probs_list = []
for seed in seeds:
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_cols,
        eval_metric="Accuracy",
        random_seed=seed,
        verbose=0,
    )
    cb_model.fit(
        X_cb_train, y_cb_train, eval_set=(X_cb_val, y_cb_val), early_stopping_rounds=50
    )
    cb_val_probs_list.append(cb_model.predict_proba(X_cb_val)[:, 1])

cb_val_probs = np.mean(cb_val_probs_list, axis=0)

# -------------------------------------------------------------
# 3. XGBoost Multi-Seed Training
# -------------------------------------------------------------
X_xgb = df_processed[all_features].copy()
for col in cat_cols:
    X_xgb[col] = X_xgb[col].astype("category")

X_xgb_train, X_xgb_val = X_xgb.iloc[idx_train], X_xgb.iloc[idx_val]
y_xgb_train, y_xgb_val = y[idx_train], y[idx_val]

xgb_val_probs_list = []
for seed in seeds:
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=seed,
        eval_metric="logloss",
    )
    xgb_model.fit(
        X_xgb_train,
        y_xgb_train,
        eval_set=[(X_xgb_val, y_xgb_val)],
        verbose=False,
    )
    xgb_val_probs_list.append(xgb_model.predict_proba(X_xgb_val)[:, 1])

xgb_val_probs = np.mean(xgb_val_probs_list, axis=0)

# -------------------------------------------------------------
# 4. Tabular Neural Network Multi-Seed Training
# -------------------------------------------------------------
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_X_cat_train = torch.tensor(X_nn_cat_train, dtype=torch.long, device=device)
t_X_num_train = torch.tensor(X_nn_num_train, dtype=torch.float32, device=device)
t_y_train = torch.tensor(y_train, dtype=torch.long, device=device)

t_X_cat_val = torch.tensor(X_nn_cat_val, dtype=torch.long, device=device)
t_X_num_val = torch.tensor(X_nn_num_val, dtype=torch.float32, device=device)

nn_val_probs_list = []
batch_size = 32
num_epochs = 150

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    nn_model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_train.shape[1]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

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
        probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
        nn_val_probs_list.append(probs)

nn_val_probs = np.mean(nn_val_probs_list, axis=0)


# -------------------------------------------------------------
# 5. Log-Odds Ensemble Blending & Simplex Optimization
# -------------------------------------------------------------
def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def from_logit(l):
    return 1.0 / (1.0 + np.exp(-l))


logit_cb = to_logit(cb_val_probs)
logit_xgb = to_logit(xgb_val_probs)
logit_nn = to_logit(nn_val_probs)

best_score = -1.0
best_weights = (1 / 3, 1 / 3, 1 / 3)
best_threshold = 0.5

w_steps = np.linspace(0.0, 1.0, 21)
thresholds = np.linspace(0.35, 0.65, 31)

for w1 in w_steps:
    for w2 in w_steps:
        if w1 + w2 > 1.0:
            continue
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0:
            continue

        blended_logit = w1 * logit_cb + w2 * logit_xgb + w3 * logit_nn
        blended_prob = from_logit(blended_logit)

        for th in thresholds:
            preds = (blended_prob >= th).astype(int)
            score = accuracy_score(y_val, preds)
            if score > best_score:
                best_score = score
                best_weights = (w1, w2, w3)
                best_threshold = th

final_blended_logit = (
    best_weights[0] * logit_cb
    + best_weights[1] * logit_xgb
    + best_weights[2] * logit_nn
)
final_val_probs = from_logit(final_blended_logit)
final_val_preds = (final_val_probs >= best_threshold).astype(int)
final_validation_score = accuracy_score(y_val, final_val_preds)

print(f"Optimal Weights (CB, XGB, NN): {best_weights}, Optimal Threshold: {best_threshold:.3f}")
print(f"Final Validation Performance: {final_validation_score}")