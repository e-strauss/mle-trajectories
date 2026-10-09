import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load dataset
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

# Train / Hold-out Validation Split
indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

df_train = df_processed.iloc[idx_train].reset_index(drop=True)
df_val = df_processed.iloc[idx_val].reset_index(drop=True)


# Neural Network Definition
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

# 5-Fold Stratified K-Fold Setup on the training partition
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_cb = np.zeros(len(df_train))
oof_preds_xgb = np.zeros(len(df_train))
oof_preds_nn = np.zeros(len(df_train))

val_preds_cb = np.zeros(len(df_val))
val_preds_xgb = np.zeros(len(df_val))
val_preds_nn = np.zeros(len(df_val))

for fold, (f_train_idx, f_val_idx) in enumerate(skf.split(df_train, y_train)):
    fold_train_df = df_train.iloc[f_train_idx]
    fold_val_df = df_train.iloc[f_val_idx]
    y_f_train = y_train[f_train_idx]
    y_f_val = y_train[f_val_idx]

    # --- 1. CatBoost ---
    X_cb_tr = fold_train_df[cat_cols + num_cols].copy()
    X_cb_va = fold_val_df[cat_cols + num_cols].copy()
    X_cb_holdout = df_val[cat_cols + num_cols].copy()
    for col in cat_cols:
        X_cb_tr[col] = X_cb_tr[col].astype(str)
        X_cb_va[col] = X_cb_va[col].astype(str)
        X_cb_holdout[col] = X_cb_holdout[col].astype(str)

    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_cols,
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_cb_tr, y_f_train, eval_set=(X_cb_va, y_f_val), early_stopping_rounds=50
    )
    oof_preds_cb[f_val_idx] = cb_model.predict_proba(X_cb_va)[:, 1]
    val_preds_cb += cb_model.predict_proba(X_cb_holdout)[:, 1] / n_splits

    # --- 2. XGBoost ---
    X_xgb_tr = fold_train_df[cat_cols + num_cols].copy()
    X_xgb_va = fold_val_df[cat_cols + num_cols].copy()
    X_xgb_holdout = df_val[cat_cols + num_cols].copy()
    for col in cat_cols:
        X_xgb_tr[col] = X_xgb_tr[col].astype("category")
        X_xgb_va[col] = pd.Categorical(
            X_xgb_va[col], categories=X_xgb_tr[col].cat.categories
        )
        X_xgb_holdout[col] = pd.Categorical(
            X_xgb_holdout[col], categories=X_xgb_tr[col].cat.categories
        )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_xgb_tr,
        y_f_train,
        eval_set=[(X_xgb_va, y_f_val)],
        verbose=False,
    )
    oof_preds_xgb[f_val_idx] = xgb_model.predict_proba(X_xgb_va)[:, 1]
    val_preds_xgb += xgb_model.predict_proba(X_xgb_holdout)[:, 1] / n_splits

    # --- 3. Tabular Neural Network ---
    cat_arrays_tr, cat_arrays_va, cat_arrays_ho, cat_dims = [], [], [], []
    for col in cat_cols:
        le = LabelEncoder()
        c_tr = le.fit_transform(fold_train_df[col].astype(str))
        mapping = {v: i for i, v in enumerate(le.classes_)}
        c_va = np.array(
            [mapping.get(v, 0) for v in fold_val_df[col].astype(str).values]
        )
        c_ho = np.array([mapping.get(v, 0) for v in df_val[col].astype(str).values])

        cat_arrays_tr.append(c_tr)
        cat_arrays_va.append(c_va)
        cat_arrays_ho.append(c_ho)
        cat_dims.append(len(le.classes_))

    X_nn_cat_tr = np.stack(cat_arrays_tr, axis=1).astype(np.int64)
    X_nn_cat_va = np.stack(cat_arrays_va, axis=1).astype(np.int64)
    X_nn_cat_ho = np.stack(cat_arrays_ho, axis=1).astype(np.int64)

    scaler = StandardScaler()
    X_nn_num_tr = scaler.fit_transform(
        fold_train_df[num_cols].values.astype(np.float32)
    )
    X_nn_num_va = scaler.transform(fold_val_df[num_cols].values.astype(np.float32))
    X_nn_num_ho = scaler.transform(df_val[num_cols].values.astype(np.float32))

    emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

    nn_model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_tr.shape[1]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    t_cat_tr = torch.tensor(X_nn_cat_tr, dtype=torch.long, device=device)
    t_num_tr = torch.tensor(X_nn_num_tr, dtype=torch.float32, device=device)
    t_y_tr = torch.tensor(y_f_train, dtype=torch.long, device=device)

    t_cat_va = torch.tensor(X_nn_cat_va, dtype=torch.long, device=device)
    t_num_va = torch.tensor(X_nn_num_va, dtype=torch.float32, device=device)

    t_cat_ho = torch.tensor(X_nn_cat_ho, dtype=torch.long, device=device)
    t_num_ho = torch.tensor(X_nn_num_ho, dtype=torch.float32, device=device)

    batch_size = 32
    num_epochs = 120
    best_loss = float("inf")
    best_weights = None

    for epoch in range(num_epochs):
        nn_model.train()
        permutation = torch.randperm(t_cat_tr.size(0))
        for i in range(0, t_cat_tr.size(0), batch_size):
            b_idx = permutation[i : i + batch_size]
            b_cat = t_cat_tr[b_idx]
            b_num = t_num_tr[b_idx]
            b_y = t_y_tr[b_idx]

            optimizer.zero_grad()
            out = nn_model(b_cat, b_num)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            va_out = nn_model(t_cat_va, t_num_va)
            va_loss = criterion(
                va_out, torch.tensor(y_f_val, dtype=torch.long, device=device)
            ).item()
            va_acc = (torch.argmax(va_out, dim=1).cpu().numpy() == y_f_val).mean()
        scheduler.step(va_acc)

        if va_loss < best_loss:
            best_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    if best_weights is not None:
        nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    nn_model.eval()
    with torch.no_grad():
        va_probs = torch.softmax(nn_model(t_cat_va, t_num_va), dim=1)[:, 1].cpu().numpy()
        ho_probs = torch.softmax(nn_model(t_cat_ho, t_num_ho), dim=1)[:, 1].cpu().numpy()

    oof_preds_nn[f_val_idx] = va_probs
    val_preds_nn += ho_probs / n_splits


# --- Meta-Feature Creation ---
def build_meta_features(p_cb, p_xgb, p_nn):
    p_cb = np.clip(p_cb, 1e-6, 1.0 - 1e-6)
    p_xgb = np.clip(p_xgb, 1e-6, 1.0 - 1e-6)
    p_nn = np.clip(p_nn, 1e-6, 1.0 - 1e-6)

    mean_prob = (p_cb + p_xgb + p_nn) / 3.0
    std_prob = np.std([p_cb, p_xgb, p_nn], axis=0)

    # Disagreement & Interaction features
    diff_cb_xgb = p_cb - p_xgb
    diff_cb_nn = p_cb - p_nn
    diff_xgb_nn = p_xgb - p_nn

    prod_cb_xgb = p_cb * p_xgb
    prod_cb_nn = p_cb * p_nn
    prod_xgb_nn = p_xgb * p_nn

    meta_X = np.column_stack(
        [
            p_cb,
            p_xgb,
            p_nn,
            mean_prob,
            std_prob,
            diff_cb_xgb,
            diff_cb_nn,
            diff_xgb_nn,
            prod_cb_xgb,
            prod_cb_nn,
            prod_xgb_nn,
        ]
    )
    return meta_X


X_meta_train = build_meta_features(oof_preds_cb, oof_preds_xgb, oof_preds_nn)
X_meta_val = build_meta_features(val_preds_cb, val_preds_xgb, val_preds_nn)

# --- Train Regularized Meta-Classifier ---
meta_model = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=1000)
meta_model.fit(X_meta_train, y_train)

meta_val_probs = meta_model.predict_proba(X_meta_val)[:, 1]
meta_train_probs = meta_model.predict_proba(X_meta_train)[:, 1]

# Base Soft-Voting Average
base_val_avg = (val_preds_cb + val_preds_xgb + val_preds_nn) / 3.0
base_train_avg = (oof_preds_cb + oof_preds_xgb + oof_preds_nn) / 3.0

# Soft Voting Fusion between meta-learner and base model average
blend_weight = 0.65
ensemble_train_probs = (
    blend_weight * meta_train_probs + (1.0 - blend_weight) * base_train_avg
)
ensemble_val_probs = blend_weight * meta_val_probs + (1.0 - blend_weight) * base_val_avg

# Find optimal threshold using OOF meta predictions
thresholds = np.linspace(0.2, 0.8, 121)
best_threshold = 0.5
best_oof_score = -1.0

for th in thresholds:
    score = accuracy_score(y_train, (ensemble_train_probs >= th).astype(int))
    if score > best_oof_score:
        best_oof_score = score
        best_threshold = th

val_preds = (ensemble_val_probs >= best_threshold).astype(int)
final_validation_score = accuracy_score(y_val, val_preds)

print(f"Optimal Threshold Selected via OOF: {best_threshold:.4f}")
print(f"Final Validation Performance: {final_validation_score}")