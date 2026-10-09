import lightgbm as lgb
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

# 1. Unified Feature Pipeline
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

# Split into train partition and holdout validation partition
indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

df_train_part = df_processed.iloc[idx_train].reset_index(drop=True)
df_val_part = df_processed.iloc[idx_val].reset_index(drop=True)


# Neural Network Architecture definition
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

# 2. Stratified Out-of-Fold (OOF) Cross-Validation on Training Split
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Matrices to store OOF probabilities and holdout predictions
oof_preds = np.zeros((len(df_train_part), 4))  # CB, XGB, LGBM, NN
val_preds_folds = np.zeros((len(df_val_part), 4, n_splits))

# Prepare datasets for CatBoost, Tree Models, and NN
# CatBoost Data
X_cb_train_full = df_train_part[cat_cols + num_cols].copy()
X_cb_val_full = df_val_part[cat_cols + num_cols].copy()
for c in cat_cols:
    X_cb_train_full[c] = X_cb_train_full[c].astype(str)
    X_cb_val_full[c] = X_cb_val_full[c].astype(str)

# XGBoost & LightGBM Data (Categorical Dtypes)
X_trees_train_full = df_train_part[cat_cols + num_cols].copy()
X_trees_val_full = df_val_part[cat_cols + num_cols].copy()
for c in cat_cols:
    X_trees_train_full[c] = X_trees_train_full[c].astype("category")
    X_trees_val_full[c] = pd.Categorical(
        X_trees_val_full[c], categories=X_trees_train_full[c].cat.categories
    )

# Pre-fit LabelEncoders and Scaler for Neural Network
nn_encoders = {}
cat_dims = []
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_train_part[col].astype(str))
    nn_encoders[col] = le
    cat_dims.append(len(le.classes_))

emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

# Loop over folds
for fold, (trn_idx, oof_idx) in enumerate(skf.split(df_train_part, y_train)):
    y_tr, y_oof = y_train[trn_idx], y_train[oof_idx]

    # Model 1: CatBoost Classifier
    X_cb_tr, X_cb_oof = (
        X_cb_train_full.iloc[trn_idx],
        X_cb_train_full.iloc[oof_idx],
    )
    cb_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_cols,
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(X_cb_tr, y_tr, eval_set=(X_cb_oof, y_oof), early_stopping_rounds=50)
    oof_preds[oof_idx, 0] = cb_model.predict_proba(X_cb_oof)[:, 1]
    val_preds_folds[:, 0, fold] = cb_model.predict_proba(X_cb_val_full)[:, 1]

    # Model 2: XGBoost Classifier
    X_xgb_tr, X_xgb_oof = (
        X_trees_train_full.iloc[trn_idx],
        X_trees_train_full.iloc[oof_idx],
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
    )
    xgb_model.fit(
        X_xgb_tr,
        y_tr,
        eval_set=[(X_xgb_oof, y_oof)],
        verbose=False,
    )
    oof_preds[oof_idx, 1] = xgb_model.predict_proba(X_xgb_oof)[:, 1]
    val_preds_folds[:, 1, fold] = xgb_model.predict_proba(X_trees_val_full)[:, 1]

    # Model 3: LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb_model.fit(
        X_xgb_tr,
        y_tr,
        eval_set=[(X_xgb_oof, y_oof)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_preds[oof_idx, 2] = lgb_model.predict_proba(X_xgb_oof)[:, 1]
    val_preds_folds[:, 2, fold] = lgb_model.predict_proba(X_trees_val_full)[:, 1]

    # Model 4: Tabular Neural Network
    cat_tr_list, cat_oof_list, cat_val_list = [], [], []
    for col in cat_cols:
        le = nn_encoders[col]
        mapping = {val: i for i, val in enumerate(le.classes_)}

        tr_c = (
            df_train_part.iloc[trn_idx][col]
            .astype(str)
            .map(lambda v: mapping.get(v, 0))
            .values
        )
        oof_c = (
            df_train_part.iloc[oof_idx][col]
            .astype(str)
            .map(lambda v: mapping.get(v, 0))
            .values
        )
        val_c = (
            df_val_part[col].astype(str).map(lambda v: mapping.get(v, 0)).values
        )

        cat_tr_list.append(tr_c)
        cat_oof_list.append(oof_c)
        cat_val_list.append(val_c)

    X_nn_cat_tr = np.stack(cat_tr_list, axis=1).astype(np.int64)
    X_nn_cat_oof = np.stack(cat_oof_list, axis=1).astype(np.int64)
    X_nn_cat_v = np.stack(cat_val_list, axis=1).astype(np.int64)

    scaler = StandardScaler()
    X_nn_num_tr = scaler.fit_transform(
        df_train_part.iloc[trn_idx][num_cols].values.astype(np.float32)
    )
    X_nn_num_oof = scaler.transform(
        df_train_part.iloc[oof_idx][num_cols].values.astype(np.float32)
    )
    X_nn_num_v = scaler.transform(df_val_part[num_cols].values.astype(np.float32))

    nn_model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_tr.shape[1]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)

    t_cat_tr = torch.tensor(X_nn_cat_tr, dtype=torch.long, device=device)
    t_num_tr = torch.tensor(X_nn_num_tr, dtype=torch.float32, device=device)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)

    t_cat_oof = torch.tensor(X_nn_cat_oof, dtype=torch.long, device=device)
    t_num_oof = torch.tensor(X_nn_num_oof, dtype=torch.float32, device=device)

    t_cat_v = torch.tensor(X_nn_cat_v, dtype=torch.long, device=device)
    t_num_v = torch.tensor(X_nn_num_v, dtype=torch.float32, device=device)

    batch_size = 32
    for epoch in range(120):
        nn_model.train()
        perm = torch.randperm(t_cat_tr.size(0))
        for b_idx in range(0, t_cat_tr.size(0), batch_size):
            b_indices = perm[b_idx : b_idx + batch_size]
            optimizer.zero_grad()
            out = nn_model(t_cat_tr[b_indices], t_num_tr[b_indices])
            loss = criterion(out, t_y_tr[b_indices])
            loss.backward()
            optimizer.step()

    nn_model.eval()
    with torch.no_grad():
        oof_nn_out = nn_model(t_cat_oof, t_num_oof)
        oof_preds[oof_idx, 3] = (
            torch.softmax(oof_nn_out, dim=1)[:, 1].cpu().numpy()
        )

        val_nn_out = nn_model(t_cat_v, t_num_v)
        val_preds_folds[:, 3, fold] = (
            torch.softmax(val_nn_out, dim=1)[:, 1].cpu().numpy()
        )

# Fold-averaged holdout validation probabilities
val_preds_avg = val_preds_folds.mean(axis=2)

# 3. Constrained Meta-Learner (Stacking) Training on OOF Probabilities
meta_learner = LogisticRegression(C=1.0, random_state=42)
meta_learner.fit(oof_preds, y_train)

# 4. Meta-Inference and Calibrated Thresholding
meta_oof_probs = meta_learner.predict_proba(oof_preds)[:, 1]
meta_val_probs = meta_learner.predict_proba(val_preds_avg)[:, 1]

# Calibrate decision threshold on training OOF predictions
best_th = 0.5
best_oof_score = -1.0
for th in np.linspace(0.2, 0.8, 61):
    score = accuracy_score(y_train, (meta_oof_probs >= th).astype(int))
    if score > best_oof_score:
        best_oof_score = score
        best_th = th

# Final prediction on holdout validation set
final_val_preds = (meta_val_probs >= best_th).astype(int)
final_validation_score = accuracy_score(y_val, final_val_preds)

print(f"Final Validation Performance: {final_validation_score}")