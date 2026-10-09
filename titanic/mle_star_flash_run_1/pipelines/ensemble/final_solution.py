import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Load Data & Unified High-Signal Feature Pipeline
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Compute global ticket frequency across both train and test
all_tickets = pd.concat([train_df["Ticket"], test_df["Ticket"]], axis=0)
ticket_counts = all_tickets.value_counts()


def prepare_features(df, is_train=True, age_lookup=None, global_age_median=None, global_fare_median=None):
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

    # Impute missing Age
    if is_train:
        age_lookup = df.groupby(["Pclass", "Sex"])["Age"].median()
        global_age_median = df["Age"].median()
        global_fare_median = df["Fare"].median()

    df["Age"] = df.set_index(["Pclass", "Sex"]).index.map(age_lookup).fillna(global_age_median).values
    df["Age"] = df["Age"].fillna(global_age_median)

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
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Simplified Cabin indicator
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Ticket frequency
    df["TicketFreq"] = df["Ticket"].map(ticket_counts).fillna(1)

    # Missing value handling for Fare and Embarked
    df["Embarked"] = df["Embarked"].fillna("Missing")
    df["Fare"] = df["Fare"].fillna(global_fare_median)

    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier"]
    num_cols = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "HasCabin", "TicketFreq"]

    if is_train:
        return df, cat_cols, num_cols, age_lookup, global_age_median, global_fare_median
    return df, cat_cols, num_cols


df_processed, cat_cols, num_cols, age_lookup, global_age_med, global_fare_med = prepare_features(
    train_df, is_train=True
)
df_test_processed, _, _ = prepare_features(
    test_df,
    is_train=False,
    age_lookup=age_lookup,
    global_age_median=global_age_med,
    global_fare_median=global_fare_med,
)

y = train_df["Survived"].values

# Train/Validation Hold-out Split
indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

df_train = df_processed.iloc[idx_train].reset_index(drop=True)
df_val = df_processed.iloc[idx_val].reset_index(drop=True)
df_test = df_test_processed.reset_index(drop=True)


# PyTorch Tabular Neural Net Architecture
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


# 2. Stratified 5-Fold Cross-Validation on df_train
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros((len(df_train), 4))  # CB, XGB, RF, NN
val_preds_folds = np.zeros((len(df_val), 4))
test_preds_folds = np.zeros((len(df_test), 4))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_f_idx, val_f_idx) in enumerate(skf.split(df_train, y_train)):
    d_tr = df_train.iloc[train_f_idx]
    d_va = df_train.iloc[val_f_idx]
    y_tr, y_va = y_train[train_f_idx], y_train[val_f_idx]

    # --- A. CatBoost ---
    X_cb_tr = d_tr[cat_cols + num_cols].copy()
    X_cb_va = d_va[cat_cols + num_cols].copy()
    X_cb_val = df_val[cat_cols + num_cols].copy()
    X_cb_test = df_test[cat_cols + num_cols].copy()
    for col in cat_cols:
        X_cb_tr[col] = X_cb_tr[col].astype(str)
        X_cb_va[col] = X_cb_va[col].astype(str)
        X_cb_val[col] = X_cb_val[col].astype(str)
        X_cb_test[col] = X_cb_test[col].astype(str)

    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_cols,
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(X_cb_tr, y_tr, eval_set=(X_cb_va, y_va), early_stopping_rounds=50)
    oof_preds[val_f_idx, 0] = cb_model.predict_proba(X_cb_va)[:, 1]
    val_preds_folds[:, 0] += cb_model.predict_proba(X_cb_val)[:, 1] / n_folds
    test_preds_folds[:, 0] += cb_model.predict_proba(X_cb_test)[:, 1] / n_folds

    # --- B. XGBoost ---
    X_xgb_tr = d_tr[cat_cols + num_cols].copy()
    X_xgb_va = d_va[cat_cols + num_cols].copy()
    X_xgb_val = df_val[cat_cols + num_cols].copy()
    X_xgb_test = df_test[cat_cols + num_cols].copy()
    for col in cat_cols:
        X_xgb_tr[col] = X_xgb_tr[col].astype("category")
        X_xgb_va[col] = X_xgb_va[col].astype("category")
        X_xgb_val[col] = X_xgb_val[col].astype("category")
        X_xgb_test[col] = X_xgb_test[col].astype("category")

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
    xgb_model.fit(X_xgb_tr, y_tr)
    oof_preds[val_f_idx, 1] = xgb_model.predict_proba(X_xgb_va)[:, 1]
    val_preds_folds[:, 1] += xgb_model.predict_proba(X_xgb_val)[:, 1] / n_folds
    test_preds_folds[:, 1] += xgb_model.predict_proba(X_xgb_test)[:, 1] / n_folds

    # --- C. Random Forest ---
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    rf_cat_tr = ohe.fit_transform(d_tr[cat_cols].astype(str))
    rf_cat_va = ohe.transform(d_va[cat_cols].astype(str))
    rf_cat_val = ohe.transform(df_val[cat_cols].astype(str))
    rf_cat_test = ohe.transform(df_test[cat_cols].astype(str))

    X_rf_tr = np.hstack([rf_cat_tr, d_tr[num_cols].values])
    X_rf_va = np.hstack([rf_cat_va, d_va[num_cols].values])
    X_rf_val = np.hstack([rf_cat_val, df_val[num_cols].values])
    X_rf_test = np.hstack([rf_cat_test, df_test[num_cols].values])

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42 + fold,
    )
    rf_model.fit(X_rf_tr, y_tr)
    oof_preds[val_f_idx, 2] = rf_model.predict_proba(X_rf_va)[:, 1]
    val_preds_folds[:, 2] += rf_model.predict_proba(X_rf_val)[:, 1] / n_folds
    test_preds_folds[:, 2] += rf_model.predict_proba(X_rf_test)[:, 1] / n_folds

    # --- D. Tabular Neural Network ---
    cat_tr_list, cat_va_list, cat_val_list, cat_test_list, cat_dims = [], [], [], [], []
    for col in cat_cols:
        le = LabelEncoder()
        c_tr = le.fit_transform(d_tr[col].astype(str))
        mapping = {val: i for i, val in enumerate(le.classes_)}
        c_va = np.array([mapping.get(v, 0) for v in d_va[col].astype(str).values])
        c_val = np.array([mapping.get(v, 0) for v in df_val[col].astype(str).values])
        c_test = np.array([mapping.get(v, 0) for v in df_test[col].astype(str).values])

        cat_tr_list.append(c_tr)
        cat_va_list.append(c_va)
        cat_val_list.append(c_val)
        cat_test_list.append(c_test)
        cat_dims.append(len(le.classes_))

    X_nn_cat_tr = np.stack(cat_tr_list, axis=1).astype(np.int64)
    X_nn_cat_va = np.stack(cat_va_list, axis=1).astype(np.int64)
    X_nn_cat_val = np.stack(cat_val_list, axis=1).astype(np.int64)
    X_nn_cat_test = np.stack(cat_test_list, axis=1).astype(np.int64)

    scaler = StandardScaler()
    X_nn_num_tr = scaler.fit_transform(d_tr[num_cols].values.astype(np.float32))
    X_nn_num_va = scaler.transform(d_va[num_cols].values.astype(np.float32))
    X_nn_num_val = scaler.transform(df_val[num_cols].values.astype(np.float32))
    X_nn_num_test = scaler.transform(df_test[num_cols].values.astype(np.float32))

    emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

    nn_model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_tr.shape[1]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)

    t_X_cat_tr = torch.tensor(X_nn_cat_tr, dtype=torch.long, device=device)
    t_X_num_tr = torch.tensor(X_nn_num_tr, dtype=torch.float32, device=device)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long, device=device)

    t_X_cat_va = torch.tensor(X_nn_cat_va, dtype=torch.long, device=device)
    t_X_num_va = torch.tensor(X_nn_num_va, dtype=torch.float32, device=device)

    t_X_cat_val = torch.tensor(X_nn_cat_val, dtype=torch.long, device=device)
    t_X_num_val = torch.tensor(X_nn_num_val, dtype=torch.float32, device=device)

    t_X_cat_test = torch.tensor(X_nn_cat_test, dtype=torch.long, device=device)
    t_X_num_test = torch.tensor(X_nn_num_test, dtype=torch.float32, device=device)

    batch_size = 32
    for epoch in range(120):
        nn_model.train()
        permutation = torch.randperm(t_X_cat_tr.size(0))
        for i in range(0, t_X_cat_tr.size(0), batch_size):
            indices_batch = permutation[i : i + batch_size]
            batch_cat = t_X_cat_tr[indices_batch]
            batch_num = t_X_num_tr[indices_batch]
            batch_y = t_y_tr[indices_batch]

            optimizer.zero_grad()
            outputs = nn_model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    nn_model.eval()
    with torch.no_grad():
        oof_nn_out = nn_model(t_X_cat_va, t_X_num_va)
        oof_preds[val_f_idx, 3] = torch.softmax(oof_nn_out, dim=1)[:, 1].cpu().numpy()

        val_nn_out = nn_model(t_X_cat_val, t_X_num_val)
        val_preds_folds[:, 3] += (
            torch.softmax(val_nn_out, dim=1)[:, 1].cpu().numpy() / n_folds
        )

        test_nn_out = nn_model(t_X_cat_test, t_X_num_test)
        test_preds_folds[:, 3] += (
            torch.softmax(test_nn_out, dim=1)[:, 1].cpu().numpy() / n_folds
        )

# 3. Probability Calibration & Meta-Feature Engineering
calibrated_oof = np.zeros_like(oof_preds)
calibrated_val = np.zeros_like(val_preds_folds)
calibrated_test = np.zeros_like(test_preds_folds)

for m_idx in range(4):
    calibrator = LogisticRegression(C=1.0, random_state=42)
    calibrator.fit(oof_preds[:, m_idx : m_idx + 1], y_train)
    calibrated_oof[:, m_idx] = calibrator.predict_proba(
        oof_preds[:, m_idx : m_idx + 1]
    )[:, 1]
    calibrated_val[:, m_idx] = calibrator.predict_proba(
        val_preds_folds[:, m_idx : m_idx + 1]
    )[:, 1]
    calibrated_test[:, m_idx] = calibrator.predict_proba(
        test_preds_folds[:, m_idx : m_idx + 1]
    )[:, 1]

anchor_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
anchor_encoder.fit(df_train[["Pclass", "Sex", "Title", "IsAlone"]].astype(str))


def build_meta_features(calib_probs, df_source):
    features = [calib_probs]
    prob_variance = np.var(calib_probs, axis=1, keepdims=True)
    prob_std = np.std(calib_probs, axis=1, keepdims=True)
    prob_mean = np.mean(calib_probs, axis=1, keepdims=True)
    features.extend([prob_mean, prob_variance, prob_std])

    anchor_ohe = anchor_encoder.transform(
        df_source[["Pclass", "Sex", "Title", "IsAlone"]].astype(str)
    )
    features.append(anchor_ohe)

    return np.hstack(features)


meta_X_train = build_meta_features(calibrated_oof, df_train)
meta_X_val = build_meta_features(calibrated_val, df_val)
meta_X_test = build_meta_features(calibrated_test, df_test)

# 4. Regularized Stacking Meta-Learner & Optimal Decision Boundary
meta_learner = LogisticRegression(C=0.5, penalty="l2", random_state=42, max_iter=1000)
meta_learner.fit(meta_X_train, y_train)

meta_oof_probs = meta_learner.predict_proba(meta_X_train)[:, 1]
meta_val_probs = meta_learner.predict_proba(meta_X_val)[:, 1]
meta_test_probs = meta_learner.predict_proba(meta_X_test)[:, 1]

best_score = -1.0
best_threshold = 0.5
for th in np.linspace(0.1, 0.9, 161):
    preds = (meta_oof_probs >= th).astype(int)
    score = accuracy_score(y_train, preds)
    if score > best_score:
        best_score = score
        best_threshold = th

final_val_preds = (meta_val_probs >= best_threshold).astype(int)
final_validation_score = accuracy_score(y_val, final_val_preds)

print(f"Optimal Decision Threshold: {best_threshold:.4f}")
print(f"Final Validation Performance: {final_validation_score}")

# 5. Generate and Save Test Predictions
final_test_preds = (meta_test_probs >= best_threshold).astype(int)

os.makedirs("./final", exist_ok=True)
submission_df = pd.DataFrame({"Survived": final_test_preds})
submission_df.to_csv("./final/submission.csv", index=False)
print("Submission saved to ./final/submission.csv successfully.")