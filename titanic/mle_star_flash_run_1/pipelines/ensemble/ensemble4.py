import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# 1. Enriched Feature Extraction
def prepare_features(df):
    df = df.copy()

    # Standardized Title categories
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

    # Cabin deck extraction
    df["Deck"] = df["Cabin"].map(
        lambda x: str(x)[0] if pd.notna(x) and len(str(x)) > 0 else "Missing"
    )
    df["HasCabin"] = df["Cabin"].notna().astype(int)

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

    # Ticket frequency to identify shared ticket groups
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts()).fillna(1)

    # Fare handling and per-person fare
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Missing value handling for Embarked
    df["Embarked"] = df["Embarked"].fillna("Missing")

    cat_cols = ["Pclass", "Sex", "Embarked", "Title", "Sex_Pclass", "FamilyTier", "Deck"]
    num_cols = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FarePerPerson",
        "FamilySize",
        "HasCabin",
        "TicketFreq",
    ]

    return df, cat_cols, num_cols


# Load data
train_df = pd.read_csv("./input/train.csv")
df_processed, cat_cols, num_cols = prepare_features(train_df)
y = train_df["Survived"].values

# Define consistent categorical types across the entire dataset
cat_dtypes = {
    col: pd.CategoricalDtype(categories=sorted(df_processed[col].astype(str).unique()))
    for col in cat_cols
}

# Train/Validation Split (Hold-out validation)
indices = np.arange(len(y))
idx_train, idx_val, y_train, y_val = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

df_train = df_processed.iloc[idx_train].reset_index(drop=True)
df_val = df_processed.iloc[idx_val].reset_index(drop=True)


# Define PyTorch Tabular Neural Network
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


# 2. Multi-Architecture Stratified K-Fold Bagging
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(len(y_train))
oof_xgb = np.zeros(len(y_train))
oof_rf = np.zeros(len(y_train))
oof_nn = np.zeros(len(y_train))

val_cb_folds = np.zeros((n_splits, len(y_val)))
val_xgb_folds = np.zeros((n_splits, len(y_val)))
val_rf_folds = np.zeros((n_splits, len(y_val)))
val_nn_folds = np.zeros((n_splits, len(y_val)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (trn_idx, val_fold_idx) in enumerate(skf.split(df_train, y_train)):
    fold_train_df = df_train.iloc[trn_idx]
    fold_val_df = df_train.iloc[val_fold_idx]

    y_f_train = y_train[trn_idx]
    y_f_val = y_train[val_fold_idx]

    # --- CatBoost Preparation & Training ---
    X_cb_train = fold_train_df[cat_cols + num_cols].copy()
    X_cb_val = fold_val_df[cat_cols + num_cols].copy()
    X_cb_test = df_val[cat_cols + num_cols].copy()

    for col in cat_cols:
        X_cb_train[col] = X_cb_train[col].astype(str)
        X_cb_val[col] = X_cb_val[col].astype(str)
        X_cb_test[col] = X_cb_test[col].astype(str)

    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=5,
        cat_features=cat_cols,
        eval_metric="Logloss",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(
        X_cb_train,
        y_f_train,
        eval_set=(X_cb_val, y_f_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    oof_cb[val_fold_idx] = cb_model.predict_proba(X_cb_val)[:, 1]
    val_cb_folds[fold] = cb_model.predict_proba(X_cb_test)[:, 1]

    # --- XGBoost Preparation & Training ---
    X_xgb_train = fold_train_df[cat_cols + num_cols].copy()
    X_xgb_val = fold_val_df[cat_cols + num_cols].copy()
    X_xgb_test = df_val[cat_cols + num_cols].copy()

    for col in cat_cols:
        X_xgb_train[col] = X_xgb_train[col].astype(str).astype(cat_dtypes[col])
        X_xgb_val[col] = X_xgb_val[col].astype(str).astype(cat_dtypes[col])
        X_xgb_test[col] = X_xgb_test[col].astype(str).astype(cat_dtypes[col])

    xgb_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42 + fold,
        eval_metric="logloss",
        early_stopping_rounds=40,
    )
    xgb_model.fit(
        X_xgb_train,
        y_f_train,
        eval_set=[(X_xgb_val, y_f_val)],
        verbose=False,
    )

    oof_xgb[val_fold_idx] = xgb_model.predict_proba(X_xgb_val)[:, 1]
    val_xgb_folds[fold] = xgb_model.predict_proba(X_xgb_test)[:, 1]

    # --- Random Forest Preparation & Training ---
    full_cat_df = pd.concat(
        [fold_train_df[cat_cols], fold_val_df[cat_cols], df_val[cat_cols]], axis=0
    )
    full_cat_dummies = pd.get_dummies(full_cat_df, drop_first=True)

    n_trn = len(fold_train_df)
    n_val_f = len(fold_val_df)
    n_val_all = len(df_val)

    X_rf_train = pd.concat(
        [
            fold_train_df[num_cols].reset_index(drop=True),
            full_cat_dummies.iloc[:n_trn].reset_index(drop=True),
        ],
        axis=1,
    )
    X_rf_val = pd.concat(
        [
            fold_val_df[num_cols].reset_index(drop=True),
            full_cat_dummies.iloc[n_trn : n_trn + n_val_f].reset_index(drop=True),
        ],
        axis=1,
    )
    X_rf_test = pd.concat(
        [
            df_val[num_cols].reset_index(drop=True),
            full_cat_dummies.iloc[n_trn + n_val_f :].reset_index(drop=True),
        ],
        axis=1,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf_model.fit(X_rf_train, y_f_train)

    oof_rf[val_fold_idx] = rf_model.predict_proba(X_rf_val)[:, 1]
    val_rf_folds[fold] = rf_model.predict_proba(X_rf_test)[:, 1]

    # --- Tabular Neural Network Preparation & Training ---
    cat_arrays_trn, cat_arrays_val_f, cat_arrays_test, cat_dims = [], [], [], []
    for col in cat_cols:
        le = LabelEncoder()
        trn_col = fold_train_df[col].astype(str)
        le.fit(trn_col)

        val_col = fold_val_df[col].astype(str).values
        test_col = df_val[col].astype(str).values

        mapping = {c: i for i, c in enumerate(le.classes_)}
        trn_encoded = np.array([mapping[c] for c in trn_col])
        val_encoded = np.array([mapping.get(c, 0) for c in val_col])
        test_encoded = np.array([mapping.get(c, 0) for c in test_col])

        cat_arrays_trn.append(trn_encoded)
        cat_arrays_val_f.append(val_encoded)
        cat_arrays_test.append(test_encoded)
        cat_dims.append(len(le.classes_))

    X_nn_cat_trn = np.stack(cat_arrays_trn, axis=1).astype(np.int64)
    X_nn_cat_val_f = np.stack(cat_arrays_val_f, axis=1).astype(np.int64)
    X_nn_cat_test = np.stack(cat_arrays_test, axis=1).astype(np.int64)

    scaler = StandardScaler()
    X_nn_num_trn = scaler.fit_transform(
        fold_train_df[num_cols].values.astype(np.float32)
    )
    X_nn_num_val_f = scaler.transform(fold_val_df[num_cols].values.astype(np.float32))
    X_nn_num_test = scaler.transform(df_val[num_cols].values.astype(np.float32))

    emb_dims = [max(2, int(round(dim**0.5 * 2))) for dim in cat_dims]

    nn_model = TabularNeuralNet(
        cat_dims, emb_dims, num_features=X_nn_num_trn.shape[1]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-2, weight_decay=1e-4)

    t_X_cat_trn = torch.tensor(X_nn_cat_trn, dtype=torch.long, device=device)
    t_X_num_trn = torch.tensor(X_nn_num_trn, dtype=torch.float32, device=device)
    t_y_trn = torch.tensor(y_f_train, dtype=torch.long, device=device)

    t_X_cat_val_f = torch.tensor(X_nn_cat_val_f, dtype=torch.long, device=device)
    t_X_num_val_f = torch.tensor(X_nn_num_val_f, dtype=torch.float32, device=device)

    t_X_cat_test = torch.tensor(X_nn_cat_test, dtype=torch.long, device=device)
    t_X_num_test = torch.tensor(X_nn_num_test, dtype=torch.float32, device=device)

    batch_size = 32
    num_epochs = 120

    for epoch in range(num_epochs):
        nn_model.train()
        permutation = torch.randperm(t_X_cat_trn.size(0))
        for i in range(0, t_X_cat_trn.size(0), batch_size):
            b_indices = permutation[i : i + batch_size]
            b_cat = t_X_cat_trn[b_indices]
            b_num = t_X_num_trn[b_indices]
            b_y = t_y_trn[b_indices]

            optimizer.zero_grad()
            outputs = nn_model(b_cat, b_num)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()

    nn_model.eval()
    with torch.no_grad():
        val_f_out = nn_model(t_X_cat_val_f, t_X_num_val_f)
        oof_nn[val_fold_idx] = torch.softmax(val_f_out, dim=1)[:, 1].cpu().numpy()

        test_out = nn_model(t_X_cat_test, t_X_num_test)
        val_nn_folds[fold] = torch.softmax(test_out, dim=1)[:, 1].cpu().numpy()

# Fold-averaged predictions on holdout validation set
val_cb = np.mean(val_cb_folds, axis=0)
val_xgb = np.mean(val_xgb_folds, axis=0)
val_rf = np.mean(val_rf_folds, axis=0)
val_nn = np.mean(val_nn_folds, axis=0)

# 3. Constrained Non-Negative Least Squares (NNLS) on Probability Simplex
OOF_matrix = np.column_stack([oof_cb, oof_xgb, oof_rf, oof_nn])
VAL_matrix = np.column_stack([val_cb, val_xgb, val_rf, val_nn])
n_models = OOF_matrix.shape[1]


def brier_objective(weights):
    w = np.array(weights)
    pred = np.dot(OOF_matrix, w)
    return np.mean((pred - y_train) ** 2)


init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    brier_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

optimal_weights = opt_res.x
optimal_weights = optimal_weights / np.sum(optimal_weights)

# 4. Calibrated Decision Boundary Optimization
oof_ensemble_probs = np.dot(OOF_matrix, optimal_weights)

best_threshold = 0.5
best_oof_acc = -1.0
threshold_grid = np.linspace(0.35, 0.65, 301)

for th in threshold_grid:
    score = accuracy_score(y_train, (oof_ensemble_probs >= th).astype(int))
    if score > best_oof_acc:
        best_oof_acc = score
        best_threshold = th

# Validation prediction and final score
val_ensemble_probs = np.dot(VAL_matrix, optimal_weights)
final_val_preds = (val_ensemble_probs >= best_threshold).astype(int)
final_validation_score = accuracy_score(y_val, final_val_preds)

print(f"Final Validation Performance: {final_validation_score}")