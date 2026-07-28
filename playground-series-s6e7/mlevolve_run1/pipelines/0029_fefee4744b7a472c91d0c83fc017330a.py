import os
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize as opt
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create required output directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Load raw datasets from ./input
train_df_raw = pd.read_csv("./input/train.csv")
test_df_raw = pd.read_csv("./input/test.csv")

# Process target variable
target_col = "health_condition"
le = LabelEncoder()
y_train = le.fit_transform(train_df_raw[target_col])
target_mapping = {str(label): int(idx) for idx, label in enumerate(le.classes_)}
inv_target_map = {v: k for k, v in target_mapping.items()}
num_classes = len(target_mapping)

test_ids = test_df_raw["id"]

# Prepare raw feature matrices
X_train_raw = train_df_raw.drop(columns=["id", target_col])
X_test_raw = test_df_raw.drop(columns=["id"])

train_len = len(train_df_raw)
combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Ordinal encodings for biological and lifestyle rankings
ordinal_maps = {
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "stress_level": {"low": 0, "medium": 1, "high": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
    "diet_type": {"veg": 0, "balanced": 1, "non-veg": 2},
    "gender": {"female": 0, "other": 1, "male": 2},
}

for col, mapping in ordinal_maps.items():
    combined_df[f"{col}_ord"] = combined_df[col].map(mapping)

# Explicit missingness flags and missingness count per sample
combined_df["missing_count"] = combined_df.isnull().sum(axis=1)
num_cols_with_na = [
    "sleep_duration",
    "calorie_expenditure",
    "water_intake",
    "bmi",
    "step_count",
    "heart_rate",
    "exercise_duration",
]
for col in num_cols_with_na:
    combined_df[f"{col}_isna"] = combined_df[col].isnull().astype(np.int8)

# Domain physiological indicator engineering
combined_df["cardiac_strain"] = (
    combined_df["heart_rate"] * combined_df["exercise_duration"]
)
combined_df["step_caloric_density"] = combined_df["calorie_expenditure"] / (
    combined_df["step_count"] + 1e-5
)
combined_df["hydration_per_bmi"] = combined_df["water_intake"] / (
    combined_df["bmi"] + 1e-5
)
combined_df["sleep_efficiency_index"] = (
    combined_df["sleep_quality_ord"] * combined_df["sleep_duration"]
)
combined_df["stress_activity_ratio"] = (combined_df["stress_level_ord"] + 1.0) / (
    combined_df["physical_activity_level_ord"] + 1.0
)
combined_df["lifestyle_risk_score"] = (
    combined_df["smoking_alcohol_ord"].fillna(1)
    + combined_df["stress_level_ord"].fillna(1)
    + (2 - combined_df["sleep_quality_ord"].fillna(1))
    + (2 - combined_df["physical_activity_level_ord"].fillna(1))
)
combined_df["estimated_bmr_proxy"] = 10 * combined_df["bmi"] + 6.25 * 165 - 5 * 35 + 5
combined_df["caloric_surplus_ratio"] = combined_df["calorie_expenditure"] / (
    combined_df["estimated_bmr_proxy"] + 1e-5
)
combined_df["step_per_minute"] = combined_df["step_count"] / (
    combined_df["exercise_duration"] + 1.0
)
combined_df["sleep_deficit"] = (8.0 - combined_df["sleep_duration"]).clip(lower=0)

# High-dimensional lifestyle categorical interaction combinations & frequency encodings
combo_cols = [
    ("stress_level", "smoking_alcohol"),
    ("diet_type", "physical_activity_level"),
    ("gender", "physical_activity_level"),
    ("sleep_quality", "stress_level"),
]

for col1, col2 in combo_cols:
    combo_name = f"{col1}_{col2}_combo"
    combined_df[combo_name] = (
        combined_df[col1].astype(str) + "_" + combined_df[col2].astype(str)
    )
    freq = combined_df.iloc[:train_len][combo_name].value_counts(normalize=True)
    combined_df[f"{combo_name}_freq"] = combined_df[combo_name].map(freq).fillna(0)
    combined_df[combo_name] = combined_df[combo_name].astype("category")

base_cat_cols = [
    "diet_type",
    "gender",
    "physical_activity_level",
    "sleep_quality",
    "smoking_alcohol",
    "stress_level",
]
for col in base_cat_cols:
    freq = combined_df.iloc[:train_len][col].value_counts(normalize=True)
    combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0)
    combined_df[col] = combined_df[col].astype(str).fillna("missing").astype("category")

# Separate train and test sets to compute cohort statistics strictly without leakage
X_train = combined_df.iloc[:train_len].copy()
X_test = combined_df.iloc[train_len:].copy()

cohort_cols = ["gender", "physical_activity_level"]
target_num_cols = [
    "bmi",
    "heart_rate",
    "step_count",
    "calorie_expenditure",
    "exercise_duration",
]

# Compute all cohort stats at once to avoid multiple dataframe merges
cohort_agg = X_train.groupby(cohort_cols, observed=False)[target_num_cols].agg(["mean", "std"])
cohort_agg.columns = [f"{col}_cohort_{stat}" for col, stat in cohort_agg.columns]
cohort_agg = cohort_agg.reset_index()

X_train = X_train.merge(cohort_agg, on=cohort_cols, how="left")
X_test = X_test.merge(cohort_agg, on=cohort_cols, how="left")

for num_col in target_num_cols:
    X_train[f"{num_col}_cohort_zscore"] = (
        X_train[num_col] - X_train[f"{num_col}_cohort_mean"]
    ) / (X_train[f"{num_col}_cohort_std"] + 1e-5)
    X_test[f"{num_col}_cohort_zscore"] = (
        X_test[num_col] - X_test[f"{num_col}_cohort_mean"]
    ) / (X_test[f"{num_col}_cohort_std"] + 1e-5)

# Construct 5-fold Stratified K-Fold splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = np.zeros(train_len, dtype=int)
for fold, (_, val_idx) in enumerate(skf.split(X_train, y_train)):
    folds[val_idx] = fold

X_train["fold"] = folds
X_train["target"] = y_train

feature_cols = [c for c in X_train.columns if c not in ["fold", "target"]]
cat_cols = [c for c in feature_cols if str(X_train[c].dtype) == "category"]

# Ensure categorical columns have missing category handled cleanly
for col in cat_cols:
    if "missing" not in X_train[col].cat.categories:
        X_train[col] = X_train[col].cat.add_categories("missing")
        X_test[col] = X_test[col].cat.add_categories("missing")
    X_train[col] = X_train[col].fillna("missing")
    X_test[col] = X_test[col].fillna("missing")

# Prepare numerical DataFrames for PyTorch Neural Network
df_train_nn = X_train[feature_cols].copy()
df_test_nn = X_test[feature_cols].copy()
for col in feature_cols:
    if str(df_train_nn[col].dtype) == "category":
        df_train_nn[col] = df_train_nn[col].cat.codes
        df_test_nn[col] = df_test_nn[col].cat.codes


# ==============================================================================
# PyTorch Deep & Cross Network v2 (DCN-v2) Architecture & Focal Loss
# ==============================================================================


class CrossNetworkLayer(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xw = torch.matmul(x, self.weight) + self.bias
        return x0 * xw + x


class TabularDCNv2(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        hidden_dims: list = [256, 128, 64],
        cross_layers: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.input_norm = nn.BatchNorm1d(num_features)
        self.cross_network = nn.ModuleList(
            [CrossNetworkLayer(num_features) for _ in range(cross_layers)]
        )

        deep_layers = []
        in_dim = num_features
        for h_dim in hidden_dims:
            deep_layers.append(nn.Linear(in_dim, h_dim))
            deep_layers.append(nn.BatchNorm1d(h_dim))
            deep_layers.append(nn.SiLU())
            deep_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.deep_network = nn.Sequential(*deep_layers)

        combined_dim = num_features + hidden_dims[-1]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(combined_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)
        x_norm = self.input_norm(x)

        x_cross = x_norm
        for layer in self.cross_network:
            x_cross = layer(x_norm, x_cross)

        x_deep = self.deep_network(x_norm)
        combined = torch.cat([x_cross, x_deep], dim=1)
        logits = self.head(self.dropout(combined))
        return logits


class FocalLossMultiClass(nn.Module):

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(target_probs + 1e-8)
        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = -focal_weight * log_probs
        return loss.mean()


class MultiClassThresholdOptimizer:

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.weights = np.ones(num_classes, dtype=np.float64)

    def fit(self, y_probs: np.ndarray, y_true: np.ndarray):

        def objective(weights):
            scaled_probs = y_probs * weights
            preds = np.argmax(scaled_probs, axis=1)
            score = metrics.balanced_accuracy_score(y_true, preds)
            return -score

        init_weights = np.ones(self.num_classes, dtype=np.float64)
        res = opt.minimize(
            objective,
            init_weights,
            method="Nelder-Mead",
            options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4},
        )
        optimized = np.maximum(res.x, 1e-5)
        self.weights = optimized / np.sum(optimized)
        return self

    def predict(self, y_probs: np.ndarray) -> np.ndarray:
        scaled_probs = y_probs * self.weights
        return np.argmax(scaled_probs, axis=1)


# Initialize arrays for fold evaluation
n_train = len(X_train)
n_test = len(X_test)
n_folds = 5

oof_lgb = np.zeros((n_train, num_classes), dtype=np.float32)
oof_xgb = np.zeros((n_train, num_classes), dtype=np.float32)
oof_cat = np.zeros((n_train, num_classes), dtype=np.float32)
oof_nn = np.zeros((n_train, num_classes), dtype=np.float32)

test_lgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_xgb = np.zeros((n_test, num_classes), dtype=np.float32)
test_cat = np.zeros((n_test, num_classes), dtype=np.float32)
test_nn = np.zeros((n_test, num_classes), dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter specifications for gradient boosted decision trees
use_gpu = torch.cuda.is_available()

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "n_estimators": 400,
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda" if use_gpu else "cpu",
    "enable_categorical": True,
}

cat_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 400,
    "learning_rate": 0.08,
    "depth": 6,
    "random_seed": SEED,
    "verbose": False,
    "thread_count": -1,
    "task_type": "GPU" if use_gpu else "CPU",
}

# Run 5-Fold Stratified Ensemble Cross-Validation Loop
for fold in range(n_folds):
    val_idx = X_train["fold"] == fold
    tr_idx = ~val_idx

    X_tr, y_tr = (
        X_train.loc[tr_idx, feature_cols],
        X_train.loc[tr_idx, "target"].values,
    )
    X_va, y_va = (
        X_train.loc[val_idx, feature_cols],
        X_train.loc[val_idx, "target"].values,
    )

    # 1. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)
    test_lgb += model_lgb.predict_proba(X_test[feature_cols]) / n_folds

    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=30)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
    test_xgb += model_xgb.predict_proba(X_test[feature_cols]) / n_folds

    # 3. CatBoost
    model_cat = CatBoostClassifier(**cat_params, early_stopping_rounds=30)
    model_cat.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_cols, verbose=False
    )
    oof_cat[val_idx] = model_cat.predict_proba(X_va)
    test_cat += model_cat.predict_proba(X_test[feature_cols]) / n_folds

    # 4. PyTorch TabularDCNv2
    X_tr_nn_df = df_train_nn.loc[tr_idx, feature_cols]
    X_va_nn_df = df_train_nn.loc[val_idx, feature_cols]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(np.nan_to_num(X_tr_nn_df.values, nan=0.0))
    X_va_scaled = scaler.transform(np.nan_to_num(X_va_nn_df.values, nan=0.0))
    X_te_scaled = scaler.transform(np.nan_to_num(df_test_nn.values, nan=0.0))

    tr_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    va_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.long),
    )

    tr_loader = DataLoader(
        tr_ds, batch_size=8192, shuffle=True, drop_last=False, num_workers=2, pin_memory=use_gpu
    )
    va_loader = DataLoader(va_ds, batch_size=16384, shuffle=False, num_workers=2, pin_memory=use_gpu)

    model_nn = TabularDCNv2(
        num_features=X_tr_scaled.shape[1], num_classes=num_classes
    ).to(device)
    criterion = FocalLossMultiClass(gamma=2.0)
    optimizer_nn = torch.optim.AdamW(model_nn.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_nn, T_max=6)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(6):
        model_nn.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_nn.zero_grad()
            logits = model_nn(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
            optimizer_nn.step()
        scheduler.step()

        model_nn.eval()
        v_loss = 0.0
        v_count = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model_nn(bx)
                v_loss += criterion(logits, by).item() * len(by)
                v_count += len(by)
        v_loss /= v_count

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_nn_weights = {k: v.clone().cpu() for k, v in model_nn.state_dict().items()}

    model_nn.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    model_nn.eval()

    with torch.no_grad():
        va_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        nn_va_probs = (
            F.softmax(model_nn(va_tensor), dim=1).cpu().numpy().astype(np.float32)
        )
        oof_nn[val_idx] = nn_va_probs

        te_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        nn_te_probs = (
            F.softmax(model_nn(te_tensor), dim=1).cpu().numpy().astype(np.float32)
        )
        test_nn += nn_te_probs / n_folds

# Multi-modal Probability Blending
oof_blend = 0.35 * oof_lgb + 0.30 * oof_xgb + 0.25 * oof_cat + 0.10 * oof_nn
test_blend = 0.35 * test_lgb + 0.30 * test_xgb + 0.25 * test_cat + 0.10 * test_nn

# Post-hoc Nelder-Mead Threshold Calibration for Balanced Accuracy Alignment
thresh_opt = MultiClassThresholdOptimizer(num_classes=num_classes)
thresh_opt.fit(oof_blend, y_train)

final_oof_preds = thresh_opt.predict(oof_blend)
final_score = metrics.balanced_accuracy_score(y_train, final_oof_preds)

# Test prediction generation and submission file export
final_test_preds = thresh_opt.predict(test_blend)
test_str_preds = [inv_target_map[int(p)] for p in final_test_preds]

sub_df = pd.DataFrame({"id": test_ids, "health_condition": test_str_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")