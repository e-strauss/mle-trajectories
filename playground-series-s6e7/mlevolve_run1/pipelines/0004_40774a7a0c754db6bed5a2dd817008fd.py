import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. DATA LOADING & STRICT TRAIN-VALIDATION SPLIT
# ==============================================================================
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "health_condition"
le_target = LabelEncoder()
y_full = le_target.fit_transform(train_df[target_col])
classes = le_target.classes_
train_df = train_df.drop(columns=[target_col])

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=y_full,
)

df_train = train_df.iloc[train_idx].copy().reset_index(drop=True)
y_train = y_full[train_idx]

df_val = train_df.iloc[val_idx].copy().reset_index(drop=True)
y_val = y_full[val_idx]

df_test = test_df.copy().reset_index(drop=True)

# ==============================================================================
# 2. FEATURE ENGINEERING & DOMAIN MAPPINGS
# ==============================================================================
stress_map = {"low": 0, "medium": 1, "high": 2}
sleep_qual_map = {"poor": 0, "average": 1, "good": 2}
activity_map = {"sedentary": 0, "moderate": 1, "active": 2}
smoking_map = {"no": 0, "occasional": 1, "yes": 2}
diet_map = {"veg": 0, "balanced": 1, "non-veg": 2}
gender_map = {"female": 0, "male": 1, "other": 2}


def engineer_features(df):
    df = df.copy()

    # Ordinal Encoding
    df["stress_level_ord"] = df["stress_level"].map(stress_map).astype("float32")
    df["sleep_quality_ord"] = df["sleep_quality"].map(sleep_qual_map).astype("float32")
    df["activity_level_ord"] = (
        df["physical_activity_level"].map(activity_map).astype("float32")
    )
    df["smoking_alcohol_ord"] = df["smoking_alcohol"].map(smoking_map).astype("float32")
    df["diet_type_ord"] = df["diet_type"].map(diet_map).astype("float32")
    df["gender_ord"] = df["gender"].map(gender_map).astype("float32")

    # Missing Value Indicators
    for col in [
        "sleep_duration",
        "calorie_expenditure",
        "water_intake",
        "bmi",
        "step_count",
        "heart_rate",
        "exercise_duration",
    ]:
        df[f"{col}_isna"] = df[col].isna().astype("float32")

    # Physiological & Behavioral Ratios
    df["caloric_burn_per_step"] = (
        df["calorie_expenditure"] / (df["step_count"] + 1.0)
    ).astype("float32")
    df["exercise_intensity"] = (
        df["calorie_expenditure"] / (df["exercise_duration"] + 1.0)
    ).astype("float32")
    df["cardiac_cost"] = (df["heart_rate"] * df["exercise_duration"]).astype("float32")
    df["heart_rate_to_activity"] = (
        df["heart_rate"] / (df["activity_level_ord"] + 1.0)
    ).astype("float32")
    df["hydration_per_bmi"] = (df["water_intake"] / (df["bmi"] + 1e-3)).astype(
        "float32"
    )
    df["hydration_per_exercise"] = (
        df["water_intake"] / (df["exercise_duration"] + 1.0)
    ).astype("float32")
    df["sleep_recovery_score"] = (
        df["sleep_duration"] * (df["sleep_quality_ord"] + 1.0)
    ).astype("float32")
    df["step_to_exercise_ratio"] = (
        df["step_count"] / (df["exercise_duration"] + 1.0)
    ).astype("float32")
    df["metabolic_rate_proxy"] = (df["calorie_expenditure"] / (df["bmi"] + 1.0)).astype(
        "float32"
    )

    # Cardiovascular Efficiency Features
    df["calorie_per_heart_rate"] = (
        df["calorie_expenditure"] / (df["heart_rate"] + 1.0)
    ).astype("float32")
    df["exercise_per_heart_rate"] = (
        df["exercise_duration"] / (df["heart_rate"] + 1.0)
    ).astype("float32")
    df["step_per_sleep_duration"] = (
        df["step_count"] / (df["sleep_duration"] + 1.0)
    ).astype("float32")

    # Composite Risk Index
    df["composite_risk"] = (
        df["stress_level_ord"].fillna(1.0)
        + df["smoking_alcohol_ord"].fillna(1.0)
        - df["sleep_quality_ord"].fillna(1.0)
        - df["activity_level_ord"].fillna(1.0)
    ).astype("float32")

    # Discrete Clinical Risk Bins
    df["high_hr_flag"] = (df["heart_rate"] > 90).astype("float32")
    df["low_sleep_flag"] = (df["sleep_duration"] < 6).astype("float32")
    df["low_steps_flag"] = (df["step_count"] < 5000).astype("float32")

    bmi_bins = [-1, 18.5, 24.9, 29.9, 100]
    df["bmi_category"] = (
        pd.cut(df["bmi"], bins=bmi_bins, labels=[0, 1, 2, 3])
        .astype("float32")
        .fillna(-1)
    )

    return df


X_train_raw = engineer_features(df_train)
X_val_raw = engineer_features(df_val)
X_test_raw = engineer_features(df_test)

# Subgroup Statistics (Fit strictly on Train Split)
group_cols = ["gender", "physical_activity_level"]
agg_cols = ["bmi", "heart_rate", "step_count", "calorie_expenditure"]

group_stats = (
    df_train.groupby(group_cols)[agg_cols].agg(["mean", "std"]).astype("float32")
)
group_stats.columns = [f"{col}_{stat}" for col, stat in group_stats.columns]
group_stats = group_stats.reset_index()

# Multi-categorical Group Aggregations (diet_type and stress_level)
group_cols_ds = ["diet_type", "stress_level"]
agg_cols_ds = ["bmi", "heart_rate", "step_count"]

group_stats_ds = (
    df_train.groupby(group_cols_ds)[agg_cols_ds].agg(["mean", "std"]).astype("float32")
)
group_stats_ds.columns = [f"{col}_ds_{stat}" for col, stat in group_stats_ds.columns]
group_stats_ds = group_stats_ds.reset_index()


def merge_group_stats(df, stats_df, group_cols):
    return pd.merge(df, stats_df, on=group_cols, how="left")


X_train = merge_group_stats(X_train_raw, group_stats, group_cols)
X_train = merge_group_stats(X_train, group_stats_ds, group_cols_ds)

X_val = merge_group_stats(X_val_raw, group_stats, group_cols)
X_val = merge_group_stats(X_val, group_stats_ds, group_cols_ds)

X_test = merge_group_stats(X_test_raw, group_stats, group_cols)
X_test = merge_group_stats(X_test, group_stats_ds, group_cols_ds)

cat_cols = [
    "gender",
    "diet_type",
    "physical_activity_level",
    "sleep_quality",
    "smoking_alcohol",
    "stress_level",
]

for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna("missing").astype("category")
    cat_type = pd.CategoricalDtype(categories=X_train[col].cat.categories, ordered=False)
    X_val[col] = X_val[col].astype(str).fillna("missing").astype(cat_type)
    X_test[col] = X_test[col].astype(str).fillna("missing").astype(cat_type)

drop_cols = ["id"]
feature_cols = [c for c in X_train.columns if c not in drop_cols]

X_train_feat = X_train[feature_cols].copy()
X_val_feat = X_val[feature_cols].copy()
X_test_feat = X_test[feature_cols].copy()

# ==============================================================================
# 3. MODEL ARCHITECTURES & DEFINITIONS
# ==============================================================================
# LightGBM
lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "class_weight": "balanced",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# XGBoost
xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "enable_categorical": True,
    "tree_method": "hist",
}
xgb_model = xgb.XGBClassifier(**xgb_params)

# CatBoost
cb_params = {
    "loss_function": "MultiClass",
    "auto_class_weights": "Balanced",
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}
cb_model = cb.CatBoostClassifier(**cb_params)


# PyTorch Tabular ResNet & Focal Loss
class ResBlock(nn.Module):

    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self, input_dim, num_classes=3, hidden_dim=256, num_blocks=3, dropout=0.2
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class WeightedFocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


# ==============================================================================
# 4. MODEL TRAINING & PREDICTION GENERATION
# ==============================================================================
# Train LightGBM
print("Training LightGBM model...")
lgb_model.fit(
    X_train_feat,
    y_train,
    eval_set=[(X_val_feat, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
lgb_val_probs = lgb_model.predict_proba(X_val_feat)
lgb_test_probs = lgb_model.predict_proba(X_test_feat)

# Train XGBoost
print("Training XGBoost model...")
xgb_model.fit(
    X_train_feat,
    y_train,
    eval_set=[(X_val_feat, y_val)],
    verbose=False,
)
xgb_val_probs = xgb_model.predict_proba(X_val_feat)
xgb_test_probs = xgb_model.predict_proba(X_test_feat)

# Train CatBoost
print("Training CatBoost model...")
cb_cat_cols = [c for c in cat_cols if c in X_train_feat.columns]
cb_model.fit(
    X_train_feat,
    y_train,
    cat_features=cb_cat_cols,
    eval_set=(X_val_feat, y_val),
    early_stopping_rounds=50,
    verbose=False,
)
cb_val_probs = cb_model.predict_proba(X_val_feat)
cb_test_probs = cb_model.predict_proba(X_test_feat)

# Train PyTorch Tabular ResNet
print("Preparing PyTorch numerical data...")
X_tr_copy = X_train_feat.copy()
X_va_copy = X_val_feat.copy()
X_te_copy = X_test_feat.copy()

for col in cat_cols:
    X_tr_copy[col] = X_tr_copy[col].cat.codes.astype(np.float32)
    X_va_copy[col] = X_va_copy[col].cat.codes.astype(np.float32)
    X_te_copy[col] = X_te_copy[col].cat.codes.astype(np.float32)

train_means = X_tr_copy.mean()
X_tr_num = X_tr_copy.fillna(train_means).values.astype(np.float32)
X_va_num = X_va_copy.fillna(train_means).values.astype(np.float32)
X_te_num = X_te_copy.fillna(train_means).values.astype(np.float32)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_num)
X_va_scaled = scaler.transform(X_va_num)
X_te_scaled = scaler.transform(X_te_num)

class_counts = np.bincount(y_train)
class_weights_arr = len(y_train) / (len(class_counts) * class_counts)
class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = TabularResNet(
    input_dim=X_tr_scaled.shape[1],
    num_classes=3,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.2,
).to(device)

criterion = WeightedFocalLoss(alpha=class_weights_tensor, gamma=2.0)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-5
)

train_dataset = TensorDataset(
    torch.tensor(X_tr_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(X_va_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0)

best_val_loss = float("inf")
best_nn_weights = None

print("Training PyTorch Tabular ResNet...")
epochs = 10
for epoch in range(1, epochs + 1):
    nn_model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = nn_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(by)
    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            out = nn_model(bx)
            loss = criterion(out, by)
            val_loss += loss.item() * len(by)

    val_loss /= len(val_dataset)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_nn_weights = {k: v.cpu().clone() for k, v in nn_model.state_dict().items()}

    print(
        f"Epoch {epoch}/{epochs} - Train Loss: {total_loss/len(train_dataset):.4f} - Val Loss: {val_loss:.4f}"
    )

nn_model.load_state_dict(best_nn_weights)
nn_model.eval()

val_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
test_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    nn_val_probs = (
        F.softmax(nn_model(val_tensor), dim=1).cpu().numpy().astype(np.float32)
    )
    nn_test_probs = (
        F.softmax(nn_model(test_tensor), dim=1).cpu().numpy().astype(np.float32)
    )

# ==============================================================================
# 5. ENSEMBLE BLENDING & THRESHOLD OPTIMIZATION
# ==============================================================================
val_probs_blend = (
    lgb_val_probs * 0.35
    + xgb_val_probs * 0.30
    + cb_val_probs * 0.25
    + nn_val_probs * 0.10
)
test_probs_blend = (
    lgb_test_probs * 0.35
    + xgb_test_probs * 0.30
    + cb_test_probs * 0.25
    + nn_test_probs * 0.10
)


def loss_func(weights):
    w_probs = val_probs_blend * weights
    preds = np.argmax(w_probs, axis=1)
    return -balanced_accuracy_score(y_val, preds)


init_weights = [1.0, 1.0, 1.0]
res = minimize(loss_func, init_weights, method="Nelder-Mead", options={"maxiter": 200})
best_weights = res.x

val_preds_opt = np.argmax(val_probs_blend * best_weights, axis=1)
val_score = balanced_accuracy_score(y_val, val_preds_opt)

# ==============================================================================
# 6. FINAL SUBMISSION GENERATION
# ==============================================================================
test_preds_opt = np.argmax(test_probs_blend * best_weights, axis=1)
test_preds_labels = classes[test_preds_opt]

os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {"id": df_test["id"], "health_condition": test_preds_labels}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_score}")