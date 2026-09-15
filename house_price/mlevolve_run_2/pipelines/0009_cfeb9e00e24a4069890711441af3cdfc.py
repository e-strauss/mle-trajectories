import gc
import json
import math
import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------------------------------------------------------------
# Directories and Paths
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

print("Starting property transaction price prediction pipeline...")

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
print("Loading train.parquet and test.csv...")
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
print(f"Loaded train: {len(df_train):,} rows | test: {len(df_test):,} rows.")

# -------------------------------------------------------------------------
# 2. Data Cleaning & Type Normalization
# -------------------------------------------------------------------------
str_cols = [
    "town",
    "district",
    "county",
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
]
for col in str_cols:
    df_train[col] = df_train[col].astype(str).str.strip().str.upper()
    df_test[col] = df_test[col].astype(str).str.strip().str.upper()

# Filter nominal non-market transfer outliers (< £500 or > £100M)
valid_mask = (df_train["price"] >= 500) & (df_train["price"] <= 100_000_000)
print(
    f"Filtered {len(df_train) - valid_mask.sum():,} anomalous records (< £500 or > £100M)."
)
df_train = df_train[valid_mask].copy()

# Parse completion dates
df_train["date"] = pd.to_datetime(df_train["date"])
df_test["date"] = pd.to_datetime(df_test["date"])

# Target variable: log10(price) aligned with the official metric
df_train["log_price"] = np.log10(df_train["price"].astype(np.float64))

# -------------------------------------------------------------------------
# 3. Global Historical Priors across the Full Corpus
# -------------------------------------------------------------------------
print("Extracting historical liquidity and appreciation priors from full history...")
hist_district_counts = df_train["district"].value_counts().to_dict()
hist_town_counts = df_train["town"].value_counts().to_dict()

# Calculate long-term appreciation slope (2012-2016 vs 2000-2005)
df_train["trans_year"] = df_train["date"].dt.year
old_period = (
    df_train[df_train["trans_year"].between(2000, 2005)]
    .groupby("district")["log_price"]
    .median()
)
recent_period = (
    df_train[df_train["trans_year"].between(2012, 2016)]
    .groupby("district")["log_price"]
    .median()
)
district_growth_prior = (recent_period - old_period).fillna(0.0).to_dict()

# -------------------------------------------------------------------------
# 4. Regime Partitioning (Modern Market Dynamic 2012-2016)
# -------------------------------------------------------------------------
df_train_modern = df_train[df_train["trans_year"] >= 2012].copy().reset_index(drop=True)
del df_train
gc.collect()
print(f"Modern regime training records (2012-2016): {len(df_train_modern):,} rows.")

# -------------------------------------------------------------------------
# 5. Temporal & Hedonic Interaction Feature Engineering
# -------------------------------------------------------------------------
base_date = pd.Timestamp("2012-01-01")


def generate_temporal_and_interaction_features(df):
    feats = pd.DataFrame(index=df.index)

    # Temporal coordinates
    feats["year"] = df["date"].dt.year.astype(np.int16)
    feats["month"] = df["date"].dt.month.astype(np.int8)
    feats["day"] = df["date"].dt.day.astype(np.int8)
    feats["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    feats["quarter"] = df["date"].dt.quarter.astype(np.int8)
    feats["dayofyear"] = df["date"].dt.dayofyear.astype(np.int16)

    # Continuous time representation (fractional years since base_date for trend extrapolation)
    feats["time_elapsed"] = ((df["date"] - base_date).dt.days / 365.25).astype(
        np.float32
    )

    # Seasonality Fourier harmonics
    feats["sin_month"] = np.sin(2 * np.pi * feats["month"] / 12.0).astype(np.float32)
    feats["cos_month"] = np.cos(2 * np.pi * feats["month"] / 12.0).astype(np.float32)
    feats["sin_doy"] = np.sin(2 * np.pi * feats["dayofyear"] / 365.25).astype(
        np.float32
    )
    feats["cos_doy"] = np.cos(2 * np.pi * feats["dayofyear"] / 365.25).astype(
        np.float32
    )

    # Historical liquidity features
    feats["hist_district_count"] = (
        df["district"].map(hist_district_counts).fillna(1).astype(np.float32)
    )
    feats["hist_town_count"] = (
        df["town"].map(hist_town_counts).fillna(1).astype(np.float32)
    )
    feats["hist_district_growth"] = (
        df["district"].map(district_growth_prior).fillna(0.0).astype(np.float32)
    )

    # High-signal interaction combinations
    feats["cat_prop_tenure"] = df["property_type"] + "_" + df["tenure"]
    feats["cat_prop_new"] = df["property_type"] + "_" + df["is_new_build"]
    feats["cat_dist_prop"] = df["district"] + "_" + df["property_type"]
    feats["cat_town_prop"] = df["town"] + "_" + df["property_type"]
    feats["cat_county_prop"] = df["county"] + "_" + df["property_type"]

    # Raw categoricals
    feats["property_type"] = df["property_type"]
    feats["is_new_build"] = df["is_new_build"]
    feats["tenure"] = df["tenure"]
    feats["sale_category"] = df["sale_category"]
    feats["county"] = df["county"]
    feats["district"] = df["district"]
    feats["town"] = df["town"]

    return feats


print("Extracting base features for train and test...")
X_train_raw = generate_temporal_and_interaction_features(df_train_modern)
X_test_raw = generate_temporal_and_interaction_features(df_test)

y_train_full = df_train_modern["log_price"].values.astype(np.float32)
train_dates = df_train_modern["date"]

# -------------------------------------------------------------------------
# 6. Hierarchical Empirical Bayes Target Encoding (Leakage-Free)
# -------------------------------------------------------------------------
print("Computing multi-resolution Hierarchical Empirical Bayes target encodings...")
target_enc_cols = [
    "district",
    "town",
    "county",
    "cat_dist_prop",
    "cat_town_prop",
    "cat_prop_tenure",
    "cat_prop_new",
    "sale_category",
]


def compute_smoothed_means(train_df, group_col, target_col, smoothing=25.0):
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(group_col)[target_col].agg(["count", "mean"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    return smoothed.to_dict(), float(global_mean)


# 5-fold Out-Of-Fold target encodings for train to prevent target leakage
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_encoded = {
    f"te_{col}": np.zeros(len(df_train_modern), dtype=np.float32)
    for col in target_enc_cols
}

for train_idx, val_idx in kf.split(df_train_modern):
    fold_train = df_train_modern.iloc[train_idx]
    for col in target_enc_cols:
        col_values_train = X_train_raw[col].iloc[train_idx]
        col_values_val = X_train_raw[col].iloc[val_idx]

        temp_df = pd.DataFrame(
            {col: col_values_train, "target": fold_train["log_price"]}
        )
        encoding_dict, global_mean = compute_smoothed_means(
            temp_df, col, "target", smoothing=25.0
        )
        oof_encoded[f"te_{col}"][val_idx] = (
            col_values_val.map(encoding_dict).fillna(global_mean).values
        )

for col in target_enc_cols:
    X_train_raw[f"te_{col}"] = oof_encoded[f"te_{col}"]

# Compute final production target statistics over entire modern training set for test transformation
for col in target_enc_cols:
    temp_df = pd.DataFrame(
        {
            col: X_train_raw[col].values,
            "target": df_train_modern["log_price"].values,
        }
    )
    encoding_dict, global_mean = compute_smoothed_means(
        temp_df, col, "target", smoothing=25.0
    )
    X_test_raw[f"te_{col}"] = (
        X_test_raw[col].map(encoding_dict).fillna(global_mean).astype(np.float32).values
    )

# -------------------------------------------------------------------------
# 7. Categorical Frequency & Ordinal Encoding
# -------------------------------------------------------------------------
print("Applying categorical frequency and label encoding...")
cat_features = [
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
    "county",
    "district",
    "town",
    "cat_prop_tenure",
    "cat_prop_new",
    "cat_dist_prop",
    "cat_town_prop",
    "cat_county_prop",
]

for col in cat_features:
    freq_map = X_train_raw[col].value_counts().to_dict()
    X_train_raw[f"freq_{col}"] = (
        X_train_raw[col].map(freq_map).fillna(0).astype(np.float32)
    )
    X_test_raw[f"freq_{col}"] = (
        X_test_raw[col].map(freq_map).fillna(0).astype(np.float32)
    )

    categories = {
        cat: idx for idx, cat in enumerate(X_train_raw[col].astype(str).unique())
    }
    X_train_raw[f"{col}_code"] = (
        X_train_raw[col].astype(str).map(categories).fillna(-1).astype(np.int32)
    )
    X_test_raw[f"{col}_code"] = (
        X_test_raw[col].astype(str).map(categories).fillna(-1).astype(np.int32)
    )

drop_cols = cat_features
X_train_final = X_train_raw.drop(columns=drop_cols)
X_test_final = X_test_raw.drop(columns=drop_cols)

# -------------------------------------------------------------------------
# 8. Define Out-of-Time Train/Validation Splits
# -------------------------------------------------------------------------
val_mask = (train_dates.dt.year == 2016).values
train_mask = (train_dates.dt.year < 2016).values

print(
    f"Split sizes: Train (2012-2015) = {train_mask.sum():,} rows | "
    f"Validation (2016) = {val_mask.sum():,} rows | "
    f"Test (2017 H1) = {len(X_test_final):,} rows."
)

# For pure out-of-time validation without any future leakage, recompute target encodings strictly on < 2016
for col in target_enc_cols:
    temp_df = pd.DataFrame(
        {
            col: X_train_raw.loc[train_mask, col].values,
            "target": y_train_full[train_mask],
        }
    )
    encoding_dict, global_mean = compute_smoothed_means(
        temp_df, col, "target", smoothing=25.0
    )
    X_train_final.loc[val_mask, f"te_{col}"] = (
        X_train_raw.loc[val_mask, col]
        .map(encoding_dict)
        .fillna(global_mean)
        .astype(np.float32)
        .values
    )

test_ids = df_test["id"].values

# -------------------------------------------------------------------------
# 9. Model Architecture Specification
# -------------------------------------------------------------------------
all_features = list(X_train_final.columns)
cat_code_cols = [c for c in all_features if c.endswith("_code")]
extrap_cols = [
    c
    for c in ["time_elapsed", "sin_month", "cos_month", "sin_doy", "cos_doy"]
    if c in all_features
]
cont_cols = [c for c in all_features if c not in cat_code_cols]

cat_cardinalities = {}
for c in cat_code_cols:
    max_val = max(int(X_train_final[c].max()), int(X_test_final[c].max()))
    cat_cardinalities[c] = max(1, max_val + 1)

extrap_indices = [cont_cols.index(col) for col in extrap_cols if col in cont_cols]


class TabularResidualBlock(nn.Module):
    """Deep residual block with Pre-LayerNorm, Mish activation, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Mish()
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.drop2(out)
        return residual + out


class HedonicExtrapolatorNet(nn.Module):
    """Hybrid Hedonic Tabular Network for Out-of-Time Real Estate Valuation."""

    def __init__(
        self,
        cat_cardinalities: dict,
        num_cont_features: int,
        extrap_indices: list,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.extrap_indices = extrap_indices

        # 1. Explicit Linear Extrapolation Head (Time trend + Fourier Seasonality)
        self.linear_trend_head = nn.Linear(len(extrap_indices), 1, bias=True)

        # 2. Entity Embeddings for Categorical Entities
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col_name, card in cat_cardinalities.items():
            emb_dim = min(48, max(4, int(math.pow(card, 0.35) * 4)))
            self.embeddings[col_name] = nn.Embedding(card + 1, emb_dim, padding_idx=0)
            total_emb_dim += emb_dim

        self.emb_dropout = nn.Dropout(dropout_rate * 0.75)

        # 3. Continuous Feature Normalization
        self.cont_norm = nn.BatchNorm1d(num_cont_features)

        # 4. Dense Projection Layer
        in_features = total_emb_dim + num_cont_features
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate),
        )

        # 5. Deep Residual Backbone
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        # 6. Non-linear Hedonic Value Head
        self.hedonic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.linear_trend_head.weight)
        nn.init.constant_(self.linear_trend_head.bias, 5.2)  # Log10(160,000) ~ 5.20
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.linear_trend_head:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, x_cont: torch.Tensor, x_cat: torch.Tensor, cat_order: list
    ) -> torch.Tensor:
        x_extrap = x_cont[:, self.extrap_indices]
        trend_pred = self.linear_trend_head(x_extrap)

        emb_outs = []
        for i, col_name in enumerate(cat_order):
            cat_idx = torch.clamp(x_cat[:, i], min=0)
            emb = self.embeddings[col_name](cat_idx)
            emb_outs.append(emb)

        emb_concat = torch.cat(emb_outs, dim=1)
        emb_concat = self.emb_dropout(emb_concat)

        cont_normed = self.cont_norm(x_cont)
        dense_in = torch.cat([emb_concat, cont_normed], dim=1)

        h = self.input_proj(dense_in)
        for block in self.res_blocks:
            h = block(h)

        hedonic_pred = self.hedonic_head(h)
        return trend_pred + hedonic_pred


class LogHuberLoss(nn.Module):
    def __init__(self, delta: float = 0.20):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=delta)

    def forward(self, pred_log: torch.Tensor, true_log: torch.Tensor) -> torch.Tensor:
        return self.huber(pred_log.squeeze(), true_log.squeeze())


def get_lgb_linear_tree_params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "linear_tree": True,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "n_jobs": 32,
        "verbosity": -1,
        "random_state": 42,
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

model = HedonicExtrapolatorNet(
    cat_cardinalities=cat_cardinalities,
    num_cont_features=len(cont_cols),
    extrap_indices=extrap_indices,
    hidden_dim=256,
    num_res_blocks=3,
    dropout_rate=0.15,
).to(device)

criterion = LogHuberLoss(delta=0.20)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if (
        "bias" in name
        or "norm" in name
        or "cont_norm" in name
        or "linear_trend_head" in name
    ):
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = AdamW(
    [
        {"params": decay_params, "weight_decay": 1e-4, "lr": 1e-3},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": 1e-3},
    ]
)

scheduler = CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)

# -------------------------------------------------------------------------
# 10. Prepare DataLoaders
# -------------------------------------------------------------------------
train_cont = np.nan_to_num(
    X_train_final.loc[train_mask, cont_cols].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
val_cont = np.nan_to_num(
    X_train_final.loc[val_mask, cont_cols].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
test_cont = np.nan_to_num(
    X_test_final[cont_cols].values.astype(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

train_cat = X_train_final.loc[train_mask, cat_code_cols].values.astype(np.int64)
val_cat = X_train_final.loc[val_mask, cat_code_cols].values.astype(np.int64)
test_cat = X_test_final[cat_code_cols].values.astype(np.int64)

for i, col in enumerate(cat_code_cols):
    card = cat_cardinalities[col]
    train_cat[:, i] = np.clip(train_cat[:, i], 0, card)
    val_cat[:, i] = np.clip(val_cat[:, i], 0, card)
    test_cat[:, i] = np.clip(test_cat[:, i], 0, card)

y_train_split = y_train_full[train_mask]
y_val_split = y_train_full[val_mask]

train_cont_t = torch.from_numpy(train_cont)
train_cat_t = torch.from_numpy(train_cat)
train_y_t = torch.from_numpy(y_train_split.astype(np.float32))

val_cont_t = torch.from_numpy(val_cont)
val_cat_t = torch.from_numpy(val_cat)

test_cont_t = torch.from_numpy(test_cont)
test_cat_t = torch.from_numpy(test_cat)

BATCH_SIZE = 8192
EVAL_BATCH_SIZE = 16384

train_dataset = TensorDataset(train_cont_t, train_cat_t, train_y_t)
val_dataset = TensorDataset(val_cont_t, val_cat_t)
test_dataset = TensorDataset(test_cont_t, test_cat_t)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=True
)

# -------------------------------------------------------------------------
# 11. Model Training Loop with Out-of-Time Metric Tracking
# -------------------------------------------------------------------------
best_model_path = os.path.join(WORKING_DIR, "best_hedonic_extrapolator.pt")
num_epochs = 12
patience = 4
best_val_rmse = float("inf")
patience_counter = 0


def compute_official_rmse(pred_log, true_log):
    pred_price = np.clip(10.0**pred_log, 1.0, None)
    true_price = np.clip(10.0**true_log, 1.0, None)
    log_pred = np.log10(pred_price)
    log_true = np.log10(true_price)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


print(f"Training HedonicExtrapolatorNet for {num_epochs} epochs on {device}...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_cont_b, x_cat_b, y_b in train_loader:
        x_cont_b = x_cont_b.to(device, non_blocking=True)
        x_cat_b = x_cat_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(x_cont_b, x_cat_b, cat_code_cols)
        loss = criterion(preds, y_b)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        total_samples += len(y_b)

    scheduler.step()
    epoch_train_loss = total_loss / total_samples

    # Out-of-time validation on 2016 holdout
    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for x_cont_b, x_cat_b in val_loader:
            x_cont_b = x_cont_b.to(device, non_blocking=True)
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            preds = model(x_cont_b, x_cat_b, cat_code_cols)
            val_preds_list.append(preds.squeeze(-1).cpu().numpy())

    val_preds_epoch = np.concatenate(val_preds_list)
    val_rmse = compute_official_rmse(val_preds_epoch, y_val_split)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        status_flag = "*"
    else:
        patience_counter += 1
        status_flag = ""

    print(
        f"Epoch {epoch+1:02d}/{num_epochs:02d} | Train Loss: {epoch_train_loss:.5f} | "
        f"Val RMSE (log10): {val_rmse:.5f} | Best: {best_val_rmse:.5f} {status_flag}"
    )

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

# -------------------------------------------------------------------------
# 12. Final Validation Evaluation using Best Saved Model
# -------------------------------------------------------------------------
print(f"Loading best checkpoint from {best_model_path}...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

final_val_preds = []
with torch.no_grad():
    for x_cont_b, x_cat_b in val_loader:
        x_cont_b = x_cont_b.to(device, non_blocking=True)
        x_cat_b = x_cat_b.to(device, non_blocking=True)
        preds = model(x_cont_b, x_cat_b, cat_code_cols)
        final_val_preds.append(preds.squeeze(-1).cpu().numpy())

final_val_preds = np.concatenate(final_val_preds)
score = compute_official_rmse(final_val_preds, y_val_split)

# -------------------------------------------------------------------------
# 13. Test Inference and Submission Generation
# -------------------------------------------------------------------------
print("Performing model inference on 375,098 test samples...")
test_preds_list = []
with torch.no_grad():
    for x_cont_b, x_cat_b in test_loader:
        x_cont_b = x_cont_b.to(device, non_blocking=True)
        x_cat_b = x_cat_b.to(device, non_blocking=True)
        preds = model(x_cont_b, x_cat_b, cat_code_cols)
        test_preds_list.append(preds.squeeze(-1).cpu().numpy())

test_preds = np.concatenate(test_preds_list)

# Transform from log10 space to raw currency units
test_price_preds = np.clip(10.0**test_preds, 1.0, 100_000_000.0)
test_price_preds = np.round(test_price_preds).astype(np.int64)

submission_df = pd.DataFrame({"id": test_ids, "price": test_price_preds})

# Submission integrity verification
assert len(submission_df) == 375098, f"Expected 375,098 rows, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    "price",
], f"Incorrect columns: {list(submission_df.columns)}"
assert not submission_df.isnull().any().any(), "Submission contains null values!"
assert (submission_df["price"] >= 1).all(), "Submission contains non-positive prices!"

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission saved successfully to {submission_path} with shape {submission_df.shape}."
)

print(f"Final Validation Score: {score}")
