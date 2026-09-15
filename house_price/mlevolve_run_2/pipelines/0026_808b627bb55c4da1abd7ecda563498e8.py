import gc
import json
import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# --- 0. Environment & Reproducibility Setup ---
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running pipeline on compute device: {device}")

# --- 1. Data Processing and Feature Engineering ---
print("--- Loading Raw Transaction Data ---")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
print(f"Loaded raw train: {train_df.shape}, raw test: {test_df.shape}")

# Standardize categorical fields
raw_cat_cols = [
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
    "county",
    "district",
]
all_str_cols = raw_cat_cols + ["town"]

for col in all_str_cols:
    train_df[col] = train_df[col].astype(str).str.strip().str.upper().fillna("MISSING")
    test_df[col] = test_df[col].astype(str).str.strip().str.upper().fillna("MISSING")

# Parse transaction dates and engineer calendar features
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

print("Extracting calendar and continuous timeline attributes...")
for df in [train_df, test_df]:
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["is_friday"] = (df["dayofweek"] == 4).astype(np.int8)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(np.int16)

    # Continuous timeline anchor and fraction of year
    df["time_elapsed_days"] = (df["date"] - pd.Timestamp("2010-01-01")).dt.days.astype(
        np.int32
    )
    df["year_fraction"] = (df["year"] + (df["dayofyear"] - 1.0) / 365.25).astype(
        np.float32
    )

    # Cyclical monthly transformations
    month_rad = 2.0 * np.pi * df["month"] / 12.0
    df["sin_month"] = np.sin(month_rad).astype(np.float32)
    df["cos_month"] = np.cos(month_rad).astype(np.float32)

    # Spatial and hedonic interaction keys
    df["town_district"] = df["town"] + "__" + df["district"]
    df["district_county"] = df["district"] + "__" + df["county"]
    df["district_type"] = df["district"] + "__" + df["property_type"]
    df["county_type"] = df["county"] + "__" + df["property_type"]
    df["hedonic_profile"] = (
        df["property_type"]
        + "_"
        + df["tenure"]
        + "_"
        + df["is_new_build"]
        + "_"
        + df["sale_category"]
    )

# Filter non-market legal transfers and extreme commercial outliers
valid_price_mask = (train_df["price"] >= 1000) & (train_df["price"] <= 50000000)
train_df = train_df[valid_price_mask].copy()
train_df["log10_price"] = np.log10(train_df["price"]).astype(np.float32)
print(f"Train rows after filtering non-market price anomalies: {len(train_df):,}")

MODERN_START_YEAR = 2012


# Leakage-free Empirical Bayes Target Encodings and Growth Momentum Computation on Stationary Residuals
def compute_hierarchical_features(fit_train_df, apply_dfs, target_col="residual"):
    stat_df = fit_train_df[fit_train_df["year"] >= MODERN_START_YEAR].copy()
    global_mean = float(stat_df[target_col].mean())

    # 1. County level encoding (shrunk to global)
    county_stats = stat_df.groupby("county")[target_col].agg(["count", "mean"])
    m_county = 50.0
    county_stats["county_te"] = (
        (county_stats["count"] * county_stats["mean"] + m_county * global_mean)
        / (county_stats["count"] + m_county)
    ).astype(np.float32)
    county_map = county_stats["county_te"].to_dict()

    # 2. District level encoding (shrunk to county level)
    district_stats = (
        stat_df.groupby(["district", "county"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    district_stats["county_prior"] = (
        district_stats["county"].map(county_map).fillna(global_mean)
    )
    m_district = 30.0
    district_stats["district_te"] = (
        (
            district_stats["count"] * district_stats["mean"]
            + m_district * district_stats["county_prior"]
        )
        / (district_stats["count"] + m_district)
    ).astype(np.float32)
    district_map = district_stats.set_index("district")["district_te"].to_dict()

    # 3. Town level encoding (shrunk to district level)
    town_stats = (
        stat_df.groupby(["town_district", "district"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    town_stats["district_prior"] = (
        town_stats["district"].map(district_map).fillna(global_mean)
    )
    m_town = 20.0
    town_stats["town_te"] = (
        (
            town_stats["count"] * town_stats["mean"]
            + m_town * town_stats["district_prior"]
        )
        / (town_stats["count"] + m_town)
    ).astype(np.float32)
    town_map = town_stats.set_index("town_district")["town_te"].to_dict()

    # 4. District x Property Type interaction encoding
    dtype_stats = (
        stat_df.groupby(["district_type", "district"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    dtype_stats["district_prior"] = (
        dtype_stats["district"].map(district_map).fillna(global_mean)
    )
    m_dtype = 25.0
    dtype_stats["district_type_te"] = (
        (
            dtype_stats["count"] * dtype_stats["mean"]
            + m_dtype * dtype_stats["district_prior"]
        )
        / (dtype_stats["count"] + m_dtype)
    ).astype(np.float32)
    dtype_map = dtype_stats.set_index("district_type")["district_type_te"].to_dict()

    # 5. Hedonic Profile encoding
    profile_stats = stat_df.groupby("hedonic_profile")[target_col].agg(
        ["count", "mean"]
    )
    m_prof = 50.0
    profile_stats["profile_te"] = (
        (profile_stats["count"] * profile_stats["mean"] + m_prof * global_mean)
        / (profile_stats["count"] + m_prof)
    ).astype(np.float32)
    profile_map = profile_stats["profile_te"].to_dict()

    # 6. Spatial Price Growth Velocity (Annual Residual Momentum)
    max_year = int(fit_train_df["year"].max())
    recent_y1 = (
        fit_train_df[fit_train_df["year"] == max_year]
        .groupby("district")[target_col]
        .mean()
    )
    recent_y2 = (
        fit_train_df[fit_train_df["year"] == (max_year - 1)]
        .groupby("district")[target_col]
        .mean()
    )
    recent_y3 = (
        fit_train_df[fit_train_df["year"] == (max_year - 2)]
        .groupby("district")[target_col]
        .mean()
    )

    growth_1y = (recent_y1 - recent_y2).to_dict()
    growth_2y = ((recent_y1 - recent_y3) / 2.0).to_dict()
    global_growth_1y = float(
        fit_train_df[fit_train_df["year"] == max_year][target_col].mean()
        - fit_train_df[fit_train_df["year"] == (max_year - 1)][target_col].mean()
    )
    global_growth_2y = float(
        (
            fit_train_df[fit_train_df["year"] == max_year][target_col].mean()
            - fit_train_df[fit_train_df["year"] == (max_year - 2)][target_col].mean()
        )
        / 2.0
    )

    results = []
    for df in apply_dfs:
        res = df.copy()
        res["te_county"] = (
            res["county"].map(county_map).fillna(global_mean).astype(np.float32)
        )
        res["te_district"] = (
            res["district"]
            .map(district_map)
            .fillna(res["te_county"])
            .astype(np.float32)
        )
        res["te_town"] = (
            res["town_district"]
            .map(town_map)
            .fillna(res["te_district"])
            .astype(np.float32)
        )
        res["te_district_type"] = (
            res["district_type"]
            .map(dtype_map)
            .fillna(res["te_district"])
            .astype(np.float32)
        )
        res["te_hedonic_profile"] = (
            res["hedonic_profile"]
            .map(profile_map)
            .fillna(global_mean)
            .astype(np.float32)
        )

        res["district_momentum_1y"] = (
            res["district"].map(growth_1y).fillna(global_growth_1y).astype(np.float32)
        )
        res["district_momentum_2y"] = (
            res["district"].map(growth_2y).fillna(global_growth_2y).astype(np.float32)
        )
        res["te_type_district_premium"] = (
            res["te_district_type"] - res["te_district"]
        ).astype(np.float32)
        results.append(res)

    return results


print("Configuring out-of-time splits and macro-trend detrending...")
val_raw_train = train_df[train_df["year"] < 2016].copy()
val_raw_holdout = train_df[train_df["year"] == 2016].copy()
prod_raw_train = train_df[train_df["year"] <= 2016].copy()
prod_raw_test = test_df.copy()

# Fit continuous macro-trend of log10_price on year_fraction over modern regime
val_modern = val_raw_train[val_raw_train["year"] >= MODERN_START_YEAR]
val_slope, val_intercept = np.polyfit(
    val_modern["year_fraction"].values, val_modern["log10_price"].values, deg=1
)
print(f"Validation macro-trend slope: {val_slope:.5f}, intercept: {val_intercept:.5f}")

val_raw_train["trend"] = (
    val_slope * val_raw_train["year_fraction"] + val_intercept
).astype(np.float32)
val_raw_train["residual"] = (
    val_raw_train["log10_price"] - val_raw_train["trend"]
).astype(np.float32)
val_raw_train["y_tilde"] = val_raw_train["residual"]

val_raw_holdout["trend"] = (
    val_slope * val_raw_holdout["year_fraction"] + val_intercept
).astype(np.float32)
val_raw_holdout["residual"] = (
    val_raw_holdout["log10_price"] - val_raw_holdout["trend"]
).astype(np.float32)
val_raw_holdout["y_tilde"] = val_raw_holdout["residual"]
val_trend = val_raw_holdout["trend"].values.astype(np.float32)

prod_modern = prod_raw_train[prod_raw_train["year"] >= MODERN_START_YEAR]
prod_slope, prod_intercept = np.polyfit(
    prod_modern["year_fraction"].values, prod_modern["log10_price"].values, deg=1
)
print(f"Production macro-trend slope: {prod_slope:.5f}, intercept: {prod_intercept:.5f}")

prod_raw_train["trend"] = (
    prod_slope * prod_raw_train["year_fraction"] + prod_intercept
).astype(np.float32)
prod_raw_train["residual"] = (
    prod_raw_train["log10_price"] - prod_raw_train["trend"]
).astype(np.float32)
prod_raw_train["y_tilde"] = prod_raw_train["residual"]

prod_raw_test["trend"] = (
    prod_slope * prod_raw_test["year_fraction"] + prod_intercept
).astype(np.float32)
prod_test_trend = prod_raw_test["trend"].values.astype(np.float32)

print(f"Validation train split (pre-2016): {len(val_raw_train):,} rows")
print(f"Validation holdout split (2016): {len(val_raw_holdout):,} rows")
print(f"Production train split (pre-2017): {len(prod_raw_train):,} rows")
print(f"Production test split (2017): {len(prod_raw_test):,} rows")

# Compute hierarchical features on stationary residuals for both regimes
val_train_feat, val_holdout_feat = compute_hierarchical_features(
    fit_train_df=val_raw_train,
    apply_dfs=[
        val_raw_train[val_raw_train["year"] >= MODERN_START_YEAR],
        val_raw_holdout,
    ],
    target_col="residual",
)

prod_train_feat, prod_test_feat = compute_hierarchical_features(
    fit_train_df=prod_raw_train,
    apply_dfs=[
        prod_raw_train[prod_raw_train["year"] >= MODERN_START_YEAR],
        prod_raw_test,
    ],
    target_col="residual",
)

# Free raw historical copies
del train_df, val_raw_train, val_raw_holdout, prod_raw_train, prod_raw_test, test_df
gc.collect()

# Encode Categorical Identifiers
cat_feature_cols = [col + "_cat" for col in raw_cat_cols]
for raw_col, cat_col in zip(raw_cat_cols, cat_feature_cols):
    categories = sorted(prod_train_feat[raw_col].unique().tolist())
    cat_to_id = {val: idx for idx, val in enumerate(categories)}

    val_train_feat[cat_col] = (
        val_train_feat[raw_col].map(cat_to_id).fillna(-1).astype(np.int32)
    )
    val_holdout_feat[cat_col] = (
        val_holdout_feat[raw_col].map(cat_to_id).fillna(-1).astype(np.int32)
    )
    prod_train_feat[cat_col] = (
        prod_train_feat[raw_col].map(cat_to_id).fillna(-1).astype(np.int32)
    )
    prod_test_feat[cat_col] = (
        prod_test_feat[raw_col].map(cat_to_id).fillna(-1).astype(np.int32)
    )

cont_feature_cols = [
    "year_fraction",
    "time_elapsed_days",
    "sin_month",
    "cos_month",
    "dayofweek",
    "is_friday",
    "te_county",
    "te_district",
    "te_town",
    "te_district_type",
    "te_hedonic_profile",
    "district_momentum_1y",
    "district_momentum_2y",
    "te_type_district_premium",
]
feature_cols = cont_feature_cols + cat_feature_cols
print(f"Total features ready for models: {len(feature_cols)}")

cat_cardinalities = {
    col: int(prod_train_feat[col].max() + 1) for col in cat_feature_cols
}

trend_feature_names = [
    "year_fraction",
    "time_elapsed_days",
    "district_momentum_1y",
    "district_momentum_2y",
]
trend_indices = [
    cont_feature_cols.index(col)
    for col in trend_feature_names
    if col in cont_feature_cols
]


# --- 2. Model Design ---
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HedonicTrendResidualNet(nn.Module):

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: dict,
        embedding_dim_mult: float = 1.0,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        trend_feature_indices: list = None,
        dropout_rate: float = 0.15,
        emb_dropout_rate: float = 0.05,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.trend_feature_indices = (
            trend_feature_indices
            if trend_feature_indices is not None
            else list(range(min(4, num_continuous)))
        )

        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = max(4, min(48, int(round((card + 1) ** 0.5 * 1.6))))
            emb_dim = int(emb_dim * embedding_dim_mult)
            # Add +2 for padding/unknown index
            self.embeddings[col] = nn.Embedding(card + 2, emb_dim, padding_idx=0)
            total_emb_dim += emb_dim

        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.cont_norm = nn.LayerNorm(num_continuous)

        # Hedonic Non-Linear Trunk directly predicting stationary residuals
        in_dim = total_emb_dim + num_continuous
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )
        self.hedonic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x_cont: torch.Tensor, x_cat_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x_cont_norm = self.cont_norm(x_cont)
        emb_list = [
            emb_layer(x_cat_dict[col]) for col, emb_layer in self.embeddings.items()
        ]
        x_emb = self.emb_dropout(torch.cat(emb_list, dim=-1))

        x_hedonic = torch.cat([x_cont_norm, x_emb], dim=-1)
        h = self.input_proj(x_hedonic)
        for block in self.res_blocks:
            h = block(h)

        hedonic_pred = self.hedonic_head(h)
        return hedonic_pred.squeeze(-1)


class MetricAlignedLog10HuberLoss(nn.Module):

    def __init__(self, delta: float = 0.25):
        super().__init__()
        self.delta = delta
        self.huber_none = nn.HuberLoss(reduction="none", delta=delta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = self.huber_none(pred, target)
        if sample_weight is not None:
            return torch.sum(loss * sample_weight) / torch.sum(sample_weight)
        return torch.mean(loss)


def build_hedonic_neural_model(
    num_continuous: int = len(cont_feature_cols),
    cat_cardinalities: dict = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = HedonicTrendResidualNet(
        num_continuous=num_continuous,
        cat_cardinalities=cat_cardinalities,
        hidden_dim=256,
        num_res_blocks=3,
        trend_feature_indices=trend_indices,
        dropout_rate=0.15,
    ).to(device)
    criterion = MetricAlignedLog10HuberLoss(delta=0.25)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


def get_lgb_hedonic_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2500,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }


# Metric Definition: Exact Task-Faithful Log10 RMSE
def compute_log10_rmse(y_true_raw, y_pred_raw):
    clipped_pred = np.clip(y_pred_raw, 1.0, None)
    clipped_true = np.clip(y_true_raw, 1.0, None)
    log_pred = np.log10(clipped_pred)
    log_true = np.log10(clipped_true)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


def prepare_tensors(
    df,
    cont_cols,
    cat_cols,
    emb_layers,
    target_col="residual",
    weights=None,
    is_test=False,
):
    x_cont = torch.as_tensor(
        df[cont_cols].fillna(0.0).values, dtype=torch.float32, device=device
    )
    x_cat_dict = {}
    for col in cat_cols:
        max_idx = emb_layers[col].num_embeddings - 1
        clipped_tokens = np.clip(df[col].values + 1, 0, max_idx).astype(np.int64)
        x_cat_dict[col] = torch.as_tensor(
            clipped_tokens, dtype=torch.long, device=device
        )
    y_tensor = None
    w_tensor = None
    if not is_test:
        y_tensor = torch.as_tensor(
            df[target_col].values, dtype=torch.float32, device=device
        )
        if weights is not None:
            w_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    return x_cont, x_cat_dict, y_tensor, w_tensor


def predict_neural_model(nn_model, x_cont, x_cat_dict, chunk_size=32768):
    nn_model.eval()
    preds = []
    n_samples = x_cont.shape[0]
    with torch.no_grad():
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            batch_cont = x_cont[start_idx:end_idx]
            batch_cat = {col: x_cat_dict[col][start_idx:end_idx] for col in x_cat_dict}
            pred_batch = nn_model(batch_cont, batch_cat)
            preds.append(pred_batch.detach().cpu().numpy())
    return np.concatenate(preds)


# --- 3. Validation Regime Training & Evaluation ---
print("--- Training HedonicTrendResidualNet on Validation Regime ---")
model, criterion, optimizer, scheduler = build_hedonic_neural_model(
    num_continuous=len(cont_feature_cols),
    cat_cardinalities=cat_cardinalities,
    device=device,
)

# Exponential recency weighting: gamma = 0.15
gamma = 0.15
max_yf_val = float(val_train_feat["year_fraction"].max())
val_weights = np.exp(gamma * (val_train_feat["year_fraction"].values - max_yf_val)).astype(np.float32)

val_x_cont_train, val_x_cat_train, val_y_train, val_w_train = prepare_tensors(
    val_train_feat,
    cont_feature_cols,
    cat_feature_cols,
    model.embeddings,
    target_col="residual",
    weights=val_weights,
)
val_x_cont_holdout, val_x_cat_holdout, _, _ = prepare_tensors(
    val_holdout_feat,
    cont_feature_cols,
    cat_feature_cols,
    model.embeddings,
    target_col="residual",
)

num_epochs = 12
batch_size = 8192
num_train_samples = len(val_train_feat)
indices = np.arange(num_train_samples)
best_val_rmse = float("inf")
best_model_weights = None
val_holdout_y_raw = val_holdout_feat["price"].values

for epoch in range(1, num_epochs + 1):
    model.train()
    np.random.shuffle(indices)
    running_loss = 0.0
    num_batches = 0

    for start_idx in range(0, num_train_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_train_samples)
        batch_idx = torch.as_tensor(
            indices[start_idx:end_idx], dtype=torch.long, device=device
        )

        batch_x_cont = val_x_cont_train[batch_idx]
        batch_x_cat = {col: val_x_cat_train[col][batch_idx] for col in cat_feature_cols}
        batch_y = val_y_train[batch_idx]
        batch_w = val_w_train[batch_idx]

        optimizer.zero_grad()
        pred_res = model(batch_x_cont, batch_x_cat)
        loss = criterion(pred_res, batch_y, sample_weight=batch_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    scheduler.step()
    epoch_loss = running_loss / max(1, num_batches)

    # Predict stationary residual and reconstruct log10 price by adding extrapolated macro trend
    val_pred_res = predict_neural_model(model, val_x_cont_holdout, val_x_cat_holdout)
    val_pred_log10 = val_pred_res + val_trend
    val_pred_raw = np.clip(10.0**val_pred_log10, 1.0, None)
    val_rmse = compute_log10_rmse(val_holdout_y_raw, val_pred_raw)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.5f} | Val RMSE: {val_rmse:.5f}"
    )

# Load best neural checkpoint and predict residuals
model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
val_nn_res = predict_neural_model(model, val_x_cont_holdout, val_x_cat_holdout)

# Clean validation tensors
del val_x_cont_train, val_x_cat_train, val_y_train, val_w_train
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# LightGBM GBDT Training & Evaluation on Stationary Residuals with Recency Weighting
print("--- Training LightGBM on Validation Regime Residuals ---")
lgb_params = get_lgb_hedonic_params()

lgb_train_data = lgb.Dataset(
    val_train_feat[feature_cols],
    label=val_train_feat["residual"],
    weight=val_weights,
    categorical_feature=cat_feature_cols,
)
lgb_val_data = lgb.Dataset(
    val_holdout_feat[feature_cols],
    label=val_holdout_feat["residual"],
    reference=lgb_train_data,
    categorical_feature=cat_feature_cols,
)

lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    num_boost_round=lgb_params.get("n_estimators", 2500),
    valid_sets=[lgb_val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

val_lgb_res = lgb_model.predict(
    val_holdout_feat[feature_cols], num_iteration=lgb_model.best_iteration
)
val_lgb_log10 = val_lgb_res + val_trend
val_lgb_rmse = compute_log10_rmse(
    val_holdout_y_raw, np.clip(10.0**val_lgb_log10, 1.0, None)
)
print(f"Validation LightGBM RMSE: {val_lgb_rmse:.5f}")

# Ensemble Calibration on Holdout
best_w = 0.5
best_ensemble_score = float("inf")
for w_cand in np.linspace(0.0, 1.0, 21):
    cand_res = w_cand * val_nn_res + (1.0 - w_cand) * val_lgb_res
    cand_log10 = cand_res + val_trend
    cand_score = compute_log10_rmse(
        val_holdout_y_raw, np.clip(10.0**cand_log10, 1.0, None)
    )
    if cand_score < best_ensemble_score:
        best_ensemble_score = cand_score
        best_w = w_cand

print(
    f"Optimal Ensemble Weight (NN): {best_w:.2f} | Holdout Validation Score: {best_ensemble_score:.5f}"
)

# Free validation structures
del (
    val_train_feat,
    val_holdout_feat,
    val_x_cont_holdout,
    val_x_cat_holdout,
    lgb_train_data,
    lgb_val_data,
)
gc.collect()

# --- 4. Production Retraining on Complete Modern Regime (2012-2016) ---
print("--- Production Retraining on Full Modern Regime (2012-2016) ---")
prod_model, prod_criterion, prod_optimizer, prod_scheduler = build_hedonic_neural_model(
    num_continuous=len(cont_feature_cols),
    cat_cardinalities=cat_cardinalities,
    device=device,
)

max_yf_prod = float(prod_train_feat["year_fraction"].max())
prod_weights = np.exp(gamma * (prod_train_feat["year_fraction"].values - max_yf_prod)).astype(np.float32)

prod_x_cont_train, prod_x_cat_train, prod_y_train, prod_w_train = prepare_tensors(
    prod_train_feat,
    cont_feature_cols,
    cat_feature_cols,
    prod_model.embeddings,
    target_col="residual",
    weights=prod_weights,
)
prod_x_cont_test, prod_x_cat_test, _, _ = prepare_tensors(
    prod_test_feat,
    cont_feature_cols,
    cat_feature_cols,
    prod_model.embeddings,
    is_test=True,
)

prod_num_samples = len(prod_train_feat)
prod_indices = np.arange(prod_num_samples)

for epoch in range(1, num_epochs + 1):
    prod_model.train()
    np.random.shuffle(prod_indices)
    running_loss = 0.0
    num_batches = 0

    for start_idx in range(0, prod_num_samples, batch_size):
        end_idx = min(start_idx + batch_size, prod_num_samples)
        batch_idx = torch.as_tensor(
            prod_indices[start_idx:end_idx], dtype=torch.long, device=device
        )

        batch_x_cont = prod_x_cont_train[batch_idx]
        batch_x_cat = {
            col: prod_x_cat_train[col][batch_idx] for col in cat_feature_cols
        }
        batch_y = prod_y_train[batch_idx]
        batch_w = prod_w_train[batch_idx]

        prod_optimizer.zero_grad()
        pred_res = prod_model(batch_x_cont, batch_x_cat)
        loss = prod_criterion(pred_res, batch_y, sample_weight=batch_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prod_model.parameters(), max_norm=1.0)
        prod_optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    prod_scheduler.step()
    epoch_loss = running_loss / max(1, num_batches)
    print(f"Production Epoch {epoch:02d} | Train Loss: {epoch_loss:.5f}")

# Inference: Neural model stationary residual predictions on test set
test_nn_res = predict_neural_model(prod_model, prod_x_cont_test, prod_x_cat_test)

# Retrain LightGBM on production modern dataset residuals with recency weighting
prod_lgb_params = lgb_params.copy()
prod_num_boost_round = max(100, int(lgb_model.best_iteration * 1.15))

full_train_lgb = lgb.Dataset(
    prod_train_feat[feature_cols],
    label=prod_train_feat["residual"],
    weight=prod_weights,
    categorical_feature=cat_feature_cols,
)
prod_lgb_model = lgb.train(
    prod_lgb_params,
    full_train_lgb,
    num_boost_round=prod_num_boost_round,
    callbacks=[lgb.log_evaluation(period=0)],
)

# Inference: LightGBM stationary residual predictions on test set
test_lgb_res = prod_lgb_model.predict(prod_test_feat[feature_cols])

# --- 5. Ensembled Inference & Verified Submission Export ---
# Blend stationary residuals and reconstruct test log10 price with extrapolated macro trend
test_final_res = best_w * test_nn_res + (1.0 - best_w) * test_lgb_res
test_final_log10 = test_final_res + prod_test_trend
test_final_price = np.clip(10.0**test_final_log10, 1.0, None)
test_final_price_rounded = np.round(test_final_price).astype(np.int64)

submission_df = pd.DataFrame(
    {"id": prod_test_feat["id"].values, "price": test_final_price_rounded}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

# Submission Integrity Verification
assert os.path.exists(submission_path), "Submission file was not created!"
assert len(submission_df) == 375098, f"Expected 375,098 rows, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    "price",
], f"Invalid columns: {submission_df.columns}"
assert (
    not submission_df["price"].isnull().any()
), "Submission contains null price values!"
assert (
    submission_df["price"] > 0
).all(), "Submission contains non-positive price values!"

# Verify alignment with sample_submission
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
assert (
    submission_df["id"].values == sample_sub["id"].values
).all(), "Submission ID order mismatch!"

print("Submission integrity verified successfully.")
print(f"Generated predictions for {len(submission_df):,} test transactions.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {best_ensemble_score:.5f}")
