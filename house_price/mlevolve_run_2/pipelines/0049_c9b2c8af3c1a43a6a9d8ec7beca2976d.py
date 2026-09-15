import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for strict reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. PATHS AND DIRECTORY SETUP
# ==============================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ==============================================================================
# 2. EVALUATION METRICS
# ==============================================================================
def compute_log10_rmse(y_true, y_pred):
    """
    Computes official competition metric: RMSE on log10(price).
    Compatible with numpy arrays and torch tensors.
    """
    if isinstance(y_true, torch.Tensor):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def compute_raw_price_rmse(raw_true, raw_pred):
    """
    Computes official competition metric on raw currency prices with >= 1 clipping.
    """
    p_true = np.clip(np.asarray(raw_true, dtype=np.float64), 1.0, None)
    p_pred = np.clip(np.asarray(raw_pred, dtype=np.float64), 1.0, None)
    return float(np.sqrt(np.mean((np.log10(p_pred) - np.log10(p_true)) ** 2)))


# ==============================================================================
# 3. MODEL ARCHITECTURES
# ==============================================================================
def get_lgb_model_params():
    """
    Returns leaf-wise LightGBM regressor configuration for log10 residual regression.
    """
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 95,
        "max_depth": 12,
        "min_child_samples": 180,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "n_estimators": 3000,
        "random_state": 42,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }


class LGBContinuousRegressor(lgb.LGBMRegressor):
    """
    Dedicated LightGBM wrapper that strictly filters out integer categorical code features
    and ignores categorical histogram splits, training exclusively on continuous features.
    """
    def _filter_cont(self, X):
        if isinstance(X, pd.DataFrame):
            cont_cols = [c for c in X.columns if not c.endswith("_code")]
            return X[cont_cols]
        return X

    def fit(self, X, y, eval_set=None, categorical_feature=None, **kwargs):
        X_filtered = self._filter_cont(X)
        eval_set_filtered = None
        if eval_set is not None:
            eval_set_filtered = [
                (self._filter_cont(ex), ey) for (ex, ey) in eval_set
            ]
        return super().fit(
            X_filtered,
            y,
            eval_set=eval_set_filtered,
            categorical_feature="auto",
            **kwargs,
        )

    def predict(self, X, **kwargs):
        X_filtered = self._filter_cont(X)
        preds = super().predict(X_filtered, **kwargs)
        return np.asarray(preds, dtype=np.float32)


def create_lgb_model():
    return LGBContinuousRegressor(**get_lgb_model_params())


def get_xgb_model_params():
    """
    Returns depth-wise GPU-accelerated XGBoost regressor configuration.
    """
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 9,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_lambda": 4.0,
        "n_estimators": 3000,
        "random_state": 42,
        "tree_method": "hist",
    }
    if torch.cuda.is_available():
        try:
            # XGBoost 2.0+ accepts device='cuda'
            _ = xgb.XGBRegressor(n_estimators=1, tree_method="hist", device="cuda")
            params["device"] = "cuda"
        except Exception:
            params["tree_method"] = "gpu_hist"
    else:
        params["n_jobs"] = -1
    return params


class XGBContinuousRegressor(xgb.XGBRegressor):
    """
    Dedicated XGBoost wrapper that strictly filters out integer categorical code features,
    training exclusively on continuous target encodings, relative premiums, and liquidity densities.
    """
    def _filter_cont(self, X):
        if isinstance(X, pd.DataFrame):
            cont_cols = [c for c in X.columns if not c.endswith("_code")]
            return X[cont_cols]
        return X

    def fit(self, X, y, eval_set=None, **kwargs):
        X_filtered = self._filter_cont(X)
        eval_set_filtered = None
        if eval_set is not None:
            eval_set_filtered = [
                (self._filter_cont(ex), ey) for (ex, ey) in eval_set
            ]
        return super().fit(X_filtered, y, eval_set=eval_set_filtered, **kwargs)

    def predict(self, X, **kwargs):
        X_filtered = self._filter_cont(X)
        preds = super().predict(X_filtered, **kwargs)
        return np.asarray(preds, dtype=np.float32)


def create_xgb_model(early_stopping_rounds=None):
    params = get_xgb_model_params()
    if early_stopping_rounds is not None:
        try:
            _ = xgb.XGBRegressor(n_estimators=1, early_stopping_rounds=early_stopping_rounds)
            params["early_stopping_rounds"] = early_stopping_rounds
        except TypeError:
            pass
    return XGBContinuousRegressor(**params)


class ResidualBlock(nn.Module):
    """
    Residual MLP block with skip addition and LayerNorm to stabilize backpropagation across deep layers.
    """
    def __init__(self, dim, dropout=0.10):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularNN(nn.Module):
    """
    Deep Tabular Neural Network with learned Entity Embeddings and Residual MLP backbone
    with Layer Normalization for continuous spatial manifold learning and log10(price) regression.
    """
    def __init__(self, cat_cardinalities, num_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, min(64, max(8, int((card ** 0.35) * 3))))
            for card in cat_cardinalities
        ])
        total_emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        in_dim = total_emb_dim + num_cont

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.res1 = ResidualBlock(256, dropout=0.10)
        self.downsample = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.res2 = ResidualBlock(128, dropout=0.08)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        # Initialize output projection layer with small Gaussian weights and zero bias for residual regression
        nn.init.normal_(self.head[2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x_cat, x_cont):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1)
        x = torch.cat([x_emb, x_cont], dim=1)
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.downsample(x)
        x = self.res2(x)
        out = self.head(x)
        return out.squeeze(-1)


def train_tabular_nn(
    X_cat_np,
    X_cont_np,
    y_np,
    sample_weights_np,
    cat_cardinalities,
    num_cont,
    target_device,
    epochs=5,
    batch_size=8192,
    lr=3e-3,
    weight_decay=1e-4,
):
    model = TabularNN(cat_cardinalities, num_cont).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    w_mean = float(np.mean(sample_weights_np))
    w_norm = (
        (sample_weights_np / w_mean).astype(np.float32)
        if w_mean > 0
        else np.ones_like(sample_weights_np, dtype=np.float32)
    )

    t_cat = torch.from_numpy(X_cat_np.astype(np.int64))
    t_cont = torch.from_numpy(X_cont_np.astype(np.float32))
    t_y = torch.from_numpy(y_np.astype(np.float32))
    t_w = torch.from_numpy(w_norm)

    dataset = TensorDataset(t_cat, t_cont, t_y, t_w)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    warmup_steps = len(loader)
    total_steps = epochs * len(loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(target_device.type == "cuda"))

    model.train()
    for epoch in range(epochs):
        for b_cat, b_cont, b_y, b_w in loader:
            b_cat = b_cat.to(target_device, non_blocking=True)
            b_cont = b_cont.to(target_device, non_blocking=True)
            b_y = b_y.to(target_device, non_blocking=True)
            b_w = b_w.to(target_device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=target_device.type, enabled=(target_device.type == "cuda")):
                preds = model(b_cat, b_cont)
                loss = torch.mean(b_w * (preds - b_y) ** 2)

            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance learning rate schedule when the optimizer step was executed
            if scale_before <= scaler.get_scale():
                scheduler.step()

    return model


def predict_tabular_nn(model, X_cat_np, X_cont_np, target_device, batch_size=16384):
    model.eval()
    preds = []
    n_samples = len(X_cat_np)
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            b_cat = torch.from_numpy(X_cat_np[start_idx:end_idx].astype(np.int64)).to(target_device)
            b_cont = torch.from_numpy(X_cont_np[start_idx:end_idx].astype(np.float32)).to(target_device)
            with torch.amp.autocast(device_type=target_device.type, enabled=(target_device.type == "cuda")):
                pred = model(b_cat, b_cont)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def predict_xgb(model, X, target_device, batch_size=131072):
    """
    Inference helper for XGBoost that matches input data device with model device,
    avoiding device mismatch warnings and fallback overhead.
    """
    if hasattr(model, "_filter_cont"):
        X = model._filter_cont(X)
    feat_names = list(X.columns) if hasattr(X, "columns") else None
    x_arr = X.values if hasattr(X, "values") else X
    x_arr = np.asarray(x_arr, dtype=np.float32)
    n_samples = len(x_arr)

    if target_device.type == "cuda":
        try:
            booster = model.get_booster()
            preds_list = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                t_chunk = torch.from_numpy(x_arr[start_idx:end_idx]).to(target_device)
                chunk_preds = booster.inplace_predict(t_chunk)
                if hasattr(chunk_preds, "cpu"):
                    chunk_preds = chunk_preds.cpu().numpy()
                elif hasattr(chunk_preds, "get"):
                    chunk_preds = chunk_preds.get()
                preds_list.append(np.asarray(chunk_preds, dtype=np.float32))
            return np.concatenate(preds_list, axis=0)
        except Exception:
            pass
    return np.asarray(model.predict(X if hasattr(X, "columns") else x_arr), dtype=np.float32)


# ==============================================================================
# 4. DATA PROCESSING & FEATURE ENGINEERING
# ==============================================================================
print("Starting data processing and feature engineering...")

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_parquet(train_path)
test_df = pd.read_csv(test_path)
print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

train_df["price"] = pd.to_numeric(train_df["price"], errors="coerce")
train_df["log10_price"] = np.log10(np.clip(train_df["price"], 1.0, None)).astype(
    np.float32
)

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df["year"] = train_df["date"].dt.year.astype(np.int16)
test_df["year"] = test_df["date"].dt.year.astype(np.int16)


# Fast vectorized categorical normalization
def clean_categoricals(df):
    for col in [
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
        "town",
        "district",
        "county",
    ]:
        uniques = df[col].dropna().unique()
        mapping = {
            v: (
                str(v).strip().upper()
                if str(v).strip().upper() not in ["NAN", "NONE", ""]
                else "UNKNOWN"
            )
            for v in uniques
        }
        df[col] = df[col].map(mapping).fillna("UNKNOWN")
    return df


train_df = clean_categoricals(train_df)
test_df = clean_categoricals(test_df)

# Construct interaction features
for df in [train_df, test_df]:
    df["prop_tenure"] = df["property_type"] + "_" + df["tenure"]
    df["prop_new"] = df["property_type"] + "_" + df["is_new_build"]
    df["prop_tenure_new"] = (
        df["property_type"] + "_" + df["tenure"] + "_" + df["is_new_build"]
    )
    df["sale_prop"] = df["sale_category"] + "_" + df["property_type"]
    df["district_prop"] = df["district"] + "_" + df["property_type"]
    df["county_prop"] = df["county"] + "_" + df["property_type"]
    df["town_prop"] = df["town"] + "_" + df["property_type"]
    df["loc_triple"] = df["town"] + "||" + df["district"] + "||" + df["county"]
    df["loc_triple_prop"] = df["loc_triple"] + "_" + df["property_type"]
    df["district_new"] = df["district"] + "_" + df["is_new_build"]
    df["district_tenure"] = df["district"] + "_" + df["tenure"]

# Extract calendar & continuous trend features
base_date = pd.Timestamp("1995-01-01")
for df in [train_df, test_df]:
    dt = df["date"]
    df["month"] = dt.dt.month.astype(np.int8)
    df["quarter"] = dt.dt.quarter.astype(np.int8)
    df["day_of_month"] = dt.dt.day.astype(np.int8)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["day_of_year"] = dt.dt.dayofyear.astype(np.int16)
    df["is_friday"] = (dt.dt.dayofweek == 4).astype(np.int8)
    df["is_month_end"] = dt.dt.is_month_end.astype(np.int8)
    df["days_since_start"] = (dt - base_date).dt.days.astype(np.int32)
    df["time_trend"] = (df["year"] + (df["day_of_year"] - 1.0) / 365.25).astype(
        np.float32
    )

# Global Categorical Encodings (Consistent across train and test)
cat_cols = [
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
    "prop_tenure",
    "prop_new",
    "prop_tenure_new",
    "sale_prop",
    "county",
    "district",
    "town",
    "district_prop",
    "county_prop",
    "town_prop",
    "loc_triple",
]

encoding_maps = {}
for col in cat_cols:
    unique_vals = sorted(train_df[col].unique().tolist())
    # 1-based indexing, reserving 0 for unknown/unseen entities
    val_to_id = {v: i + 1 for i, v in enumerate(unique_vals)}
    encoding_maps[col] = val_to_id
    train_df[f"{col}_code"] = train_df[col].map(val_to_id).fillna(0).astype(np.int32)
    test_df[f"{col}_code"] = test_df[col].map(val_to_id).fillna(0).astype(np.int32)

# Frequency encoding computed on pre-validation reference period (2014-2015) to avoid validation leakage
ref_mask_freq = (train_df["year"] >= 2014) & (train_df["year"] <= 2015)
ref_df_freq = train_df[ref_mask_freq]

for col in ["district", "town", "county", "district_prop", "town_prop"]:
    counts = ref_df_freq[col].value_counts().to_dict()
    train_df[f"freq_log_{col}"] = np.log1p(train_df[col].map(counts).fillna(0)).astype(
        np.float32
    )
    test_df[f"freq_log_{col}"] = np.log1p(test_df[col].map(counts).fillna(0)).astype(
        np.float32
    )

del ref_df_freq
gc.collect()

# Dual-Horizon Hierarchical Empirical Bayes Target Encoding (1-Year Velocity + 3-Year Valuation)
print("Computing Dual-Horizon Empirical Bayes Target Encodings with Segment Momentum and Liquidity...")

years_to_process = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
te_records_train = []
te_records_test = None

for eval_year in years_to_process:
    ref_3y_start = eval_year - 3
    ref_3y_end = eval_year - 1
    ref_1y = eval_year - 1
    ref_t2 = eval_year - 2

    hist_3y_mask = (train_df["year"] >= ref_3y_start) & (train_df["year"] <= ref_3y_end)
    hist_3y = train_df.loc[
        hist_3y_mask,
        [
            "log10_price",
            "county",
            "district",
            "town",
            "property_type",
            "is_new_build",
            "tenure",
            "prop_tenure",
            "prop_new",
            "prop_tenure_new",
            "district_prop",
            "town_prop",
            "loc_triple",
            "loc_triple_prop",
            "district_new",
            "district_tenure",
            "sale_category",
            "year",
            "month",
            "time_trend",
        ],
    ].copy()

    global_mean_3y = float(hist_3y["log10_price"].mean())
    global_std_3y = float(hist_3y["log10_price"].std())

    # --- 1. 3-Year Stabilized Spatial Valuation Hierarchical Priors ---
    c_grp_3y = hist_3y.groupby("county")["log10_price"].agg(count="count", mean="mean").reset_index()
    c_grp_3y["te_c_3y"] = (c_grp_3y["count"] * c_grp_3y["mean"] + 50.0 * global_mean_3y) / (c_grp_3y["count"] + 50.0)
    county_map_3y = dict(zip(c_grp_3y["county"], c_grp_3y["te_c_3y"]))
    hist_3y["c_prior"] = hist_3y["county"].map(county_map_3y).fillna(global_mean_3y)

    d_grp_3y = hist_3y.groupby("district").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        c_prior=("c_prior", "mean")
    ).reset_index()
    d_grp_3y["te_d_3y"] = (d_grp_3y["count"] * d_grp_3y["mean"] + 25.0 * d_grp_3y["c_prior"]) / (d_grp_3y["count"] + 25.0)
    dist_map_3y = dict(zip(d_grp_3y["district"], d_grp_3y["te_d_3y"]))
    dist_counts_3y = dict(zip(d_grp_3y["district"], d_grp_3y["count"]))
    hist_3y["d_prior"] = hist_3y["district"].map(dist_map_3y).fillna(hist_3y["c_prior"])

    t_grp_3y = hist_3y.groupby("town").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        d_prior=("d_prior", "mean")
    ).reset_index()
    t_grp_3y["te_t_3y"] = (t_grp_3y["count"] * t_grp_3y["mean"] + 20.0 * t_grp_3y["d_prior"]) / (t_grp_3y["count"] + 20.0)
    town_map_3y = dict(zip(t_grp_3y["town"], t_grp_3y["te_t_3y"]))
    town_counts_3y = dict(zip(t_grp_3y["town"], t_grp_3y["count"]))

    # Micro-spatial Empirical Bayes Encodings: loc_triple shrunk to town prior (K=15)
    hist_3y["t_prior"] = hist_3y["town"].map(town_map_3y).fillna(hist_3y["d_prior"])
    loc_grp_3y = hist_3y.groupby("loc_triple").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        t_prior=("t_prior", "mean")
    ).reset_index()
    loc_grp_3y["te_loc_3y"] = (loc_grp_3y["count"] * loc_grp_3y["mean"] + 15.0 * loc_grp_3y["t_prior"]) / (loc_grp_3y["count"] + 15.0)
    loc_map_3y = dict(zip(loc_grp_3y["loc_triple"], loc_grp_3y["te_loc_3y"]))
    loc_diff_map_3y = dict(
        zip(
            loc_grp_3y["loc_triple"],
            (loc_grp_3y["count"] * (loc_grp_3y["mean"] - loc_grp_3y["t_prior"])) / (loc_grp_3y["count"] + 15.0),
        )
    )
    hist_3y["loc_prior"] = hist_3y["loc_triple"].map(loc_map_3y).fillna(hist_3y["t_prior"])

    dp_grp_3y = hist_3y.groupby("district_prop").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        std=("log10_price", "std"),
        d_prior=("d_prior", "mean")
    ).reset_index()
    dp_grp_3y["te_dp_3y"] = (dp_grp_3y["count"] * dp_grp_3y["mean"] + 15.0 * dp_grp_3y["d_prior"]) / (dp_grp_3y["count"] + 15.0)
    dp_map_3y = dict(zip(dp_grp_3y["district_prop"], dp_grp_3y["te_dp_3y"]))
    dp_counts_3y = dict(zip(dp_grp_3y["district_prop"], dp_grp_3y["count"]))
    dp_std_filled = dp_grp_3y["std"].fillna(global_std_3y)
    dp_disp_3y = (dp_grp_3y["count"] * dp_std_filled + 20.0 * global_std_3y) / (dp_grp_3y["count"] + 20.0)
    dp_disp_map_3y = dict(zip(dp_grp_3y["district_prop"], dp_disp_3y))

    tp_grp_3y = hist_3y.groupby("town_prop").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        t_prior=("t_prior", "mean")
    ).reset_index()
    tp_grp_3y["te_tp_3y"] = (tp_grp_3y["count"] * tp_grp_3y["mean"] + 10.0 * tp_grp_3y["t_prior"]) / (tp_grp_3y["count"] + 10.0)
    tp_map_3y = dict(zip(tp_grp_3y["town_prop"], tp_grp_3y["te_tp_3y"]))

    # Micro-spatial Empirical Bayes Encodings: loc_triple_prop shrunk toward loc_prior & town_prop (K=10)
    locp_grp_3y = hist_3y.groupby("loc_triple_prop").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        loc_prior=("loc_prior", "mean")
    ).reset_index()
    locp_grp_3y["te_locp_3y"] = (locp_grp_3y["count"] * locp_grp_3y["mean"] + 10.0 * locp_grp_3y["loc_prior"]) / (locp_grp_3y["count"] + 10.0)
    loc_prop_map_3y = dict(zip(locp_grp_3y["loc_triple_prop"], locp_grp_3y["te_locp_3y"]))

    # Localized property interactions: district_new and district_tenure
    dn_grp_3y = hist_3y.groupby("district_new").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        d_prior=("d_prior", "mean")
    ).reset_index()
    dn_grp_3y["te_dn_3y"] = (dn_grp_3y["count"] * dn_grp_3y["mean"] + 20.0 * dn_grp_3y["d_prior"]) / (dn_grp_3y["count"] + 20.0)
    d_new_map_3y = dict(zip(dn_grp_3y["district_new"], dn_grp_3y["te_dn_3y"]))

    dt_grp_3y = hist_3y.groupby("district_tenure").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        d_prior=("d_prior", "mean")
    ).reset_index()
    dt_grp_3y["te_dt_3y"] = (dt_grp_3y["count"] * dt_grp_3y["mean"] + 20.0 * dt_grp_3y["d_prior"]) / (dt_grp_3y["count"] + 20.0)
    d_tenure_map_3y = dict(zip(dt_grp_3y["district_tenure"], dt_grp_3y["te_dt_3y"]))

    pt_grp = hist_3y.groupby("prop_tenure")["log10_price"].agg(["count", "mean"]).reset_index()
    pt_map = dict(zip(pt_grp["prop_tenure"], (pt_grp["count"] * pt_grp["mean"] + 30.0 * global_mean_3y) / (pt_grp["count"] + 30.0)))

    pn_grp = hist_3y.groupby("prop_new")["log10_price"].agg(["count", "mean"]).reset_index()
    pn_map = dict(zip(pn_grp["prop_new"], (pn_grp["count"] * pn_grp["mean"] + 30.0 * global_mean_3y) / (pn_grp["count"] + 30.0)))

    ptn_grp = hist_3y.groupby("prop_tenure_new")["log10_price"].agg(["count", "mean"]).reset_index()
    ptn_map = dict(zip(ptn_grp["prop_tenure_new"], (ptn_grp["count"] * ptn_grp["mean"] + 30.0 * global_mean_3y) / (ptn_grp["count"] + 30.0)))

    sc_grp = hist_3y.groupby("sale_category")["log10_price"].agg(["count", "mean"]).reset_index()
    sc_map = dict(zip(sc_grp["sale_category"], (sc_grp["count"] * sc_grp["mean"] + 30.0 * global_mean_3y) / (sc_grp["count"] + 30.0)))

    # --- 2. 1-Year Responsive Velocity Window (t-1) & Semi-Annual Windows ---
    hist_1y = hist_3y.loc[hist_3y["year"] == ref_1y].copy()
    global_mean_1y = float(hist_1y["log10_price"].mean())

    c_grp_1y = hist_1y.groupby("county")["log10_price"].agg(count="count", mean="mean").reset_index()
    c_grp_1y["prior"] = c_grp_1y["county"].map(county_map_3y).fillna(global_mean_1y)
    c_grp_1y["te_c_1y"] = (c_grp_1y["count"] * c_grp_1y["mean"] + 30.0 * c_grp_1y["prior"]) / (c_grp_1y["count"] + 30.0)
    county_map_1y = dict(zip(c_grp_1y["county"], c_grp_1y["te_c_1y"]))
    hist_1y["c_prior"] = hist_1y["county"].map(county_map_1y).fillna(global_mean_1y)

    d_grp_1y = hist_1y.groupby("district").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        c_prior=("c_prior", "mean")
    ).reset_index()
    d_grp_1y["d_prior_3y"] = d_grp_1y["district"].map(dist_map_3y).fillna(d_grp_1y["c_prior"])
    d_grp_1y["te_d_1y"] = (d_grp_1y["count"] * d_grp_1y["mean"] + 20.0 * d_grp_1y["d_prior_3y"]) / (d_grp_1y["count"] + 20.0)
    dist_map_1y = dict(zip(d_grp_1y["district"], d_grp_1y["te_d_1y"]))
    dist_counts_1y = dict(zip(d_grp_1y["district"], d_grp_1y["count"]))
    hist_1y["d_prior"] = hist_1y["district"].map(dist_map_1y).fillna(hist_1y["c_prior"])

    t_grp_1y = hist_1y.groupby("town").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        d_prior=("d_prior", "mean")
    ).reset_index()
    t_grp_1y["t_prior_3y"] = t_grp_1y["town"].map(town_map_3y).fillna(t_grp_1y["d_prior"])
    t_grp_1y["te_t_1y"] = (t_grp_1y["count"] * t_grp_1y["mean"] + 15.0 * t_grp_1y["t_prior_3y"]) / (t_grp_1y["count"] + 15.0)
    town_map_1y = dict(zip(t_grp_1y["town"], t_grp_1y["te_t_1y"]))
    town_counts_1y = dict(zip(t_grp_1y["town"], t_grp_1y["count"]))

    dp_grp_1y = hist_1y.groupby("district_prop").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        d_prior=("d_prior", "mean")
    ).reset_index()
    dp_grp_1y["dp_prior_3y"] = dp_grp_1y["district_prop"].map(dp_map_3y).fillna(dp_grp_1y["d_prior"])
    dp_grp_1y["te_dp_1y"] = (dp_grp_1y["count"] * dp_grp_1y["mean"] + 12.0 * dp_grp_1y["dp_prior_3y"]) / (dp_grp_1y["count"] + 12.0)
    dp_map_1y = dict(zip(dp_grp_1y["district_prop"], dp_grp_1y["te_dp_1y"]))
    dp_counts_1y = dict(zip(dp_grp_1y["district_prop"], dp_grp_1y["count"]))

    # Empirical Bayes shrunk 1-year town-property premiums over district-property (K=12)
    hist_1y["dp_val_1y"] = hist_1y["district_prop"].map(dp_map_1y).fillna(global_mean_1y)
    hist_1y["tp_over_dp"] = hist_1y["log10_price"] - hist_1y["dp_val_1y"]
    tp_prem_grp_1y = hist_1y.groupby("town_prop")["tp_over_dp"].agg(
        count="count",
        mean="mean"
    ).reset_index()
    tp_prem_grp_1y["prem_shrunk"] = (tp_prem_grp_1y["count"] * tp_prem_grp_1y["mean"]) / (tp_prem_grp_1y["count"] + 12.0)
    town_prem_map_1y = dict(zip(tp_prem_grp_1y["town_prop"], tp_prem_grp_1y["prem_shrunk"]))

    # Property prior and Empirical Bayes shrunk attribute differentials
    p_grp_1y = hist_1y.groupby("property_type")["log10_price"].agg(count="count", mean="mean").reset_index()
    p_prior_map_1y = dict(zip(p_grp_1y["property_type"], p_grp_1y["mean"]))
    hist_1y["p_prior"] = hist_1y["property_type"].map(p_prior_map_1y).fillna(global_mean_1y)
    hist_1y["diff_from_prop"] = hist_1y["log10_price"] - hist_1y["p_prior"]

    nb_grp = hist_1y.groupby("is_new_build")["diff_from_prop"].agg(count="count", mean="mean").reset_index()
    nb_grp["offset"] = (nb_grp["count"] * nb_grp["mean"]) / (nb_grp["count"] + 25.0)
    nb_offset_map = dict(zip(nb_grp["is_new_build"], nb_grp["offset"]))

    ten_grp = hist_1y.groupby("tenure")["diff_from_prop"].agg(count="count", mean="mean").reset_index()
    ten_grp["offset"] = (ten_grp["count"] * ten_grp["mean"]) / (ten_grp["count"] + 25.0)
    tenure_offset_map = dict(zip(ten_grp["tenure"], ten_grp["offset"]))

    sc_grp_1y = hist_1y.groupby("sale_category")["diff_from_prop"].agg(count="count", mean="mean").reset_index()
    sc_grp_1y["offset"] = (sc_grp_1y["count"] * sc_grp_1y["mean"]) / (sc_grp_1y["count"] + 20.0)
    sc_offset_map = dict(zip(sc_grp_1y["sale_category"], sc_grp_1y["offset"]))

    # Trailing semi-annual valuation window (H2 vs H1 of t-1)
    hist_1y_h1 = hist_1y.loc[hist_1y["month"] <= 6]
    hist_1y_h2 = hist_1y.loc[hist_1y["month"] >= 7]
    m_h1 = hist_1y_h1["log10_price"].mean()
    m_h2 = hist_1y_h2["log10_price"].mean()
    accel_national = float(m_h2 - m_h1) if (pd.notna(m_h1) and pd.notna(m_h2)) else 0.0
    accel_national = float(np.clip(accel_national, -0.15, 0.15))

    d_h1 = hist_1y_h1.groupby("district")["log10_price"].agg(count="count", mean="mean")
    d_h2 = hist_1y_h2.groupby("district")["log10_price"].agg(count="count", mean="mean")
    d_semi = d_h2.join(d_h1, lsuffix="_h2", rsuffix="_h1")
    d_semi_eff = np.minimum(d_semi["count_h2"].fillna(0), d_semi["count_h1"].fillna(0))
    d_semi_raw = (d_semi["mean_h2"] - d_semi["mean_h1"]).fillna(0.0)
    d_semi_accel = (d_semi_eff * d_semi_raw + 25.0 * accel_national) / (d_semi_eff + 25.0)
    dist_accel_map = d_semi_accel.fillna(accel_national).clip(-0.25, 0.25).to_dict()

    c_h2 = d_semi["count_h2"].fillna(0)
    m_h2_dist = d_semi["mean_h2"].fillna(0.0)
    d_prior_1y = pd.Series(d_semi.index.map(dist_map_1y).fillna(global_mean_1y), index=d_semi.index)
    te_d_h2_series = (c_h2 * m_h2_dist + 20.0 * d_prior_1y) / (c_h2 + 20.0)
    dist_h2_map = te_d_h2_series.fillna(d_prior_1y).to_dict()

    dp_h1 = hist_1y_h1.groupby("district_prop")["log10_price"].agg(count="count", mean="mean")
    dp_h2 = hist_1y_h2.groupby("district_prop")["log10_price"].agg(count="count", mean="mean")
    dp_semi = dp_h2.join(dp_h1, lsuffix="_h2", rsuffix="_h1")
    c_dp_h2 = dp_semi["count_h2"].fillna(0)
    m_dp_h2 = dp_semi["mean_h2"].fillna(0.0)
    dp_prior_series = pd.Series(dp_semi.index.map(dp_map_1y).fillna(global_mean_1y), index=dp_semi.index)
    te_dp_h2_series = (c_dp_h2 * m_dp_h2 + 15.0 * dp_prior_series) / (c_dp_h2 + 15.0)
    dist_prop_h2_map = te_dp_h2_series.fillna(dp_prior_series).to_dict()

    c_h1 = d_semi["count_h1"].fillna(0)
    liq_semi = (c_h2 - c_h1) / (c_h1 + 10.0)
    dist_semi_liq_map = liq_semi.fillna(0.0).clip(-3.0, 3.0).to_dict()

    # --- 3. Segment-Specific Momentum Indicators (t-2 to t-1) ---
    hist_t2 = hist_3y.loc[hist_3y["year"] == ref_t2]
    m_t2 = hist_t2["log10_price"].mean()
    m_t1 = hist_1y["log10_price"].mean()
    momentum_national = (
        float(m_t1 - m_t2) if (pd.notna(m_t1) and pd.notna(m_t2)) else 0.0
    )
    momentum_national = float(np.clip(momentum_national, -0.15, 0.15))

    c_t2 = hist_t2.groupby("county")["log10_price"].agg(count="count", mean="mean")
    c_t1 = hist_1y.groupby("county")["log10_price"].agg(count="count", mean="mean")
    c_growth = c_t1.join(c_t2, lsuffix="_end", rsuffix="_start")
    c_eff = np.minimum(c_growth["count_end"].fillna(0), c_growth["count_start"].fillna(0))
    c_raw_delta = (c_growth["mean_end"] - c_growth["mean_start"]).fillna(0.0)
    c_mom = (c_eff * c_raw_delta + 50.0 * momentum_national) / (c_eff + 50.0)
    county_momentum_map = c_mom.fillna(momentum_national).clip(-0.20, 0.20).to_dict()

    d_t2 = hist_t2.groupby("district")["log10_price"].agg(count="count", mean="mean")
    d_t1 = hist_1y.groupby("district")["log10_price"].agg(count="count", mean="mean")
    d_growth = d_t1.join(d_t2, lsuffix="_end", rsuffix="_start")
    d_eff = np.minimum(d_growth["count_end"].fillna(0), d_growth["count_start"].fillna(0))
    d_raw_delta = (d_growth["mean_end"] - d_growth["mean_start"]).fillna(0.0)
    dist_to_county = hist_3y.groupby("district")["county"].first().to_dict()
    d_prior_mom = (
        d_growth.index.to_series()
        .map(dist_to_county)
        .map(county_momentum_map)
        .fillna(momentum_national)
    )
    d_mom = (d_eff * d_raw_delta + 30.0 * d_prior_mom) / (d_eff + 30.0)
    dist_momentum_map = d_mom.fillna(d_prior_mom).clip(-0.25, 0.25).to_dict()

    # Property-segment momentum: district x property_type
    dp_t2 = hist_t2.groupby("district_prop")["log10_price"].agg(count="count", mean="mean")
    dp_t1 = hist_1y.groupby("district_prop")["log10_price"].agg(count="count", mean="mean")
    dp_growth = dp_t1.join(dp_t2, lsuffix="_end", rsuffix="_start")
    dp_eff = np.minimum(dp_growth["count_end"].fillna(0), dp_growth["count_start"].fillna(0))
    dp_raw_delta = (dp_growth["mean_end"] - dp_growth["mean_start"]).fillna(0.0)
    dp_to_dist = hist_3y.groupby("district_prop")["district"].first().to_dict()
    dp_prior_mom = (
        dp_growth.index.to_series()
        .map(dp_to_dist)
        .map(dist_momentum_map)
        .fillna(momentum_national)
    )
    dp_mom = (dp_eff * dp_raw_delta + 20.0 * dp_prior_mom) / (dp_eff + 20.0)
    dp_momentum_map = dp_mom.fillna(dp_prior_mom).clip(-0.30, 0.30).to_dict()

    if eval_year == 2017:
        target_df = test_df
    else:
        target_df = train_df.loc[train_df["year"] == eval_year]

    # Target Mapping: 3-Year Valuations
    te_c = target_df["county"].map(county_map_3y).fillna(global_mean_3y).astype(np.float32)
    te_d = target_df["district"].map(dist_map_3y).fillna(te_c).astype(np.float32)
    te_t = target_df["town"].map(town_map_3y).fillna(te_d).astype(np.float32)
    te_loc = target_df["loc_triple"].map(loc_map_3y).fillna(te_t).astype(np.float32)
    te_dp = target_df["district_prop"].map(dp_map_3y).fillna(te_d).astype(np.float32)
    te_tp = target_df["town_prop"].map(tp_map_3y).fillna(te_t).astype(np.float32)
    te_loc_p = target_df["loc_triple_prop"].map(loc_prop_map_3y).fillna(te_loc).astype(np.float32)

    # Localized property interactions
    te_dn = target_df["district_new"].map(d_new_map_3y).fillna(te_d).astype(np.float32)
    te_dt = target_df["district_tenure"].map(d_tenure_map_3y).fillna(te_d).astype(np.float32)

    # Target Mapping: 1-Year & Semi-Annual Valuations
    te_c_1y = target_df["county"].map(county_map_1y).fillna(te_c).astype(np.float32)
    te_d_1y = target_df["district"].map(dist_map_1y).fillna(te_d).astype(np.float32)
    te_t_1y = target_df["town"].map(town_map_1y).fillna(te_t).astype(np.float32)
    te_dp_1y = target_df["district_prop"].map(dp_map_1y).fillna(te_dp).astype(np.float32)
    te_d_h2 = target_df["district"].map(dist_h2_map).fillna(te_d_1y).astype(np.float32)
    te_dp_h2 = target_df["district_prop"].map(dist_prop_h2_map).fillna(te_dp_1y).astype(np.float32)
    accel_d = target_df["district"].map(dist_accel_map).fillna(accel_national).astype(np.float32)
    liq_delta_semi = target_df["district"].map(dist_semi_liq_map).fillna(0.0).astype(np.float32)

    offset_nb = target_df["is_new_build"].map(nb_offset_map).fillna(0.0).astype(np.float32)
    offset_tenure = target_df["tenure"].map(tenure_offset_map).fillna(0.0).astype(np.float32)
    offset_sc = target_df["sale_category"].map(sc_offset_map).fillna(0.0).astype(np.float32)

    te_pt = target_df["prop_tenure"].map(pt_map).fillna(global_mean_3y).astype(np.float32)
    te_pn = target_df["prop_new"].map(pn_map).fillna(global_mean_3y).astype(np.float32)
    te_ptn = target_df["prop_tenure_new"].map(ptn_map).fillna(global_mean_3y).astype(np.float32)
    te_sc = target_df["sale_category"].map(sc_map).fillna(global_mean_3y).astype(np.float32)
    te_disp_dp = target_df["district_prop"].map(dp_disp_map_3y).fillna(global_std_3y).astype(np.float32)

    ref_1y_center = (
        float(hist_1y["time_trend"].mean())
        if len(hist_1y) > 0
        else float(ref_1y) + 0.5
    )
    dt_diff = np.clip((target_df["time_trend"] - ref_1y_center).astype(np.float32), 0.0, 3.0)
    mom_c = target_df["county"].map(county_momentum_map).fillna(momentum_national).astype(np.float32)
    mom_d = target_df["district"].map(dist_momentum_map).fillna(mom_c).astype(np.float32)
    mom_dp = target_df["district_prop"].map(dp_momentum_map).fillna(mom_d).astype(np.float32)

    # Empirical Bayes shrunk 1-year town-property premiums and 3-year loc_triple differentials
    town_prem_shrunk = target_df["town_prop"].map(town_prem_map_1y).fillna(0.0).astype(np.float32)
    loc_triple_diff_shrunk = target_df["loc_triple"].map(loc_diff_map_3y).fillna(0.0).astype(np.float32)

    # Analytical hierarchical spatial & structural baseline projection
    base_proj = (
        te_dp_1y
        + town_prem_shrunk
        + loc_triple_diff_shrunk
        + offset_nb
        + offset_tenure
        + offset_sc
        + mom_dp * dt_diff
    ).astype(np.float32)

    # Localized transaction liquidity frequency ratios
    c1_d = target_df["district"].map(dist_counts_1y).fillna(0).astype(np.float32)
    c3_d = target_df["district"].map(dist_counts_3y).fillna(0).astype(np.float32)
    liq_d = np.clip(c1_d / (c3_d / 3.0 + 1.0), 0.0, 10.0).astype(np.float32)

    c1_dp = target_df["district_prop"].map(dp_counts_1y).fillna(0).astype(np.float32)
    c3_dp = target_df["district_prop"].map(dp_counts_3y).fillna(0).astype(np.float32)
    liq_dp = np.clip(c1_dp / (c3_dp / 3.0 + 1.0), 0.0, 10.0).astype(np.float32)

    c1_t = target_df["town"].map(town_counts_1y).fillna(0).astype(np.float32)
    c3_t = target_df["town"].map(town_counts_3y).fillna(0).astype(np.float32)
    liq_t = np.clip(c1_t / (c3_t / 3.0 + 1.0), 0.0, 10.0).astype(np.float32)

    te_block = pd.DataFrame(
        {
            "id": target_df["id"].values,
            "dt_diff": dt_diff.values,
            "base_proj": base_proj.values,
            "hist_mean_national": np.float32(global_mean_3y),
            "hist_trend_national": np.float32(momentum_national),
            "momentum_county": mom_c.values,
            "momentum_district": mom_d.values,
            "momentum_district_prop": mom_dp.values,
            "te_county": te_c.values,
            "te_district": te_d.values,
            "te_town": te_t.values,
            "te_loc_triple": te_loc.values,
            "te_district_prop": te_dp.values,
            "te_town_prop": te_tp.values,
            "te_loc_triple_prop": te_loc_p.values,
            "te_district_new": te_dn.values,
            "te_district_tenure": te_dt.values,
            "te_county_1y": te_c_1y.values,
            "te_district_1y": te_d_1y.values,
            "te_town_1y": te_t_1y.values,
            "te_district_prop_1y": te_dp_1y.values,
            "te_district_h2": te_d_h2.values,
            "te_dp_h2": te_dp_h2.values,
            "delta_1y_3y_district": (te_d_1y - te_d).values,
            "delta_1y_3y_district_prop": (te_dp_1y - te_dp).values,
            "delta_h2_1y_district": (te_d_h2 - te_d_1y).values,
            "delta_h2_1y_district_prop": (te_dp_h2 - te_dp_1y).values,
            "accel_district_semi": accel_d.values,
            "liquidity_delta_semi_district": liq_delta_semi.values,
            "liquidity_ratio_district": liq_d.values,
            "liquidity_ratio_district_prop": liq_dp.values,
            "liquidity_ratio_town": liq_t.values,
            "te_prop_tenure": te_pt.values,
            "te_prop_new": te_pn.values,
            "te_prop_tenure_new": te_ptn.values,
            "te_sale_category": te_sc.values,
            "dispersion_district_prop": te_disp_dp.values,
            "te_national_projected": (global_mean_1y + momentum_national * dt_diff).values,
            "te_county_projected": (te_c_1y + mom_c * dt_diff).values,
            "te_district_projected": (te_d_1y + mom_d * dt_diff).values,
            "te_town_projected": (te_t_1y + mom_d * dt_diff).values,
            "te_district_prop_projected": base_proj.values,
            "te_town_prop_projected": (te_tp + mom_dp * dt_diff).values,
            "town_prem_shrunk": town_prem_shrunk.values,
            "loc_triple_diff_shrunk": loc_triple_diff_shrunk.values,
            "rel_premium_district": (te_d - global_mean_3y).values,
            "rel_premium_town": (te_t - te_d).values,
            "rel_premium_loc_triple": (te_loc - te_t).values,
            "rel_premium_loc_triple_prop": (te_loc_p - te_loc).values,
            "rel_premium_district_prop": (te_dp - te_d).values,
            "rel_premium_town_prop": (te_tp - te_t).values,
            "rel_premium_district_new": (te_dn - te_d).values,
            "rel_premium_district_tenure": (te_dt - te_d).values,
            "rel_premium_prop_tenure": (te_pt - global_mean_3y).values,
            "rel_premium_prop_new": (te_pn - global_mean_3y).values,
            "rel_premium_prop_tenure_new": (te_ptn - global_mean_3y).values,
            "rel_diff_town_to_district_prop": (te_tp - te_dp).values,
            "rel_diff_loc_to_district_prop": (te_loc_p - te_dp).values,
            "rel_diff_new_to_district": (te_dn - te_d).values,
            "rel_diff_new_build": offset_nb.values,
            "rel_diff_sale_b": offset_sc.values,
            "rel_diff_tenure": offset_tenure.values,
        }
    )

    if eval_year == 2017:
        te_records_test = te_block
    else:
        te_records_train.append(te_block)

te_df_train = pd.concat(te_records_train, ignore_index=True)
del te_records_train
gc.collect()

# Filter train to modern market regime (2011-2016) and merge TE features
print("Filtering training set to modern regime (2011-2016)...")
train_modern = train_df[(train_df["year"] >= 2011) & (train_df["year"] <= 2016)].copy()
del train_df
gc.collect()

train_modern = train_modern.merge(te_df_train, on="id", how="inner")
del te_df_train
gc.collect()

test_df = test_df.merge(te_records_test, on="id", how="inner")
del te_records_test
gc.collect()

# Define validation splits and clean train mask
train_modern["is_val_2016"] = (train_modern["year"] == 2016).astype(bool)
train_modern["is_val_h2_2016"] = (train_modern["date"] >= "2016-07-01").astype(bool)
train_modern["is_clean_train"] = (train_modern["price"] >= 1000) & (
    train_modern["price"] <= 50000000
)

feature_columns = [
    # Relative Temporal Features
    "dt_diff",
    "month",
    "quarter",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "is_friday",
    "is_month_end",
    # Categorical Encoded IDs
    "property_type_code",
    "is_new_build_code",
    "tenure_code",
    "sale_category_code",
    "prop_tenure_code",
    "prop_new_code",
    "prop_tenure_new_code",
    "sale_prop_code",
    "county_code",
    "district_code",
    "town_code",
    "district_prop_code",
    "town_prop_code",
    "loc_triple_code",
    # Frequency / Volume Densities
    "freq_log_district",
    "freq_log_town",
    "freq_log_county",
    "freq_log_district_prop",
    "freq_log_town_prop",
    # Multi-Scale Target Encodings & Momentum
    "hist_mean_national",
    "hist_trend_national",
    "momentum_county",
    "momentum_district",
    "momentum_district_prop",
    "te_county",
    "te_district",
    "te_town",
    "te_loc_triple",
    "te_district_prop",
    "te_town_prop",
    "te_loc_triple_prop",
    "te_district_new",
    "te_district_tenure",
    "te_county_1y",
    "te_district_1y",
    "te_town_1y",
    "te_district_prop_1y",
    "te_district_h2",
    "te_dp_h2",
    "delta_1y_3y_district",
    "delta_1y_3y_district_prop",
    "delta_h2_1y_district",
    "delta_h2_1y_district_prop",
    "accel_district_semi",
    "liquidity_delta_semi_district",
    "liquidity_ratio_district",
    "liquidity_ratio_district_prop",
    "liquidity_ratio_town",
    "te_prop_tenure",
    "te_prop_new",
    "te_prop_tenure_new",
    "te_sale_category",
    "dispersion_district_prop",
    "te_national_projected",
    "te_county_projected",
    "te_district_projected",
    "te_town_projected",
    "te_district_prop_projected",
    "te_town_prop_projected",
    # Relative Micro & Regional Price Premiums & Contrasts
    "town_prem_shrunk",
    "loc_triple_diff_shrunk",
    "rel_premium_district",
    "rel_premium_town",
    "rel_premium_loc_triple",
    "rel_premium_loc_triple_prop",
    "rel_premium_district_prop",
    "rel_premium_town_prop",
    "rel_premium_district_new",
    "rel_premium_district_tenure",
    "rel_premium_prop_tenure",
    "rel_premium_prop_new",
    "rel_premium_prop_tenure_new",
    "rel_diff_town_to_district_prop",
    "rel_diff_loc_to_district_prop",
    "rel_diff_new_to_district",
    "rel_diff_new_build",
    "rel_diff_sale_b",
    "rel_diff_tenure",
]

categorical_columns = [
    "property_type_code",
    "is_new_build_code",
    "tenure_code",
    "sale_category_code",
    "prop_tenure_code",
    "prop_new_code",
    "prop_tenure_new_code",
    "sale_prop_code",
    "county_code",
    "district_code",
    "town_code",
    "district_prop_code",
    "town_prop_code",
    "loc_triple_code",
]

cont_cols = [c for c in feature_columns if c not in categorical_columns]
cat_cardinalities = [len(encoding_maps[col.replace("_code", "")]) for col in categorical_columns]

# Ensure zero NaNs across all feature columns
train_modern[cont_cols] = train_modern[cont_cols].fillna(0.0).astype(np.float32)
test_df[cont_cols] = test_df[cont_cols].fillna(0.0).astype(np.float32)
train_modern[categorical_columns] = train_modern[categorical_columns].fillna(0).astype(np.int32)
test_df[categorical_columns] = test_df[categorical_columns].fillna(0).astype(np.int32)

# Save metadata for reproducibility
meta_dict = {
    "feature_columns": feature_columns,
    "categorical_columns": categorical_columns,
    "target_column": "log10_price",
    "train_rows": len(train_modern),
    "test_rows": len(test_df),
    "num_features": len(feature_columns),
    "cat_cardinalities": cat_cardinalities,
    "val_2016_rows": int(train_modern["is_val_2016"].sum()),
}
with open(os.path.join(WORKING_DIR, "feature_meta.json"), "w") as f:
    json.dump(meta_dict, f, indent=2)

# ==============================================================================
# 5. MODEL ARCHITECTURE VERIFICATION
# ==============================================================================
print("Verifying LightGBM, GPU XGBoost, and TabularNN model initializations...")
dummy_lgb = create_lgb_model()
dummy_xgb = create_xgb_model()
dummy_nn = TabularNN(cat_cardinalities, len(cont_cols)).to(device)
print(
    f"LightGBM verified. XGBoost device: {dummy_xgb.get_params().get('device', 'cpu')}, "
    f"tree_method: {dummy_xgb.get_params().get('tree_method')}. "
    f"TabularNN initialized on {device}."
)
del dummy_lgb, dummy_xgb, dummy_nn
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 6. TRAINING & OUT-OF-TIME VALIDATION
# ==============================================================================
# Gentle recency sample weighting: exp(0.10 * (t - 2011))
train_modern["sample_weight"] = np.exp(
    0.10 * (train_modern["time_trend"] - 2011.0)
).astype(np.float32)

val_mask = train_modern["is_val_2016"].values
train_mask = (~val_mask) & (train_modern["is_clean_train"].values)

base_proj_train = train_modern.loc[train_mask, "base_proj"].values.astype(np.float32)
base_proj_val = train_modern.loc[val_mask, "base_proj"].values.astype(np.float32)

X_train = train_modern.loc[train_mask, feature_columns]
y_train = train_modern.loc[train_mask, "log10_price"].values.astype(np.float32)
y_train_res = (y_train - base_proj_train).astype(np.float32)
w_train = train_modern.loc[train_mask, "sample_weight"].values.astype(np.float32)

X_val = train_modern.loc[val_mask, feature_columns]
y_val = train_modern.loc[val_mask, "log10_price"].values.astype(np.float32)
y_val_res = (y_val - base_proj_val).astype(np.float32)
val_true_price = train_modern.loc[val_mask, "price"].values

print(f"Validation split: {len(X_train):,} train samples, {len(X_val):,} val samples.")

# Compute continuous feature standardization on train split
cont_mean = train_modern.loc[train_mask, cont_cols].mean().values.astype(np.float32)
cont_std = train_modern.loc[train_mask, cont_cols].std().values.astype(np.float32)
cont_std = np.where(cont_std < 1e-6, 1.0, cont_std)

X_train_cont_norm = np.nan_to_num(
    ((X_train[cont_cols].values - cont_mean) / cont_std).astype(np.float32)
)
X_train_cat = X_train[categorical_columns].values.astype(np.int32)

X_val_cont_norm = np.nan_to_num(
    ((X_val[cont_cols].values - cont_mean) / cont_std).astype(np.float32)
)
X_val_cat = X_val[categorical_columns].values.astype(np.int32)

# 6.1 Train Validation LightGBM Model on residuals
print("Training Validation LightGBM Regressor on residuals...")
val_lgb_model = create_lgb_model()
lgb_callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

val_lgb_model.fit(
    X_train,
    y_train_res,
    sample_weight=w_train,
    eval_set=[(X_val, y_val_res)],
    categorical_feature=categorical_columns,
    callbacks=lgb_callbacks,
)

lgb_best_iteration = (
    val_lgb_model.best_iteration_
    if hasattr(val_lgb_model, "best_iteration_") and val_lgb_model.best_iteration_ > 0
    else 1500
)
print(f"Validation LightGBM converged at iteration: {lgb_best_iteration}")

# 6.2 Train Validation GPU XGBoost Model on residuals
print("Training Validation GPU-accelerated XGBoost Regressor on residuals...")
val_xgb_model = create_xgb_model(early_stopping_rounds=100)
xgb_fit_kwargs = {
    "sample_weight": w_train,
    "eval_set": [(X_val, y_val_res)],
    "verbose": False,
}
if "early_stopping_rounds" not in val_xgb_model.get_params():
    xgb_fit_kwargs["early_stopping_rounds"] = 100

val_xgb_model.fit(X_train, y_train_res, **xgb_fit_kwargs)

xgb_best_iter_raw = getattr(val_xgb_model, "best_iteration", None)
if xgb_best_iter_raw is not None and xgb_best_iter_raw > 0:
    xgb_best_iteration = xgb_best_iter_raw + 1
else:
    xgb_best_iteration = getattr(val_xgb_model, "best_ntree_limit", 500)
print(f"Validation XGBoost converged at iteration: {xgb_best_iteration}")

# 6.3 Train Validation Deep Tabular Neural Network on residuals
print("Training Validation Deep Tabular Neural Network on residuals...")
val_nn_model = train_tabular_nn(
    X_train_cat,
    X_train_cont_norm,
    y_train_res,
    w_train,
    cat_cardinalities,
    len(cont_cols),
    device,
    epochs=5,
    batch_size=8192,
)

# 6.4 Out-of-Time Residual Prediction and Constrained Convex Blending Optimization
val_pred_res_lgb = val_lgb_model.predict(X_val)
val_pred_res_xgb = predict_xgb(val_xgb_model, X_val, device)
val_pred_res_nn = predict_tabular_nn(val_nn_model, X_val_cat, X_val_cont_norm, device)

lgb_val_pred_log10 = base_proj_val + val_pred_res_lgb
xgb_val_pred_log10 = base_proj_val + val_pred_res_xgb
nn_val_pred_log10 = base_proj_val + val_pred_res_nn

lgb_val_price = np.clip(10.0 ** lgb_val_pred_log10, 1.0, None)
xgb_val_price = np.clip(10.0 ** xgb_val_pred_log10, 1.0, None)
nn_val_price = np.clip(10.0 ** nn_val_pred_log10, 1.0, None)

lgb_val_score = compute_raw_price_rmse(val_true_price, lgb_val_price)
xgb_val_score = compute_raw_price_rmse(val_true_price, xgb_val_price)
nn_val_score = compute_raw_price_rmse(val_true_price, nn_val_price)
print(f"Standalone Validation LGBM Score: {lgb_val_score:.6f}")
print(f"Standalone Validation XGBoost Score: {xgb_val_score:.6f}")
print(f"Standalone Validation TabularNN Score: {nn_val_score:.6f}")

def blend_objective(weights):
    w = np.maximum(weights, 0.0)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    blend_res = w[0] * val_pred_res_lgb + w[1] * val_pred_res_xgb + w[2] * val_pred_res_nn
    blend_log10 = base_proj_val + blend_res
    blend_price = np.clip(10.0 ** blend_log10, 1.0, None)
    return compute_raw_price_rmse(val_true_price, blend_price)

init_weights = [0.45, 0.35, 0.20]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"eps": 1e-3, "maxiter": 100, "ftol": 1e-6},
)
best_weights = np.maximum(opt_res.x, 0.0)
best_weights = best_weights / np.sum(best_weights)
w_lgb, w_xgb, w_nn = float(best_weights[0]), float(best_weights[1]), float(best_weights[2])

final_val_score = blend_objective(best_weights)
print(f"Optimal Ensemble Blend: {w_lgb:.3f} LGBM + {w_xgb:.3f} XGBoost + {w_nn:.3f} TabularNN")
print(f"Blended Holdout Validation Score: {final_val_score:.6f}")

# Clean validation resources before full retraining
del val_lgb_model, val_xgb_model, val_nn_model
del X_train, y_train, y_train_res, w_train, X_val, y_val, y_val_res, val_true_price
del X_train_cont_norm, X_train_cat, X_val_cont_norm, X_val_cat
del val_pred_res_lgb, val_pred_res_xgb, val_pred_res_nn, lgb_val_price, xgb_val_price, nn_val_price
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 7. PRODUCTION MODEL RETRAINING & INFERENCE
# ==============================================================================
full_lgb_estimators = max(lgb_best_iteration, int(lgb_best_iteration * 1.05))
full_xgb_estimators = max(xgb_best_iteration, int(xgb_best_iteration * 1.05))

print(
    f"Retraining production tri-model ensemble on full modern history residuals (2011-2016)... "
    f"(LGBM estimators: {full_lgb_estimators}, XGBoost estimators: {full_xgb_estimators}, TabularNN epochs: 5)"
)

full_clean_mask = train_modern["is_clean_train"].values
base_proj_full = train_modern.loc[full_clean_mask, "base_proj"].values.astype(np.float32)
X_full = train_modern.loc[full_clean_mask, feature_columns]
y_full = train_modern.loc[full_clean_mask, "log10_price"].values.astype(np.float32)
y_full_res = (y_full - base_proj_full).astype(np.float32)
w_full = train_modern.loc[full_clean_mask, "sample_weight"].values.astype(np.float32)

# Standardize continuous features across full modern dataset
full_cont_mean = train_modern.loc[full_clean_mask, cont_cols].mean().values.astype(np.float32)
full_cont_std = train_modern.loc[full_clean_mask, cont_cols].std().values.astype(np.float32)
full_cont_std = np.where(full_cont_std < 1e-6, 1.0, full_cont_std)

X_full_cont_norm = np.nan_to_num(
    ((X_full[cont_cols].values - full_cont_mean) / full_cont_std).astype(np.float32)
)
X_full_cat = X_full[categorical_columns].values.astype(np.int32)

# Retrain full LightGBM model
prod_lgb_model = create_lgb_model()
prod_lgb_model.set_params(n_estimators=full_lgb_estimators)
prod_lgb_model.fit(
    X_full,
    y_full_res,
    sample_weight=w_full,
    categorical_feature=categorical_columns,
)
prod_lgb_path = os.path.join(WORKING_DIR, "prod_lgb_model.txt")
prod_lgb_model.booster_.save_model(prod_lgb_path)

# Retrain full XGBoost model
prod_xgb_model = create_xgb_model()
prod_xgb_model.set_params(n_estimators=full_xgb_estimators)
prod_xgb_model.fit(
    X_full,
    y_full_res,
    sample_weight=w_full,
    verbose=False,
)
prod_xgb_path = os.path.join(WORKING_DIR, "prod_xgb_model.json")
prod_xgb_model.save_model(prod_xgb_path)

# Retrain full PyTorch Tabular Neural Network
print("Retraining production TabularNN on full modern dataset residuals...")
prod_nn_model = train_tabular_nn(
    X_full_cat,
    X_full_cont_norm,
    y_full_res,
    w_full,
    cat_cardinalities,
    len(cont_cols),
    device,
    epochs=5,
    batch_size=8192,
)
torch.save(prod_nn_model.state_dict(), os.path.join(WORKING_DIR, "prod_nn_model.pt"))

del X_full, y_full, y_full_res, w_full, X_full_cat, X_full_cont_norm, train_modern, base_proj_full
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Generating test predictions on 2017 holdout transactions...")
base_proj_test = test_df["base_proj"].values.astype(np.float32)
X_test = test_df[feature_columns].copy()
X_test[cont_cols] = X_test[cont_cols].fillna(0.0).astype(np.float32)
X_test[categorical_columns] = X_test[categorical_columns].fillna(0).astype(np.int32)

X_test_cont_norm = np.nan_to_num(
    ((X_test[cont_cols].values - full_cont_mean) / full_cont_std).astype(np.float32)
)
X_test_cat = X_test[categorical_columns].values.astype(np.int32)

test_pred_res_lgb = prod_lgb_model.predict(X_test)
test_pred_res_xgb = predict_xgb(prod_xgb_model, X_test, device)
test_pred_res_nn = predict_tabular_nn(prod_nn_model, X_test_cat, X_test_cont_norm, device)

blend_test_res = w_lgb * test_pred_res_lgb + w_xgb * test_pred_res_xgb + w_nn * test_pred_res_nn
test_pred_log10 = base_proj_test + blend_test_res
test_pred_price = np.clip(10.0 ** test_pred_log10, 1.0, None)

# ==============================================================================
# 8. SUBMISSION GENERATION & VERIFICATION
# ==============================================================================
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

submission_raw = pd.DataFrame(
    {"id": test_df["id"].values, "price": np.round(test_pred_price, 2)}
)

submission = sample_sub[["id"]].merge(submission_raw, on="id", how="left")

assert len(submission) == len(
    sample_sub
), f"Row count mismatch: {len(submission)} vs expected {len(sample_sub)}"
assert not submission["price"].isna().any(), "NaN values detected in final predictions!"
assert (submission["price"] >= 1.0).all(), "Predictions contain values < 1.0!"

final_sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(final_sub_path, index=False)
print(f"Saved submission with {len(submission)} rows to {final_sub_path}")

print(f"Final Validation Score: {final_val_score}")