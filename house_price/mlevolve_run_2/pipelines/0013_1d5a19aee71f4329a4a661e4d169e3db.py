import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

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
        "min_child_samples": 150,
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


def create_lgb_model():
    return lgb.LGBMRegressor(**get_lgb_model_params())


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
        "reg_lambda": 3.0,
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


def create_xgb_model(early_stopping_rounds=None):
    params = get_xgb_model_params()
    if early_stopping_rounds is not None:
        try:
            _ = xgb.XGBRegressor(n_estimators=1, early_stopping_rounds=early_stopping_rounds)
            params["early_stopping_rounds"] = early_stopping_rounds
        except TypeError:
            pass
    return xgb.XGBRegressor(**params)


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
    df["sale_prop"] = df["sale_category"] + "_" + df["property_type"]
    df["district_prop"] = df["district"] + "_" + df["property_type"]
    df["county_prop"] = df["county"] + "_" + df["property_type"]
    df["town_prop"] = df["town"] + "_" + df["property_type"]
    df["loc_triple"] = df["town"] + "||" + df["district"] + "||" + df["county"]

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
    val_to_id = {v: i for i, v in enumerate(unique_vals)}
    encoding_maps[col] = val_to_id
    train_df[f"{col}_code"] = train_df[col].map(val_to_id).fillna(-1).astype(np.int32)
    test_df[f"{col}_code"] = test_df[col].map(val_to_id).fillna(-1).astype(np.int32)

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
            "prop_tenure",
            "prop_new",
            "district_prop",
            "town_prop",
            "sale_category",
            "year",
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

    hist_3y["t_prior"] = hist_3y["town"].map(town_map_3y).fillna(hist_3y["d_prior"])
    tp_grp_3y = hist_3y.groupby("town_prop").agg(
        count=("log10_price", "count"),
        mean=("log10_price", "mean"),
        t_prior=("t_prior", "mean")
    ).reset_index()
    tp_grp_3y["te_tp_3y"] = (tp_grp_3y["count"] * tp_grp_3y["mean"] + 10.0 * tp_grp_3y["t_prior"]) / (tp_grp_3y["count"] + 10.0)
    tp_map_3y = dict(zip(tp_grp_3y["town_prop"], tp_grp_3y["te_tp_3y"]))

    pt_grp = hist_3y.groupby("prop_tenure")["log10_price"].agg(["count", "mean"]).reset_index()
    pt_map = dict(zip(pt_grp["prop_tenure"], (pt_grp["count"] * pt_grp["mean"] + 30.0 * global_mean_3y) / (pt_grp["count"] + 30.0)))

    pn_grp = hist_3y.groupby("prop_new")["log10_price"].agg(["count", "mean"]).reset_index()
    pn_map = dict(zip(pn_grp["prop_new"], (pn_grp["count"] * pn_grp["mean"] + 30.0 * global_mean_3y) / (pn_grp["count"] + 30.0)))

    sc_grp = hist_3y.groupby("sale_category")["log10_price"].agg(["count", "mean"]).reset_index()
    sc_map = dict(zip(sc_grp["sale_category"], (sc_grp["count"] * sc_grp["mean"] + 30.0 * global_mean_3y) / (sc_grp["count"] + 30.0)))

    # --- 2. 1-Year Responsive Velocity Window (t-1) ---
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
    te_dp = target_df["district_prop"].map(dp_map_3y).fillna(te_d).astype(np.float32)
    te_tp = target_df["town_prop"].map(tp_map_3y).fillna(te_t).astype(np.float32)

    # Target Mapping: 1-Year Valuations
    te_c_1y = target_df["county"].map(county_map_1y).fillna(te_c).astype(np.float32)
    te_d_1y = target_df["district"].map(dist_map_1y).fillna(te_d).astype(np.float32)
    te_t_1y = target_df["town"].map(town_map_1y).fillna(te_t).astype(np.float32)
    te_dp_1y = target_df["district_prop"].map(dp_map_1y).fillna(te_dp).astype(np.float32)

    te_pt = target_df["prop_tenure"].map(pt_map).fillna(global_mean_3y).astype(np.float32)
    te_pn = target_df["prop_new"].map(pn_map).fillna(global_mean_3y).astype(np.float32)
    te_sc = target_df["sale_category"].map(sc_map).fillna(global_mean_3y).astype(np.float32)
    te_disp_dp = target_df["district_prop"].map(dp_disp_map_3y).fillna(global_std_3y).astype(np.float32)

    dt_diff = np.clip((target_df["time_trend"] - float(ref_1y)).astype(np.float32), 0.0, 3.0)
    mom_c = target_df["county"].map(county_momentum_map).fillna(momentum_national).astype(np.float32)
    mom_d = target_df["district"].map(dist_momentum_map).fillna(mom_c).astype(np.float32)
    mom_dp = target_df["district_prop"].map(dp_momentum_map).fillna(mom_d).astype(np.float32)

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
            "hist_mean_national": np.float32(global_mean_3y),
            "hist_trend_national": np.float32(momentum_national),
            "momentum_county": mom_c.values,
            "momentum_district": mom_d.values,
            "momentum_district_prop": mom_dp.values,
            "te_county": te_c.values,
            "te_district": te_d.values,
            "te_town": te_t.values,
            "te_district_prop": te_dp.values,
            "te_town_prop": te_tp.values,
            "te_county_1y": te_c_1y.values,
            "te_district_1y": te_d_1y.values,
            "te_town_1y": te_t_1y.values,
            "te_district_prop_1y": te_dp_1y.values,
            "delta_1y_3y_district": (te_d_1y - te_d).values,
            "delta_1y_3y_district_prop": (te_dp_1y - te_dp).values,
            "liquidity_ratio_district": liq_d.values,
            "liquidity_ratio_district_prop": liq_dp.values,
            "liquidity_ratio_town": liq_t.values,
            "te_prop_tenure": te_pt.values,
            "te_prop_new": te_pn.values,
            "te_sale_category": te_sc.values,
            "dispersion_district_prop": te_disp_dp.values,
            "te_national_projected": (global_mean_1y + momentum_national * dt_diff).values,
            "te_county_projected": (te_c_1y + mom_c * dt_diff).values,
            "te_district_projected": (te_d_1y + mom_d * dt_diff).values,
            "te_town_projected": (te_t_1y + mom_d * dt_diff).values,
            "te_district_prop_projected": (te_dp_1y + mom_dp * dt_diff).values,
            "te_town_prop_projected": (te_tp + mom_dp * dt_diff).values,
            "rel_premium_district": (te_d - global_mean_3y).values,
            "rel_premium_town": (te_t - te_d).values,
            "rel_premium_district_prop": (te_dp - te_d).values,
            "rel_premium_town_prop": (te_tp - te_t).values,
            "rel_premium_prop_tenure": (te_pt - global_mean_3y).values,
            "rel_premium_prop_new": (te_pn - global_mean_3y).values,
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
    "sale_prop_code",
    "county_code",
    "district_code",
    "town_code",
    "district_prop_code",
    # Frequency / Volume Densities
    "freq_log_district",
    "freq_log_town",
    "freq_log_county",
    "freq_log_district_prop",
    "freq_log_town_prop",
    # Dual-Horizon Target Encodings & Momentum
    "hist_mean_national",
    "hist_trend_national",
    "momentum_county",
    "momentum_district",
    "momentum_district_prop",
    "te_county",
    "te_district",
    "te_town",
    "te_district_prop",
    "te_town_prop",
    "te_county_1y",
    "te_district_1y",
    "te_town_1y",
    "te_district_prop_1y",
    "delta_1y_3y_district",
    "delta_1y_3y_district_prop",
    "liquidity_ratio_district",
    "liquidity_ratio_district_prop",
    "liquidity_ratio_town",
    "te_prop_tenure",
    "te_prop_new",
    "te_sale_category",
    "dispersion_district_prop",
    "te_national_projected",
    "te_county_projected",
    "te_district_projected",
    "te_town_projected",
    "te_district_prop_projected",
    "te_town_prop_projected",
    # Relative Price Premiums
    "rel_premium_district",
    "rel_premium_town",
    "rel_premium_district_prop",
    "rel_premium_town_prop",
    "rel_premium_prop_tenure",
    "rel_premium_prop_new",
]

categorical_columns = [
    "property_type_code",
    "is_new_build_code",
    "tenure_code",
    "sale_category_code",
    "prop_tenure_code",
    "prop_new_code",
    "sale_prop_code",
    "county_code",
    "district_code",
    "town_code",
    "district_prop_code",
]

cont_cols = [c for c in feature_columns if c not in categorical_columns]

# Save metadata for reproducibility
meta_dict = {
    "feature_columns": feature_columns,
    "categorical_columns": categorical_columns,
    "target_column": "log10_price",
    "train_rows": len(train_modern),
    "test_rows": len(test_df),
    "num_features": len(feature_columns),
    "val_2016_rows": int(train_modern["is_val_2016"].sum()),
}
with open(os.path.join(WORKING_DIR, "feature_meta.json"), "w") as f:
    json.dump(meta_dict, f, indent=2)

# ==============================================================================
# 5. MODEL ARCHITECTURE VERIFICATION
# ==============================================================================
print("Verifying LightGBM and GPU-accelerated XGBoost model initializations...")
dummy_lgb = create_lgb_model()
dummy_xgb = create_xgb_model()
print(f"LightGBM params verified. XGBoost device: {dummy_xgb.get_params().get('device', 'cpu')}, tree_method: {dummy_xgb.get_params().get('tree_method')}")
del dummy_lgb, dummy_xgb
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 6. TRAINING & OUT-OF-TIME VALIDATION
# ==============================================================================
# Exponential recency sample weighting
train_modern["sample_weight"] = np.exp(
    0.20 * (train_modern["time_trend"] - 2011.0)
).astype(np.float32)

val_mask = train_modern["is_val_2016"].values
train_mask = (~val_mask) & (train_modern["is_clean_train"].values)

X_train = train_modern.loc[train_mask, feature_columns]
y_train_res = (
    train_modern.loc[train_mask, "log10_price"]
    - train_modern.loc[train_mask, "te_district_prop_projected"]
).values.astype(np.float32)
w_train = train_modern.loc[train_mask, "sample_weight"].values

X_val = train_modern.loc[val_mask, feature_columns]
y_val_res = (
    train_modern.loc[val_mask, "log10_price"]
    - train_modern.loc[val_mask, "te_district_prop_projected"]
).values.astype(np.float32)
val_base = train_modern.loc[val_mask, "te_district_prop_projected"].values
val_true_price = train_modern.loc[val_mask, "price"].values

print(f"Validation split: {len(X_train):,} train samples, {len(X_val):,} val samples.")

# 6.1 Train Validation LightGBM Model
print("Training Validation LightGBM Regressor...")
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

# 6.2 Train Validation GPU XGBoost Model
print("Training Validation GPU-accelerated XGBoost Regressor...")
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

# 6.3 Out-of-Time Prediction and Convex Blending Optimization
val_pred_res_lgb = val_lgb_model.predict(X_val)
val_pred_res_xgb = val_xgb_model.predict(X_val)

lgb_val_price = np.clip(10.0 ** (val_base + val_pred_res_lgb), 1.0, None)
xgb_val_price = np.clip(10.0 ** (val_base + val_pred_res_xgb), 1.0, None)
lgb_val_score = compute_raw_price_rmse(val_true_price, lgb_val_price)
xgb_val_score = compute_raw_price_rmse(val_true_price, xgb_val_price)
print(f"Standalone Validation LGBM Score: {lgb_val_score:.6f}")
print(f"Standalone Validation XGBoost Score: {xgb_val_score:.6f}")

best_w = 0.5
best_val_rmse = float("inf")
for w in np.linspace(0.0, 1.0, 101):
    blend_res = w * val_pred_res_lgb + (1.0 - w) * val_pred_res_xgb
    blend_price = np.clip(10.0 ** (val_base + blend_res), 1.0, None)
    score = compute_raw_price_rmse(val_true_price, blend_price)
    if score < best_val_rmse:
        best_val_rmse = score
        best_w = float(w)

final_val_score = best_val_rmse
print(f"Optimal Ensemble Blend: {best_w:.2f} LGBM + {1.0 - best_w:.2f} XGBoost")
print(f"Blended Holdout Validation Score: {final_val_score:.6f}")

# Clean validation resources before full retraining
del val_lgb_model, val_xgb_model
del X_train, y_train_res, w_train, X_val, y_val_res, val_base, val_true_price
del val_pred_res_lgb, val_pred_res_xgb, lgb_val_price, xgb_val_price
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 7. PRODUCTION MODEL RETRAINING & INFERENCE
# ==============================================================================
full_lgb_estimators = max(lgb_best_iteration, int(lgb_best_iteration * 1.05))
full_xgb_estimators = max(xgb_best_iteration, int(xgb_best_iteration * 1.05))

print(
    f"Retraining production ensemble on full modern history (2011-2016)... "
    f"(LGBM estimators: {full_lgb_estimators}, XGBoost estimators: {full_xgb_estimators})"
)

full_clean_mask = train_modern["is_clean_train"].values
X_full = train_modern.loc[full_clean_mask, feature_columns]
y_full_res = (
    train_modern.loc[full_clean_mask, "log10_price"]
    - train_modern.loc[full_clean_mask, "te_district_prop_projected"]
).values.astype(np.float32)
w_full = train_modern.loc[full_clean_mask, "sample_weight"].values

# Fit full LightGBM model
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

# Fit full XGBoost model
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

del X_full, y_full_res, w_full, train_modern
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Generating test predictions on 2017 holdout transactions...")
X_test = test_df[feature_columns]
if X_test.isna().any().any():
    X_test = X_test.fillna(0)

test_pred_res_lgb = prod_lgb_model.predict(X_test)
test_pred_res_xgb = prod_xgb_model.predict(X_test)
test_pred_res = best_w * test_pred_res_lgb + (1.0 - best_w) * test_pred_res_xgb

test_base = test_df["te_district_prop_projected"].values
test_pred_log10 = test_base + test_pred_res
test_pred_price = np.clip(10.0**test_pred_log10, 1.0, None)

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