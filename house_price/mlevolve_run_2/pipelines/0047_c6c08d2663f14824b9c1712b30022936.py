import gc
import json
import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb


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


def fit_spatio_temporal_trend(df: pd.DataFrame, m_shrink: float = 300.0) -> dict:
    """
    Fits a robust hierarchical linear trend mu(t, county) = alpha + beta * year_fraction
    at national and county levels with shrinkage toward the national slope.
    """
    t = df["year_fraction"].values.astype(np.float64)
    y = df["log10_price"].values.astype(np.float64)
    counties = df["county"].values

    t_mean_nat = np.mean(t)
    y_mean_nat = np.mean(y)
    t_dev = t - t_mean_nat
    var_nat = np.mean(t_dev**2)
    cov_nat = np.mean(t_dev * (y - y_mean_nat))
    beta_nat = cov_nat / (var_nat + 1e-9)
    alpha_nat = y_mean_nat - beta_nat * t_mean_nat

    county_agg = pd.DataFrame(
        {
            "county": counties,
            "t": t,
            "y": y,
            "ty": t * y,
            "t2": t * t,
        }
    )
    c_stats = county_agg.groupby("county").agg(
        n=("t", "count"),
        sum_t=("t", "sum"),
        sum_y=("y", "sum"),
        sum_ty=("ty", "sum"),
        sum_t2=("t2", "sum"),
    ).reset_index()

    c_stats["t_mean"] = c_stats["sum_t"] / c_stats["n"]
    c_stats["y_mean"] = c_stats["sum_y"] / c_stats["n"]
    c_stats["var_t"] = (c_stats["sum_t2"] / c_stats["n"]) - (c_stats["t_mean"] ** 2)
    c_stats["cov_ty"] = (c_stats["sum_ty"] / c_stats["n"]) - (c_stats["t_mean"] * c_stats["y_mean"])

    c_stats["beta_raw"] = np.where(
        c_stats["var_t"] > 1e-6,
        c_stats["cov_ty"] / (c_stats["var_t"] + 1e-9),
        beta_nat,
    )
    c_stats["beta_shrunk"] = (
        (c_stats["n"] * c_stats["beta_raw"] + m_shrink * beta_nat)
        / (c_stats["n"] + m_shrink)
    )
    c_stats["alpha"] = c_stats["y_mean"] - c_stats["beta_shrunk"] * c_stats["t_mean"]

    return {
        "alpha_nat": np.float32(alpha_nat),
        "beta_nat": np.float32(beta_nat),
        "county_alpha": dict(zip(c_stats["county"], c_stats["alpha"].astype(np.float32))),
        "county_beta": dict(zip(c_stats["county"], c_stats["beta_shrunk"].astype(np.float32))),
    }


def predict_trend(df: pd.DataFrame, trend_model: dict) -> np.ndarray:
    """
    Computes extrapolated trend mu(t, county) = alpha_county + beta_county * year_fraction.
    """
    alpha_nat = trend_model["alpha_nat"]
    beta_nat = trend_model["beta_nat"]
    c_alpha = trend_model["county_alpha"]
    c_beta = trend_model["county_beta"]

    t = df["year_fraction"].values.astype(np.float32)
    county_series = df["county"]

    alpha_vec = county_series.map(c_alpha).fillna(alpha_nat).values.astype(np.float32)
    beta_vec = county_series.map(c_beta).fillna(beta_nat).values.astype(np.float32)
    return alpha_vec + beta_vec * t


# Leakage-free Empirical Bayes Target Encodings on Stationary Residuals
def compute_hierarchical_features(fit_train_df, apply_dfs):
    stat_df = fit_train_df[fit_train_df["year"] >= MODERN_START_YEAR].copy()

    # Fit hierarchical spatio-temporal trend decomposition
    trend_model = fit_spatio_temporal_trend(stat_df)
    stat_df["trend"] = predict_trend(stat_df, trend_model)
    stat_df["y_res"] = (stat_df["log10_price"] - stat_df["trend"]).astype(np.float32)

    global_mean = stat_df["y_res"].mean()

    # 1. County level encoding on stationary residuals
    county_stats = stat_df.groupby("county")["y_res"].agg(["count", "mean"])
    m_county = 50.0
    county_stats["county_te"] = (
        (county_stats["count"] * county_stats["mean"] + m_county * global_mean)
        / (county_stats["count"] + m_county)
    ).astype(np.float32)
    county_map = county_stats["county_te"].to_dict()

    # 2. District level encoding (shrunk to county level)
    district_stats = (
        stat_df.groupby(["district", "county"])["y_res"]
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
        stat_df.groupby(["town_district", "district"])["y_res"]
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
        stat_df.groupby(["district_type", "district"])["y_res"]
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
    profile_stats = stat_df.groupby("hedonic_profile")["y_res"].agg(
        ["count", "mean"]
    )
    m_prof = 50.0
    profile_stats["profile_te"] = (
        (profile_stats["count"] * profile_stats["mean"] + m_prof * global_mean)
        / (profile_stats["count"] + m_prof)
    ).astype(np.float32)
    profile_map = profile_stats["profile_te"].to_dict()

    # 6. Spatial Residual Momentum
    max_year = int(stat_df["year"].max())
    recent_y1 = (
        stat_df[stat_df["year"] == max_year]
        .groupby("district")["y_res"]
        .mean()
    )
    recent_y2 = (
        stat_df[stat_df["year"] == (max_year - 1)]
        .groupby("district")["y_res"]
        .mean()
    )
    recent_y3 = (
        stat_df[stat_df["year"] == (max_year - 2)]
        .groupby("district")["y_res"]
        .mean()
    )

    growth_1y = (recent_y1 - recent_y2).to_dict()
    growth_2y = ((recent_y1 - recent_y3) / 2.0).to_dict()
    global_growth_1y = float(
        stat_df[stat_df["year"] == max_year]["y_res"].mean()
        - stat_df[stat_df["year"] == (max_year - 1)]["y_res"].mean()
    )
    global_growth_2y = float(
        (
            stat_df[stat_df["year"] == max_year]["y_res"].mean()
            - stat_df[stat_df["year"] == (max_year - 2)]["y_res"].mean()
        )
        / 2.0
    )

    results = []
    for df in apply_dfs:
        res = df.copy()
        res["trend"] = predict_trend(res, trend_model).astype(np.float32)
        if "log10_price" in res.columns:
            res["y_res"] = (res["log10_price"] - res["trend"]).astype(np.float32)

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


print("Configuring out-of-time splits...")
val_raw_train = train_df[train_df["year"] < 2016].copy()
val_raw_holdout = train_df[train_df["year"] == 2016].copy()
prod_raw_train = train_df[train_df["year"] <= 2016].copy()
prod_raw_test = test_df.copy()

print(f"Validation train split (pre-2016): {len(val_raw_train):,} rows")
print(f"Validation holdout split (2016): {len(val_raw_holdout):,} rows")
print(f"Production train split (pre-2017): {len(prod_raw_train):,} rows")
print(f"Production test split (2017): {len(prod_raw_test):,} rows")

# Compute hierarchical features for both regimes
val_train_feat, val_holdout_feat = compute_hierarchical_features(
    fit_train_df=val_raw_train,
    apply_dfs=[
        val_raw_train[val_raw_train["year"] >= MODERN_START_YEAR],
        val_raw_holdout,
    ],
)

prod_train_feat, prod_test_feat = compute_hierarchical_features(
    fit_train_df=prod_raw_train,
    apply_dfs=[
        prod_raw_train[prod_raw_train["year"] >= MODERN_START_YEAR],
        prod_raw_test,
    ],
)

# Export precomputed extrapolated trend series
val_trend = val_holdout_feat["trend"].values.astype(np.float32)
test_trend = prod_test_feat["trend"].values.astype(np.float32)

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

# --- 2. Model Design ---
def get_lgb_hedonic_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 95,
        "max_depth": 9,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "n_estimators": 2500,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }


def get_xgb_hedonic_params() -> dict:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 8,
        "colsample_bytree": 0.75,
        "subsample": 0.8,
        "reg_lambda": 3.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if torch.cuda.is_available():
        try:
            test_model = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
            test_model.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
            params["device"] = "cuda"
        except Exception:
            try:
                test_model = xgb.XGBRegressor(tree_method="gpu_hist", n_estimators=1)
                test_model.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
                params["tree_method"] = "gpu_hist"
            except Exception:
                pass
    return params


# Metric Definition: Exact Task-Faithful Log10 RMSE
def compute_log10_rmse(y_true_raw, y_pred_raw):
    clipped_pred = np.clip(y_pred_raw, 1.0, None)
    clipped_true = np.clip(y_true_raw, 1.0, None)
    log_pred = np.log10(clipped_pred)
    log_true = np.log10(clipped_true)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


# --- 3. Validation Regime Training & Evaluation ---
print("--- Training Dual Heterogeneous Tree Architecture on Validation Regime ---")
# Compute exponential recency sample weights (gamma = 0.15)
max_yf_val = val_train_feat["year_fraction"].max()
val_train_weights = np.exp(0.15 * (val_train_feat["year_fraction"].values - max_yf_val)).astype(np.float32)

val_holdout_y_raw = val_holdout_feat["price"].values

# 3A. Train LightGBM on Stationary Residuals
print("Training leaf-wise LightGBM on residuals...")
lgb_params = get_lgb_hedonic_params()

lgb_train_data = lgb.Dataset(
    val_train_feat[feature_cols],
    label=val_train_feat["y_res"],
    weight=val_train_weights,
    categorical_feature=cat_feature_cols,
)
lgb_val_data = lgb.Dataset(
    val_holdout_feat[feature_cols],
    label=val_holdout_feat["y_res"],
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
val_lgb_log10 = val_trend + val_lgb_res
val_lgb_price = np.clip(10.0**val_lgb_log10, 1.0, None)
val_lgb_rmse = compute_log10_rmse(val_holdout_y_raw, val_lgb_price)
print(f"Validation LightGBM RMSE: {val_lgb_rmse:.5f} (Best Iter: {lgb_model.best_iteration})")

# 3B. Train Depth-wise XGBoost on Stationary Residuals
print("Training depth-wise XGBoost on residuals...")
xgb_params = get_xgb_hedonic_params()
dxgb_train = xgb.DMatrix(
    val_train_feat[feature_cols],
    label=val_train_feat["y_res"],
    weight=val_train_weights,
)
dxgb_val = xgb.DMatrix(
    val_holdout_feat[feature_cols],
    label=val_holdout_feat["y_res"],
)

xgb_model = xgb.train(
    xgb_params,
    dxgb_train,
    num_boost_round=2500,
    evals=[(dxgb_val, "val")],
    early_stopping_rounds=40,
    verbose_eval=False,
)

val_xgb_res = xgb_model.predict(dxgb_val)
val_xgb_log10 = val_trend + val_xgb_res
val_xgb_price = np.clip(10.0**val_xgb_log10, 1.0, None)
val_xgb_rmse = compute_log10_rmse(val_holdout_y_raw, val_xgb_price)
print(f"Validation XGBoost RMSE: {val_xgb_rmse:.5f} (Best Iter: {xgb_model.best_iteration})")

# 3C. Holdout Ensemble Blend Weight Optimization
best_w = 0.5
best_ensemble_score = float("inf")
for w_cand in np.linspace(0.0, 1.0, 101):
    cand_res = w_cand * val_lgb_res + (1.0 - w_cand) * val_xgb_res
    cand_log10 = val_trend + cand_res
    cand_price = np.clip(10.0**cand_log10, 1.0, None)
    cand_score = compute_log10_rmse(val_holdout_y_raw, cand_price)
    if cand_score < best_ensemble_score:
        best_ensemble_score = cand_score
        best_w = float(w_cand)

print(
    f"Optimal Holdout Blend: {best_w:.2f} LGB + {1.0 - best_w:.2f} XGB | Holdout Validation Score: {best_ensemble_score:.5f}"
)

# Free validation datasets
del (
    val_train_feat,
    val_holdout_feat,
    lgb_train_data,
    lgb_val_data,
    dxgb_train,
    dxgb_val,
)
gc.collect()

# --- 4. Production Retraining on Complete Modern Regime (2012-2016) ---
print("--- Production Retraining on Full Modern Regime (2012-2016) ---")
max_yf_prod = prod_train_feat["year_fraction"].max()
prod_train_weights = np.exp(0.15 * (prod_train_feat["year_fraction"].values - max_yf_prod)).astype(np.float32)

# Retrain LightGBM
prod_lgb_params = lgb_params.copy()
prod_lgb_rounds = max(100, int(lgb_model.best_iteration * 1.15))
full_train_lgb = lgb.Dataset(
    prod_train_feat[feature_cols],
    label=prod_train_feat["y_res"],
    weight=prod_train_weights,
    categorical_feature=cat_feature_cols,
)
prod_lgb_model = lgb.train(
    prod_lgb_params,
    full_train_lgb,
    num_boost_round=prod_lgb_rounds,
    callbacks=[lgb.log_evaluation(period=0)],
)
test_lgb_res = prod_lgb_model.predict(prod_test_feat[feature_cols])

# Retrain XGBoost
prod_xgb_rounds = max(100, int(xgb_model.best_iteration * 1.15))
dprod_train = xgb.DMatrix(
    prod_train_feat[feature_cols],
    label=prod_train_feat["y_res"],
    weight=prod_train_weights,
)
dprod_test = xgb.DMatrix(prod_test_feat[feature_cols])
prod_xgb_model = xgb.train(
    xgb_params,
    dprod_train,
    num_boost_round=prod_xgb_rounds,
    verbose_eval=False,
)
test_xgb_res = prod_xgb_model.predict(dprod_test)

del dprod_train, dprod_test, full_train_lgb
gc.collect()

# --- 5. Ensembled Inference & Verified Submission Export ---
# Blend residuals and reconstruct raw prices via extrapolated test trend
test_blend_res = best_w * test_lgb_res + (1.0 - best_w) * test_xgb_res
test_final_log10 = test_trend + test_blend_res
test_final_price = np.clip(10.0**test_final_log10, 1.0, 100000000.0)
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
