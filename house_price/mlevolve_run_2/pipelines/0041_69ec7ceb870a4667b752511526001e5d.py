import gc
import json
import math
import os
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

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

# Filter nominal non-market transfer outliers (< £2,000 or > £100M)
valid_mask = (df_train["price"] >= 2000) & (df_train["price"] <= 100_000_000)
print(
    f"Filtered {len(df_train) - valid_mask.sum():,} anomalous records (< £2,000 or > £100M)."
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
print("Extracting inflation-normalized price anchors from full 22M history...")
df_train["trans_year"] = df_train["date"].dt.year
hist_district_counts = df_train["district"].value_counts().to_dict()
hist_town_counts = df_train["town"].value_counts().to_dict()

# 3-year period medians across 1995-2016 to normalize 4x macro price inflation into invariant differentials
anchor_smoothing = 50.0
trans_period = ((df_train["trans_year"] - 1995) // 3).clip(lower=0)
period_medians = df_train.groupby(trans_period)["log_price"].median().to_dict()
df_train["rel_log_price"] = (
    df_train["log_price"] - trans_period.map(period_medians)
).astype(np.float32)

dist_stats = df_train.groupby("district")["rel_log_price"].agg(["median", "count"])
hist_dist_anchor = (
    (dist_stats["count"] * dist_stats["median"])
    / (dist_stats["count"] + anchor_smoothing)
).to_dict()

town_stats = df_train.groupby("town")["rel_log_price"].agg(["median", "count"])
hist_town_anchor = (
    (town_stats["count"] * town_stats["median"])
    / (town_stats["count"] + anchor_smoothing)
).to_dict()

# Micro-location and property type interaction anchors from 22M corpus
dist_prop_stats = df_train.groupby(["district", "property_type"])["rel_log_price"].agg(["median", "count"])
dist_prop_anchor_val = (
    dist_prop_stats["count"] * dist_prop_stats["median"]
) / (dist_prop_stats["count"] + anchor_smoothing)
hist_dist_prop_anchor = {
    f"{d}_{p}": float(v) for (d, p), v in zip(dist_prop_stats.index, dist_prop_anchor_val)
}

town_prop_stats = df_train.groupby(["town", "property_type"])["rel_log_price"].agg(["median", "count"])
town_prop_anchor_val = (
    town_prop_stats["count"] * town_prop_stats["median"]
) / (town_prop_stats["count"] + anchor_smoothing)
hist_town_prop_anchor = {
    f"{t}_{p}": float(v) for (t, p), v in zip(town_prop_stats.index, town_prop_anchor_val)
}

# Disambiguated micro-location anchors from full 22M history
df_train["district_town"] = df_train["district"] + "_" + df_train["town"]
hist_dist_town_counts = df_train["district_town"].value_counts().to_dict()

dist_town_stats = df_train.groupby("district_town")["rel_log_price"].agg(["median", "count"])
dist_town_anchor_val = (
    dist_town_stats["count"] * dist_town_stats["median"]
) / (dist_town_stats["count"] + anchor_smoothing)
hist_dist_town_anchor = dict(zip(dist_town_stats.index, dist_town_anchor_val.astype(float)))

df_train["cat_dist_town_prop"] = df_train["district_town"] + "_" + df_train["property_type"]
dist_town_prop_stats = df_train.groupby("cat_dist_town_prop")["rel_log_price"].agg(["median", "count"])
dist_town_prop_anchor_val = (
    dist_town_prop_stats["count"] * dist_town_prop_stats["median"]
) / (dist_town_prop_stats["count"] + anchor_smoothing)
hist_dist_town_prop_anchor = dict(zip(dist_town_prop_stats.index, dist_town_prop_anchor_val.astype(float)))

df_train.drop(columns=["district_town", "cat_dist_town_prop"], inplace=True)
gc.collect()

# Calculate long-term appreciation slope (2012-2016 vs 2000-2005)
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

# Trailing 12-month versus 36-month median log-price momentum differentials at district and property levels
print("Computing multi-scale trailing price momentum indicators across historical windows...")
dist_momentum_dict = {}
prop_momentum_dict = {}
dist_prop_momentum_dict = {}

for target_yr in range(2012, 2018):
    sub_12m = df_train[df_train["trans_year"] == (target_yr - 1)]
    sub_36m = df_train[df_train["trans_year"].between(target_yr - 3, target_yr - 1)]

    # District level momentum: genuine log-price differentials
    d_med_12 = sub_12m.groupby("district")["log_price"].median()
    d_med_36 = sub_36m.groupby("district")["log_price"].median()
    d_ratio = (d_med_12 - d_med_36).clip(-0.25, 0.25)
    for dist_name, r_val in d_ratio.items():
        dist_momentum_dict[f"{dist_name}_{target_yr}"] = float(r_val)

    # Property level momentum: genuine log-price differentials
    p_med_12 = sub_12m.groupby("property_type")["log_price"].median()
    p_med_36 = sub_36m.groupby("property_type")["log_price"].median()
    p_ratio = (p_med_12 - p_med_36).clip(-0.25, 0.25)
    for prop_name, r_val in p_ratio.items():
        prop_momentum_dict[f"{prop_name}_{target_yr}"] = float(r_val)

    # District x Property level momentum: genuine log-price differentials
    dp_med_12 = sub_12m.groupby(["district", "property_type"])["log_price"].median()
    dp_med_36 = sub_36m.groupby(["district", "property_type"])["log_price"].median()
    dp_ratio = (dp_med_12 - dp_med_36).clip(-0.25, 0.25)
    for (d_name, p_name), r_val in dp_ratio.items():
        dist_prop_momentum_dict[f"{d_name}_{p_name}_{target_yr}"] = float(r_val)

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

    # Disambiguated micro-location key
    feats["district_town"] = df["district"] + "_" + df["town"]

    # Historical liquidity features and 22M corpus price anchors
    feats["hist_district_count"] = (
        df["district"].map(hist_district_counts).fillna(1).astype(np.float32)
    )
    feats["hist_town_count"] = (
        df["town"].map(hist_town_counts).fillna(1).astype(np.float32)
    )
    feats["hist_dist_town_count"] = (
        feats["district_town"].map(hist_dist_town_counts).fillna(1).astype(np.float32)
    )
    feats["hist_district_growth"] = (
        df["district"].map(district_growth_prior).fillna(0.0).astype(np.float32)
    )
    feats["hist_dist_anchor"] = (
        df["district"].map(hist_dist_anchor).fillna(0.0).astype(np.float32)
    )
    feats["hist_town_anchor"] = (
        df["town"].map(hist_town_anchor).fillna(0.0).astype(np.float32)
    )
    feats["hist_dist_town_anchor"] = (
        feats["district_town"]
        .map(hist_dist_town_anchor)
        .fillna(feats["hist_town_anchor"])
        .fillna(feats["hist_dist_anchor"])
        .fillna(0.0)
        .astype(np.float32)
    )

    # High-signal interaction combinations
    feats["cat_prop_tenure"] = df["property_type"] + "_" + df["tenure"]
    feats["cat_prop_new"] = df["property_type"] + "_" + df["is_new_build"]
    feats["cat_archetype"] = (
        df["property_type"] + "_" + df["is_new_build"] + "_" + df["tenure"]
    )
    feats["cat_dist_prop"] = df["district"] + "_" + df["property_type"]
    feats["cat_town_prop"] = df["town"] + "_" + df["property_type"]
    feats["cat_dist_town_prop"] = feats["district_town"] + "_" + df["property_type"]
    feats["cat_dist_tenure"] = df["district"] + "_" + df["tenure"]
    feats["cat_county_prop"] = df["county"] + "_" + df["property_type"]
    feats["cat_dist_new"] = df["district"] + "_" + df["is_new_build"]
    feats["cat_sale_prop"] = df["sale_category"] + "_" + df["property_type"]

    # Micro-location and hedonic interaction anchors from 22M corpus
    feats["hist_dist_prop_anchor"] = (
        feats["cat_dist_prop"]
        .map(hist_dist_prop_anchor)
        .fillna(feats["hist_dist_anchor"])
        .astype(np.float32)
    )
    feats["hist_dist_town_prop_anchor"] = (
        feats["cat_dist_town_prop"]
        .map(hist_dist_town_prop_anchor)
        .fillna(feats["hist_dist_prop_anchor"])
        .fillna(feats["hist_dist_town_anchor"])
        .fillna(0.0)
        .astype(np.float32)
    )
    feats["hist_town_prop_anchor"] = (
        feats["cat_town_prop"]
        .map(hist_town_prop_anchor)
        .fillna(feats["hist_dist_prop_anchor"])
        .fillna(feats["hist_town_anchor"])
        .fillna(0.0)
        .astype(np.float32)
    )

    # UK hedonic structural interactions & penalty flags
    feats["is_house_leasehold"] = (
        (df["property_type"].isin(["D", "S", "T"])) & (df["tenure"] == "L")
    ).astype(np.int32)
    feats["is_flat_freehold"] = (
        (df["property_type"] == "F") & (df["tenure"] == "F")
    ).astype(np.int32)
    feats["is_new_detached"] = (
        (df["property_type"] == "D") & (df["is_new_build"] == "Y")
    ).astype(np.int32)
    feats["is_new_flat"] = (
        (df["property_type"] == "F") & (df["is_new_build"] == "Y")
    ).astype(np.int32)
    feats["is_commercial_other"] = (
        (df["property_type"] == "O") | (df["sale_category"] == "B")
    ).astype(np.int32)

    # Multi-scale trailing 12-month versus 36-month median log-price momentum differentials
    yr_str = df["date"].dt.year.astype(str)
    dist_yr_key = df["district"] + "_" + yr_str
    prop_yr_key = df["property_type"] + "_" + yr_str
    dist_prop_yr_key = df["district"] + "_" + df["property_type"] + "_" + yr_str

    feats["momentum_dist_12_36"] = (
        dist_yr_key.map(dist_momentum_dict).fillna(0.0).astype(np.float32)
    )
    feats["momentum_prop_12_36"] = (
        prop_yr_key.map(prop_momentum_dict).fillna(0.0).astype(np.float32)
    )
    feats["momentum_dist_prop_12_36"] = (
        dist_prop_yr_key.map(dist_prop_momentum_dict)
        .fillna(feats["momentum_dist_12_36"])
        .fillna(0.0)
        .astype(np.float32)
    )

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

# Attach continuous chronological coordinates to base DataFrames for trend modeling
df_train_modern["time_elapsed"] = X_train_raw["time_elapsed"].values
df_test["time_elapsed"] = X_test_raw["time_elapsed"].values

y_train_full = df_train_modern["log_price"].values.astype(np.float32)
train_dates = df_train_modern["date"]

# -------------------------------------------------------------------------
# 6. Hierarchical Empirical Bayes Target Encoding with Recency Weighting
# -------------------------------------------------------------------------
print("Computing multi-resolution Hierarchical Empirical Bayes target encodings...")
target_enc_cols = [
    "district",
    "town",
    "district_town",
    "county",
    "county_std",
    "district_std",
    "cat_dist_prop",
    "cat_dist_town_prop",
    "cat_town_prop",
    "cat_dist_tenure",
    "cat_prop_tenure",
    "cat_prop_new",
    "cat_archetype",
    "cat_dist_new",
    "cat_sale_prop",
    "sale_category",
]


def compute_smoothed_means(
    train_df,
    group_col,
    target_col="target",
    weight_col="weight",
    prior_series=None,
    smoothing=25.0,
):
    """Multi-level empirical Bayes smoothing with recency weights and prior cascading."""
    w = (
        train_df[weight_col].values.astype(np.float64)
        if weight_col in train_df.columns
        else np.ones(len(train_df), dtype=np.float64)
    )
    y = train_df[target_col].values.astype(np.float64)
    global_mean = float(np.sum(w * y) / np.sum(w))

    if prior_series is None:
        calc_df = pd.DataFrame(
            {"grp": train_df[group_col].values, "wy": w * y, "w": w}
        )
        grouped = calc_df.groupby("grp", sort=False).sum()
        smoothed = (grouped["wy"] + smoothing * global_mean) / (
            grouped["w"] + smoothing
        )
    else:
        calc_df = pd.DataFrame(
            {
                "grp": train_df[group_col].values,
                "wy": w * y,
                "w": w,
                "wprior": w * prior_series.values.astype(np.float64),
            }
        )
        grouped = calc_df.groupby("grp", sort=False).sum()
        p_g = (grouped["wprior"] / grouped["w"]).fillna(global_mean)
        smoothed = (grouped["wy"] + smoothing * p_g) / (grouped["w"] + smoothing)

    return smoothed.to_dict(), global_mean


def compute_hierarchical_target_encodings(
    train_features_df, target_series, weight_series=None, smoothing=25.0
):
    """Computes nested hierarchical target encodings: county -> district -> district_town -> cat_dist_town_prop."""
    train_calc = train_features_df.copy()
    train_calc["target"] = target_series.values
    if weight_series is not None:
        train_calc["weight"] = weight_series.values
    else:
        train_calc["weight"] = 1.0

    # 1. County stats shrink to global mean
    county_map, global_mean = compute_smoothed_means(
        train_calc, "county", "target", "weight", prior_series=None, smoothing=smoothing
    )

    # 2. District stats shrink to enclosing county prior
    row_county_prior = train_calc["county"].map(county_map).fillna(global_mean)
    district_map, _ = compute_smoothed_means(
        train_calc,
        "district",
        "target",
        "weight",
        prior_series=row_county_prior,
        smoothing=smoothing,
    )
    row_dist_prior = train_calc["district"].map(district_map).fillna(row_county_prior)

    # 3. Disambiguated district_town stats shrink to enclosing district prior
    district_town_map, _ = compute_smoothed_means(
        train_calc,
        "district_town",
        "target",
        "weight",
        prior_series=row_dist_prior,
        smoothing=smoothing,
    )
    row_dist_town_prior = (
        train_calc["district_town"].map(district_town_map).fillna(row_dist_prior)
    )

    # 4. Town stats shrink to enclosing district prior
    town_map, _ = compute_smoothed_means(
        train_calc,
        "town",
        "target",
        "weight",
        prior_series=row_dist_prior,
        smoothing=smoothing,
    )

    # 5. cat_dist_prop shrinks to district prior
    cat_dist_prop_map, _ = compute_smoothed_means(
        train_calc,
        "cat_dist_prop",
        "target",
        "weight",
        prior_series=row_dist_prior,
        smoothing=smoothing,
    )
    row_cdp_prior = (
        train_calc["cat_dist_prop"].map(cat_dist_prop_map).fillna(row_dist_prior)
    )

    # 6. cat_dist_town_prop shrinks to cat_dist_prop prior (cascading district property effect into micro-location)
    cat_dist_town_prop_map, _ = compute_smoothed_means(
        train_calc,
        "cat_dist_town_prop",
        "target",
        "weight",
        prior_series=row_cdp_prior,
        smoothing=smoothing,
    )

    # 7. cat_town_prop shrinks to cat_dist_prop prior
    cat_town_prop_map, _ = compute_smoothed_means(
        train_calc,
        "cat_town_prop",
        "target",
        "weight",
        prior_series=row_cdp_prior,
        smoothing=smoothing,
    )

    # 8. cat_dist_tenure shrinks to district prior
    cat_dist_tenure_map, _ = compute_smoothed_means(
        train_calc,
        "cat_dist_tenure",
        "target",
        "weight",
        prior_series=row_dist_prior,
        smoothing=smoothing,
    )

    # 9. cat_dist_new shrinks to district prior
    cat_dist_new_map, _ = compute_smoothed_means(
        train_calc,
        "cat_dist_new",
        "target",
        "weight",
        prior_series=row_dist_prior,
        smoothing=smoothing,
    )

    # 10. Other categorical combinations shrink to global mean
    other_maps = {}
    for col in ["cat_prop_tenure", "cat_prop_new", "cat_archetype", "cat_sale_prop", "sale_category"]:
        col_map, _ = compute_smoothed_means(
            train_calc,
            col,
            "target",
            "weight",
            prior_series=None,
            smoothing=smoothing,
        )
        other_maps[col] = col_map

    # Empirical Bayes target dispersion (residual std per district and county)
    global_std = float(np.std(train_calc["target"].values.astype(np.float64)))
    county_std = train_calc.groupby("county", sort=False)["target"].std().fillna(global_std)
    county_cnt = train_calc.groupby("county", sort=False)["target"].count()
    county_disp_map = (
        (county_cnt * county_std + smoothing * global_std) / (county_cnt + smoothing)
    ).to_dict()

    dist_to_county = train_calc.groupby("district", sort=False)["county"].first().to_dict()
    district_std = train_calc.groupby("district", sort=False)["target"].std()
    district_cnt = train_calc.groupby("district", sort=False)["target"].count()
    dist_priors = pd.Series(
        [county_disp_map.get(dist_to_county.get(d, ""), global_std) for d in district_std.index],
        index=district_std.index,
    )
    district_std_clean = district_std.fillna(dist_priors)
    district_disp_map = (
        (district_cnt * district_std_clean + smoothing * dist_priors) / (district_cnt + smoothing)
    ).to_dict()

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "county": county_map,
        "district": district_map,
        "district_town": district_town_map,
        "town": town_map,
        "cat_dist_prop": cat_dist_prop_map,
        "cat_dist_town_prop": cat_dist_town_prop_map,
        "cat_town_prop": cat_town_prop_map,
        "cat_dist_tenure": cat_dist_tenure_map,
        "cat_dist_new": cat_dist_new_map,
        "county_disp": county_disp_map,
        "district_disp": district_disp_map,
        **other_maps,
    }


def apply_hierarchical_encodings(df, enc_dict):
    """Applies hierarchical encodings with fallback cascading."""
    gm = enc_dict["global_mean"]
    gm_std = enc_dict.get("global_std", 0.25)
    te_results = {}

    c_enc = pd.to_numeric(df["county"].map(enc_dict["county"]), errors="coerce").fillna(gm)
    te_results["te_county"] = c_enc.values.astype(np.float32)

    c_disp = pd.to_numeric(df["county"].map(enc_dict.get("county_disp", {})), errors="coerce").fillna(gm_std)
    te_results["te_county_std"] = c_disp.values.astype(np.float32)

    d_enc = pd.to_numeric(df["district"].map(enc_dict["district"]), errors="coerce").fillna(c_enc)
    te_results["te_district"] = d_enc.values.astype(np.float32)

    d_disp = pd.to_numeric(df["district"].map(enc_dict.get("district_disp", {})), errors="coerce").fillna(c_disp)
    te_results["te_district_std"] = d_disp.values.astype(np.float32)

    dt_enc = pd.to_numeric(df["district_town"].map(enc_dict.get("district_town", {})), errors="coerce").fillna(d_enc)
    te_results["te_district_town"] = dt_enc.values.astype(np.float32)

    t_enc = pd.to_numeric(df["town"].map(enc_dict["town"]), errors="coerce").fillna(d_enc)
    te_results["te_town"] = t_enc.values.astype(np.float32)

    cdp_enc = pd.to_numeric(df["cat_dist_prop"].map(enc_dict["cat_dist_prop"]), errors="coerce").fillna(d_enc)
    te_results["te_cat_dist_prop"] = cdp_enc.values.astype(np.float32)

    cdtp_enc = (
        pd.to_numeric(df["cat_dist_town_prop"].map(enc_dict.get("cat_dist_town_prop", {})), errors="coerce")
        .fillna(cdp_enc)
        .fillna(dt_enc)
    )
    te_results["te_cat_dist_town_prop"] = cdtp_enc.values.astype(np.float32)

    ctp_enc = (
        pd.to_numeric(df["cat_town_prop"].map(enc_dict["cat_town_prop"]), errors="coerce")
        .fillna(cdp_enc)
        .fillna(t_enc)
    )
    te_results["te_cat_town_prop"] = ctp_enc.values.astype(np.float32)

    cdt_enc = pd.to_numeric(df["cat_dist_tenure"].map(enc_dict.get("cat_dist_tenure", {})), errors="coerce").fillna(d_enc)
    te_results["te_cat_dist_tenure"] = cdt_enc.values.astype(np.float32)

    cdn_enc = pd.to_numeric(df["cat_dist_new"].map(enc_dict["cat_dist_new"]), errors="coerce").fillna(d_enc)
    te_results["te_cat_dist_new"] = cdn_enc.values.astype(np.float32)

    csp_enc = pd.to_numeric(df["cat_sale_prop"].map(enc_dict["cat_sale_prop"]), errors="coerce").fillna(gm)
    te_results["te_cat_sale_prop"] = csp_enc.values.astype(np.float32)

    for col in ["cat_prop_tenure", "cat_prop_new", "cat_archetype", "sale_category"]:
        res = pd.to_numeric(df[col].map(enc_dict[col]), errors="coerce").fillna(gm)
        te_results[f"te_{col}"] = res.values.astype(np.float32)

    return te_results


# -------------------------------------------------------------------------
# Macro Regional Trend Model (Tri-Level Hierarchical Bayesian Trend Model)
# -------------------------------------------------------------------------
class TriLevelHierarchicalTrendModel:
    """Tri-Level Hierarchical Bayesian Trend Extrapolator:
    Jointly estimates baseline intercepts and annualized appreciation slopes across
    National -> County -> District and Property Type tiers with empirical Bayes slope shrinkage
    and non-negativity clipping (slope >= 0.0).
    Centering on t_bar ensures orthogonality between intercept and slope.
    Omits localized intercepts to preserve cross-sectional spatial price variance in residuals.
    """

    def __init__(
        self,
        lambda_slope_county=200.0,
        lambda_slope_district=250.0,
        lambda_slope_prop=500.0,
        **kwargs,
    ):
        self.lambda_slope_county = float(lambda_slope_county)
        self.lambda_slope_district = float(lambda_slope_district)
        self.lambda_slope_prop = float(lambda_slope_prop)
        self.t_bar = 0.0
        self.global_intercept = 5.25
        self.global_slope = 0.03
        self.county_deltas = {}
        self.district_deltas = {}
        self.prop_deltas = {}

    def fit(self, df, y, weights=None):
        t = df["time_elapsed"].values.astype(np.float64)
        y = np.asarray(y, dtype=np.float64)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
        else:
            w = np.ones(len(df), dtype=np.float64)

        # Centered time coordinates (t - t_bar)
        w_sum = np.sum(w)
        self.t_bar = float(np.sum(w * t) / w_sum)
        t_c = t - self.t_bar

        # 1. National baseline intercept and slope via weighted least squares
        wt_sum = np.sum(w * t_c)
        wy_sum = np.sum(w * y)
        wtt_sum = np.sum(w * t_c * t_c)
        wty_sum = np.sum(w * t_c * y)

        ss_t = np.maximum(1e-6, wtt_sum - (wt_sum**2) / w_sum)
        sp_ty = wty_sum - (wt_sum * wy_sum) / w_sum

        raw_nat_slope = float(sp_ty / ss_t)
        self.global_slope = float(max(0.0, raw_nat_slope))
        self.global_intercept = float((wy_sum - self.global_slope * wt_sum) / w_sum)

        county_vals = df["county"].values.astype(str)
        district_vals = df["district"].values.astype(str)
        prop_vals = df["property_type"].values.astype(str)

        calc_df = pd.DataFrame(
            {
                "county": county_vals,
                "district": district_vals,
                "property_type": prop_vals,
                "w": w,
                "wt": w * t_c,
                "wy": w * y,
                "wtt": w * t_c * t_c,
                "wty": w * t_c * y,
            }
        )

        # 2. County-level appreciation slope deviations shrunk to national slope
        grp_c = calc_df.groupby("county", sort=False)[["w", "wt", "wy", "wtt", "wty"]].sum()
        w_c = grp_c["w"].values
        wt_c = grp_c["wt"].values
        wy_c = grp_c["wy"].values
        wtt_c = grp_c["wtt"].values
        wty_c = grp_c["wty"].values

        ss_t_c = np.maximum(0.0, wtt_c - (wt_c**2) / w_c)
        sp_ty_c = wty_c - (wt_c * wy_c) / w_c
        c_slopes = (sp_ty_c + self.lambda_slope_county * self.global_slope) / (
            ss_t_c + self.lambda_slope_county
        )
        c_deltas = c_slopes - self.global_slope
        self.county_deltas = dict(zip(grp_c.index, c_deltas.astype(float)))

        # 3. District-level slope deviations shrunk toward parent county slope
        dist_to_county = calc_df.groupby("district", sort=False)["county"].first().to_dict()
        grp_d = calc_df.groupby("district", sort=False)[["w", "wt", "wy", "wtt", "wty"]].sum()
        w_d = grp_d["w"].values
        wt_d = grp_d["wt"].values
        wy_d = grp_d["wy"].values
        wtt_d = grp_d["wtt"].values
        wty_d = grp_d["wty"].values

        ss_t_d = np.maximum(0.0, wtt_d - (wt_d**2) / w_d)
        sp_ty_d = wty_d - (wt_d * wy_d) / w_d

        beta_c = np.array(
            [
                self.global_slope + self.county_deltas.get(dist_to_county.get(d, ""), 0.0)
                for d in grp_d.index
            ],
            dtype=np.float64,
        )
        beta_d = (sp_ty_d + self.lambda_slope_district * beta_c) / (
            ss_t_d + self.lambda_slope_district
        )
        d_deltas = beta_d - beta_c
        self.district_deltas = dict(zip(grp_d.index, d_deltas.astype(float)))

        # 4. Property-type appreciation slope deviations shrunk to national slope
        grp_p = calc_df.groupby("property_type", sort=False)[["w", "wt", "wy", "wtt", "wty"]].sum()
        w_p = grp_p["w"].values
        wt_p = grp_p["wt"].values
        wy_p = grp_p["wy"].values
        wtt_p = grp_p["wtt"].values
        wty_p = grp_p["wty"].values

        ss_t_p = np.maximum(0.0, wtt_p - (wt_p**2) / w_p)
        sp_ty_p = wty_p - (wt_p * wy_p) / w_p
        p_slopes = (sp_ty_p + self.lambda_slope_prop * self.global_slope) / (
            ss_t_p + self.lambda_slope_prop
        )
        p_deltas = p_slopes - self.global_slope
        self.prop_deltas = dict(zip(grp_p.index, p_deltas.astype(float)))

        del calc_df, grp_c, grp_d, grp_p
        gc.collect()
        return self

    def predict(self, df):
        t = df["time_elapsed"].values.astype(np.float64)
        t_c = t - self.t_bar

        c_series = df["county"].astype(str)
        c_deltas = (
            pd.to_numeric(c_series.map(self.county_deltas), errors="coerce")
            .fillna(0.0)
            .values.astype(np.float64)
        )

        d_series = df["district"].astype(str)
        d_deltas = (
            pd.to_numeric(d_series.map(self.district_deltas), errors="coerce")
            .fillna(0.0)
            .values.astype(np.float64)
        )

        p_series = df["property_type"].astype(str)
        p_deltas = (
            pd.to_numeric(p_series.map(self.prop_deltas), errors="coerce")
            .fillna(0.0)
            .values.astype(np.float64)
        )

        slopes = np.maximum(0.0, self.global_slope + c_deltas + d_deltas + p_deltas)
        return (self.global_intercept + slopes * t_c).astype(np.float32)


MacroRegionalTrendModel = TriLevelHierarchicalTrendModel
HierarchicalTrendModel = TriLevelHierarchicalTrendModel


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
    "district_town",
    "cat_prop_tenure",
    "cat_prop_new",
    "cat_archetype",
    "cat_dist_prop",
    "cat_town_prop",
    "cat_dist_town_prop",
    "cat_dist_tenure",
    "cat_county_prop",
    "cat_dist_new",
    "cat_sale_prop",
]

# Continuous frequency encodings for all categorical combinations
for col in cat_features:
    freq_map = X_train_raw[col].value_counts().to_dict()
    X_train_raw[f"freq_{col}"] = (
        X_train_raw[col].map(freq_map).fillna(0).astype(np.float32)
    )
    X_test_raw[f"freq_{col}"] = (
        X_test_raw[col].map(freq_map).fillna(0).astype(np.float32)
    )

# Restrict categorical feature indexing strictly to low-cardinality structural variables
low_card_cat_features = [
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
    "county",
]

for col in low_card_cat_features:
    categories = {
        cat: idx + 1 for idx, cat in enumerate(X_train_raw[col].astype(str).unique())
    }
    X_train_raw[f"{col}_code"] = (
        X_train_raw[col].astype(str).map(categories).fillna(0).astype(np.int32)
    )
    X_test_raw[f"{col}_code"] = (
        X_test_raw[col].astype(str).map(categories).fillna(0).astype(np.int32)
    )

drop_cols = cat_features + ["year", "time_elapsed"]
X_train_base = X_train_raw.drop(columns=drop_cols)
X_test_base = X_test_raw.drop(columns=drop_cols)

# -------------------------------------------------------------------------
# 8. Define Out-of-Time Train/Validation Splits & Strictly Isolated Encodings
# -------------------------------------------------------------------------
val_mask = (train_dates.dt.year == 2016).values
train_mask = (train_dates.dt.year < 2016).values

print(
    f"Split sizes: Train (2012-2015) = {train_mask.sum():,} rows | "
    f"Validation (2016) = {val_mask.sum():,} rows | "
    f"Test (2017 H1) = {len(X_test_base):,} rows."
)

# Linear temporal recency weighting across 2012-2015 train regime (bounded [0.35, 1.0])
df_train_sub = df_train_modern.loc[train_mask]
t_sub = df_train_sub["time_elapsed"].values
t_min_sub = float(t_sub.min())
t_max_sub = float(t_sub.max())
w_val_train = (
    0.35 + 0.65 * (t_sub - t_min_sub) / np.maximum(1e-5, t_max_sub - t_min_sub)
).astype(np.float32)

# Symmetrical tail outlier down-weighting (0.20x) for lower-tail (< £15k), luxury outliers (> £5M), atypical ('O'), and Category B
downweight_sub = (
    (df_train_sub["price"] < 15_000)
    | (df_train_sub["price"] > 5_000_000)
    | (df_train_sub["property_type"] == "O")
    | (df_train_sub["sale_category"] == "B")
).values
w_val_train = np.where(downweight_sub, w_val_train * 0.20, w_val_train).astype(np.float32)

print("Fitting validation Macro Regional Trend Model strictly on 2012-2015 data with recency weights...")
val_trend_model = MacroRegionalTrendModel(
    lambda_slope_county=200.0,
    lambda_slope_district=250.0,
)
val_trend_model.fit(df_train_sub, y_train_full[train_mask], weights=w_val_train)

T_hat_val_train = val_trend_model.predict(df_train_sub)
T_hat_val_holdout = val_trend_model.predict(df_train_modern.loc[val_mask])

y_detrended_val_train = (y_train_full[train_mask] - T_hat_val_train).astype(np.float32)
y_detrended_val_holdout = (y_train_full[val_mask] - T_hat_val_holdout).astype(np.float32)

print(
    f"Validation macro trend: intercept={val_trend_model.global_intercept:.4f}, "
    f"slope={val_trend_model.global_slope:.4f}. "
    f"Detrended target std: {np.std(y_detrended_val_train):.4f}"
)

print(
    "Computing strictly leak-free 5-fold CV hierarchical target encodings on 2012-2015 detrended targets..."
)
pre2016_raw = X_train_raw.loc[train_mask].reset_index(drop=True)
pre2016_y_detrended = pd.Series(y_detrended_val_train)
pre2016_weights = pd.Series(w_val_train)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_val_train = {
    f"te_{col}": np.zeros(len(pre2016_raw), dtype=np.float32)
    for col in target_enc_cols
}

for fold, (trn_idx, hold_idx) in enumerate(kf.split(pre2016_raw)):
    fold_trn_feats = pre2016_raw.iloc[trn_idx]
    fold_trn_target = pre2016_y_detrended.iloc[trn_idx]
    fold_trn_weights = pre2016_weights.iloc[trn_idx]
    fold_hold_feats = pre2016_raw.iloc[hold_idx]

    fold_enc = compute_hierarchical_target_encodings(
        fold_trn_feats,
        fold_trn_target,
        weight_series=fold_trn_weights,
        smoothing=25.0,
    )
    fold_hold_te = apply_hierarchical_encodings(fold_hold_feats, fold_enc)

    for col in target_enc_cols:
        oof_val_train[f"te_{col}"][hold_idx] = fold_hold_te[f"te_{col}"]

# Map full pre-2016 target encodings forward to 2016 holdout
pre2016_full_enc = compute_hierarchical_target_encodings(
    pre2016_raw, pre2016_y_detrended, weight_series=pre2016_weights, smoothing=25.0
)
holdout_te = apply_hierarchical_encodings(
    X_train_raw.loc[val_mask], pre2016_full_enc
)

train_split_X = X_train_base.loc[train_mask].copy().reset_index(drop=True)
for col in target_enc_cols:
    train_split_X[f"te_{col}"] = oof_val_train[f"te_{col}"]

val_split_X = X_train_base.loc[val_mask].copy().reset_index(drop=True)
for col in target_enc_cols:
    val_split_X[f"te_{col}"] = holdout_te[f"te_{col}"]

assert not train_split_X.isnull().any().any(), "NaN found in train_split_X"
assert not val_split_X.isnull().any().any(), "NaN found in val_split_X"

y_val_split = y_train_full[val_mask]


def compute_official_rmse(pred_log, true_log):
    log_pred = np.maximum(0.0, np.asarray(pred_log, dtype=np.float64))
    log_true = np.maximum(0.0, np.asarray(true_log, dtype=np.float64))
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


# -------------------------------------------------------------------------
# 9. Dual-Engine Spatio-Temporal Hybrid Architecture
# -------------------------------------------------------------------------
class DualEngineSpatioTemporalHybridModel:
    """Dual-Engine Spatio-Temporal Hybrid Architecture:
    Macro Regional Trend Extrapolator +
    Convex Blend of Leaf-wise LightGBM & Depth-wise GPU XGBoost Regressors.
    """

    def __init__(self, trend_model, lgb_model, xgb_model, w_lgb=0.5, w_xgb=0.5):
        self.trend_model = trend_model
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.w_lgb = float(w_lgb)
        self.w_xgb = float(w_xgb)

    def predict(self, df_spatial, X_features, num_iteration=None):
        """Unified prediction interface outputting 1D continuous log10 predictions."""
        t_hat = self.trend_model.predict(df_spatial)
        if num_iteration is not None:
            lgb_res = self.lgb_model.predict(
                X_features, num_iteration=num_iteration
            )
        else:
            lgb_res = self.lgb_model.predict(X_features)
        xgb_res = self.xgb_model.predict(X_features)
        y_res_hat = self.w_lgb * lgb_res + self.w_xgb * xgb_res
        return (t_hat + y_res_hat).astype(np.float32)


DualSpatioTemporalHybridModel = DualEngineSpatioTemporalHybridModel


def build_lgb_model(
    num_leaves=127,
    max_depth=10,
    min_child_samples=80,
    feature_fraction=0.70,
    learning_rate=0.035,
):
    """Builds LightGBM hyperparameter configuration dictionary with expanded tree capacity."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "feature_fraction": feature_fraction,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "min_child_samples": min_child_samples,
        "cat_l2": 20.0,
        "cat_smooth": 20.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": 16,
        "verbosity": -1,
        "random_state": 42,
    }


def optimize_blend_weights(pred_lgb, pred_xgb, t_hat, true_y):
    """Solves for non-negative convex ensemble weights (w_lgb, w_xgb) minimizing official RMSE."""
    def objective(w):
        blend = w[0] * pred_lgb + w[1] * pred_xgb
        pred_log = t_hat + blend
        return compute_official_rmse(pred_log, true_y)

    init_weights = [0.5, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    constraints = {"type": "eq", "fun": lambda w: w[0] + w[1] - 1.0}

    res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    w_lgb = float(np.clip(res.x[0], 0.0, 1.0))
    w_xgb = float(np.clip(1.0 - w_lgb, 0.0, 1.0))
    return w_lgb, w_xgb


lgb_params = build_lgb_model(
    num_leaves=127,
    max_depth=10,
    min_child_samples=80,
    feature_fraction=0.70,
    learning_rate=0.035,
)

# -------------------------------------------------------------------------
# 10. Training Regularized Hedonic Model on 2012-2015 & Holdout Validation
# -------------------------------------------------------------------------
cat_code_cols = [
    "property_type_code",
    "is_new_build_code",
    "tenure_code",
    "sale_category_code",
    "county_code",
]

trn_data = lgb.Dataset(
    train_split_X,
    label=y_detrended_val_train,
    weight=w_val_train,
    categorical_feature=cat_code_cols,
    free_raw_data=False,
)
val_data = lgb.Dataset(
    val_split_X,
    label=y_detrended_val_holdout,
    weight=None,
    categorical_feature=cat_code_cols,
    reference=trn_data,
    free_raw_data=False,
)


def eval_official_rmse(preds, dataset):
    if len(preds) != len(y_val_split):
        true_log = dataset.get_label()
        rmse = float(np.sqrt(np.mean((preds - true_log) ** 2)))
        return "official_rmse", rmse, False
    val_preds_log = T_hat_val_holdout + preds
    val_rmse = compute_official_rmse(val_preds_log, y_val_split)
    return "official_rmse", val_rmse, False


print(
    "Training regularized LightGBM hedonic model on 2012-2015 detrended targets with 50-round early stopping on 2016 holdout..."
)
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=50),
]

lgb_val_model = lgb.train(
    lgb_params,
    trn_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    valid_names=["val"],
    feval=eval_official_rmse,
    callbacks=callbacks,
)

best_iter_lgb = lgb_val_model.best_iteration
if best_iter_lgb is None or best_iter_lgb <= 0:
    best_iter_lgb = 1000
best_iteration_lgb = max(50, int(best_iter_lgb))
print(f"Optimal LightGBM iteration count determined: {best_iteration_lgb}")
lgb_val_res_preds = lgb_val_model.predict(val_split_X, num_iteration=best_iteration_lgb)

print(
    "Training depth-wise GPU XGBoost regressor on 2012-2015 detrended targets (max_depth=9, learning_rate=0.035)..."
)
xgb_device_kwargs = {"tree_method": "hist", "device": "cuda"}
try:
    xgb_val_model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=9,
        learning_rate=0.035,
        colsample_bytree=0.70,
        subsample=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=8,
        **xgb_device_kwargs,
    )
    xgb_val_model.fit(
        train_split_X,
        y_detrended_val_train,
        sample_weight=w_val_train,
        eval_set=[(val_split_X, y_detrended_val_holdout)],
        verbose=100,
    )
except Exception as e:
    print(f"XGBoost device='cuda' fit failed ({e}), attempting tree_method='gpu_hist'...")
    try:
        xgb_device_kwargs = {"tree_method": "gpu_hist"}
        xgb_val_model = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=9,
            learning_rate=0.035,
            colsample_bytree=0.70,
            subsample=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=42,
            n_jobs=8,
            **xgb_device_kwargs,
        )
        xgb_val_model.fit(
            train_split_X,
            y_detrended_val_train,
            sample_weight=w_val_train,
            eval_set=[(val_split_X, y_detrended_val_holdout)],
            verbose=100,
        )
    except Exception as e2:
        print(f"XGBoost gpu_hist failed ({e2}), falling back to CPU hist...")
        xgb_device_kwargs = {"tree_method": "hist"}
        xgb_val_model = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=9,
            learning_rate=0.035,
            colsample_bytree=0.70,
            subsample=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=42,
            n_jobs=8,
            **xgb_device_kwargs,
        )
        xgb_val_model.fit(
            train_split_X,
            y_detrended_val_train,
            sample_weight=w_val_train,
            eval_set=[(val_split_X, y_detrended_val_holdout)],
            verbose=100,
        )

best_iter_xgb = xgb_val_model.best_iteration
if best_iter_xgb is None or best_iter_xgb <= 0:
    best_iter_xgb = 500
best_iteration_xgb = max(50, int(best_iter_xgb) + 1)
print(f"Optimal XGBoost iteration count determined: {best_iteration_xgb}")
xgb_val_res_preds = xgb_val_model.predict(val_split_X)

print("Solving for optimal convex ensemble blend weights (w_lgb, w_xgb) via SLSQP...")
w_lgb_opt, w_xgb_opt = optimize_blend_weights(
    lgb_val_res_preds, xgb_val_res_preds, T_hat_val_holdout, y_val_split
)
print(f"Optimal Ensemble Weights: w_lgb = {w_lgb_opt:.4f}, w_xgb = {w_xgb_opt:.4f}")

hybrid_val_model = DualEngineSpatioTemporalHybridModel(
    val_trend_model, lgb_val_model, xgb_val_model, w_lgb=w_lgb_opt, w_xgb=w_xgb_opt
)
val_preds = hybrid_val_model.predict(
    df_train_modern.loc[val_mask], val_split_X, num_iteration=best_iteration_lgb
)
score = compute_official_rmse(val_preds, y_val_split)
print(f"Holdout Validation Score (2016): {score:.6f}")

del trn_data, val_data, train_split_X, val_split_X, lgb_val_model, xgb_val_model
gc.collect()

# -------------------------------------------------------------------------
# 11. Full Modern Regime Retraining (2012-2016)
# -------------------------------------------------------------------------
t_full = df_train_modern["time_elapsed"].values
t_min_full = float(t_full.min())
t_max_full = float(t_full.max())
w_full = (
    0.35 + 0.65 * (t_full - t_min_full) / np.maximum(1e-5, t_max_full - t_min_full)
).astype(np.float32)

# Symmetrical tail outlier down-weighting (0.20x) for lower-tail (< £15k), luxury outliers (> £5M), atypical ('O'), and Category B
downweight_full = (
    (df_train_modern["price"] < 15_000)
    | (df_train_modern["price"] > 5_000_000)
    | (df_train_modern["property_type"] == "O")
    | (df_train_modern["sale_category"] == "B")
).values
w_full = np.where(downweight_full, w_full * 0.20, w_full).astype(np.float32)

print("Fitting full Macro Regional Trend Model on modern transactions (2012-2016) with recency weights...")
full_trend_model = MacroRegionalTrendModel(
    lambda_slope_county=200.0,
    lambda_slope_district=250.0,
)
full_trend_model.fit(df_train_modern, y_train_full, weights=w_full)

T_hat_full = full_trend_model.predict(df_train_modern)
y_detrended_full = (y_train_full - T_hat_full).astype(np.float32)

print(
    f"Full macro trend: intercept={full_trend_model.global_intercept:.4f}, "
    f"slope={full_trend_model.global_slope:.4f}. "
    f"Detrended target std: {np.std(y_detrended_full):.4f}"
)

print(
    "Computing 5-fold OOF hierarchical target encodings on full 2012-2016 detrended targets..."
)
oof_full_train = {
    f"te_{col}": np.zeros(len(df_train_modern), dtype=np.float32)
    for col in target_enc_cols
}
full_y_detrended_series = pd.Series(y_detrended_full)
full_weights_series = pd.Series(w_full)

for fold, (trn_idx, hold_idx) in enumerate(kf.split(df_train_modern)):
    fold_trn_feats = X_train_raw.iloc[trn_idx]
    fold_trn_target = full_y_detrended_series.iloc[trn_idx]
    fold_trn_weights = full_weights_series.iloc[trn_idx]
    fold_hold_feats = X_train_raw.iloc[hold_idx]

    fold_enc = compute_hierarchical_target_encodings(
        fold_trn_feats,
        fold_trn_target,
        weight_series=fold_trn_weights,
        smoothing=25.0,
    )
    fold_hold_te = apply_hierarchical_encodings(fold_hold_feats, fold_enc)

    for col in target_enc_cols:
        oof_full_train[f"te_{col}"][hold_idx] = fold_hold_te[f"te_{col}"]

# Compute final production target encodings for test set transformation
full_enc = compute_hierarchical_target_encodings(
    X_train_raw,
    full_y_detrended_series,
    weight_series=full_weights_series,
    smoothing=25.0,
)
test_te = apply_hierarchical_encodings(X_test_raw, full_enc)

X_train_final = X_train_base.copy()
for col in target_enc_cols:
    X_train_final[f"te_{col}"] = oof_full_train[f"te_{col}"]

X_test_final = X_test_base.copy()
for col in target_enc_cols:
    X_test_final[f"te_{col}"] = test_te[f"te_{col}"]

assert not X_train_final.isnull().any().any(), "NaN found in X_train_final"
assert not X_test_final.isnull().any().any(), "NaN found in X_test_final"

print(
    f"Executing full modern regime LightGBM retraining (2012-2016, {len(X_train_final):,} samples) for {best_iteration_lgb} iterations..."
)
full_trn_data = lgb.Dataset(
    X_train_final,
    label=y_detrended_full,
    weight=w_full,
    categorical_feature=cat_code_cols,
    free_raw_data=False,
)

lgb_full_model = lgb.train(
    lgb_params,
    full_trn_data,
    num_boost_round=best_iteration_lgb,
)

print(
    f"Executing full modern regime XGBoost retraining (2012-2016, {len(X_train_final):,} samples) for {best_iteration_xgb} iterations..."
)
xgb_full_model = xgb.XGBRegressor(
    n_estimators=best_iteration_xgb,
    max_depth=9,
    learning_rate=0.035,
    colsample_bytree=0.70,
    subsample=0.80,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=8,
    **xgb_device_kwargs,
)
xgb_full_model.fit(
    X_train_final,
    y_detrended_full,
    sample_weight=w_full,
    verbose=100,
)

hybrid_full_model = DualEngineSpatioTemporalHybridModel(
    full_trend_model, lgb_full_model, xgb_full_model, w_lgb=w_lgb_opt, w_xgb=w_xgb_opt
)

# -------------------------------------------------------------------------
# 12. Test Inference and Submission Generation
# -------------------------------------------------------------------------
print("Generating test predictions on 375,098 test samples for 2017 H1...")
test_preds = hybrid_full_model.predict(df_test, X_test_final)

# Clip to [1, 100,000,000] and convert from log10 to raw currency units
test_preds_clipped = np.clip(test_preds, 0.0, 8.0)
test_price_preds = np.clip(10.0**test_preds_clipped, 1.0, 100_000_000.0)
test_price_preds = np.round(test_price_preds).astype(np.int64)

test_ids = df_test["id"].values
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
