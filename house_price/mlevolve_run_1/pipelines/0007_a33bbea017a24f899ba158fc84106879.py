import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------------------
# 0. Setup and Directory Verification
# -------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Starting Property Transaction Price Prediction Pipeline...")

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.csv"

print(f"Loading raw data from {train_path} and {test_path}...")
df_train = pd.read_parquet(train_path)
df_test = pd.read_csv(test_path)
print(f"Loaded train shape: {df_train.shape}, test shape: {df_test.shape}")

# -------------------------------------------------------------------------
# 2. String Standardization & Missing Value Handling
# -------------------------------------------------------------------------
cat_cols = [
    "property_type",
    "is_new_build",
    "tenure",
    "town",
    "district",
    "county",
    "sale_category",
]

for col in cat_cols:
    df_train[col] = df_train[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df_test[col] = df_test[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

# Ensure dates are parsed as datetime
df_train["date"] = pd.to_datetime(df_train["date"])
df_test["date"] = pd.to_datetime(df_test["date"])

# Compute log10 target on train (clipped lower to 1 as per competition rules)
df_train["target"] = np.log10(np.maximum(df_train["price"].values, 1.0)).astype(
    np.float32
)

# -------------------------------------------------------------------------
# 3. Temporal Out-of-Time Split Definition
# -------------------------------------------------------------------------
# Test covers 2017-01-01 to 2017-06-29 (~6 months).
# Validation covers exactly the preceding 6 months: 2016-07-01 to 2016-12-31.
# All feature engineering statistics are computed strictly prior to 2016-07-01.
VAL_START_DATE = pd.Timestamp("2016-07-01")
VAL_END_DATE = pd.Timestamp("2016-12-31")
TRAIN_WINDOW_START = pd.Timestamp("2011-01-01")

val_mask = (df_train["date"] >= VAL_START_DATE) & (df_train["date"] <= VAL_END_DATE)
hist_mask = df_train["date"] < VAL_START_DATE

# For modeling partition, use modern post-2010 price regime with extreme noise removed.
# Validation partition remains 100% unfiltered to faithfully represent evaluation.
train_model_mask = (
    (df_train["date"] >= TRAIN_WINDOW_START)
    & (df_train["date"] < VAL_START_DATE)
    & (df_train["price"] >= 5000)
    & (df_train["price"] <= 25000000)
)

print(f"Historical stats partition size (pre-val): {hist_mask.sum():,} rows")
print(f"Train modeling partition size: {train_model_mask.sum():,} rows")
print(f"Validation partition size: {val_mask.sum():,} rows")


# -------------------------------------------------------------------------
# 4. Temporal Featurization
# -------------------------------------------------------------------------
def extract_temporal_features(df):
    dt = df["date"]
    year = dt.dt.year.astype(np.int16)
    month = dt.dt.month.astype(np.int8)
    day = dt.dt.day.astype(np.int8)
    dayofweek = dt.dt.dayofweek.astype(np.int8)
    quarter = dt.dt.quarter.astype(np.int8)
    dayofyear = dt.dt.dayofyear.astype(np.int16)

    # Continuous time index (years since 2010-01-01) for macro trend extrapolation
    ref_date = pd.Timestamp("2010-01-01")
    time_continuous = ((dt - ref_date).dt.days / 365.25).astype(np.float32)

    # Seasonality cyclical transformations
    month_sin = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    month_cos = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365.25).astype(np.float32)
    dayofyear_cos = np.cos(2 * np.pi * dayofyear / 365.25).astype(np.float32)

    is_friday = (dayofweek == 4).astype(np.int8)

    return pd.DataFrame(
        {
            "year": year,
            "month": month,
            "day": day,
            "dayofweek": dayofweek,
            "quarter": quarter,
            "dayofyear": dayofyear,
            "is_friday": is_friday,
            "time_continuous": time_continuous,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "dayofyear_sin": dayofyear_sin,
            "dayofyear_cos": dayofyear_cos,
        },
        index=df.index,
    )


print("Extracting temporal features...")
temp_train = extract_temporal_features(df_train)
temp_test = extract_temporal_features(df_test)

for c in temp_train.columns:
    df_train[c] = temp_train[c]
    df_test[c] = temp_test[c]

del temp_train, temp_test
gc.collect()

# -------------------------------------------------------------------------
# 5. Categorical Encoding & Hierarchy Interaction Keys
# -------------------------------------------------------------------------
print("Constructing spatial interaction keys and categorical rankings...")
for df in (df_train, df_test):
    df["district_prop"] = df["district"] + "___" + df["property_type"]
    df["town_prop"] = df["town"] + "___" + df["property_type"]
    df["county_prop"] = df["county"] + "___" + df["property_type"]
    df["prop_tenure"] = df["property_type"] + "___" + df["tenure"]
    df["prop_new"] = df["property_type"] + "___" + df["is_new_build"]
    df["prop_sale_cat"] = df["property_type"] + "___" + df["sale_category"]

prop_type_map = {"D": 0, "S": 1, "T": 2, "F": 3, "O": 4}
size_rank_map = {"D": 4, "S": 3, "T": 2, "F": 1, "O": 2}
new_build_map = {"N": 0, "Y": 1}
tenure_map = {"F": 0, "L": 1, "U": 2}
sale_cat_map = {"A": 0, "B": 1}

for df in (df_train, df_test):
    df["enc_prop_type"] = (
        df["property_type"].map(prop_type_map).fillna(4).astype(np.int8)
    )
    df["enc_size_rank"] = (
        df["property_type"].map(size_rank_map).fillna(2).astype(np.int8)
    )
    df["enc_new_build"] = (
        df["is_new_build"].map(new_build_map).fillna(0).astype(np.int8)
    )
    df["enc_tenure"] = df["tenure"].map(tenure_map).fillna(2).astype(np.int8)
    df["enc_sale_cat"] = df["sale_category"].map(sale_cat_map).fillna(0).astype(np.int8)

# -------------------------------------------------------------------------
# 6. Spatial Composition & Frequency Features (Pre-Val History Only)
# -------------------------------------------------------------------------
print("Computing spatial frequency and composition features...")
df_hist = df_train[hist_mask]
total_hist = len(df_hist)

county_counts = df_hist["county"].value_counts()
district_counts = df_hist["district"].value_counts()
town_counts = df_hist["town"].value_counts()
dist_prop_counts = df_hist["district_prop"].value_counts()

for df in (df_train, df_test):
    df["freq_county"] = (df["county"].map(county_counts).fillna(1) / total_hist).astype(
        np.float32
    )
    df["freq_district"] = (
        df["district"].map(district_counts).fillna(1) / total_hist
    ).astype(np.float32)
    df["freq_town"] = (df["town"].map(town_counts).fillna(1) / total_hist).astype(
        np.float32
    )
    df["freq_dist_prop"] = (
        df["district_prop"].map(dist_prop_counts).fillna(1) / total_hist
    ).astype(np.float32)

# District structural composition ratios
district_totals = df_hist.groupby("district").size()
flat_in_dist = df_hist[df_hist["property_type"] == "F"].groupby("district").size()
detached_in_dist = df_hist[df_hist["property_type"] == "D"].groupby("district").size()
new_in_dist = df_hist[df_hist["is_new_build"] == "Y"].groupby("district").size()

dist_flat_ratio = (flat_in_dist / district_totals).fillna(0).astype(np.float32)
dist_detached_ratio = (detached_in_dist / district_totals).fillna(0).astype(np.float32)
dist_new_ratio = (new_in_dist / district_totals).fillna(0).astype(np.float32)

for df in (df_train, df_test):
    df["ratio_dist_flat"] = (
        df["district"].map(dist_flat_ratio).fillna(0.2).astype(np.float32)
    )
    df["ratio_dist_detached"] = (
        df["district"].map(dist_detached_ratio).fillna(0.2).astype(np.float32)
    )
    df["ratio_dist_new_build"] = (
        df["district"].map(dist_new_ratio).fillna(0.1).astype(np.float32)
    )

# -------------------------------------------------------------------------
# 7. Hierarchical Bayesian Target Encodings
# -------------------------------------------------------------------------
print("Building Hierarchical Bayesian Target Encodings...")


def compute_bayesian_target_encodings(source_df, smoothing_weights):
    global_mean = float(source_df["target"].mean())

    # Level 1: County
    county_stats = source_df.groupby("county")["target"].agg(["count", "mean"])
    m_c = smoothing_weights.get("county", 40.0)
    county_smooth = (
        county_stats["count"] * county_stats["mean"] + m_c * global_mean
    ) / (county_stats["count"] + m_c)
    county_dict = county_smooth.to_dict()

    # Vectorized district-to-county mode mapping
    dist_to_county = (
        source_df.groupby(["district", "county"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["district", "cnt"])
        .drop_duplicates("district", keep="last")
        .set_index("district")["county"]
        .to_dict()
    )

    # Level 2: District
    dist_stats = source_df.groupby("district")["target"].agg(["count", "mean"])
    dist_priors = pd.Series(
        dist_stats.index.map(
            lambda d: county_dict.get(dist_to_county.get(d, ""), global_mean)
        ),
        index=dist_stats.index,
    )
    m_d = smoothing_weights.get("district", 25.0)
    dist_smooth = (dist_stats["count"] * dist_stats["mean"] + m_d * dist_priors) / (
        dist_stats["count"] + m_d
    )
    dist_dict = dist_smooth.to_dict()

    # Vectorized town-to-district mode mapping
    town_to_dist = (
        source_df.groupby(["town", "district"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["town", "cnt"])
        .drop_duplicates("town", keep="last")
        .set_index("town")["district"]
        .to_dict()
    )

    # Level 3: Town
    town_stats = source_df.groupby("town")["target"].agg(["count", "mean"])
    town_priors = pd.Series(
        town_stats.index.map(
            lambda t: dist_dict.get(town_to_dist.get(t, ""), global_mean)
        ),
        index=town_stats.index,
    )
    m_t = smoothing_weights.get("town", 15.0)
    town_smooth = (town_stats["count"] * town_stats["mean"] + m_t * town_priors) / (
        town_stats["count"] + m_t
    )
    town_dict = town_smooth.to_dict()

    # Level 4: Property Type
    prop_stats = source_df.groupby("property_type")["target"].agg(["count", "mean"])
    m_p = smoothing_weights.get("property_type", 50.0)
    prop_smooth = (prop_stats["count"] * prop_stats["mean"] + m_p * global_mean) / (
        prop_stats["count"] + m_p
    )
    prop_dict = prop_smooth.to_dict()

    # Level 5: District x Property Type
    dist_prop_stats = source_df.groupby("district_prop")["target"].agg(
        ["count", "mean"]
    )

    def get_dist_prop_prior(key):
        parts = key.split("___")
        if len(parts) == 2:
            d, p = parts
            base = dist_dict.get(d, global_mean)
            p_adj = prop_dict.get(p, global_mean) - global_mean
            return base + p_adj
        return global_mean

    dist_prop_priors = pd.Series(
        [get_dist_prop_prior(k) for k in dist_prop_stats.index],
        index=dist_prop_stats.index,
    )
    m_dp = smoothing_weights.get("district_prop", 20.0)
    dist_prop_smooth = (
        dist_prop_stats["count"] * dist_prop_stats["mean"] + m_dp * dist_prop_priors
    ) / (dist_prop_stats["count"] + m_dp)
    dist_prop_dict = dist_prop_smooth.to_dict()

    # Level 6: Town x Property Type
    town_prop_stats = source_df.groupby("town_prop")["target"].agg(["count", "mean"])

    def get_town_prop_prior(key):
        parts = key.split("___")
        if len(parts) == 2:
            t, p = parts
            d = town_to_dist.get(t, "")
            return dist_prop_dict.get(f"{d}___{p}", dist_dict.get(d, global_mean))
        return global_mean

    town_prop_priors = pd.Series(
        [get_town_prop_prior(k) for k in town_prop_stats.index],
        index=town_prop_stats.index,
    )
    m_tp = smoothing_weights.get("town_prop", 10.0)
    town_prop_smooth = (
        town_prop_stats["count"] * town_prop_stats["mean"] + m_tp * town_prop_priors
    ) / (town_prop_stats["count"] + m_tp)
    town_prop_dict = town_prop_smooth.to_dict()

    return {
        "global_mean": global_mean,
        "county": county_dict,
        "district": dist_dict,
        "town": town_dict,
        "prop": prop_dict,
        "dist_prop": dist_prop_dict,
        "town_prop": town_prop_dict,
    }


# 1. Recent Market Window (2014-01-01 to 2016-06-30)
recent_mask = (df_train["date"] >= pd.Timestamp("2014-01-01")) & (
    df_train["date"] < VAL_START_DATE
)
enc_recent = compute_bayesian_target_encodings(
    df_train[recent_mask],
    {
        "county": 30.0,
        "district": 20.0,
        "town": 12.0,
        "property_type": 30.0,
        "district_prop": 15.0,
        "town_prop": 8.0,
    },
)

# 2. All-Time Historical Window (pre-val)
enc_alltime = compute_bayesian_target_encodings(
    df_hist,
    {
        "county": 50.0,
        "district": 30.0,
        "town": 20.0,
        "property_type": 50.0,
        "district_prop": 25.0,
        "town_prop": 12.0,
    },
)


def apply_target_encodings(df, enc, prefix):
    gm = enc["global_mean"]
    col_dist_prop = f"{prefix}_dist_prop"
    col_town = f"{prefix}_town"
    col_dist = f"{prefix}_dist"
    col_county = f"{prefix}_county"

    dist_prop_vals = df["district_prop"].map(enc["dist_prop"])
    dist_vals = df["district"].map(enc["district"])
    county_vals = df["county"].map(enc["county"])
    town_vals = df["town"].map(enc["town"])

    res = pd.DataFrame(index=df.index)
    res[col_dist_prop] = (
        dist_prop_vals.fillna(dist_vals)
        .fillna(county_vals)
        .fillna(gm)
        .astype(np.float32)
    )
    res[col_town] = (
        town_vals.fillna(dist_vals).fillna(county_vals).fillna(gm).astype(np.float32)
    )
    res[col_dist] = dist_vals.fillna(county_vals).fillna(gm).astype(np.float32)
    res[col_county] = county_vals.fillna(gm).astype(np.float32)
    return res


print("Applying hierarchical Bayesian encodings to full dataset...")
te_train_recent = apply_target_encodings(df_train, enc_recent, "te_rec")
te_train_alltime = apply_target_encodings(df_train, enc_alltime, "te_all")
te_test_recent = apply_target_encodings(df_test, enc_recent, "te_rec")
te_test_alltime = apply_target_encodings(df_test, enc_alltime, "te_all")

for c in te_train_recent.columns:
    df_train[c] = te_train_recent[c]
    df_test[c] = te_test_recent[c]

for c in te_train_alltime.columns:
    df_train[c] = te_train_alltime[c]
    df_test[c] = te_test_alltime[c]

del te_train_recent, te_train_alltime, te_test_recent, te_test_alltime
gc.collect()

# -------------------------------------------------------------------------
# 8. Out-of-Fold (OOF) Encodings for Train Modeling Partition
# -------------------------------------------------------------------------
print("Generating 5-Fold Out-of-Fold target encodings to prevent training leakage...")
train_indices = df_train[train_model_mask].index.to_numpy()
n_folds = 5
fold_assignments = np.random.randint(0, n_folds, size=len(train_indices))

for fold in range(n_folds):
    val_fold_idx = train_indices[fold_assignments == fold]
    train_fold_idx = train_indices[fold_assignments != fold]

    fold_train_df = df_train.loc[train_fold_idx]
    fold_enc = compute_bayesian_target_encodings(
        fold_train_df,
        {
            "county": 30.0,
            "district": 20.0,
            "town": 12.0,
            "property_type": 30.0,
            "district_prop": 15.0,
            "town_prop": 8.0,
        },
    )

    fold_val_df = df_train.loc[val_fold_idx]
    fold_res = apply_target_encodings(fold_val_df, fold_enc, "te_rec")
    for c in fold_res.columns:
        df_train.loc[val_fold_idx, c] = fold_res[c]

# Local geographic price momentum and spatial differential features
for df in (df_train, df_test):
    df["price_momentum_dist"] = (df["te_rec_dist"] - df["te_all_dist"]).astype(
        np.float32
    )
    df["price_momentum_town"] = (df["te_rec_town"] - df["te_all_town"]).astype(
        np.float32
    )
    df["town_premium_dist"] = (df["te_rec_town"] - df["te_rec_dist"]).astype(np.float32)
    df["dist_premium_county"] = (df["te_rec_dist"] - df["te_rec_county"]).astype(
        np.float32
    )

# -------------------------------------------------------------------------
# 9. Hierarchical Hedonic Base Level & Trend Decomposition
# -------------------------------------------------------------------------
print("Fitting hierarchical linear time-trend estimator across national/county/district...")
train_trend_df = df_train.loc[
    train_model_mask, ["county", "district", "time_continuous", "target"]
].copy()
t_arr = train_trend_df["time_continuous"].to_numpy(dtype=np.float64)
y_arr = train_trend_df["target"].to_numpy(dtype=np.float64)

# 1. National level linear trend
mean_t_nat = float(np.mean(t_arr))
mean_y_nat = float(np.mean(y_arr))
s_tt_nat = float(np.sum((t_arr - mean_t_nat) ** 2))
s_ty_nat = float(np.sum((t_arr - mean_t_nat) * (y_arr - mean_y_nat)))
slope_nat = float(s_ty_nat / s_tt_nat)
intercept_nat = mean_y_nat - slope_nat * mean_t_nat
print(
    f"National Trend: slope = {slope_nat:.5f}, intercept = {intercept_nat:.5f}"
)

# 2. County level trend with empirical Bayes shrinkage toward national slope
train_trend_df["t2"] = t_arr**2
train_trend_df["ty"] = t_arr * y_arr

county_stats = train_trend_df.groupby("county").agg(
    n=("target", "count"),
    sum_t=("time_continuous", "sum"),
    sum_y=("target", "sum"),
    sum_t2=("t2", "sum"),
    sum_ty=("ty", "sum"),
)
c_mean_t = county_stats["sum_t"] / county_stats["n"]
c_mean_y = county_stats["sum_y"] / county_stats["n"]
c_s_tt = county_stats["sum_t2"] - county_stats["n"] * (c_mean_t**2)
c_s_ty = county_stats["sum_ty"] - county_stats["n"] * c_mean_t * c_mean_y
c_raw_slope = c_s_ty / np.maximum(c_s_tt, 1e-4)

M_county = 100.0
county_smooth_slope = (county_stats["n"] * c_raw_slope + M_county * slope_nat) / (
    county_stats["n"] + M_county
)
county_slope_dict = county_smooth_slope.to_dict()

# Vectorized district-to-county mode mapping
dist_to_county = (
    train_trend_df.groupby(["district", "county"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["district", "cnt"])
    .drop_duplicates("district", keep="last")
    .set_index("district")["county"]
    .to_dict()
)

# 3. District level trend with empirical Bayes shrinkage toward county slope
dist_stats = train_trend_df.groupby("district").agg(
    n=("target", "count"),
    sum_t=("time_continuous", "sum"),
    sum_y=("target", "sum"),
    sum_t2=("t2", "sum"),
    sum_ty=("ty", "sum"),
)
d_mean_t = dist_stats["sum_t"] / dist_stats["n"]
d_mean_y = dist_stats["sum_y"] / dist_stats["n"]
d_s_tt = dist_stats["sum_t2"] - dist_stats["n"] * (d_mean_t**2)
d_s_ty = dist_stats["sum_ty"] - dist_stats["n"] * d_mean_t * d_mean_y
d_raw_slope = d_s_ty / np.maximum(d_s_tt, 1e-4)

d_county_priors = dist_stats.index.map(
    lambda d: county_slope_dict.get(dist_to_county.get(d, ""), slope_nat)
)
M_dist = 50.0
dist_smooth_slope = (dist_stats["n"] * d_raw_slope + M_dist * d_county_priors) / (
    dist_stats["n"] + M_dist
)
dist_slope_dict = dist_smooth_slope.to_dict()

del train_trend_df, t_arr, y_arr
gc.collect()

# Vectorized town-to-district mode mapping on training partition
town_to_dist = (
    df_train.loc[train_model_mask, ["town", "district"]]
    .groupby(["town", "district"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["town", "cnt"])
    .drop_duplicates("town", keep="last")
    .set_index("town")["district"]
    .to_dict()
)

# Continuous time relative to Reference Date T_ref = 2016-01-01 (in fractional years)
T_REF = pd.Timestamp("2016-01-01")
for df in (df_train, df_test):
    d_slope = df["district"].map(dist_slope_dict)
    c_slope = df["county"].map(county_slope_dict)
    df["trend_slope"] = d_slope.fillna(c_slope).fillna(slope_nat).astype(np.float32)
    df["t_from_ref"] = ((df["date"] - T_REF).dt.days / 365.25).astype(np.float32)

# Estimate Bayesian smoothed base levels B(loc, prop_type) at T_ref
print("Estimating hierarchical Bayesian hedonic base levels anchored at T_ref (2016-01-01)...")
clean_train_idx = df_train[train_model_mask].index
y_detrended_clean = (
    df_train.loc[clean_train_idx, "target"]
    - df_train.loc[clean_train_idx, "trend_slope"] * df_train.loc[clean_train_idx, "t_from_ref"]
).to_numpy(dtype=np.float64)

# Hierarchy Level 0: Global Mean Base Level
global_base = float(np.mean(y_detrended_clean))

# Hierarchy Level 1: Property Type Base Level
prop_series = df_train.loc[clean_train_idx, "property_type"]
prop_stats = pd.DataFrame({"y": y_detrended_clean, "p": prop_series}).groupby("p")["y"].agg(["count", "mean"])
m_p_base = 50.0
prop_smooth = (prop_stats["count"] * prop_stats["mean"] + m_p_base * global_base) / (prop_stats["count"] + m_p_base)
prop_base_dict = prop_smooth.to_dict()

# Hierarchy Level 2: County x Property Type
cp_series = df_train.loc[clean_train_idx, "county_prop"]
cp_stats = pd.DataFrame({"y": y_detrended_clean, "cp": cp_series}).groupby("cp")["y"].agg(["count", "mean"])
cp_priors = pd.Series([prop_base_dict.get(k.split("___")[1], global_base) for k in cp_stats.index], index=cp_stats.index)
m_cp_base = 40.0
cp_smooth = (cp_stats["count"] * cp_stats["mean"] + m_cp_base * cp_priors) / (cp_stats["count"] + m_cp_base)
cp_base_dict = cp_smooth.to_dict()

# Hierarchy Level 3: District x Property Type
dp_series = df_train.loc[clean_train_idx, "district_prop"]
dp_stats = pd.DataFrame({"y": y_detrended_clean, "dp": dp_series}).groupby("dp")["y"].agg(["count", "mean"])
dp_priors = pd.Series([
    cp_base_dict.get(
        f"{dist_to_county.get(k.split('___')[0], '')}___{k.split('___')[1]}",
        prop_base_dict.get(k.split('___')[1], global_base),
    )
    for k in dp_stats.index
], index=dp_stats.index)
m_dp_base = 25.0
dp_smooth = (dp_stats["count"] * dp_stats["mean"] + m_dp_base * dp_priors) / (dp_stats["count"] + m_dp_base)
dp_base_dict = dp_smooth.to_dict()

# Hierarchy Level 4: Town x Property Type
tp_series = df_train.loc[clean_train_idx, "town_prop"]
tp_stats = pd.DataFrame({"y": y_detrended_clean, "tp": tp_series}).groupby("tp")["y"].agg(["count", "mean"])
tp_priors = pd.Series([
    dp_base_dict.get(
        f"{town_to_dist.get(k.split('___')[0], '')}___{k.split('___')[1]}",
        cp_base_dict.get(
            f"{dist_to_county.get(town_to_dist.get(k.split('___')[0], ''), '')}___{k.split('___')[1]}",
            prop_base_dict.get(k.split('___')[1], global_base),
        ),
    )
    for k in tp_stats.index
], index=tp_stats.index)
m_tp_base = 15.0
tp_smooth = (tp_stats["count"] * tp_stats["mean"] + m_tp_base * tp_priors) / (tp_stats["count"] + m_tp_base)
tp_base_dict = tp_smooth.to_dict()

print(f"Hierarchical Hedonic Base Level at T_ref: Global={global_base:.5f}, coverage: {len(tp_base_dict)} towns, {len(dp_base_dict)} districts")

# Compute base price levels y_base = B(loc, type) + slope(district) * (t - T_ref)
for df in (df_train, df_test):
    b_tp = df["town_prop"].map(tp_base_dict)
    b_dp = df["district_prop"].map(dp_base_dict)
    b_cp = df["county_prop"].map(cp_base_dict)
    b_p = df["property_type"].map(prop_base_dict)
    df["base_level"] = b_tp.fillna(b_dp).fillna(b_cp).fillna(b_p).fillna(global_base).astype(np.float32)
    df["y_base"] = (df["base_level"] + df["trend_slope"] * df["t_from_ref"]).astype(np.float32)

del y_detrended_clean, prop_stats, cp_stats, dp_stats, tp_stats
gc.collect()

# -------------------------------------------------------------------------
# 10. Prepare Partitions for Modeling & Categorical Types
# -------------------------------------------------------------------------
cat_cols_lgb = [
    "district",
    "county",
    "property_type",
    "tenure",
    "is_new_build",
    "sale_category",
    "prop_tenure",
    "prop_new",
    "prop_sale_cat",
]

for col in cat_cols_lgb:
    common_categories = sorted(
        list(set(df_train[col].dropna()).union(set(df_test[col].dropna())))
    )
    df_train[col] = pd.Categorical(df_train[col], categories=common_categories)
    df_test[col] = pd.Categorical(df_test[col], categories=common_categories)

# Stationary feature set: drop saturating time features (year, time_continuous, macro_trend)
feature_columns = [
    # Stationary / cyclical temporal features
    "month",
    "day",
    "dayofweek",
    "quarter",
    "dayofyear",
    "is_friday",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    # Categorical encoded & ranks
    "enc_prop_type",
    "enc_size_rank",
    "enc_new_build",
    "enc_tenure",
    "enc_sale_cat",
    # Frequency & spatial structural features
    "freq_county",
    "freq_district",
    "freq_town",
    "freq_dist_prop",
    "ratio_dist_flat",
    "ratio_dist_detached",
    "ratio_dist_new_build",
    # Target encodings (recent & all-time)
    "te_rec_dist_prop",
    "te_rec_town",
    "te_rec_dist",
    "te_rec_county",
    "te_all_dist_prop",
    "te_all_town",
    "te_all_dist",
    "te_all_county",
    # Momentum & spatial differentials
    "price_momentum_dist",
    "price_momentum_town",
    "town_premium_dist",
    "dist_premium_county",
    # Trend slope
    "trend_slope",
] + cat_cols_lgb

print(f"Total engineered features: {len(feature_columns)}")

X_train = df_train.loc[train_model_mask, feature_columns].reset_index(drop=True)
y_train = df_train.loc[train_model_mask, "target"].to_numpy(dtype=np.float32)
y_train_base = df_train.loc[train_model_mask, "y_base"].to_numpy(dtype=np.float32)
y_train_residual = y_train - y_train_base

# Compute temporal recency deltas and sample weights for training
days_to_val = (
    (VAL_START_DATE - df_train.loc[train_model_mask, "date"]).dt.days.to_numpy(
        dtype=np.float32
    )
)
sample_weight = np.exp(-0.0003 * days_to_val).astype(np.float32)

X_val = df_train.loc[val_mask, feature_columns].reset_index(drop=True)
y_val = df_train.loc[val_mask, "target"].to_numpy(dtype=np.float32)
y_val_base = df_train.loc[val_mask, "y_base"].to_numpy(dtype=np.float32)
y_val_residual = y_val - y_val_base
val_true_price = df_train.loc[val_mask, "price"].to_numpy(dtype=np.float64)

X_test = df_test[feature_columns].reset_index(drop=True)
y_test_base = df_test["y_base"].to_numpy(dtype=np.float32)
test_ids = df_test["id"].to_numpy()

# Free large raw dataframe to optimize RAM
del df_train, df_test, df_hist
gc.collect()

print(
    f"Training partition: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}"
)

# -------------------------------------------------------------------------
# 11. Model Architecture & Hyperparameter Configuration
# -------------------------------------------------------------------------
model_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "cat_smooth": 20.0,
    "min_data_per_group": 20,
    "n_jobs": 32,
    "random_state": SEED,
    "verbose": -1,
}

print("Instantiating and fitting LightGBM model on stationary residuals...")
model = lgb.LGBMRegressor(**model_params)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=0),
]

model.fit(
    X_train,
    y_train_residual,
    sample_weight=sample_weight,
    eval_set=[(X_val, y_val_residual)],
    eval_names=["valid"],
    callbacks=callbacks,
    categorical_feature=cat_cols_lgb,
)

best_iteration = (
    model.best_iteration_ if hasattr(model, "best_iteration_") else model.n_estimators
)
print(f"Optimal model checkpoint reached at tree #{best_iteration}.")

# Save and reload booster to verify inference fidelity
model_file_path = "./working/lgbm_best_model.txt"
model.booster_.save_model(model_file_path)
loaded_booster = lgb.Booster(model_file=model_file_path)

# -------------------------------------------------------------------------
# 12. Hold-Out Out-of-Time Validation Evaluation
# -------------------------------------------------------------------------


def official_log10_rmse(y_true_price, y_pred_price):
    pred_clipped = np.maximum(np.asarray(y_pred_price, dtype=np.float64), 1.0)
    true_clipped = np.maximum(np.asarray(y_true_price, dtype=np.float64), 1.0)
    log_pred = np.log10(pred_clipped)
    log_true = np.log10(true_clipped)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


print("Computing additive validation predictions and official metric...")
val_residual_preds = loaded_booster.predict(X_val, num_iteration=best_iteration)
val_preds_log = y_val_base + val_residual_preds
val_preds_price = np.maximum(10.0**val_preds_log, 1.0)

final_val_score = official_log10_rmse(val_true_price, val_preds_price)
print(f"Hold-out Validation RMSE on log10(price): {final_val_score:.5f}")

# -------------------------------------------------------------------------
# 13. Test Set Inference & Submission Generation
# -------------------------------------------------------------------------
print("Generating additive test predictions on held-out transactions...")
test_residual_preds = loaded_booster.predict(X_test, num_iteration=best_iteration)
test_preds_log = y_test_base + test_residual_preds
test_preds_price = np.round(np.maximum(10.0**test_preds_log, 1.0)).astype(np.int64)

submission_df = pd.DataFrame({"id": test_ids, "price": test_preds_price})

sub_path = "./submission/submission.csv"
submission_df.to_csv(sub_path, index=False)
print(f"Saved submission file to {sub_path} with {len(submission_df):,} rows.")

# -------------------------------------------------------------------------
# 14. Rigorous Integrity Sanity Checks
# -------------------------------------------------------------------------
sample_sub = pd.read_csv("./input/sample_submission.csv")
assert os.path.exists(sub_path), f"Submission file {sub_path} does not exist!"
assert len(submission_df) == len(
    sample_sub
), f"Row count mismatch: {len(submission_df)} vs {len(sample_sub)}"
assert list(submission_df.columns) == list(
    sample_sub.columns
), f"Column mismatch: {list(submission_df.columns)}"
assert submission_df["price"].isna().sum() == 0, "Submission contains NaN values!"
assert (submission_df["price"] > 0).all(), "Submission contains non-positive prices!"
assert (
    submission_df["id"].values == sample_sub["id"].values
).all(), "ID alignment mismatch with sample submission!"

print("Integrity verification passed successfully.")

# Clean up memory
del X_train, y_train, y_train_residual, X_val, y_val, y_val_residual, X_test, y_val_base, y_test_base
gc.collect()

# -------------------------------------------------------------------------
# 15. Final Validation Score Output (Mandatory Format)
# -------------------------------------------------------------------------
print(f"Final Validation Score: {final_val_score:.5f}")
