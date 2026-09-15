import gc
import json
import os
import random
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
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
    df["town_type"] = df["town_district"] + "__" + df["property_type"]
    df["district_new_build"] = df["district"] + "__" + df["is_new_build"]
    df["district_tenure"] = df["district"] + "__" + df["tenure"]
    df["district_sale_cat"] = df["district"] + "__" + df["sale_category"]
    df["sale_cat_type"] = df["sale_category"] + "__" + df["property_type"]
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


# --- Autoregressive Damped Quarterly Constant-Quality HPI Extrapolator ---
class HierarchicalRegionalTrendModel:
    """
    Quarterly Constant-Quality Hedonic Price Index (HPI) Decomposition
    with Autoregressive Damped Forward Extrapolation (damping factor 0.88).
    Decouples hedonic mix shifts via Ridge regression, estimates quarterly
    price indices across National -> County -> District -> District_Type with
    Empirical Bayes shrinkage, and extrapolates forward with multi-quarter
    momentum damped by 0.88 to prevent post-2015 UK housing deceleration overshoot.
    """

    def __init__(
        self,
        t0: float = 2014.0,
        modern_start_year: int = 2012,
        damping_factor: float = 0.88,
    ):
        self.t0 = float(t0)
        self.modern_start_year = int(modern_start_year)
        self.damping_factor = float(damping_factor)
        self.hedonic_cols = ["property_type", "is_new_build", "tenure", "sale_category"]
        self.ohe = None
        self.ridge = None
        self.hedonic_intercept = 0.0
        self.national_intercept = 0.0
        self.national_slope = 0.02
        self.district_slopes = {}
        self.district_type_slopes = {}
        self.max_q = 0
        self.nat_q_index = {}
        self.nat_last_level = 0.0
        self.nat_momentum = 0.005
        self.county_last_level = {}
        self.county_momentum = {}
        self.district_last_level = {}
        self.district_momentum = {}
        self.dt_last_level = {}
        self.dt_momentum = {}

    def _get_hedonic_effect(self, df: pd.DataFrame) -> np.ndarray:
        if self.ohe is None or self.ridge is None:
            return np.zeros(len(df), dtype=np.float64)
        X_ohe = self.ohe.transform(df[self.hedonic_cols])
        hedonic_pred = self.ridge.predict(X_ohe)
        return (hedonic_pred - self.hedonic_intercept).astype(np.float64)

    def fit(self, df: pd.DataFrame):
        fit_df = (
            df[df["year"] >= self.modern_start_year].copy()
            if "year" in df.columns
            else df.copy()
        )

        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        X_hedonic = self.ohe.fit_transform(fit_df[self.hedonic_cols])
        y_raw = fit_df["log10_price"].values.astype(np.float64)

        self.ridge = Ridge(alpha=50.0, fit_intercept=True)
        self.ridge.fit(X_hedonic, y_raw)
        self.hedonic_intercept = float(np.mean(self.ridge.predict(X_hedonic)))

        hedonic_effect = self.ridge.predict(X_hedonic) - self.hedonic_intercept
        cq_y = y_raw - hedonic_effect
        fit_df["_cq_y"] = cq_y

        # Quarterly continuous and discrete indices
        q_int = (
            (fit_df["year"] - self.modern_start_year) * 4 + (fit_df["quarter"] - 1)
        ).astype(np.int32)
        fit_df["_q"] = q_int
        self.max_q = int(fit_df["_q"].max())

        # 1. National Quarterly Series & Multi-Quarter Momentum
        nat_q_stats = fit_df.groupby("_q")["_cq_y"].mean().to_dict()
        for q in range(self.max_q + 1):
            self.nat_q_index[q] = float(
                nat_q_stats.get(q, self.nat_q_index.get(q - 1, 5.30))
            )

        self.nat_last_level = self.nat_q_index[self.max_q]
        lookback = min(4, self.max_q)
        raw_nat_mom = (
            self.nat_last_level - self.nat_q_index[self.max_q - lookback]
        ) / max(1, lookback)
        self.nat_momentum = max(0.001, float(raw_nat_mom))
        self.national_slope = float(self.nat_momentum * 4.0)
        self.national_intercept = float(self.nat_last_level)

        # 2. County Level Aggregations & EB Momentum
        c_last = (
            fit_df[fit_df["_q"] == self.max_q]
            .groupby("county")["_cq_y"]
            .agg(["count", "mean"])
            .reset_index()
        )
        c_prev = (
            fit_df[fit_df["_q"] == max(0, self.max_q - lookback)]
            .groupby("county")["_cq_y"]
            .agg(["count", "mean"])
            .reset_index()
        )

        c_merged = pd.merge(
            c_last, c_prev, on="county", how="outer", suffixes=("_last", "_prev")
        ).fillna({"count_last": 0, "count_prev": 0})
        m_c = 100.0
        for _, r in c_merged.iterrows():
            c = r["county"]
            n_l = r["count_last"]
            y_l = r["mean_last"] if n_l > 0 else self.nat_last_level
            shrunk_last = (n_l * y_l + m_c * self.nat_last_level) / (n_l + m_c)
            self.county_last_level[c] = float(shrunk_last)

            n_p = r["count_prev"]
            if n_l > 10 and n_p > 10:
                raw_mom = (r["mean_last"] - r["mean_prev"]) / max(1, lookback)
                shrunk_mom = (
                    min(n_l, n_p) * raw_mom + m_c * self.nat_momentum
                ) / (min(n_l, n_p) + m_c)
            else:
                shrunk_mom = self.nat_momentum
            self.county_momentum[c] = max(0.001, float(shrunk_mom))

        # 3. District Level Aggregations & EB Momentum
        d_last = (
            fit_df[fit_df["_q"] == self.max_q]
            .groupby(["district", "county"])["_cq_y"]
            .agg(["count", "mean"])
            .reset_index()
        )
        d_prev = (
            fit_df[fit_df["_q"] == max(0, self.max_q - lookback)]
            .groupby(["district", "county"])["_cq_y"]
            .agg(["count", "mean"])
            .reset_index()
        )
        d_merged = pd.merge(
            d_last,
            d_prev,
            on=["district", "county"],
            how="outer",
            suffixes=("_last", "_prev"),
        ).fillna({"count_last": 0, "count_prev": 0})

        m_d = 50.0
        for _, r in d_merged.iterrows():
            d = r["district"]
            c = r["county"]
            c_prior_lvl = self.county_last_level.get(c, self.nat_last_level)
            c_prior_mom = self.county_momentum.get(c, self.nat_momentum)

            n_l = r["count_last"]
            y_l = r["mean_last"] if n_l > 0 else c_prior_lvl
            shrunk_last = (n_l * y_l + m_d * c_prior_lvl) / (n_l + m_d)
            self.district_last_level[d] = float(shrunk_last)

            n_p = r["count_prev"]
            if n_l > 5 and n_p > 5:
                raw_mom = (r["mean_last"] - r["mean_prev"]) / max(1, lookback)
                shrunk_mom = (
                    min(n_l, n_p) * raw_mom + m_d * c_prior_mom
                ) / (min(n_l, n_p) + m_d)
            else:
                shrunk_mom = c_prior_mom
            shrunk_mom = max(0.001, float(shrunk_mom))
            self.district_momentum[d] = shrunk_mom
            self.district_slopes[d] = shrunk_mom * 4.0

        # 4. District x Property Type Level
        dt_last = (
            fit_df[fit_df["_q"] == self.max_q]
            .groupby(["district_type", "district"])["_cq_y"]
            .agg(["count", "mean"])
            .reset_index()
        )
        dt_prev = (
            fit_df[fit_df["_q"] == max(0, self.max_q - lookback)]
            .groupby(["district_type", "district"])["_cq_y"]
            .agg(["count", "mean"])
            .reset_index()
        )
        dt_merged = pd.merge(
            dt_last,
            dt_prev,
            on=["district_type", "district"],
            how="outer",
            suffixes=("_last", "_prev"),
        ).fillna({"count_last": 0, "count_prev": 0})

        m_dt = 25.0
        for _, r in dt_merged.iterrows():
            dt = r["district_type"]
            d = r["district"]
            d_prior_lvl = self.district_last_level.get(d, self.nat_last_level)
            d_prior_mom = self.district_momentum.get(d, self.nat_momentum)

            n_l = r["count_last"]
            y_l = r["mean_last"] if n_l > 0 else d_prior_lvl
            shrunk_last = (n_l * y_l + m_dt * d_prior_lvl) / (n_l + m_dt)
            self.dt_last_level[dt] = float(shrunk_last)

            n_p = r["count_prev"]
            if n_l > 3 and n_p > 3:
                raw_mom = (r["mean_last"] - r["mean_prev"]) / max(1, lookback)
                shrunk_mom = (
                    min(n_l, n_p) * raw_mom + m_dt * d_prior_mom
                ) / (min(n_l, n_p) + m_dt)
            else:
                shrunk_mom = d_prior_mom
            shrunk_mom = max(0.001, float(shrunk_mom))
            self.dt_momentum[dt] = shrunk_mom
            self.district_type_slopes[dt] = shrunk_mom * 4.0

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        q_float = (
            df["year_fraction"].values.astype(np.float64) - self.modern_start_year
        ) * 4.0

        # Retrieve base levels and quarterly momentums
        c_lvl = (
            df["county"]
            .map(self.county_last_level)
            .fillna(self.nat_last_level)
            .values.astype(np.float64)
        )
        c_mom = (
            df["county"]
            .map(self.county_momentum)
            .fillna(self.nat_momentum)
            .values.astype(np.float64)
        )

        d_lvl = (
            df["district"]
            .map(self.district_last_level)
            .fillna(pd.Series(c_lvl, index=df.index))
            .values.astype(np.float64)
        )
        d_mom = (
            df["district"]
            .map(self.district_momentum)
            .fillna(pd.Series(c_mom, index=df.index))
            .values.astype(np.float64)
        )

        dt_lvl = (
            df["district_type"]
            .map(self.dt_last_level)
            .fillna(pd.Series(d_lvl, index=df.index))
            .values.astype(np.float64)
        )
        dt_mom = (
            df["district_type"]
            .map(self.dt_momentum)
            .fillna(pd.Series(d_mom, index=df.index))
            .values.astype(np.float64)
        )

        # Autoregressive forward extrapolation with geometric damping
        h = q_float - float(self.max_q)
        phi = self.damping_factor

        in_sample_mask = h <= 0.0
        out_sample_mask = ~in_sample_mask

        trend = np.zeros(len(df), dtype=np.float64)

        if np.any(out_sample_mask):
            h_out = h[out_sample_mask]
            # Cumulative damped appreciation: phi * (1 - phi^h) / (1 - phi)
            damped_increments = dt_mom[out_sample_mask] * (
                phi * (1.0 - np.power(phi, h_out)) / (1.0 - phi)
            )
            trend[out_sample_mask] = dt_lvl[out_sample_mask] + damped_increments

        if np.any(in_sample_mask):
            trend[in_sample_mask] = (
                dt_lvl[in_sample_mask] + dt_mom[in_sample_mask] * h[in_sample_mask]
            )

        hedonic_effect = self._get_hedonic_effect(df)
        combined_trend = trend + hedonic_effect
        return combined_trend.astype(np.float32)

    def get_district_slopes(self) -> dict:
        return self.district_slopes

    def get_district_type_slopes(self) -> dict:
        return self.district_type_slopes


# 22-Year (1995-2016) Historical Spatial Price Anchors Extraction
def compute_historical_spatial_anchors(hist_df: pd.DataFrame):
    """
    Extracts 22-year persistent relative spatial price anchors across the full historical dataset.
    Computes Empirical Bayes shrunk relative differentials:
    1. Town to District price differential
    2. District to National price differential
    3. Town to National price differential
    """
    nat_mean = float(hist_df["log10_price"].mean())

    # District vs National
    dist_stats = hist_df.groupby("district")["log10_price"].agg(["count", "mean"])
    m_dist = 50.0
    shrunk_dist_mean = (
        (dist_stats["count"] * dist_stats["mean"] + m_dist * nat_mean)
        / (dist_stats["count"] + m_dist)
    ).astype(np.float32)
    dist_anchor_map = (shrunk_dist_mean - nat_mean).astype(np.float32).to_dict()
    dist_mean_map = shrunk_dist_mean.to_dict()

    # Town vs District and Town vs National
    town_stats = (
        hist_df.groupby(["town_district", "district"])["log10_price"]
        .agg(["count", "mean"])
        .reset_index()
    )
    town_stats["dist_prior"] = (
        town_stats["district"].map(dist_mean_map).fillna(nat_mean)
    )
    m_town = 25.0
    town_stats["shrunk_town_mean"] = (
        (town_stats["count"] * town_stats["mean"] + m_town * town_stats["dist_prior"])
        / (town_stats["count"] + m_town)
    ).astype(np.float32)

    town_dist_diff = (
        town_stats["shrunk_town_mean"] - town_stats["dist_prior"]
    ).astype(np.float32)
    town_nat_diff = (town_stats["shrunk_town_mean"] - nat_mean).astype(np.float32)

    town_dist_diff_map = dict(zip(town_stats["town_district"], town_dist_diff))
    town_nat_diff_map = dict(zip(town_stats["town_district"], town_nat_diff))

    return dist_anchor_map, town_dist_diff_map, town_nat_diff_map


def _extract_te_maps(stat_df: pd.DataFrame, target_col: str = "residual") -> dict:
    global_mean = float(stat_df[target_col].mean())

    # 1. County level encoding (shrunk to global)
    county_stats = stat_df.groupby("county")[target_col].agg(["count", "mean"])
    m_county = 50.0
    county_map = (
        (county_stats["count"] * county_stats["mean"] + m_county * global_mean)
        / (county_stats["count"] + m_county)
    ).astype(np.float32).to_dict()

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
    district_te = (
        (
            district_stats["count"] * district_stats["mean"]
            + m_district * district_stats["county_prior"]
        )
        / (district_stats["count"] + m_district)
    ).astype(np.float32)
    district_map = dict(zip(district_stats["district"], district_te))

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
    town_te = (
        (
            town_stats["count"] * town_stats["mean"]
            + m_town * town_stats["district_prior"]
        )
        / (town_stats["count"] + m_town)
    ).astype(np.float32)
    town_map = dict(zip(town_stats["town_district"], town_te))

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
    dtype_te = (
        (
            dtype_stats["count"] * dtype_stats["mean"]
            + m_dtype * dtype_stats["district_prior"]
        )
        / (dtype_stats["count"] + m_dtype)
    ).astype(np.float32)
    dtype_map = dict(zip(dtype_stats["district_type"], dtype_te))

    # 5. Town x Property Type interaction encoding (shrunk to district_type)
    ttype_stats = (
        stat_df.groupby(["town_type", "district_type"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    ttype_stats["dtype_prior"] = (
        ttype_stats["district_type"].map(dtype_map).fillna(global_mean)
    )
    m_ttype = 15.0
    ttype_te = (
        (
            ttype_stats["count"] * ttype_stats["mean"]
            + m_ttype * ttype_stats["dtype_prior"]
        )
        / (ttype_stats["count"] + m_ttype)
    ).astype(np.float32)
    ttype_map = dict(zip(ttype_stats["town_type"], ttype_te))

    # 6. District x New Build interaction encoding
    dnew_stats = (
        stat_df.groupby(["district_new_build", "district"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    dnew_stats["district_prior"] = (
        dnew_stats["district"].map(district_map).fillna(global_mean)
    )
    m_dnew = 20.0
    dnew_te = (
        (
            dnew_stats["count"] * dnew_stats["mean"]
            + m_dnew * dnew_stats["district_prior"]
        )
        / (dnew_stats["count"] + m_dnew)
    ).astype(np.float32)
    dnew_map = dict(zip(dnew_stats["district_new_build"], dnew_te))

    # 7. District x Tenure interaction encoding
    dten_stats = (
        stat_df.groupby(["district_tenure", "district"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    dten_stats["district_prior"] = (
        dten_stats["district"].map(district_map).fillna(global_mean)
    )
    m_dten = 20.0
    dten_te = (
        (
            dten_stats["count"] * dten_stats["mean"]
            + m_dten * dten_stats["district_prior"]
        )
        / (dten_stats["count"] + m_dten)
    ).astype(np.float32)
    dten_map = dict(zip(dten_stats["district_tenure"], dten_te))

    # 8. Hedonic Profile encoding
    profile_stats = stat_df.groupby("hedonic_profile")[target_col].agg(
        ["count", "mean"]
    )
    m_prof = 50.0
    profile_map = (
        (profile_stats["count"] * profile_stats["mean"] + m_prof * global_mean)
        / (profile_stats["count"] + m_prof)
    ).astype(np.float32).to_dict()

    # 9. District x Sale Category interaction encoding
    dsc_stats = (
        stat_df.groupby(["district_sale_cat", "district"])[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    dsc_stats["district_prior"] = (
        dsc_stats["district"].map(district_map).fillna(global_mean)
    )
    m_dsc = 25.0
    dsc_te = (
        (
            dsc_stats["count"] * dsc_stats["mean"]
            + m_dsc * dsc_stats["district_prior"]
        )
        / (dsc_stats["count"] + m_dsc)
    ).astype(np.float32)
    dsc_map = dict(zip(dsc_stats["district_sale_cat"], dsc_te))

    # 10. Sale Category x Property Type interaction encoding
    sct_stats = (
        stat_df.groupby("sale_cat_type")[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    m_sct = 30.0
    sct_te = (
        (
            sct_stats["count"] * sct_stats["mean"]
            + m_sct * global_mean
        )
        / (sct_stats["count"] + m_sct)
    ).astype(np.float32)
    sct_map = dict(zip(sct_stats["sale_cat_type"], sct_te))

    return {
        "global_mean": global_mean,
        "county_map": county_map,
        "district_map": district_map,
        "town_map": town_map,
        "dtype_map": dtype_map,
        "ttype_map": ttype_map,
        "dnew_map": dnew_map,
        "dten_map": dten_map,
        "profile_map": profile_map,
        "dsc_map": dsc_map,
        "sct_map": sct_map,
    }


def _apply_te_maps(df: pd.DataFrame, maps: dict) -> dict:
    gm = maps["global_mean"]
    te_county = df["county"].map(maps["county_map"]).fillna(gm).astype(np.float32)
    te_district = (
        df["district"].map(maps["district_map"]).fillna(te_county).astype(np.float32)
    )
    te_town = (
        df["town_district"].map(maps["town_map"]).fillna(te_district).astype(np.float32)
    )
    te_district_type = (
        df["district_type"]
        .map(maps["dtype_map"])
        .fillna(te_district)
        .astype(np.float32)
    )
    te_town_type = (
        df["town_type"].map(maps["ttype_map"]).fillna(te_district_type).astype(np.float32)
    )
    te_district_new_build = (
        df["district_new_build"]
        .map(maps["dnew_map"])
        .fillna(te_district)
        .astype(np.float32)
    )
    te_district_tenure = (
        df["district_tenure"]
        .map(maps["dten_map"])
        .fillna(te_district)
        .astype(np.float32)
    )
    te_district_sale_cat = (
        df["district_sale_cat"]
        .map(maps["dsc_map"])
        .fillna(te_district)
        .astype(np.float32)
    )
    te_sale_cat_type = (
        df["sale_cat_type"].map(maps["sct_map"]).fillna(gm).astype(np.float32)
    )
    te_hedonic_profile = (
        df["hedonic_profile"].map(maps["profile_map"]).fillna(gm).astype(np.float32)
    )

    return {
        "te_county": te_county.values,
        "te_district": te_district.values,
        "te_town": te_town.values,
        "te_district_type": te_district_type.values,
        "te_town_type": te_town_type.values,
        "te_district_new_build": te_district_new_build.values,
        "te_district_tenure": te_district_tenure.values,
        "te_district_sale_cat": te_district_sale_cat.values,
        "te_sale_cat_type": te_sale_cat_type.values,
        "te_hedonic_profile": te_hedonic_profile.values,
    }


def compute_hierarchical_features(
    fit_train_df: pd.DataFrame,
    apply_dfs: list,
    target_col: str = "residual",
    trend_model=None,
    hist_anchors: tuple = None,
):
    """
    Computes strict 5-fold Out-Of-Fold (OOF) Empirical Bayes target encodings for the modern training slice,
    and applies full-train target encodings to holdout / test sets. Integrates 22-year persistent spatial anchors.
    """
    stat_df = fit_train_df[fit_train_df["year"] >= MODERN_START_YEAR].copy()
    global_mean = float(stat_df[target_col].mean())

    if trend_model is not None:
        district_slopes = trend_model.get_district_slopes()
        default_slope = float(trend_model.national_slope)
    else:
        district_slopes = {}
        default_slope = 0.02

    # Recency-weighted Micro-Location Target Encodings (Half-life ~2 years, gamma = 0.35)
    max_yf_stat = float(stat_df["year_fraction"].max())
    stat_df["_rec_w"] = np.exp(
        0.35 * (stat_df["year_fraction"] - max_yf_stat)
    ).astype(np.float32)
    stat_df["_rec_wy"] = stat_df["_rec_w"] * stat_df[target_col]

    global_rec_mean = float(
        stat_df["_rec_wy"].sum() / max(stat_df["_rec_w"].sum(), 1e-6)
    )

    # District Recency Residual
    rec_agg = stat_df.groupby("district")[["_rec_w", "_rec_wy"]].sum()
    rec_m = 30.0
    rec_district_map = (
        (rec_agg["_rec_wy"] + rec_m * global_rec_mean) / (rec_agg["_rec_w"] + rec_m)
    ).astype(np.float32).to_dict()

    # District x Property Type Recency Residual (EB shrunk to District Recency)
    dt_rec_agg = (
        stat_df.groupby(["district_type", "district"])[["_rec_w", "_rec_wy"]]
        .sum()
        .reset_index()
    )
    dt_rec_agg["d_prior"] = (
        dt_rec_agg["district"].map(rec_district_map).fillna(global_rec_mean)
    )
    rec_dt_m = 20.0
    rec_dt_te = (
        (dt_rec_agg["_rec_wy"] + rec_dt_m * dt_rec_agg["d_prior"])
        / (dt_rec_agg["_rec_w"] + rec_dt_m)
    ).astype(np.float32)
    rec_dtype_map = dict(zip(dt_rec_agg["district_type"], rec_dt_te))

    # Town x Property Type Recency Residual (EB shrunk to District_Type Recency)
    tt_rec_agg = (
        stat_df.groupby(["town_type", "district_type"])[["_rec_w", "_rec_wy"]]
        .sum()
        .reset_index()
    )
    tt_rec_agg["dt_prior"] = (
        tt_rec_agg["district_type"].map(rec_dtype_map).fillna(global_rec_mean)
    )
    rec_tt_m = 15.0
    rec_tt_te = (
        (tt_rec_agg["_rec_wy"] + rec_tt_m * tt_rec_agg["dt_prior"])
        / (tt_rec_agg["_rec_w"] + rec_tt_m)
    ).astype(np.float32)
    rec_ttype_map = dict(zip(tt_rec_agg["town_type"], rec_tt_te))

    # District x New Build Recency Residual
    dnb_rec_agg = (
        stat_df.groupby(["district_new_build", "district"])[["_rec_w", "_rec_wy"]]
        .sum()
        .reset_index()
    )
    dnb_rec_agg["d_prior"] = (
        dnb_rec_agg["district"].map(rec_district_map).fillna(global_rec_mean)
    )
    rec_dnb_m = 15.0
    rec_dnb_te = (
        (dnb_rec_agg["_rec_wy"] + rec_dnb_m * dnb_rec_agg["d_prior"])
        / (dnb_rec_agg["_rec_w"] + rec_dnb_m)
    ).astype(np.float32)
    rec_dnew_map = dict(zip(dnb_rec_agg["district_new_build"], rec_dnb_te))

    # Spatial Volatility (EB Shrunk Standard Deviation of Residuals)
    global_std = float(stat_df[target_col].std())
    county_vol_raw = stat_df.groupby("county")[target_col].std().fillna(global_std)
    county_vol_map = county_vol_raw.to_dict()

    dist_vol_stats = (
        stat_df.groupby(["district", "county"])[target_col]
        .agg(["count", "std"])
        .reset_index()
    )
    dist_vol_stats["std"] = dist_vol_stats["std"].fillna(global_std)
    dist_vol_stats["c_prior"] = (
        dist_vol_stats["county"].map(county_vol_map).fillna(global_std)
    )
    m_vol = 30.0
    shrunk_dist_vol = (
        (dist_vol_stats["count"] * dist_vol_stats["std"] + m_vol * dist_vol_stats["c_prior"])
        / (dist_vol_stats["count"] + m_vol)
    ).astype(np.float32)
    dist_volatility_map = dict(zip(dist_vol_stats["district"], shrunk_dist_vol))

    # Log Transaction Frequencies
    district_counts = (
        np.log1p(stat_df["district"].value_counts()).astype(np.float32).to_dict()
    )
    town_counts = (
        np.log1p(stat_df["town_district"].value_counts()).astype(np.float32).to_dict()
    )

    # Spatial Price Growth Velocity (Annual Residual Momentum)
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

    district_type_slopes = (
        trend_model.get_district_type_slopes()
        if trend_model and hasattr(trend_model, "get_district_type_slopes")
        else {}
    )

    if hist_anchors is not None:
        dist_anchor_map, town_dist_diff_map, town_nat_diff_map = hist_anchors
    else:
        dist_anchor_map, town_dist_diff_map, town_nat_diff_map = {}, {}, {}

    te_cols = [
        "te_county",
        "te_district",
        "te_town",
        "te_district_type",
        "te_town_type",
        "te_district_new_build",
        "te_district_tenure",
        "te_district_sale_cat",
        "te_sale_cat_type",
        "te_hedonic_profile",
    ]

    # 5-Fold OOF Target Encoding on Training Slice (apply_dfs[0])
    train_modern_df = apply_dfs[0].copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_te_arrays = {
        col: np.zeros(len(train_modern_df), dtype=np.float32) for col in te_cols
    }

    for tr_idx, oof_idx in kf.split(train_modern_df):
        tr_slice = train_modern_df.iloc[tr_idx]
        oof_slice = train_modern_df.iloc[oof_idx]
        fold_maps = _extract_te_maps(tr_slice, target_col=target_col)
        fold_te = _apply_te_maps(oof_slice, fold_maps)
        for col in te_cols:
            oof_te_arrays[col][oof_idx] = fold_te[col]

    for col in te_cols:
        train_modern_df[col] = oof_te_arrays[col]

    # Full-train target encodings for holdout and test sets (apply_dfs[1:])
    full_maps = _extract_te_maps(stat_df, target_col=target_col)
    processed_other_dfs = []
    for other_df in apply_dfs[1:]:
        other_copy = other_df.copy()
        other_te = _apply_te_maps(other_copy, full_maps)
        for col in te_cols:
            other_copy[col] = other_te[col]
        processed_other_dfs.append(other_copy)

    # Attach interaction, recency, velocity, volatility, and 22-year persistent spatial anchors
    results = []
    for res in [train_modern_df] + processed_other_dfs:
        res["te_type_district_premium"] = (
            res["te_district_type"] - res["te_district"]
        ).astype(np.float32)
        res["district_recency_residual"] = (
            res["district"].map(rec_district_map).fillna(global_rec_mean).astype(np.float32)
        )
        res["rec_district_type_res"] = (
            res["district_type"].map(rec_dtype_map).fillna(res["district_recency_residual"]).astype(np.float32)
        )
        res["rec_town_type_res"] = (
            res["town_type"].map(rec_ttype_map).fillna(res["rec_district_type_res"]).astype(np.float32)
        )
        res["rec_dnew_res"] = (
            res["district_new_build"].map(rec_dnew_map).fillna(res["district_recency_residual"]).astype(np.float32)
        )
        res["district_volatility"] = (
            res["district"].map(dist_volatility_map).fillna(global_std).astype(np.float32)
        )
        res["log_count_district"] = (
            res["district"].map(district_counts).fillna(0.0).astype(np.float32)
        )
        res["log_count_town"] = (
            res["town_district"].map(town_counts).fillna(0.0).astype(np.float32)
        )
        res["district_trend_slope"] = (
            res["district"].map(district_slopes).fillna(default_slope).astype(np.float32)
        )
        res["district_type_trend_slope"] = (
            res["district_type"]
            .map(district_type_slopes)
            .fillna(res["district_trend_slope"])
            .astype(np.float32)
        )

        res["district_momentum_1y"] = (
            res["district"].map(growth_1y).fillna(global_growth_1y).astype(np.float32)
        )
        res["district_momentum_2y"] = (
            res["district"].map(growth_2y).fillna(global_growth_2y).astype(np.float32)
        )

        # 22-Year persistent historical spatial price anchors
        res["hist_anchor_town_dist_diff"] = (
            res["town_district"].map(town_dist_diff_map).fillna(0.0).astype(np.float32)
        )
        res["hist_anchor_dist_nat_diff"] = (
            res["district"].map(dist_anchor_map).fillna(0.0).astype(np.float32)
        )
        res["hist_anchor_town_nat_diff"] = (
            res["town_district"].map(town_nat_diff_map).fillna(0.0).astype(np.float32)
        )

        results.append(res)

    return results


print("Configuring out-of-time splits and constant-quality hierarchical regional detrending...")
val_raw_train = train_df[train_df["year"] < 2016].copy()
val_raw_holdout = train_df[train_df["year"] == 2016].copy()
prod_raw_train = train_df[train_df["year"] <= 2016].copy()
prod_raw_test = test_df.copy()

# Precalculate 22-year persistent historical spatial anchors
print("Computing 22-year persistent historical spatial price anchors...")
val_hist_anchors = compute_historical_spatial_anchors(val_raw_train)
prod_hist_anchors = compute_historical_spatial_anchors(prod_raw_train)

# Fit Hedonic-Decoupled Hierarchical Regional Trend Model over modern regime
val_trend_model = HierarchicalRegionalTrendModel(
    t0=2014.0, modern_start_year=MODERN_START_YEAR
)
val_trend_model.fit(val_raw_train)
print(
    f"Validation Hierarchical Trend national slope: {val_trend_model.national_slope:.5f}, intercept: {val_trend_model.national_intercept:.5f}"
)

val_raw_train["trend"] = val_trend_model.predict(val_raw_train)
val_raw_train["residual"] = (
    val_raw_train["log10_price"] - val_raw_train["trend"]
).astype(np.float32)
val_raw_train["y_tilde"] = val_raw_train["residual"]

val_raw_holdout["trend"] = val_trend_model.predict(val_raw_holdout)
val_raw_holdout["residual"] = (
    val_raw_holdout["log10_price"] - val_raw_holdout["trend"]
).astype(np.float32)
val_raw_holdout["y_tilde"] = val_raw_holdout["residual"]
val_trend = val_raw_holdout["trend"].values.astype(np.float32)

prod_trend_model = HierarchicalRegionalTrendModel(
    t0=2014.0, modern_start_year=MODERN_START_YEAR
)
prod_trend_model.fit(prod_raw_train)
print(
    f"Production Hierarchical Trend national slope: {prod_trend_model.national_slope:.5f}, intercept: {prod_trend_model.national_intercept:.5f}"
)

prod_raw_train["trend"] = prod_trend_model.predict(prod_raw_train)
prod_raw_train["residual"] = (
    prod_raw_train["log10_price"] - prod_raw_train["trend"]
).astype(np.float32)
prod_raw_train["y_tilde"] = prod_raw_train["residual"]

prod_raw_test["trend"] = prod_trend_model.predict(prod_raw_test)
prod_test_trend = prod_raw_test["trend"].values.astype(np.float32)

print(f"Validation train split (pre-2016): {len(val_raw_train):,} rows")
print(f"Validation holdout split (2016): {len(val_raw_holdout):,} rows")
print(f"Production train split (pre-2017): {len(prod_raw_train):,} rows")
print(f"Production test split (2017): {len(prod_raw_test):,} rows")

# Compute 5-fold OOF hierarchical features and persistent spatial anchors
val_train_feat, val_holdout_feat = compute_hierarchical_features(
    fit_train_df=val_raw_train,
    apply_dfs=[
        val_raw_train[val_raw_train["year"] >= MODERN_START_YEAR],
        val_raw_holdout,
    ],
    target_col="residual",
    trend_model=val_trend_model,
    hist_anchors=val_hist_anchors,
)

prod_train_feat, prod_test_feat = compute_hierarchical_features(
    fit_train_df=prod_raw_train,
    apply_dfs=[
        prod_raw_train[prod_raw_train["year"] >= MODERN_START_YEAR],
        prod_raw_test,
    ],
    target_col="residual",
    trend_model=prod_trend_model,
    hist_anchors=prod_hist_anchors,
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
    "district_trend_slope",
    "district_type_trend_slope",
    "te_county",
    "te_district",
    "te_town",
    "te_district_type",
    "te_town_type",
    "te_district_new_build",
    "te_district_tenure",
    "te_district_sale_cat",
    "te_sale_cat_type",
    "te_hedonic_profile",
    "te_type_district_premium",
    "district_recency_residual",
    "rec_district_type_res",
    "rec_town_type_res",
    "rec_dnew_res",
    "district_volatility",
    "log_count_district",
    "log_count_town",
    "district_momentum_1y",
    "district_momentum_2y",
    "hist_anchor_town_dist_diff",
    "hist_anchor_dist_nat_diff",
    "hist_anchor_town_nat_diff",
]
feature_cols = cont_feature_cols + cat_feature_cols
print(f"Total features ready for models: {len(feature_cols)}")

cat_cardinalities = {
    col: int(prod_train_feat[col].max() + 1) for col in cat_feature_cols
}

trend_feature_names = [
    "year_fraction",
    "time_elapsed_days",
    "district_trend_slope",
    "district_type_trend_slope",
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
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "n_estimators": 2500,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }


def get_catboost_hedonic_params() -> dict:
    params = {
        "iterations": 1600,
        "learning_rate": 0.06,
        "depth": 7,
        "l2_leaf_reg": 6.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
    }
    if torch.cuda.is_available():
        params["task_type"] = "GPU"
    return params


# --- Spatially-Conditioned Non-Negative Meta-Learner ---
class SpatiallyConditionedMetaLearner:
    """
    Spatially-Conditioned Non-Negative Meta-Learner.
    Dynamically predicts non-negative blending weights for LightGBM, CatBoost,
    and HedonicTrendResidualNet conditioned on spatial transaction density
    (log_count_district), property type, and volatility.
    Guarantees non-negative weights summing to 1.0 everywhere via softmax gating.
    """

    def __init__(self, n_models: int = 3, l2_reg: float = 1e-2):
        self.n_models = n_models
        self.l2_reg = l2_reg
        self.W = None
        self.z_mean = None
        self.z_std = None

    def _extract_conditioning_matrix(self, df: pd.DataFrame) -> np.ndarray:
        log_count = df["log_count_district"].values.astype(np.float64)
        volatility = df["district_volatility"].values.astype(np.float64)
        is_new = (df["is_new_build"] == "Y").astype(np.float64).values

        p_type = df["property_type"].values
        is_flat = (p_type == "F").astype(np.float64)
        is_detached = (p_type == "D").astype(np.float64)
        is_semi = (p_type == "S").astype(np.float64)
        is_terraced = (p_type == "T").astype(np.float64)

        feats = np.column_stack(
            [log_count, volatility, is_new, is_flat, is_detached, is_semi, is_terraced]
        )
        return feats

    def fit(self, oof_preds: np.ndarray, y_true: np.ndarray, train_df: pd.DataFrame):
        raw_z = self._extract_conditioning_matrix(train_df)
        self.z_mean = np.mean(raw_z, axis=0, keepdims=True)
        self.z_std = np.std(raw_z, axis=0, keepdims=True) + 1e-5
        norm_z = (raw_z - self.z_mean) / self.z_std
        N, K = norm_z.shape
        Z = np.column_stack([np.ones((N, 1), dtype=np.float64), norm_z])
        num_params_per_model = K + 1

        # Global optimal weights as prior
        def global_obj(w):
            pred = np.sum(oof_preds * w, axis=1)
            return np.mean((pred - y_true) ** 2)

        res_g = minimize(
            global_obj,
            x0=[1.0 / self.n_models] * self.n_models,
            bounds=[(0.0, 1.0)] * self.n_models,
            constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
            method="SLSQP",
        )
        w_global = res_g.x / np.sum(res_g.x)
        print(
            f"Global Prior Weights: LGB={w_global[0]:.4f}, CB={w_global[1]:.4f}, NN={w_global[2]:.4f}"
        )

        init_W = np.zeros((self.n_models, num_params_per_model), dtype=np.float64)
        for m in range(self.n_models):
            init_W[m, 0] = np.log(max(1e-4, w_global[m]))

        flat_init = init_W.ravel()

        def loss_and_grad(flat_w):
            curr_W = flat_w.reshape((self.n_models, num_params_per_model))
            logits = Z @ curr_W.T
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            pred = np.sum(oof_preds * weights, axis=1)
            error = pred - y_true
            loss = np.mean(error**2)

            l2_loss = 0.5 * self.l2_reg * np.sum(curr_W[:, 1:] ** 2)
            total_loss = loss + l2_loss

            dL_dpred = (2.0 / N) * error[:, None]
            d_logits = weights * (oof_preds - pred[:, None]) * dL_dpred
            grad_W = d_logits.T @ Z
            grad_W[:, 1:] += self.l2_reg * curr_W[:, 1:]

            return float(total_loss), grad_W.ravel().astype(np.float64)

        opt_res = minimize(
            loss_and_grad,
            x0=flat_init,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 60, "ftol": 1e-7, "disp": False},
        )
        self.W = opt_res.x.reshape((self.n_models, num_params_per_model))
        print("Spatially-Conditioned Meta-Learner calibrated successfully.")
        return self

    def predict_weights(self, df: pd.DataFrame) -> np.ndarray:
        raw_z = self._extract_conditioning_matrix(df)
        norm_z = (raw_z - self.z_mean) / self.z_std
        N = len(df)
        Z = np.column_stack([np.ones((N, 1), dtype=np.float64), norm_z])
        logits = Z @ self.W.T
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, model_preds: list, df: pd.DataFrame) -> np.ndarray:
        preds_stack = np.column_stack(model_preds)
        weights = self.predict_weights(df)
        blended = np.sum(preds_stack * weights, axis=1)
        return blended.astype(np.float32)


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


# --- 3. Out-Of-Fold Cross-Validation & Spatially-Conditioned Meta-Stacking ---
print("--- Generating Out-of-Fold Residual Predictions on Modern Regime (2012-2015) ---")
gamma = 0.20
max_yf_val = float(val_train_feat["year_fraction"].max())
val_weights = np.exp(gamma * (val_train_feat["year_fraction"].values - max_yf_val)).astype(np.float32)

n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(val_train_feat), dtype=np.float32)
oof_cat = np.zeros(len(val_train_feat), dtype=np.float32)
oof_nn = np.zeros(len(val_train_feat), dtype=np.float32)

holdout_lgb_folds = []
holdout_cat_folds = []
holdout_nn_folds = []

lgb_params = get_lgb_hedonic_params()
cb_params = get_catboost_hedonic_params()

# Prepare holdout neural tensors
dummy_nn, _, _, _ = build_hedonic_neural_model(
    num_continuous=len(cont_feature_cols),
    cat_cardinalities=cat_cardinalities,
    device=device,
)
val_x_cont_holdout, val_x_cat_holdout, _, _ = prepare_tensors(
    val_holdout_feat,
    cont_feature_cols,
    cat_feature_cols,
    dummy_nn.embeddings,
    target_col="residual",
)
del dummy_nn
if torch.cuda.is_available():
    torch.cuda.empty_cache()

best_iters_lgb = []
best_iters_cat = []

for fold, (tr_idx, oof_idx) in enumerate(kf.split(val_train_feat), 1):
    print(f"--- Processing OOF Fold {fold}/{n_splits} ---")
    tr_df = val_train_feat.iloc[tr_idx]
    va_df = val_train_feat.iloc[oof_idx]
    w_tr = val_weights[tr_idx]

    # 1. Neural Model on Fold
    fold_nn, fold_criterion, fold_optimizer, fold_scheduler = build_hedonic_neural_model(
        num_continuous=len(cont_feature_cols),
        cat_cardinalities=cat_cardinalities,
        device=device,
    )
    f_x_cont_tr, f_x_cat_tr, f_y_tr, f_w_tr = prepare_tensors(
        tr_df,
        cont_feature_cols,
        cat_feature_cols,
        fold_nn.embeddings,
        target_col="residual",
        weights=w_tr,
    )
    f_x_cont_va, f_x_cat_va, _, _ = prepare_tensors(
        va_df,
        cont_feature_cols,
        cat_feature_cols,
        fold_nn.embeddings,
        target_col="residual",
    )

    f_samples = len(tr_df)
    f_indices = np.arange(f_samples)
    f_epochs = 8
    f_batch_size = 8192

    for epoch in range(1, f_epochs + 1):
        fold_nn.train()
        np.random.shuffle(f_indices)
        for s_idx in range(0, f_samples, f_batch_size):
            e_idx = min(s_idx + f_batch_size, f_samples)
            b_idx = torch.as_tensor(f_indices[s_idx:e_idx], dtype=torch.long, device=device)
            fold_optimizer.zero_grad()
            p_res = fold_nn(f_x_cont_tr[b_idx], {col: f_x_cat_tr[col][b_idx] for col in cat_feature_cols})
            loss = fold_criterion(p_res, f_y_tr[b_idx], sample_weight=f_w_tr[b_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fold_nn.parameters(), max_norm=1.0)
            fold_optimizer.step()
        fold_scheduler.step()

    oof_nn[oof_idx] = predict_neural_model(fold_nn, f_x_cont_va, f_x_cat_va)
    holdout_nn_folds.append(predict_neural_model(fold_nn, val_x_cont_holdout, val_x_cat_holdout))

    del f_x_cont_tr, f_x_cat_tr, f_y_tr, f_w_tr, f_x_cont_va, f_x_cat_va, fold_nn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. LightGBM on Fold
    lgb_tr = lgb.Dataset(
        tr_df[feature_cols],
        label=tr_df["residual"],
        weight=w_tr,
        categorical_feature=cat_feature_cols,
    )
    lgb_va = lgb.Dataset(
        va_df[feature_cols],
        label=va_df["residual"],
        reference=lgb_tr,
        categorical_feature=cat_feature_cols,
    )
    f_lgb = lgb.train(
        lgb_params,
        lgb_tr,
        num_boost_round=1200,
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[oof_idx] = f_lgb.predict(va_df[feature_cols], num_iteration=f_lgb.best_iteration)
    holdout_lgb_folds.append(f_lgb.predict(val_holdout_feat[feature_cols], num_iteration=f_lgb.best_iteration))
    best_iters_lgb.append(f_lgb.best_iteration)
    del lgb_tr, lgb_va, f_lgb

    # 3. CatBoost on Fold
    f_cb_params = cb_params.copy()
    f_cb_params["iterations"] = 1000
    f_cb = CatBoostRegressor(**f_cb_params)
    try:
        f_cb.fit(
            tr_df[feature_cols],
            tr_df["residual"],
            sample_weight=w_tr,
            eval_set=(va_df[feature_cols], va_df["residual"]),
            early_stopping_rounds=30,
            verbose=False,
        )
    except Exception:
        f_cb_params["task_type"] = "CPU"
        f_cb_params["thread_count"] = -1
        f_cb = CatBoostRegressor(**f_cb_params)
        f_cb.fit(
            tr_df[feature_cols],
            tr_df["residual"],
            sample_weight=w_tr,
            eval_set=(va_df[feature_cols], va_df["residual"]),
            early_stopping_rounds=30,
            verbose=False,
        )
    oof_cat[oof_idx] = f_cb.predict(va_df[feature_cols]).astype(np.float32)
    holdout_cat_folds.append(f_cb.predict(val_holdout_feat[feature_cols]).astype(np.float32))
    cb_best = f_cb.get_best_iteration()
    best_iters_cat.append(cb_best if (cb_best and cb_best > 0) else 1000)
    del f_cb

    gc.collect()

# Average holdout base model predictions across folds
val_nn_res = np.mean(holdout_nn_folds, axis=0)
val_lgb_res = np.mean(holdout_lgb_folds, axis=0)
val_cat_res = np.mean(holdout_cat_folds, axis=0)

print("--- Calibrating Spatially-Conditioned Meta-Learner on OOF Predictions ---")
meta_learner = SpatiallyConditionedMetaLearner(n_models=3, l2_reg=1e-2)
oof_matrix = np.column_stack([oof_lgb, oof_cat, oof_nn])
meta_learner.fit(oof_matrix, val_train_feat["residual"].values, val_train_feat)

# Evaluate Spatially-Conditioned Ensemble on 2016 Holdout
val_meta_res = meta_learner.predict([val_lgb_res, val_cat_res, val_nn_res], val_holdout_feat)
val_meta_log10 = val_meta_res + val_trend
val_meta_raw = np.clip(10.0**val_meta_log10, 1.0, None)
holdout_score = compute_log10_rmse(val_holdout_feat["price"].values, val_meta_raw)
print(f"Holdout 2016 Spatially-Conditioned Meta-Ensemble RMSE: {holdout_score:.5f}")

del (
    val_train_feat,
    val_holdout_feat,
    val_x_cont_holdout,
    val_x_cat_holdout,
    holdout_lgb_folds,
    holdout_cat_folds,
    holdout_nn_folds,
    oof_matrix,
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
prod_num_epochs = 10
batch_size = 8192

for epoch in range(1, prod_num_epochs + 1):
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

test_nn_res = predict_neural_model(prod_model, prod_x_cont_test, prod_x_cat_test)

del prod_x_cont_train, prod_x_cat_train, prod_y_train, prod_w_train, prod_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Retrain LightGBM on complete modern dataset
avg_lgb_round = int(np.mean(best_iters_lgb) * 1.15) if best_iters_lgb else 1200
prod_lgb_params = lgb_params.copy()

full_train_lgb = lgb.Dataset(
    prod_train_feat[feature_cols],
    label=prod_train_feat["residual"],
    weight=prod_weights,
    categorical_feature=cat_feature_cols,
)
prod_lgb_model = lgb.train(
    prod_lgb_params,
    full_train_lgb,
    num_boost_round=avg_lgb_round,
    callbacks=[lgb.log_evaluation(period=0)],
)
test_lgb_res = prod_lgb_model.predict(prod_test_feat[feature_cols]).astype(np.float32)

# Retrain CatBoost on complete modern dataset
avg_cat_round = int(np.mean(best_iters_cat) * 1.15) if best_iters_cat else 1000
prod_cb_params = cb_params.copy()
prod_cb_params["iterations"] = avg_cat_round
if cb_params.get("task_type") == "CPU":
    prod_cb_params["task_type"] = "CPU"
    prod_cb_params["thread_count"] = -1

prod_cb_model = CatBoostRegressor(**prod_cb_params)
try:
    prod_cb_model.fit(
        prod_train_feat[feature_cols],
        prod_train_feat["residual"],
        sample_weight=prod_weights,
        verbose=False,
    )
except Exception:
    prod_cb_params["task_type"] = "CPU"
    prod_cb_params["thread_count"] = -1
    prod_cb_model = CatBoostRegressor(**prod_cb_params)
    prod_cb_model.fit(
        prod_train_feat[feature_cols],
        prod_train_feat["residual"],
        sample_weight=prod_weights,
        verbose=False,
    )

test_cat_res = prod_cb_model.predict(prod_test_feat[feature_cols]).astype(np.float32)

# --- 5. Spatially-Conditioned Meta-Stacking Inference & Export ---
test_final_res = meta_learner.predict(
    [test_lgb_res, test_cat_res, test_nn_res], prod_test_feat
)
test_final_log10 = test_final_res + prod_test_trend
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

sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
assert (
    submission_df["id"].values == sample_sub["id"].values
).all(), "Submission ID order mismatch!"

print("Submission integrity verified successfully.")
print(f"Generated predictions for {len(submission_df):,} test transactions.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Final Validation Score: {holdout_score:.5f}")
