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


# --- Hedonic-Stratified Closed-Form Linear Trend Extrapolator ---
class HierarchicalRegionalTrendModel:
    """
    Hedonic-Stratified Closed-Form Linear Trend Extrapolator.
    Decouples hedonic property mix shifts via Ridge regression on modern transactions,
    then estimates robust closed-form WLS linear appreciation slopes across
    National -> County -> District -> (District x Property Type) with Empirical Bayes shrinkage
    in centered time coordinates.
    Guarantees steady, non-negative trend extrapolation into out-of-time horizons (2016/2017).
    """

    def __init__(
        self,
        t0: float = 2014.0,
        modern_start_year: int = 2012,
        min_slope: float = 0.008,
    ):
        self.t0 = float(t0)
        self.modern_start_year = int(modern_start_year)
        self.min_slope = float(min_slope)
        self.hedonic_cols = ["property_type", "is_new_build", "tenure", "sale_category"]
        self.ohe = None
        self.ridge = None
        self.hedonic_intercept = 0.0
        self.nat_alpha = 5.30
        self.nat_beta = 0.026
        self.national_slope = 0.026
        self.national_intercept = 5.30
        self.county_alpha = {}
        self.county_beta = {}
        self.district_alpha = {}
        self.district_beta = {}
        self.dt_alpha = {}
        self.dt_beta = {}
        self.district_slopes = {}
        self.district_type_slopes = {}

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

        # Centered time coordinate in float64
        t_vals = fit_df["year_fraction"].values.astype(np.float64) - self.t0
        fit_df["_t"] = t_vals
        fit_df["_t2"] = t_vals * t_vals
        fit_df["_ty"] = t_vals * cq_y

        # 1. National Closed-Form WLS Linear Trend
        n_nat = float(len(fit_df))
        sum_t = float(fit_df["_t"].sum())
        sum_y = float(fit_df["_cq_y"].sum())
        sum_t2 = float(fit_df["_t2"].sum())
        sum_ty = float(fit_df["_ty"].sum())

        var_t_nat = sum_t2 - (sum_t * sum_t) / n_nat
        cov_ty_nat = sum_ty - (sum_t * sum_y) / n_nat
        raw_beta_nat = cov_ty_nat / max(1e-6, var_t_nat)
        raw_alpha_nat = (sum_y / n_nat) - raw_beta_nat * (sum_t / n_nat)

        self.nat_beta = float(np.clip(raw_beta_nat, 0.015, 0.045))
        self.nat_alpha = float(raw_alpha_nat)
        self.national_slope = self.nat_beta
        self.national_intercept = self.nat_alpha

        # 2. County Level EB Trend
        c_agg = (
            fit_df.groupby("county")
            .agg({"_cq_y": ["count", "sum"], "_t": "sum", "_t2": "sum", "_ty": "sum"})
            .reset_index()
        )
        c_agg.columns = ["county", "count", "sum_y", "sum_t", "sum_t2", "sum_ty"]
        m_c = 100.0

        for _, r in c_agg.iterrows():
            c = r["county"]
            n = float(r["count"])
            v_t = float(r["sum_t2"]) - (float(r["sum_t"]) ** 2) / n
            c_ty = float(r["sum_ty"]) - (float(r["sum_t"]) * float(r["sum_y"])) / n
            b_raw = c_ty / max(1e-6, v_t) if v_t > 1e-4 else self.nat_beta
            a_raw = (float(r["sum_y"]) / n) - b_raw * (float(r["sum_t"]) / n)

            s = n / (n + m_c)
            b_shrunk = max(self.min_slope, float(s * b_raw + (1.0 - s) * self.nat_beta))
            a_shrunk = float(s * a_raw + (1.0 - s) * self.nat_alpha)
            self.county_beta[c] = b_shrunk
            self.county_alpha[c] = a_shrunk

        # 3. District Level EB Trend (shrunk to County)
        d_agg = (
            fit_df.groupby(["district", "county"])
            .agg({"_cq_y": ["count", "sum"], "_t": "sum", "_t2": "sum", "_ty": "sum"})
            .reset_index()
        )
        d_agg.columns = [
            "district",
            "county",
            "count",
            "sum_y",
            "sum_t",
            "sum_t2",
            "sum_ty",
        ]
        m_d = 50.0

        for _, r in d_agg.iterrows():
            d = r["district"]
            c = r["county"]
            c_b = self.county_beta.get(c, self.nat_beta)
            c_a = self.county_alpha.get(c, self.nat_alpha)

            n = float(r["count"])
            v_t = float(r["sum_t2"]) - (float(r["sum_t"]) ** 2) / n
            c_ty = float(r["sum_ty"]) - (float(r["sum_t"]) * float(r["sum_y"])) / n
            b_raw = c_ty / max(1e-6, v_t) if v_t > 1e-4 else c_b
            a_raw = (float(r["sum_y"]) / n) - b_raw * (float(r["sum_t"]) / n)

            s = n / (n + m_d)
            b_shrunk = max(self.min_slope, float(s * b_raw + (1.0 - s) * c_b))
            a_shrunk = float(s * a_raw + (1.0 - s) * c_a)
            self.district_beta[d] = b_shrunk
            self.district_alpha[d] = a_shrunk
            self.district_slopes[d] = b_shrunk

        # 4. District x Property Type Level EB Trend (shrunk to District)
        dt_agg = (
            fit_df.groupby(["district_type", "district"])
            .agg({"_cq_y": ["count", "sum"], "_t": "sum", "_t2": "sum", "_ty": "sum"})
            .reset_index()
        )
        dt_agg.columns = [
            "district_type",
            "district",
            "count",
            "sum_y",
            "sum_t",
            "sum_t2",
            "sum_ty",
        ]
        m_dt = 25.0

        for _, r in dt_agg.iterrows():
            dt = r["district_type"]
            d = r["district"]
            d_b = self.district_beta.get(d, self.nat_beta)
            d_a = self.district_alpha.get(d, self.nat_alpha)

            n = float(r["count"])
            v_t = float(r["sum_t2"]) - (float(r["sum_t"]) ** 2) / n
            c_ty = float(r["sum_ty"]) - (float(r["sum_t"]) * float(r["sum_y"])) / n
            b_raw = c_ty / max(1e-6, v_t) if v_t > 1e-4 else d_b
            a_raw = (float(r["sum_y"]) / n) - b_raw * (float(r["sum_t"]) / n)

            s = n / (n + m_dt)
            b_shrunk = max(self.min_slope, float(s * b_raw + (1.0 - s) * d_b))
            a_shrunk = float(s * a_raw + (1.0 - s) * d_a)
            self.dt_beta[dt] = b_shrunk
            self.dt_alpha[dt] = a_shrunk
            self.district_type_slopes[dt] = b_shrunk

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        t = df["year_fraction"].values.astype(np.float64) - self.t0

        c_a = (
            df["county"]
            .map(self.county_alpha)
            .fillna(self.nat_alpha)
            .values.astype(np.float64)
        )
        c_b = (
            df["county"]
            .map(self.county_beta)
            .fillna(self.nat_beta)
            .values.astype(np.float64)
        )

        d_a = (
            df["district"]
            .map(self.district_alpha)
            .fillna(pd.Series(c_a, index=df.index))
            .values.astype(np.float64)
        )
        d_b = (
            df["district"]
            .map(self.district_beta)
            .fillna(pd.Series(c_b, index=df.index))
            .values.astype(np.float64)
        )

        dt_a = (
            df["district_type"]
            .map(self.dt_alpha)
            .fillna(pd.Series(d_a, index=df.index))
            .values.astype(np.float64)
        )
        dt_b = (
            df["district_type"]
            .map(self.dt_beta)
            .fillna(pd.Series(d_b, index=df.index))
            .values.astype(np.float64)
        )

        trend_linear = dt_a + dt_b * t
        hedonic_effect = self._get_hedonic_effect(df)
        combined_trend = trend_linear + hedonic_effect
        return combined_trend.astype(np.float32)

    def get_district_slopes(self) -> dict:
        return self.district_slopes

    def get_district_type_slopes(self) -> dict:
        return self.district_type_slopes


# 22-Year (1995-2016) Historical Spatial Price Anchors and Drift Extraction
def compute_historical_spatial_anchors_and_drift(
    hist_df: pd.DataFrame, modern_start_year: int = 2012
):
    """
    Extracts 22-year persistent relative spatial price anchors and modern spatial drift.
    Computes normalized log-prices relative to national annual medians across 1995-2016
    to isolate pure spatial premiums and measure town-level gentrification/decay.
    """
    # 1. Annual National Medians for Macro Normalization
    nat_year_medians = hist_df.groupby("year")["log10_price"].median().to_dict()
    norm_log_price = (
        hist_df["log10_price"].values - hist_df["year"].map(nat_year_medians).values
    ).astype(np.float32)
    hist_df["_norm_lp"] = norm_log_price

    # 2. 22-Year District Relative Premium (EB shrunk to 0.0)
    dist_stats = hist_df.groupby("district")["_norm_lp"].agg(["count", "mean"])
    m_dist = 50.0
    shrunk_dist_mean = (
        (dist_stats["count"] * dist_stats["mean"]) / (dist_stats["count"] + m_dist)
    ).astype(np.float32)
    dist_anchor_map = shrunk_dist_mean.to_dict()

    # 3. 22-Year Town to District and Town to National Premium
    town_stats = (
        hist_df.groupby(["town_district", "district"])["_norm_lp"]
        .agg(["count", "mean"])
        .reset_index()
    )
    town_stats["dist_prior"] = (
        town_stats["district"].map(dist_anchor_map).fillna(0.0)
    )
    m_town = 25.0
    shrunk_town_mean = (
        (town_stats["count"] * town_stats["mean"] + m_town * town_stats["dist_prior"])
        / (town_stats["count"] + m_town)
    ).astype(np.float32)

    town_dist_diff = (shrunk_town_mean - town_stats["dist_prior"]).astype(np.float32)
    town_nat_diff = shrunk_town_mean

    town_dist_diff_map = dict(zip(town_stats["town_district"], town_dist_diff))
    town_nat_diff_map = dict(zip(town_stats["town_district"], town_nat_diff))

    # 4. Modern (2012+) Town to District Premium
    modern_df = hist_df[hist_df["year"] >= modern_start_year]
    mod_dist_stats = modern_df.groupby("district")["_norm_lp"].agg(["count", "mean"])
    mod_dist_mean = (
        (mod_dist_stats["count"] * mod_dist_stats["mean"])
        / (mod_dist_stats["count"] + 30.0)
    ).astype(np.float32).to_dict()

    mod_town_stats = (
        modern_df.groupby(["town_district", "district"])["_norm_lp"]
        .agg(["count", "mean"])
        .reset_index()
    )
    mod_town_stats["dist_prior"] = (
        mod_town_stats["district"].map(mod_dist_mean).fillna(0.0)
    )
    m_mod_town = 15.0
    mod_shrunk_town_mean = (
        (
            mod_town_stats["count"] * mod_town_stats["mean"]
            + m_mod_town * mod_town_stats["dist_prior"]
        )
        / (mod_town_stats["count"] + m_mod_town)
    ).astype(np.float32)
    mod_town_dist_diff = (mod_shrunk_town_mean - mod_town_stats["dist_prior"]).astype(
        np.float32
    )
    mod_town_dist_diff_map = dict(
        zip(mod_town_stats["town_district"], mod_town_dist_diff)
    )

    # 5. Spatial Drift: Modern vs 22-Year Town Premium
    town_spatial_drift_map = {}
    for td, hist_diff in town_dist_diff_map.items():
        mod_diff = mod_town_dist_diff_map.get(td, hist_diff)
        town_spatial_drift_map[td] = float(mod_diff - hist_diff)

    hist_df.drop(columns=["_norm_lp"], inplace=True)

    return (
        dist_anchor_map,
        town_dist_diff_map,
        town_nat_diff_map,
        mod_town_dist_diff_map,
        town_spatial_drift_map,
    )


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
        (
            dist_anchor_map,
            town_dist_diff_map,
            town_nat_diff_map,
            mod_town_dist_diff_map,
            town_spatial_drift_map,
        ) = hist_anchors
    else:
        dist_anchor_map, town_dist_diff_map, town_nat_diff_map = {}, {}, {}
        mod_town_dist_diff_map, town_spatial_drift_map = {}, {}

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

        # 22-Year persistent historical spatial price anchors and spatial drift
        res["hist_anchor_town_dist_diff"] = (
            res["town_district"].map(town_dist_diff_map).fillna(0.0).astype(np.float32)
        )
        res["hist_anchor_dist_nat_diff"] = (
            res["district"].map(dist_anchor_map).fillna(0.0).astype(np.float32)
        )
        res["hist_anchor_town_nat_diff"] = (
            res["town_district"].map(town_nat_diff_map).fillna(0.0).astype(np.float32)
        )
        res["modern_town_dist_diff"] = (
            res["town_district"]
            .map(mod_town_dist_diff_map)
            .fillna(res["hist_anchor_town_dist_diff"])
            .astype(np.float32)
        )
        res["town_spatial_drift"] = (
            res["town_district"]
            .map(town_spatial_drift_map)
            .fillna(0.0)
            .astype(np.float32)
        )

        results.append(res)

    return results


print("Configuring out-of-time splits and constant-quality hierarchical regional detrending...")
val_raw_train = train_df[train_df["year"] < 2016].copy()
val_raw_holdout = train_df[train_df["year"] == 2016].copy()
prod_raw_train = train_df[train_df["year"] <= 2016].copy()
prod_raw_test = test_df.copy()

# Precalculate 22-year persistent historical spatial anchors and spatial drift
print("Computing 22-year persistent historical spatial price anchors and spatial drift...")
val_hist_anchors = compute_historical_spatial_anchors_and_drift(val_raw_train)
prod_hist_anchors = compute_historical_spatial_anchors_and_drift(prod_raw_train)

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
    "modern_town_dist_diff",
    "town_spatial_drift",
]
feature_cols = cont_feature_cols + cat_feature_cols
print(f"Total features ready for models: {len(feature_cols)}")


# --- 2. Model Design: Heterogeneous Tri-GBDT Architecture ---
def get_lgb_hedonic_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 192,
        "max_depth": 11,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "n_estimators": 2500,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }


def get_catboost_hedonic_params() -> dict:
    params = {
        "iterations": 1800,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
    }
    if torch.cuda.is_available():
        params["task_type"] = "GPU"
    return params


def get_xgboost_hedonic_params() -> dict:
    params = {
        "n_estimators": 1800,
        "learning_rate": 0.05,
        "max_depth": 9,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if torch.cuda.is_available():
        try:
            major_ver = int(xgb.__version__.split(".")[0])
            if major_ver >= 2:
                params["tree_method"] = "hist"
                params["device"] = "cuda"
            else:
                params["tree_method"] = "gpu_hist"
        except Exception:
            params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist"
    return params


# Metric Definition: Exact Task-Faithful Log10 RMSE
def compute_log10_rmse(y_true_raw, y_pred_raw):
    clipped_pred = np.clip(y_pred_raw, 1.0, None)
    clipped_true = np.clip(y_true_raw, 1.0, None)
    log_pred = np.log10(clipped_pred)
    log_true = np.log10(clipped_true)
    return float(np.sqrt(np.mean((log_pred - log_true) ** 2)))


# --- 3. Out-of-Time Validation & Ensemble Weight Calibration ---
print(
    "--- Training Tri-GBDT Ensemble on Modern Regime (2012-2015) with 2016 Holdout ---"
)
gamma = 0.18
max_yf_val = float(val_train_feat["year_fraction"].max())
val_weights = np.exp(
    gamma * (val_train_feat["year_fraction"].values - max_yf_val)
).astype(np.float32)

lgb_params = get_lgb_hedonic_params()
cb_params = get_catboost_hedonic_params()
xgb_params = get_xgboost_hedonic_params()

# 1. Train Leaf-Wise LightGBM
print("Training leaf-wise LightGBM...")
lgb_tr = lgb.Dataset(
    val_train_feat[feature_cols],
    label=val_train_feat["residual"],
    weight=val_weights,
    categorical_feature=cat_feature_cols,
)
lgb_va = lgb.Dataset(
    val_holdout_feat[feature_cols],
    label=val_holdout_feat["residual"],
    reference=lgb_tr,
    categorical_feature=cat_feature_cols,
)
val_lgb = lgb.train(
    lgb_params,
    lgb_tr,
    num_boost_round=lgb_params["n_estimators"],
    valid_sets=[lgb_va],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
)
best_iter_lgb = val_lgb.best_iteration
val_pred_lgb = val_lgb.predict(
    val_holdout_feat[feature_cols], num_iteration=best_iter_lgb
).astype(np.float32)
del lgb_tr, lgb_va, val_lgb
gc.collect()

lgb_score = compute_log10_rmse(
    val_holdout_feat["price"].values,
    np.clip(10.0 ** (val_pred_lgb + val_trend), 1.0, None),
)
print(
    f"Holdout 2016 LightGBM RMSE: {lgb_score:.5f} (best iteration: {best_iter_lgb})"
)

# 2. Train Symmetric Oblivious CatBoost
print("Training symmetric oblivious CatBoost...")
val_cb = CatBoostRegressor(**cb_params, early_stopping_rounds=30)
try:
    val_cb.fit(
        val_train_feat[feature_cols],
        val_train_feat["residual"],
        sample_weight=val_weights,
        eval_set=(val_holdout_feat[feature_cols], val_holdout_feat["residual"]),
        verbose=False,
    )
except Exception as e:
    print(f"CatBoost GPU fallback to CPU: {e}")
    cb_params["task_type"] = "CPU"
    cb_params["thread_count"] = -1
    val_cb = CatBoostRegressor(**cb_params, early_stopping_rounds=30)
    val_cb.fit(
        val_train_feat[feature_cols],
        val_train_feat["residual"],
        sample_weight=val_weights,
        eval_set=(val_holdout_feat[feature_cols], val_holdout_feat["residual"]),
        verbose=False,
    )

cb_best = val_cb.get_best_iteration()
best_iter_cb = cb_best if (cb_best and cb_best > 0) else cb_params["iterations"]
val_pred_cb = val_cb.predict(val_holdout_feat[feature_cols]).astype(np.float32)
del val_cb
gc.collect()

cb_score = compute_log10_rmse(
    val_holdout_feat["price"].values,
    np.clip(10.0 ** (val_pred_cb + val_trend), 1.0, None),
)
print(
    f"Holdout 2016 CatBoost RMSE: {cb_score:.5f} (best iteration: {best_iter_cb})"
)

# 3. Train Depth-Wise Histogram XGBoost
print("Training depth-wise hist XGBoost...")
try:
    val_xgb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=30)
    val_xgb.fit(
        val_train_feat[feature_cols],
        val_train_feat["residual"],
        sample_weight=val_weights,
        eval_set=[(val_holdout_feat[feature_cols], val_holdout_feat["residual"])],
        verbose=False,
    )
except Exception as e:
    print(f"XGBoost GPU error: {e}, falling back to CPU hist...")
    xgb_params_cpu = xgb_params.copy()
    xgb_params_cpu.pop("device", None)
    xgb_params_cpu["tree_method"] = "hist"
    val_xgb = xgb.XGBRegressor(**xgb_params_cpu, early_stopping_rounds=30)
    val_xgb.fit(
        val_train_feat[feature_cols],
        val_train_feat["residual"],
        sample_weight=val_weights,
        eval_set=[(val_holdout_feat[feature_cols], val_holdout_feat["residual"])],
        verbose=False,
    )

xgb_best = getattr(val_xgb, "best_iteration", None)
best_iter_xgb = xgb_best if (xgb_best and xgb_best > 0) else xgb_params["n_estimators"]
val_pred_xgb = val_xgb.predict(val_holdout_feat[feature_cols]).astype(np.float32)
del val_xgb
gc.collect()

xgb_score = compute_log10_rmse(
    val_holdout_feat["price"].values,
    np.clip(10.0 ** (val_pred_xgb + val_trend), 1.0, None),
)
print(
    f"Holdout 2016 XGBoost RMSE: {xgb_score:.5f} (best iteration: {best_iter_xgb})"
)

# 4. Calibrate Non-Negative Ensemble Weights via SLSQP
print("--- Calibrating Tri-GBDT Ensemble Weights via SLSQP on 2016 Holdout ---")
y_true_holdout = val_holdout_feat["price"].values.astype(np.float64)
val_trend_64 = val_trend.astype(np.float64)
pred_matrix = np.column_stack(
    [
        val_pred_lgb.astype(np.float64),
        val_pred_cb.astype(np.float64),
        val_pred_xgb.astype(np.float64),
    ]
)


def holdout_ensemble_obj(w):
    w_norm = w / max(np.sum(w), 1e-8)
    blended_res = np.sum(pred_matrix * w_norm, axis=1)
    blended_price = np.clip(10.0 ** (blended_res + val_trend_64), 1.0, None)
    return compute_log10_rmse(y_true_holdout, blended_price)


res_opt = minimize(
    holdout_ensemble_obj,
    x0=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    method="SLSQP",
)
opt_w = res_opt.x / np.sum(res_opt.x)
w_lgb, w_cb, w_xgb = float(opt_w[0]), float(opt_w[1]), float(opt_w[2])
print(
    f"Optimal Tri-GBDT Weights: LGB={w_lgb:.4f}, CB={w_cb:.4f}, XGB={w_xgb:.4f}"
)

val_ensemble_res = (
    w_lgb * val_pred_lgb + w_cb * val_pred_cb + w_xgb * val_pred_xgb
)
val_ensemble_raw = np.clip(10.0 ** (val_ensemble_res + val_trend), 1.0, None)
holdout_score = compute_log10_rmse(y_true_holdout, val_ensemble_raw)
print(f"Holdout 2016 Tri-GBDT Ensemble RMSE: {holdout_score:.5f}")

del val_train_feat, val_holdout_feat, pred_matrix
gc.collect()

# --- 4. Production Retraining on Full Modern Regime (2012-2016) ---
print("--- Production Retraining on Complete Modern Regime (2012-2016) ---")
max_yf_prod = float(prod_train_feat["year_fraction"].max())
prod_weights = np.exp(
    gamma * (prod_train_feat["year_fraction"].values - max_yf_prod)
).astype(np.float32)

# Retrain LightGBM
prod_lgb_round = max(500, int(best_iter_lgb * 1.15))
print(f"Retraining LightGBM for {prod_lgb_round} rounds...")
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
    num_boost_round=prod_lgb_round,
    callbacks=[lgb.log_evaluation(period=0)],
)
test_lgb_res = prod_lgb_model.predict(prod_test_feat[feature_cols]).astype(
    np.float32
)
del full_train_lgb, prod_lgb_model
gc.collect()

# Retrain CatBoost
prod_cb_round = max(500, int(best_iter_cb * 1.15))
print(f"Retraining CatBoost for {prod_cb_round} iterations...")
prod_cb_params = cb_params.copy()
prod_cb_params["iterations"] = prod_cb_round
prod_cb_model = CatBoostRegressor(**prod_cb_params)
try:
    prod_cb_model.fit(
        prod_train_feat[feature_cols],
        prod_train_feat["residual"],
        sample_weight=prod_weights,
        verbose=False,
    )
except Exception as e:
    print(f"CatBoost production fallback: {e}")
    prod_cb_params["task_type"] = "CPU"
    prod_cb_params["thread_count"] = -1
    prod_cb_model = CatBoostRegressor(**prod_cb_params)
    prod_cb_model.fit(
        prod_train_feat[feature_cols],
        prod_train_feat["residual"],
        sample_weight=prod_weights,
        verbose=False,
    )
test_cat_res = prod_cb_model.predict(prod_test_feat[feature_cols]).astype(
    np.float32
)
del prod_cb_model
gc.collect()

# Retrain XGBoost
prod_xgb_round = max(500, int(best_iter_xgb * 1.15))
print(f"Retraining XGBoost for {prod_xgb_round} trees...")
prod_xgb_params = xgb_params.copy()
prod_xgb_params["n_estimators"] = prod_xgb_round
try:
    prod_xgb_model = xgb.XGBRegressor(**prod_xgb_params)
    prod_xgb_model.fit(
        prod_train_feat[feature_cols],
        prod_train_feat["residual"],
        sample_weight=prod_weights,
        verbose=False,
    )
except Exception as e:
    print(f"XGBoost production fallback: {e}")
    prod_xgb_params_cpu = prod_xgb_params.copy()
    prod_xgb_params_cpu.pop("device", None)
    prod_xgb_params_cpu["tree_method"] = "hist"
    prod_xgb_model = xgb.XGBRegressor(**prod_xgb_params_cpu)
    prod_xgb_model.fit(
        prod_train_feat[feature_cols],
        prod_train_feat["residual"],
        sample_weight=prod_weights,
        verbose=False,
    )
test_xgb_res = prod_xgb_model.predict(prod_test_feat[feature_cols]).astype(
    np.float32
)
del prod_xgb_model
gc.collect()

# --- 5. Tri-GBDT Ensemble Inference & Submission Export ---
print("--- Computing Final Ensemble Predictions for 2017 Test Horizon ---")
test_final_res = (
    w_lgb * test_lgb_res + w_cb * test_cat_res + w_xgb * test_xgb_res
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