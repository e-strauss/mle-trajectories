import gc
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from catboost import CatBoostRegressor
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb

# ==============================================================================
# Setup directories
# ==============================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==============================================================================
# Step 1: Data Processing and Feature Engineering
# ==============================================================================
print("Starting data processing and feature engineering...")

train_path = "./input/train.parquet"
test_path = "./input/test.csv"

df_train = pd.read_parquet(train_path)
df_test = pd.read_csv(test_path)
print(f"Loaded train: {df_train.shape[0]:,} rows | test: {df_test.shape[0]:,} rows")


def clean_and_build_composites(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "town",
        "district",
        "county",
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
    ]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    for col in ["town", "district", "county"]:
        df[col] = df[col].replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})

    df["district_county"] = df["district"] + "__" + df["county"]
    df["town_district"] = df["town"] + "__" + df["district"]
    df["town_district_county"] = (
        df["town"] + "__" + df["district"] + "__" + df["county"]
    )

    df["type_new_build"] = df["property_type"] + "_" + df["is_new_build"]
    df["type_tenure"] = df["property_type"] + "_" + df["tenure"]
    df["type_sale_cat"] = df["property_type"] + "_" + df["sale_category"]

    df["dist_new_build"] = df["district"] + "___" + df["is_new_build"]
    df["dist_sale_cat"] = df["district"] + "___" + df["sale_category"]
    return df


df_train = clean_and_build_composites(df_train)
df_test = clean_and_build_composites(df_test)


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    date_dt = pd.to_datetime(df["date"])
    df["year"] = date_dt.dt.year.astype(np.int16)
    df["month"] = date_dt.dt.month.astype(np.int8)
    df["day"] = date_dt.dt.day.astype(np.int8)
    df["dayofweek"] = date_dt.dt.dayofweek.astype(np.int8)
    df["quarter"] = date_dt.dt.quarter.astype(np.int8)
    df["dayofyear"] = date_dt.dt.dayofyear.astype(np.int16)

    df["is_friday"] = (df["dayofweek"] == 4).astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["is_month_end"] = date_dt.dt.is_month_end.astype(np.int8)

    df["time_elapsed"] = (
        (date_dt - pd.Timestamp("1995-01-01")).dt.total_seconds() / (365.25 * 86400.0)
    ).astype(np.float32)

    month_rad = 2.0 * np.pi * (df["month"] - 1) / 12.0
    df["month_sin"] = np.sin(month_rad).astype(np.float32)
    df["month_cos"] = np.cos(month_rad).astype(np.float32)

    doy_rad = 2.0 * np.pi * (df["dayofyear"] - 1) / 365.25
    df["doy_sin"] = np.sin(doy_rad).astype(np.float32)
    df["doy_cos"] = np.cos(doy_rad).astype(np.float32)
    return df


df_train = extract_temporal_features(df_train)
df_test = extract_temporal_features(df_test)

# Ordinal encodings
property_type_map = {"D": 4, "S": 3, "T": 2, "F": 1, "O": 0}
tenure_map = {"F": 1, "L": 0, "U": -1}
new_build_map = {"Y": 1, "N": 0}
sale_cat_map = {"A": 0, "B": 1}

for df in [df_train, df_test]:
    df["ptype_ord"] = (
        df["property_type"].map(property_type_map).fillna(-1).astype(np.int8)
    )
    df["tenure_ord"] = df["tenure"].map(tenure_map).fillna(-1).astype(np.int8)
    df["new_build_ord"] = (
        df["is_new_build"].map(new_build_map).fillna(-1).astype(np.int8)
    )
    df["sale_cat_ord"] = (
        df["sale_category"].map(sale_cat_map).fillna(-1).astype(np.int8)
    )

# Frequency encodings strictly from training history
freq_cols = [
    "county",
    "district",
    "town",
    "district_county",
    "town_district_county",
    "property_type",
]
for col in freq_cols:
    freq_map = df_train[col].value_counts()
    df_train[f"freq_{col}"] = np.log1p(
        df_train[col].map(freq_map).fillna(0).values
    ).astype(np.float32)
    df_test[f"freq_{col}"] = np.log1p(
        df_test[col].map(freq_map).fillna(0).values
    ).astype(np.float32)

# Target preparation on log10 scale
df_train["log10_price"] = np.log10(np.clip(df_train["price"].values, 1.0, None)).astype(
    np.float32
)
clean_train_mask = (df_train["price"] >= 500) & (df_train["price"] <= 50_000_000)


class MacroTrendModel:
    """
    Fits a regularized hierarchical Empirical Bayes linear macro trend on time_elapsed across
    National, County, and District tiers with property-type slope offsets strictly on training data.
    """

    def __init__(
        self,
        reg_county: float = 50.0,
        m_county: float = 30.0,
        reg_district: float = 100.0,
        m_district: float = 50.0,
        reg_ptype: float = 200.0,
        m_ptype: float = 100.0,
        **kwargs,
    ):
        self.reg_county = reg_county
        self.m_county = m_county
        self.reg_district = reg_district
        self.m_district = m_district
        self.reg_ptype = reg_ptype
        self.m_ptype = m_ptype
        self.global_intercept = 0.0
        self.global_slope = 0.0
        self.county_slopes = {}
        self.county_intercepts = {}
        self.district_slopes = {}
        self.district_intercepts = {}
        self.ptype_slopes = {}
        self.ptype_intercepts = {}

    def fit(
        self,
        df: pd.DataFrame,
        time_col: str = "time_elapsed",
        target_col: str = "log10_price",
        district_col: str = "district",
        county_col: str = "county",
        ptype_col: str = "property_type",
        **kwargs,
    ):
        t = df[time_col].to_numpy(dtype=np.float64)
        y = df[target_col].to_numpy(dtype=np.float64)

        t_mean = float(np.mean(t))
        y_mean = float(np.mean(y))

        t_diff = t - t_mean
        y_diff = y - y_mean

        s_xx = float(np.sum(t_diff**2))
        s_xy = float(np.sum(t_diff * y_diff))

        self.global_slope = float(s_xy / (s_xx + 1e-8))
        self.global_intercept = float(y_mean - self.global_slope * t_mean)

        # Tier 2: County slopes regularized toward National
        df_c = pd.DataFrame(
            {
                county_col: df[county_col].values,
                "t": t,
                "y": y,
                "t2": t**2,
                "ty": t * y,
            }
        )
        agg_c = (
            df_c.groupby(county_col)
            .agg(
                n=("t", "count"),
                t_sum=("t", "sum"),
                y_sum=("y", "sum"),
                t2_sum=("t2", "sum"),
                ty_sum=("ty", "sum"),
            )
            .reset_index()
        )

        self.county_slopes = {}
        self.county_intercepts = {}

        for _, row in agg_c.iterrows():
            grp = row[county_col]
            n_c = float(row["n"])
            t_bar = float(row["t_sum"] / n_c)
            y_bar = float(row["y_sum"] / n_c)
            s_xx_c = float(row["t2_sum"] - n_c * (t_bar**2))
            s_xy_c = float(row["ty_sum"] - n_c * t_bar * y_bar)

            slope_c = (s_xy_c + self.reg_county * self.global_slope) / (
                s_xx_c + self.reg_county
            )
            nat_y_at_tbar = self.global_intercept + self.global_slope * t_bar
            smoothed_y_bar = (n_c * y_bar + self.m_county * nat_y_at_tbar) / (
                n_c + self.m_county
            )
            intercept_c = smoothed_y_bar - slope_c * t_bar

            self.county_slopes[grp] = float(slope_c)
            self.county_intercepts[grp] = float(intercept_c)

        # Tier 3: District slopes regularized toward County
        df_d = pd.DataFrame(
            {
                district_col: df[district_col].values,
                county_col: df[county_col].values,
                "t": t,
                "y": y,
                "t2": t**2,
                "ty": t * y,
            }
        )
        agg_d = (
            df_d.groupby([district_col, county_col])
            .agg(
                n=("t", "count"),
                t_sum=("t", "sum"),
                y_sum=("y", "sum"),
                t2_sum=("t2", "sum"),
                ty_sum=("ty", "sum"),
            )
            .reset_index()
            .sort_values("n", ascending=True)
        )

        self.district_slopes = {}
        self.district_intercepts = {}

        for _, row in agg_d.iterrows():
            d_grp = row[district_col]
            c_grp = row[county_col]
            n_d = float(row["n"])
            t_bar = float(row["t_sum"] / n_d)
            y_bar = float(row["y_sum"] / n_d)
            s_xx_d = float(row["t2_sum"] - n_d * (t_bar**2))
            s_xy_d = float(row["ty_sum"] - n_d * t_bar * y_bar)

            c_slope = self.county_slopes.get(c_grp, self.global_slope)
            c_intercept = self.county_intercepts.get(c_grp, self.global_intercept)

            slope_d = (s_xy_d + self.reg_district * c_slope) / (
                s_xx_d + self.reg_district
            )
            county_y_at_tbar = c_intercept + c_slope * t_bar
            smoothed_y_bar = (n_d * y_bar + self.m_district * county_y_at_tbar) / (
                n_d + self.m_district
            )
            intercept_d = smoothed_y_bar - slope_d * t_bar

            self.district_slopes[d_grp] = float(slope_d)
            self.district_intercepts[d_grp] = float(intercept_d)

        # Tier 4: Property-type slope adjustments on district trend residuals
        c_slopes_vec = (
            df[county_col]
            .map(self.county_slopes)
            .fillna(self.global_slope)
            .to_numpy(dtype=np.float64)
        )
        c_intercepts_vec = (
            df[county_col]
            .map(self.county_intercepts)
            .fillna(self.global_intercept)
            .to_numpy(dtype=np.float64)
        )

        d_slopes_vec = (
            df[district_col]
            .map(self.district_slopes)
            .to_numpy(dtype=np.float64)
        )
        d_intercepts_vec = (
            df[district_col]
            .map(self.district_intercepts)
            .to_numpy(dtype=np.float64)
        )

        nan_m = np.isnan(d_slopes_vec)
        d_slopes_vec[nan_m] = c_slopes_vec[nan_m]
        d_intercepts_vec[nan_m] = c_intercepts_vec[nan_m]

        pred_dist = d_intercepts_vec + d_slopes_vec * t
        res = y - pred_dist

        df_p = pd.DataFrame(
            {
                ptype_col: df[ptype_col].values,
                "t": t,
                "r": res,
                "t2": t**2,
                "tr": t * res,
            }
        )
        agg_p = (
            df_p.groupby(ptype_col)
            .agg(
                n=("t", "count"),
                t_sum=("t", "sum"),
                r_sum=("r", "sum"),
                t2_sum=("t2", "sum"),
                tr_sum=("tr", "sum"),
            )
            .reset_index()
        )

        self.ptype_slopes = {}
        self.ptype_intercepts = {}

        for _, row in agg_p.iterrows():
            p_grp = row[ptype_col]
            n_p = float(row["n"])
            t_bar = float(row["t_sum"] / n_p)
            r_bar = float(row["r_sum"] / n_p)
            s_xx_p = float(row["t2_sum"] - n_p * (t_bar**2))
            s_xr_p = float(row["tr_sum"] - n_p * t_bar * r_bar)

            slope_offset = s_xr_p / (s_xx_p + self.reg_ptype)
            smoothed_r_bar = (n_p * r_bar) / (n_p + self.m_ptype)
            intercept_offset = smoothed_r_bar - slope_offset * t_bar

            self.ptype_slopes[p_grp] = float(slope_offset)
            self.ptype_intercepts[p_grp] = float(intercept_offset)

        return self

    def predict(
        self,
        df: pd.DataFrame,
        time_col: str = "time_elapsed",
        district_col: str = "district",
        county_col: str = "county",
        ptype_col: str = "property_type",
        **kwargs,
    ) -> np.ndarray:
        t = df[time_col].to_numpy(dtype=np.float64)
        c_series = df[county_col]
        d_series = df[district_col]
        p_series = df[ptype_col]

        c_slopes = (
            c_series.map(self.county_slopes)
            .fillna(self.global_slope)
            .to_numpy(dtype=np.float64)
        )
        c_intercepts = (
            c_series.map(self.county_intercepts)
            .fillna(self.global_intercept)
            .to_numpy(dtype=np.float64)
        )

        d_slopes = d_series.map(self.district_slopes).to_numpy(dtype=np.float64)
        d_intercepts = d_series.map(self.district_intercepts).to_numpy(dtype=np.float64)

        nan_m = np.isnan(d_slopes)
        d_slopes[nan_m] = c_slopes[nan_m]
        d_intercepts[nan_m] = c_intercepts[nan_m]

        p_slopes = (
            p_series.map(self.ptype_slopes)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        p_intercepts = (
            p_series.map(self.ptype_intercepts)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        preds = (d_intercepts + p_intercepts) + (d_slopes + p_slopes) * t
        return preds.astype(np.float32)


class HierarchicalEBEncoder:
    """
    Computes smoothed target encodings with Empirical Bayes shrinkage across nested hierarchies:
    town -> district -> county -> global, district x property_type, town x property_type,
    district x is_new_build, property_type x is_new_build, and district x sale_category interactions.
    """

    def __init__(
        self,
        m_global: float = 50.0,
        m_county: float = 30.0,
        m_district: float = 20.0,
        m_dist_ptype: float = 15.0,
        m_town_ptype: float = 15.0,
        m_type_tenure: float = 20.0,
        m_dist_nb: float = 15.0,
        m_type_nb: float = 20.0,
        m_dist_sc: float = 15.0,
    ):
        self.m_global = m_global
        self.m_county = m_county
        self.m_district = m_district
        self.m_dist_ptype = m_dist_ptype
        self.m_town_ptype = m_town_ptype
        self.m_type_tenure = m_type_tenure
        self.m_dist_nb = m_dist_nb
        self.m_type_nb = m_type_nb
        self.m_dist_sc = m_dist_sc

    def fit(self, df_ref: pd.DataFrame, target_col: str = "price_residual"):
        self.global_mean = float(df_ref[target_col].mean())

        # Level 1: County smoothed towards global
        c_stats = df_ref.groupby("county")[target_col].agg(["count", "sum"])
        self.county_te = (
            (c_stats["sum"] + self.m_global * self.global_mean)
            / (c_stats["count"] + self.m_global)
        ).to_dict()

        # Level 2: District smoothed towards County
        d_stats = (
            df_ref.groupby(["district", "county"])[target_col]
            .agg(["count", "sum"])
            .reset_index()
        )
        d_stats["county_prior"] = (
            d_stats["county"].map(self.county_te).fillna(self.global_mean)
        )
        d_stats["te"] = (d_stats["sum"] + self.m_county * d_stats["county_prior"]) / (
            d_stats["count"] + self.m_county
        )
        d_stats = d_stats.sort_values("count", ascending=True)
        self.district_te = dict(zip(d_stats["district"], d_stats["te"]))

        # Level 3: Town smoothed towards District
        t_stats = (
            df_ref.groupby(["town", "district"])[target_col]
            .agg(["count", "sum"])
            .reset_index()
        )
        t_stats["dist_prior"] = (
            t_stats["district"].map(self.district_te).fillna(self.global_mean)
        )
        t_stats["te"] = (t_stats["sum"] + self.m_district * t_stats["dist_prior"]) / (
            t_stats["count"] + self.m_district
        )
        t_stats = t_stats.sort_values("count", ascending=True)
        self.town_te = dict(zip(t_stats["town"], t_stats["te"]))

        # Level 4: District x Property Type smoothed towards District
        dp_stats = (
            df_ref.groupby(["district", "property_type"])[target_col]
            .agg(["count", "sum"])
            .reset_index()
        )
        dp_stats["dist_prior"] = (
            dp_stats["district"].map(self.district_te).fillna(self.global_mean)
        )
        dp_stats["te"] = (
            dp_stats["sum"] + self.m_dist_ptype * dp_stats["dist_prior"]
        ) / (dp_stats["count"] + self.m_dist_ptype)
        dp_stats["key"] = dp_stats["district"] + "___" + dp_stats["property_type"]
        dp_stats = dp_stats.sort_values("count", ascending=True)
        self.dist_ptype_te = dict(zip(dp_stats["key"], dp_stats["te"]))

        # Level 4.5: Town x Property Type smoothed towards District x Property Type
        tp_stats = (
            df_ref.groupby(["town", "district", "property_type"])[target_col]
            .agg(["count", "sum"])
            .reset_index()
        )
        tp_stats["dp_key"] = (
            tp_stats["district"] + "___" + tp_stats["property_type"]
        )
        tp_stats["dp_prior"] = (
            tp_stats["dp_key"].map(self.dist_ptype_te).fillna(self.global_mean)
        )
        tp_stats["te"] = (
            tp_stats["sum"] + self.m_town_ptype * tp_stats["dp_prior"]
        ) / (tp_stats["count"] + self.m_town_ptype)
        tp_stats["key"] = tp_stats["town"] + "___" + tp_stats["property_type"]
        tp_stats = tp_stats.sort_values("count", ascending=True)
        self.town_ptype_te = dict(zip(tp_stats["key"], tp_stats["te"]))

        # Level 5: Type x Tenure smoothed towards global
        tt_stats = df_ref.groupby("type_tenure")[target_col].agg(["count", "sum"])
        self.type_tenure_te = (
            (tt_stats["sum"] + self.m_type_tenure * self.global_mean)
            / (tt_stats["count"] + self.m_type_tenure)
        ).to_dict()

        # Level 6: District x is_new_build smoothed towards District
        dnb_stats = (
            df_ref.groupby(["district", "is_new_build"])[target_col]
            .agg(["count", "sum"])
            .reset_index()
        )
        dnb_stats["dist_prior"] = (
            dnb_stats["district"].map(self.district_te).fillna(self.global_mean)
        )
        dnb_stats["te"] = (
            dnb_stats["sum"] + self.m_dist_nb * dnb_stats["dist_prior"]
        ) / (dnb_stats["count"] + self.m_dist_nb)
        dnb_stats["key"] = dnb_stats["district"] + "___" + dnb_stats["is_new_build"]
        dnb_stats = dnb_stats.sort_values("count", ascending=True)
        self.dist_nb_te = dict(zip(dnb_stats["key"], dnb_stats["te"]))

        # Level 7: Property Type x is_new_build smoothed towards global
        tnb_stats = df_ref.groupby("type_new_build")[target_col].agg(["count", "sum"])
        self.type_nb_te = (
            (tnb_stats["sum"] + self.m_type_nb * self.global_mean)
            / (tnb_stats["count"] + self.m_type_nb)
        ).to_dict()

        # Level 8: District x sale_category smoothed towards District
        dsc_stats = (
            df_ref.groupby(["district", "sale_category"])[target_col]
            .agg(["count", "sum"])
            .reset_index()
        )
        dsc_stats["dist_prior"] = (
            dsc_stats["district"].map(self.district_te).fillna(self.global_mean)
        )
        dsc_stats["te"] = (
            dsc_stats["sum"] + self.m_dist_sc * dsc_stats["dist_prior"]
        ) / (dsc_stats["count"] + self.m_dist_sc)
        dsc_stats["key"] = dsc_stats["district"] + "___" + dsc_stats["sale_category"]
        dsc_stats = dsc_stats.sort_values("count", ascending=True)
        self.dist_sc_te = dict(zip(dsc_stats["key"], dsc_stats["te"]))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        c_mapped = df["county"].map(self.county_te).fillna(self.global_mean)
        d_mapped = df["district"].map(self.district_te).fillna(c_mapped)
        t_mapped = df["town"].map(self.town_te).fillna(d_mapped)

        dp_key = df["district"] + "___" + df["property_type"]
        dp_mapped = dp_key.map(self.dist_ptype_te).fillna(d_mapped)

        tp_key = df["town"] + "___" + df["property_type"]
        tp_mapped = tp_key.map(self.town_ptype_te).fillna(dp_mapped)

        tt_mapped = df["type_tenure"].map(self.type_tenure_te).fillna(self.global_mean)

        dnb_key = df["district"] + "___" + df["is_new_build"]
        dnb_mapped = dnb_key.map(self.dist_nb_te).fillna(d_mapped)

        tnb_mapped = df["type_new_build"].map(self.type_nb_te).fillna(self.global_mean)

        dsc_key = df["district"] + "___" + df["sale_category"]
        dsc_mapped = dsc_key.map(self.dist_sc_te).fillna(d_mapped)

        out["te_county"] = c_mapped.astype(np.float32).values
        out["te_district"] = d_mapped.astype(np.float32).values
        out["te_town"] = t_mapped.astype(np.float32).values
        out["te_dist_ptype"] = dp_mapped.astype(np.float32).values
        out["te_town_ptype"] = tp_mapped.astype(np.float32).values
        out["te_type_tenure"] = tt_mapped.astype(np.float32).values
        out["te_dist_new_build"] = dnb_mapped.astype(np.float32).values
        out["te_type_new_build"] = tnb_mapped.astype(np.float32).values
        out["te_dist_sale_cat"] = dsc_mapped.astype(np.float32).values
        return out


print(
    "Fitting macro de-trending model and leakage-free hierarchical target encodings on residuals..."
)

val_mask = df_train["year"] == 2016
train_mask = (df_train["year"] >= 2011) & (df_train["year"] <= 2015)
full_train_mask = (df_train["year"] >= 2011) & (df_train["year"] <= 2016)

# Extract validation and production train splits
df_train_sub = df_train[train_mask].copy()
clean_sub = clean_train_mask.loc[df_train_sub.index].values

df_val = df_train[val_mask].copy()

df_full_train = df_train[full_train_mask].copy()
clean_full = clean_train_mask.loc[df_full_train.index].values

df_test_proc = df_test.copy()

# Fit validation macro-trend model strictly on 2011-2015 training partition
macro_trend_val = MacroTrendModel().fit(df_train_sub[clean_sub])
train_sub_trend = macro_trend_val.predict(df_train_sub)
df_train_sub["price_residual"] = (
    df_train_sub["log10_price"] - train_sub_trend
).astype(np.float32)

val_trend = macro_trend_val.predict(df_val)
df_val["price_residual"] = (df_val["log10_price"] - val_trend).astype(
    np.float32
)

# Reference sets for validation set encodings
ref_val_clean = df_train_sub[clean_sub]
ref_val_recent = ref_val_clean[ref_val_clean["year"] == 2015]

eb_val_modern = HierarchicalEBEncoder().fit(
    ref_val_clean, target_col="price_residual"
)
eb_val_recent = HierarchicalEBEncoder().fit(
    ref_val_recent, target_col="price_residual"
)

# Validation set transformation
te_val_modern = eb_val_modern.transform(df_val)
te_val_recent = eb_val_recent.transform(df_val)

for c in te_val_modern.columns:
    df_val[f"{c}_modern"] = te_val_modern[c].values
    df_val[f"{c}_recent"] = te_val_recent[c].values
df_val["te_district_momentum"] = (
    df_val["te_district_recent"] - df_val["te_district_modern"]
).astype(np.float32)

# Fit production macro-trend model strictly on 2011-2016 full training partition
macro_trend_prod = MacroTrendModel().fit(df_full_train[clean_full])
full_train_trend = macro_trend_prod.predict(df_full_train)
df_full_train["price_residual"] = (
    df_full_train["log10_price"] - full_train_trend
).astype(np.float32)

test_trend = macro_trend_prod.predict(df_test_proc)

ref_test_clean = df_full_train[clean_full]
ref_test_recent = ref_test_clean[ref_test_clean["year"] == 2016]

eb_test_modern = HierarchicalEBEncoder().fit(
    ref_test_clean, target_col="price_residual"
)
eb_test_recent = HierarchicalEBEncoder().fit(
    ref_test_recent, target_col="price_residual"
)

# Test set transformation
te_test_modern = eb_test_modern.transform(df_test_proc)
te_test_recent = eb_test_recent.transform(df_test_proc)

for c in te_test_modern.columns:
    df_test_proc[f"{c}_modern"] = te_test_modern[c].values
    df_test_proc[f"{c}_recent"] = te_test_recent[c].values
df_test_proc["te_district_momentum"] = (
    df_test_proc["te_district_recent"] - df_test_proc["te_district_modern"]
).astype(np.float32)

# Out-of-fold target encodings on residuals for validation training (2011-2015)
oof_cols = [
    "te_county",
    "te_district",
    "te_town",
    "te_dist_ptype",
    "te_town_ptype",
    "te_type_tenure",
    "te_dist_new_build",
    "te_type_new_build",
    "te_dist_sale_cat",
]

for c in oof_cols:
    df_train_sub[f"{c}_modern"] = np.float32(0.0)
    df_train_sub[f"{c}_recent"] = np.float32(0.0)

np.random.seed(42)
fold_assignments = np.random.randint(0, 5, size=len(df_train_sub))

for f in range(5):
    f_val_mask = fold_assignments == f
    f_trn_mask = (~f_val_mask) & clean_sub

    trn_fold_clean = df_train_sub[f_trn_mask]
    trn_fold_recent = trn_fold_clean[trn_fold_clean["year"] == 2015]

    eb_fold_modern = HierarchicalEBEncoder().fit(
        trn_fold_clean, target_col="price_residual"
    )
    eb_fold_recent = HierarchicalEBEncoder().fit(
        trn_fold_recent if len(trn_fold_recent) > 0 else trn_fold_clean,
        target_col="price_residual",
    )

    val_fold_df = df_train_sub[f_val_mask]
    trans_m = eb_fold_modern.transform(val_fold_df)
    trans_r = eb_fold_recent.transform(val_fold_df)

    for c in oof_cols:
        df_train_sub.loc[val_fold_df.index, f"{c}_modern"] = trans_m[c].values
        df_train_sub.loc[val_fold_df.index, f"{c}_recent"] = trans_r[c].values

df_train_sub["te_district_momentum"] = (
    df_train_sub["te_district_recent"] - df_train_sub["te_district_modern"]
).astype(np.float32)

# Out-of-fold target encodings on residuals for full training (2011-2016)
for c in oof_cols:
    df_full_train[f"{c}_modern"] = np.float32(0.0)
    df_full_train[f"{c}_recent"] = np.float32(0.0)

fold_assignments_full = np.random.randint(0, 5, size=len(df_full_train))

for f in range(5):
    f_val_mask = fold_assignments_full == f
    f_trn_mask = (~f_val_mask) & clean_full

    trn_fold_clean = df_full_train[f_trn_mask]
    trn_fold_recent = trn_fold_clean[trn_fold_clean["year"] == 2016]

    eb_fold_m = HierarchicalEBEncoder().fit(
        trn_fold_clean, target_col="price_residual"
    )
    eb_fold_r = HierarchicalEBEncoder().fit(
        trn_fold_recent if len(trn_fold_recent) > 0 else trn_fold_clean,
        target_col="price_residual",
    )

    val_fold_df = df_full_train[f_val_mask]
    trans_m = eb_fold_m.transform(val_fold_df)
    trans_r = eb_fold_r.transform(val_fold_df)

    for c in oof_cols:
        df_full_train.loc[val_fold_df.index, f"{c}_modern"] = trans_m[c].values
        df_full_train.loc[val_fold_df.index, f"{c}_recent"] = trans_r[c].values

df_full_train["te_district_momentum"] = (
    df_full_train["te_district_recent"] - df_full_train["te_district_modern"]
).astype(np.float32)

# Free raw 22M train dataframe to release memory
del df_train, df_test
gc.collect()

feature_cols = [
    "year",
    "month",
    "day",
    "dayofweek",
    "quarter",
    "dayofyear",
    "is_friday",
    "is_weekend",
    "is_month_end",
    "time_elapsed",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "ptype_ord",
    "tenure_ord",
    "new_build_ord",
    "sale_cat_ord",
    "freq_county",
    "freq_district",
    "freq_town",
    "freq_district_county",
    "freq_town_district_county",
    "freq_property_type",
    "te_county_modern",
    "te_district_modern",
    "te_town_modern",
    "te_dist_ptype_modern",
    "te_town_ptype_modern",
    "te_type_tenure_modern",
    "te_dist_new_build_modern",
    "te_type_new_build_modern",
    "te_dist_sale_cat_modern",
    "te_county_recent",
    "te_district_recent",
    "te_town_recent",
    "te_dist_ptype_recent",
    "te_town_ptype_recent",
    "te_type_tenure_recent",
    "te_dist_new_build_recent",
    "te_type_new_build_recent",
    "te_dist_sale_cat_recent",
    "te_district_momentum",
]

cat_cols = [
    "property_type",
    "is_new_build",
    "tenure",
    "sale_category",
    "district",
    "county",
    "town",
]

# Unify categorical levels across all splits
for col in cat_cols:
    unified_categories = (
        pd.concat([df_full_train[col], df_test_proc[col]], axis=0).dropna().unique()
    )
    df_train_sub[col] = pd.Categorical(df_train_sub[col], categories=unified_categories)
    df_val[col] = pd.Categorical(df_val[col], categories=unified_categories)
    df_full_train[col] = pd.Categorical(
        df_full_train[col], categories=unified_categories
    )
    df_test_proc[col] = pd.Categorical(df_test_proc[col], categories=unified_categories)

# Ensure no nulls in continuous feature sets
for df_obj in [df_train_sub, df_val, df_full_train, df_test_proc]:
    null_count = df_obj[feature_cols].isnull().sum().sum()
    if null_count > 0:
        df_obj[feature_cols] = df_obj[feature_cols].fillna(
            df_obj[feature_cols].median()
        )

all_features = feature_cols + cat_cols
print(f"Feature engineering completed. Active features: {len(all_features)}")


# ==============================================================================
# Step 2: Model Design (Architectures & Criteria)
# ==============================================================================
class ResidualDenseBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.act2(out + residual)
        return out


class HedonicEmbeddingNet(nn.Module):

    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        num_continuous_features: int,
        embedding_dim_factor: float = 1.6,
        max_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.cat_keys = sorted(list(cat_cardinalities.keys()))

        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = int(
                min(
                    max_embedding_dim,
                    max(4, math.ceil((card**0.35) * embedding_dim_factor)),
                )
            )
            self.embeddings[col] = nn.Embedding(
                num_embeddings=card + 1,
                embedding_dim=emb_dim,
                padding_idx=0,
            )
            total_emb_dim += emb_dim

        self.emb_dropout = nn.Dropout(dropout_rate)

        self.num_cont = num_continuous_features
        if self.num_cont > 0:
            self.cont_norm = nn.BatchNorm1d(num_continuous_features)
            self.cont_proj = nn.Linear(num_continuous_features, hidden_dim // 2)
            fusion_in_dim = total_emb_dim + (hidden_dim // 2)
        else:
            fusion_in_dim = total_emb_dim

        self.fusion_fc = nn.Linear(fusion_in_dim, hidden_dim)
        self.fusion_ln = nn.LayerNorm(hidden_dim)
        self.fusion_act = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_inputs: Dict[str, torch.Tensor],
        cont_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb_tensors = []
        for col in self.cat_keys:
            emb_tensors.append(self.embeddings[col](cat_inputs[col]))

        cat_repr = torch.cat(emb_tensors, dim=-1)
        cat_repr = self.emb_dropout(cat_repr)

        if self.num_cont > 0 and cont_inputs is not None:
            cont_normed = self.cont_norm(cont_inputs)
            cont_repr = F.silu(self.cont_proj(cont_normed))
            x = torch.cat([cat_repr, cont_repr], dim=-1)
        else:
            x = cat_repr

        x = self.fusion_fc(x)
        x = self.fusion_ln(x)
        x = self.fusion_act(x)

        for block in self.res_blocks:
            x = block(x)

        out = self.head(x).squeeze(-1)
        return out


class Log10RMSELoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((y_pred - y_true) ** 2)
        return torch.sqrt(mse + self.eps)


def build_lgbm_model(
    custom_params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 4000,
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)

    return lgb.LGBMRegressor(**default_params)


def build_catboost_model(
    custom_params: Optional[Dict[str, Any]] = None,
) -> CatBoostRegressor:
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 800,
        "learning_rate": 0.08,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "border_count": 64,
        "random_seed": 42,
        "verbose": 100,
        "thread_count": -1,
    }
    if custom_params:
        default_params.update(custom_params)

    return CatBoostRegressor(**default_params)


class EnsembleResidualPredictor:
    """
    Dual GBDT ensemble combining LightGBM (leaf-wise growth) and CatBoost (symmetric oblivious trees)
    to predict stationary price residuals.
    """

    def __init__(
        self,
        lgb_model: Any,
        cb_model: Any,
        lgb_weight: float = 0.5,
        cb_weight: float = 0.5,
    ):
        self.lgb_model = lgb_model
        self.cb_model = cb_model
        self.lgb_weight = lgb_weight
        self.cb_weight = cb_weight

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds_lgb = self.lgb_model.predict(X).astype(np.float64)
        preds_cb = self.cb_model.predict(X).astype(np.float64)
        blended = self.lgb_weight * preds_lgb + self.cb_weight * preds_cb
        return blended.astype(np.float32)


# ==============================================================================
# Step 3: Training, Evaluation & Submission Generation
# ==============================================================================
print("Starting training and evaluation pipeline...")

# Clean outlier filter applied to training sets to eliminate nominal non-market transfers (£1-£100)
X_train = df_train_sub.loc[clean_sub, all_features]
y_train = df_train_sub.loc[clean_sub, "price_residual"].values

# Validation set remains unconstrained on full holdout to match evaluation conditions
X_val = df_val[all_features]
y_val = df_val["price_residual"].values
val_true_price = df_val["price"].values

# Exponential recency weighting prioritizing modern market regime
gamma = 0.15
sample_weight_train = np.exp(
    gamma * (df_train_sub.loc[clean_sub, "time_elapsed"].values - df_train_sub["time_elapsed"].max())
).astype(np.float32)

print(
    f"Filtered validation train set: {len(X_train):,} samples | Val set: {len(X_val):,} samples"
)

# Train LightGBM with early stopping on validation residuals
print("Training LightGBM model...")
val_lgb = build_lgbm_model()
val_lgb.fit(
    X_train,
    y_train,
    sample_weight=sample_weight_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
)
lgb_best_iteration = val_lgb.best_iteration_ if val_lgb.best_iteration_ > 0 else 3000
print(f"Validation LightGBM best iteration: {lgb_best_iteration}")

joblib.dump(val_lgb, "./working/lgbm_val_model.pkl")
loaded_val_lgb = joblib.load("./working/lgbm_val_model.pkl")

# Train CatBoost with early stopping on validation residuals
print("Training CatBoost model...")
val_cb = build_catboost_model()
val_cb.fit(
    X_train,
    y_train,
    sample_weight=sample_weight_train,
    eval_set=(X_val, y_val),
    cat_features=cat_cols,
    early_stopping_rounds=40,
    verbose=100,
)
cb_best_iteration = getattr(val_cb, "best_iteration_", None)
if cb_best_iteration is None:
    cb_best_iteration = val_cb.get_best_iteration()
if cb_best_iteration is None or cb_best_iteration <= 0:
    cb_best_iteration = 600
print(f"Validation CatBoost best iteration: {cb_best_iteration}")

joblib.dump(val_cb, "./working/catboost_val_model.pkl")
loaded_val_cb = joblib.load("./working/catboost_val_model.pkl")

# Evaluate individual models and dual ensemble
ensemble_val = EnsembleResidualPredictor(loaded_val_lgb, loaded_val_cb, 0.5, 0.5)
val_residual_preds = ensemble_val.predict(X_val)

# Out-of-time validation metric evaluation: reconstruct price by adding extrapolated trend
val_preds_log10 = val_residual_preds + val_trend
val_preds_price = np.power(10.0, val_preds_log10)

val_preds_clipped = np.clip(val_preds_price, 1.0, None)
val_true_clipped = np.clip(val_true_price, 1.0, None)

val_rmse = float(
    np.sqrt(np.mean((np.log10(val_preds_clipped) - np.log10(val_true_clipped)) ** 2))
)
print(f"Validation Out-of-Time RMSE (log10 scale): {val_rmse:.5f}")

# Free validation models to conserve memory before production training
del val_lgb, val_cb, loaded_val_lgb, loaded_val_cb, X_train, y_train, sample_weight_train
gc.collect()

# Production retraining on complete modern historical data (2011-2016)
print("Retraining production dual models on full modern clean period (2011-2016)...")
X_full = df_full_train.loc[clean_full, all_features]
y_full = df_full_train.loc[clean_full, "price_residual"].values

sample_weight_full = np.exp(
    gamma * (df_full_train.loc[clean_full, "time_elapsed"].values - df_full_train["time_elapsed"].max())
).astype(np.float32)

# Retrain LightGBM
full_lgb_n_estimators = int(lgb_best_iteration * 1.20)
prod_lgb_params = {
    "n_estimators": full_lgb_n_estimators,
    "random_state": 42,
}
full_lgb = build_lgbm_model(prod_lgb_params)
full_lgb.fit(
    X_full,
    y_full,
    sample_weight=sample_weight_full,
    categorical_feature=cat_cols,
)
joblib.dump(full_lgb, "./working/lgbm_production_model.pkl")

# Retrain CatBoost
full_cb_n_estimators = int(cb_best_iteration * 1.20)
prod_cb_params = {
    "iterations": full_cb_n_estimators,
    "random_seed": 42,
    "verbose": 100,
}
full_cb = build_catboost_model(prod_cb_params)
full_cb.fit(
    X_full,
    y_full,
    sample_weight=sample_weight_full,
    cat_features=cat_cols,
    verbose=100,
)
joblib.dump(full_cb, "./working/catboost_production_model.pkl")

loaded_prod_lgb = joblib.load("./working/lgbm_production_model.pkl")
loaded_prod_cb = joblib.load("./working/catboost_production_model.pkl")
ensemble_prod = EnsembleResidualPredictor(loaded_prod_lgb, loaded_prod_cb, 0.5, 0.5)

# Free full train objects
del full_lgb, full_cb, loaded_prod_lgb, loaded_prod_cb, X_full, y_full, sample_weight_full
gc.collect()

# Inference on unseen 2017 H1 test set: extrapolate macro trend forward
print("Performing model inference on unseen test set (2017 H1)...")
X_test = df_test_proc[all_features]
test_residual_preds = ensemble_prod.predict(X_test)
test_preds_log10 = test_residual_preds + test_trend

test_preds_price = np.power(10.0, test_preds_log10)
test_preds_price = np.clip(test_preds_price, 1.0, None)

assert not np.isnan(test_preds_price).any(), "NaN values found in test predictions!"
assert not np.isinf(
    test_preds_price
).any(), "Infinite values found in test predictions!"
assert len(test_preds_price) == len(
    df_test_proc
), f"Mismatch in test count: {len(test_preds_price)} vs {len(df_test_proc)}"

submission_path = "./submission/submission.csv"
submission = pd.DataFrame(
    {"id": df_test_proc["id"].values, "price": np.round(test_preds_price, 2)}
)
submission.to_csv(submission_path, index=False)

sample_sub = pd.read_csv("./input/sample_submission.csv")
assert list(submission.columns) == list(
    sample_sub.columns
), f"Submission columns {submission.columns} do not match sample {sample_sub.columns}"
assert len(submission) == len(
    sample_sub
), f"Row count mismatch: expected {len(sample_sub)}, got {len(submission)}"
assert (
    submission["id"] == sample_sub["id"]
).all(), "ID order in submission does not match sample submission!"

print(f"Final Validation Score: {val_rmse}")
