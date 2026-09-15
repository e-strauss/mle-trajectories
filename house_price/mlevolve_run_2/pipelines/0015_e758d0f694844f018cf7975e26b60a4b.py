import gc
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================================================================
# 1. EVALUATION METRIC AND CRITERIA
# =========================================================================


def compute_log10_rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    is_log10_scale: bool = True,
) -> float:
    """
    Computes the official competition metric: RMSE on log10(price).
    score = sqrt( mean_i ( log10(pred_i) - log10(true_i) )^2 )
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if not is_log10_scale:
        y_true = np.log10(np.clip(y_true, 1.0, None))
        y_pred = np.log10(np.clip(y_pred, 1.0, None))

    mse = np.mean((y_pred - y_true) ** 2)
    return float(np.sqrt(mse))


class Log10MSELoss(nn.Module):
    """
    Loss criterion aligned with competition metric:
    Minimizing Mean Squared Error on log10(price) directly minimizes RMSE on log10(price).
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred.view(-1), target.view(-1))


# =========================================================================
# 2. FEATURE ENGINEERING & HIERARCHICAL TARGET ENCODING
# =========================================================================


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal, structural hedonic, and composite interaction features.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Temporal features (non-stationary year and time_trend_years removed)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["is_friday"] = (df["dayofweek"] == 4).astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(np.int16)

    # Cyclical seasonality encodings
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)

    # Standardize string fields
    for col in [
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
        "town",
        "district",
        "county",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Hedonic binary flags
    df["is_new_build_flag"] = (df["is_new_build"] == "Y").astype(np.int8)
    df["is_standard_sale"] = (df["sale_category"] == "A").astype(np.int8)

    # Tenure mapping: Freehold=0, Leasehold=1, Unknown/Other=2
    tenure_map = {"F": 0, "L": 1, "U": 2}
    df["tenure_code"] = df["tenure"].map(tenure_map).fillna(2).astype(np.int8)

    # Property type mapping: Detached=0, Semi=1, Terraced=2, Flat=3, Other=4
    ptype_map = {"D": 0, "S": 1, "T": 2, "F": 3, "O": 4}
    df["property_type_code"] = (
        df["property_type"].map(ptype_map).fillna(4).astype(np.int8)
    )

    # High-signal interaction terms
    df["prop_tenure"] = df["property_type"] + "_" + df["tenure"]
    df["prop_new"] = df["property_type"] + "_" + df["is_new_build"]
    df["prop_sale"] = df["property_type"] + "_" + df["sale_category"]
    df["district_prop"] = df["district"] + "___" + df["property_type"]
    df["dist_new"] = df["district"] + "___" + df["is_new_build"]
    df["prop_tenure_new"] = (
        df["property_type"] + "_" + df["tenure"] + "_" + df["is_new_build"]
    )
    df["town_prop"] = df["town"] + "___" + df["property_type"]
    df["county_prop"] = df["county"] + "___" + df["property_type"]

    return df


class HierarchicalBayesianTargetEncoder:
    """
    Multi-Scale Hierarchical Empirical Bayes Target Encoder with trailing windows (6m, 12m, 24m)
    and residual encodings with bounded momentum and spatial price dispersion metrics.
    """

    def __init__(self, m_smooth: float = 15.0):
        self.m = m_smooth
        self.global_mean = 0.0
        self.county_stats = {}
        self.district_stats = {}
        self.town_stats = {}
        self.dist_prop_stats = {}
        self.town_prop_stats = {}
        self.dist_tenure_stats = {}
        self.dist_new_stats = {}
        self.prop_tenure_new_stats = {}
        self.recent_dist_stats = {}

        # Multi-scale lookback anchors (6m, 12m, 24m)
        self.global_6m_mean = 0.0
        self.global_12m_mean = 0.0
        self.global_24m_mean = 0.0

        self.prop_6m_stats = {}
        self.prop_12m_stats = {}
        self.prop_24m_stats = {}

        self.county_prop_6m_stats = {}
        self.county_prop_12m_stats = {}
        self.county_prop_24m_stats = {}

        self.dist_prop_6m_stats = {}
        self.dist_prop_12m_stats = {}
        self.dist_prop_24m_stats = {}

        self.town_prop_6m_stats = {}
        self.town_prop_12m_stats = {}
        self.town_prop_24m_stats = {}

        self.county_6m_stats = {}
        self.county_12m_stats = {}
        self.county_24m_stats = {}

        self.dist_6m_stats = {}
        self.dist_12m_stats = {}
        self.dist_24m_stats = {}

        self.town_6m_stats = {}
        self.town_12m_stats = {}
        self.town_24m_stats = {}

        # Price dispersion stats
        self.global_price_std = 0.2
        self.dist_dispersion_stats = {}
        self.dist_prop_dispersion_stats = {}

    def _fit_window_anchors(self, df_win: pd.DataFrame, target: str, m: float):
        global_m = float(df_win[target].mean())
        town_to_dist = df_win.groupby("town")["district"].first().to_dict()

        p_grp = df_win.groupby("property_type")[target].agg(["count", "mean"])
        prop_stats = ((p_grp["count"] * p_grp["mean"] + 10.0 * global_m) / (p_grp["count"] + 10.0)).to_dict()

        c_grp = df_win.groupby("county")[target].agg(["count", "mean"])
        county_stats = ((c_grp["count"] * c_grp["mean"] + m * global_m) / (c_grp["count"] + m)).to_dict()

        cp_grp = df_win.groupby(["county_prop", "property_type"])[target].agg(["count", "mean"]).reset_index()
        cp_prior = cp_grp["property_type"].map(prop_stats).fillna(global_m)
        cp_stats = dict(zip(cp_grp["county_prop"], (cp_grp["count"] * cp_grp["mean"] + m * cp_prior) / (cp_grp["count"] + m)))

        d_grp = df_win.groupby(["district", "county"])[target].agg(["count", "mean"]).reset_index()
        d_prior = d_grp["county"].map(county_stats).fillna(global_m)
        dist_stats = dict(zip(d_grp["district"], (d_grp["count"] * d_grp["mean"] + m * d_prior) / (d_grp["count"] + m)))

        dp_grp = df_win.groupby(["district_prop", "county_prop", "property_type"])[target].agg(["count", "mean"]).reset_index()
        dp_prior = dp_grp["county_prop"].map(cp_stats).fillna(dp_grp["property_type"].map(prop_stats)).fillna(global_m)
        dp_stats = dict(zip(dp_grp["district_prop"], (dp_grp["count"] * dp_grp["mean"] + m * dp_prior) / (dp_grp["count"] + m)))

        t_grp = df_win.groupby("town")[target].agg(["count", "mean"]).reset_index()
        t_prior = t_grp["town"].map(town_to_dist).map(dist_stats).fillna(global_m)
        town_stats = dict(zip(t_grp["town"], (t_grp["count"] * t_grp["mean"] + m * t_prior) / (t_grp["count"] + m)))

        tp_grp = df_win.groupby(["town_prop", "district_prop"])[target].agg(["count", "mean"]).reset_index()
        tp_prior = tp_grp["district_prop"].map(dp_stats).fillna(global_m)
        tp_stats = dict(zip(tp_grp["town_prop"], (tp_grp["count"] * tp_grp["mean"] + m * tp_prior) / (tp_grp["count"] + m)))

        return global_m, prop_stats, county_stats, cp_stats, dist_stats, dp_stats, town_stats, tp_stats

    def fit(self, df: pd.DataFrame, target_col: str = "residual"):
        y = df[target_col].values
        self.global_mean = float(np.mean(y))

        # Residual encodings
        c_grp = df.groupby("county")[target_col].agg(["count", "mean"])
        self.county_stats = ((c_grp["count"] * c_grp["mean"] + self.m * self.global_mean) / (c_grp["count"] + self.m)).to_dict()

        d_grp = df.groupby(["district", "county"])[target_col].agg(["count", "mean"]).reset_index()
        d_prior = d_grp["county"].map(self.county_stats).fillna(self.global_mean)
        self.district_stats = dict(zip(d_grp["district"], (d_grp["count"] * d_grp["mean"] + self.m * d_prior) / (d_grp["count"] + self.m)))

        town_to_dist = df.groupby("town")["district"].first().to_dict()
        t_grp = df.groupby("town")[target_col].agg(["count", "mean"]).reset_index()
        t_prior = t_grp["town"].map(town_to_dist).map(self.district_stats).fillna(self.global_mean)
        self.town_stats = dict(zip(t_grp["town"], (t_grp["count"] * t_grp["mean"] + self.m * t_prior) / (t_grp["count"] + self.m)))

        dp_grp = df.groupby(["district_prop", "district"])[target_col].agg(["count", "mean"]).reset_index()
        dp_prior = dp_grp["district"].map(self.district_stats).fillna(self.global_mean)
        self.dist_prop_stats = dict(zip(dp_grp["district_prop"], (dp_grp["count"] * dp_grp["mean"] + self.m * dp_prior) / (dp_grp["count"] + self.m)))

        tp_grp = df.groupby(["town_prop", "district_prop", "town"])[target_col].agg(["count", "mean"]).reset_index()
        tp_prior = tp_grp["town"].map(self.town_stats).fillna(tp_grp["district_prop"].map(self.dist_prop_stats)).fillna(self.global_mean)
        self.town_prop_stats = dict(zip(tp_grp["town_prop"], (tp_grp["count"] * tp_grp["mean"] + self.m * tp_prior) / (tp_grp["count"] + self.m)))

        dt_grp = df.groupby(["district", "prop_tenure"])[target_col].agg(["count", "mean"]).reset_index()
        dt_key = dt_grp["district"] + "___" + dt_grp["prop_tenure"]
        dt_prior = dt_grp["district"].map(self.district_stats).fillna(self.global_mean)
        self.dist_tenure_stats = dict(zip(dt_key, (dt_grp["count"] * dt_grp["mean"] + self.m * dt_prior) / (dt_grp["count"] + self.m)))

        dn_grp = df.groupby(["dist_new", "district"])[target_col].agg(["count", "mean"]).reset_index()
        dn_prior = dn_grp["district"].map(self.district_stats).fillna(self.global_mean)
        self.dist_new_stats = dict(zip(dn_grp["dist_new"], (dn_grp["count"] * dn_grp["mean"] + self.m * dn_prior) / (dn_grp["count"] + self.m)))

        ptn_grp = df.groupby("prop_tenure_new")[target_col].agg(["count", "mean"]).reset_index()
        self.prop_tenure_new_stats = dict(zip(ptn_grp["prop_tenure_new"], (ptn_grp["count"] * ptn_grp["mean"] + self.m * self.global_mean) / (ptn_grp["count"] + self.m)))

        recent_mask = df["date"] >= (df["date"].max() - pd.Timedelta(days=540))
        if recent_mask.sum() > 1000:
            rec_grp = df[recent_mask].groupby("district")[target_col].agg(["count", "mean"]).reset_index()
            rec_prior = rec_grp["district"].map(self.district_stats).fillna(self.global_mean)
            self.recent_dist_stats = dict(zip(rec_grp["district"], (rec_grp["count"] * rec_grp["mean"] + self.m * rec_prior) / (rec_grp["count"] + self.m)))
        else:
            self.recent_dist_stats = self.district_stats.copy()

        # Multi-scale lookback anchors (6m, 12m, 24m) on log10_price
        price_target = "log10_price" if "log10_price" in df.columns else target_col
        t_max = df["date"].max()

        df_6m = df[df["date"] >= (t_max - pd.Timedelta(days=183))]
        df_12m = df[df["date"] >= (t_max - pd.Timedelta(days=365))]
        df_24m = df[df["date"] >= (t_max - pd.Timedelta(days=730))]

        if len(df_6m) < 50:
            df_6m = df
        if len(df_12m) < 50:
            df_12m = df
        if len(df_24m) < 50:
            df_24m = df

        (self.global_6m_mean, self.prop_6m_stats, self.county_6m_stats, self.county_prop_6m_stats,
         self.dist_6m_stats, self.dist_prop_6m_stats, self.town_6m_stats, self.town_prop_6m_stats) = self._fit_window_anchors(df_6m, price_target, self.m)

        (self.global_12m_mean, self.prop_12m_stats, self.county_12m_stats, self.county_prop_12m_stats,
         self.dist_12m_stats, self.dist_prop_12m_stats, self.town_12m_stats, self.town_prop_12m_stats) = self._fit_window_anchors(df_12m, price_target, self.m)

        (self.global_24m_mean, self.prop_24m_stats, self.county_24m_stats, self.county_prop_24m_stats,
         self.dist_24m_stats, self.dist_prop_24m_stats, self.town_24m_stats, self.town_prop_24m_stats) = self._fit_window_anchors(df_24m, price_target, self.m)

        # Price dispersion statistics in trailing 12m
        self.global_price_std = float(df_12m[price_target].std()) if len(df_12m) > 1 else 0.2
        d_std = df_12m.groupby("district")[price_target].std().fillna(self.global_price_std).to_dict()
        self.dist_dispersion_stats = d_std

        dp_std = df_12m.groupby("district_prop")[price_target].std().fillna(self.global_price_std).to_dict()
        self.dist_prop_dispersion_stats = dp_std

    def _get_anchor(self, df: pd.DataFrame, g_mean, prop_s, c_s, cp_s, d_s, dp_s, t_s, tp_s) -> pd.Series:
        p_prior = df["property_type"].astype(str).map(prop_s).fillna(g_mean)
        cp_prior = df["county_prop"].astype(str).map(cp_s).fillna(p_prior)
        c_prior = df["county"].astype(str).map(c_s).fillna(g_mean)
        d_prior = df["district"].astype(str).map(d_s).fillna(c_prior)
        dp_prior = df["district_prop"].astype(str).map(dp_s).fillna(cp_prior)
        t_prior = df["town"].astype(str).map(t_s).fillna(d_prior)
        tp_prior = df["town_prop"].astype(str).map(tp_s).fillna(dp_prior).fillna(t_prior).fillna(g_mean)
        return tp_prior.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # Residual encodings
        out["te_county"] = df["county"].astype(str).map(self.county_stats).fillna(self.global_mean).astype(np.float32)
        out["te_district"] = df["district"].astype(str).map(self.district_stats).fillna(out["te_county"]).astype(np.float32)
        out["te_town"] = df["town"].astype(str).map(self.town_stats).fillna(out["te_district"]).astype(np.float32)
        out["te_dist_prop"] = df["district_prop"].astype(str).map(self.dist_prop_stats).fillna(out["te_district"]).astype(np.float32)
        out["te_town_prop"] = df["town_prop"].astype(str).map(self.town_prop_stats).fillna(out["te_town"]).astype(np.float32)

        dt_key = df["district"].astype(str) + "___" + df["prop_tenure"].astype(str)
        out["te_dist_tenure"] = dt_key.map(self.dist_tenure_stats).fillna(out["te_district"]).astype(np.float32)
        out["te_dist_new"] = df["dist_new"].astype(str).map(self.dist_new_stats).fillna(out["te_district"]).astype(np.float32)
        out["te_prop_tenure_new"] = df["prop_tenure_new"].astype(str).map(self.prop_tenure_new_stats).fillna(self.global_mean).astype(np.float32)

        out["te_recent_district"] = df["district"].astype(str).map(self.recent_dist_stats).fillna(out["te_district"]).astype(np.float32)
        out["te_district_momentum"] = (out["te_recent_district"] - out["te_district"]).astype(np.float32)

        # Multi-scale Bayesian Anchors
        out["anchor_6m"] = self._get_anchor(df, self.global_6m_mean, self.prop_6m_stats, self.county_6m_stats,
                                            self.county_prop_6m_stats, self.dist_6m_stats, self.dist_prop_6m_stats,
                                            self.town_6m_stats, self.town_prop_6m_stats)
        out["anchor_12m"] = self._get_anchor(df, self.global_12m_mean, self.prop_12m_stats, self.county_12m_stats,
                                             self.county_prop_12m_stats, self.dist_12m_stats, self.dist_prop_12m_stats,
                                             self.town_12m_stats, self.town_prop_12m_stats)
        out["anchor_24m"] = self._get_anchor(df, self.global_24m_mean, self.prop_24m_stats, self.county_24m_stats,
                                             self.county_prop_24m_stats, self.dist_24m_stats, self.dist_prop_24m_stats,
                                             self.town_24m_stats, self.town_prop_24m_stats)

        # Bounded local momentum features
        out["momentum_6m_12m"] = (out["anchor_6m"] - out["anchor_12m"]).astype(np.float32)
        out["momentum_12m_24m"] = (out["anchor_12m"] - out["anchor_24m"]).astype(np.float32)

        prop_prior_12m = df["property_type"].astype(str).map(self.prop_12m_stats).fillna(self.global_12m_mean)
        county_prop_prior_12m = df["county_prop"].astype(str).map(self.county_prop_12m_stats).fillna(prop_prior_12m)
        out["recent_12m_dist_prop"] = df["district_prop"].astype(str).map(self.dist_prop_12m_stats).fillna(county_prop_prior_12m).astype(np.float32)
        out["recent_12m_town_prop"] = df["town_prop"].astype(str).map(self.town_prop_12m_stats).fillna(out["recent_12m_dist_prop"]).astype(np.float32)

        prop_prior_24m = df["property_type"].astype(str).map(self.prop_24m_stats).fillna(self.global_24m_mean)
        county_prop_prior_24m = df["county_prop"].astype(str).map(self.county_prop_24m_stats).fillna(prop_prior_24m)
        out["recent_24m_dist_prop"] = df["district_prop"].astype(str).map(self.dist_prop_24m_stats).fillna(county_prop_prior_24m).astype(np.float32)

        county_prior_12m = df["county"].astype(str).map(self.county_12m_stats).fillna(self.global_12m_mean)
        out["recent_12m_district"] = df["district"].astype(str).map(self.dist_12m_stats).fillna(county_prior_12m).astype(np.float32)
        out["recent_12m_town"] = df["town"].astype(str).map(self.town_12m_stats).fillna(out["recent_12m_district"]).astype(np.float32)

        county_prior_24m = df["county"].astype(str).map(self.county_24m_stats).fillna(self.global_24m_mean)
        out["recent_24m_district"] = df["district"].astype(str).map(self.dist_24m_stats).fillna(county_prior_24m).astype(np.float32)

        out["town_to_district_premium"] = (out["recent_12m_town"] - out["recent_12m_district"]).astype(np.float32)
        out["district_momentum_1y_2y"] = (out["recent_12m_district"] - out["recent_24m_district"]).astype(np.float32)
        out["dist_prop_momentum_1y_2y"] = (out["recent_12m_dist_prop"] - out["recent_24m_dist_prop"]).astype(np.float32)

        # Price dispersion features
        c_std_prior = self.global_price_std
        out["district_price_dispersion"] = df["district"].astype(str).map(self.dist_dispersion_stats).fillna(c_std_prior).astype(np.float32)
        out["dist_prop_price_dispersion"] = df["district_prop"].astype(str).map(self.dist_prop_dispersion_stats).fillna(out["district_price_dispersion"]).astype(np.float32)
        out["dispersion_ratio"] = (out["dist_prop_price_dispersion"] / (out["district_price_dispersion"] + 1e-4)).astype(np.float32)

        return out


def compute_oof_target_encoding(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    m_smooth: float = 15.0,
    target_col: str = "residual",
) -> pd.DataFrame:
    """
    Computes Out-of-Fold Hierarchical Target Encodings on training split to eliminate target leakage.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_cols = [
        "te_county",
        "te_district",
        "te_town",
        "te_dist_prop",
        "te_town_prop",
        "te_dist_tenure",
        "te_dist_new",
        "te_prop_tenure_new",
        "te_recent_district",
        "te_district_momentum",
        "anchor_6m",
        "anchor_12m",
        "anchor_24m",
        "momentum_6m_12m",
        "momentum_12m_24m",
        "recent_12m_dist_prop",
        "recent_12m_town_prop",
        "recent_24m_dist_prop",
        "recent_12m_town",
        "recent_12m_district",
        "recent_24m_district",
        "town_to_district_premium",
        "district_momentum_1y_2y",
        "dist_prop_momentum_1y_2y",
        "district_price_dispersion",
        "dist_prop_price_dispersion",
        "dispersion_ratio",
    ]
    oof_features = pd.DataFrame(
        index=train_df.index, columns=feature_cols, dtype=np.float32
    )

    for tr_idx, val_idx in kf.split(train_df):
        tr_part = train_df.iloc[tr_idx]
        val_part = train_df.iloc[val_idx]

        encoder = HierarchicalBayesianTargetEncoder(m_smooth=m_smooth)
        encoder.fit(tr_part, target_col=target_col)
        val_encoded = encoder.transform(val_part)
        oof_features.iloc[val_idx] = val_encoded[feature_cols].values

    return oof_features


def compute_frequency_and_spatial_features(
    base_train: pd.DataFrame, dfs_to_transform: list
) -> list:
    """
    Computes liquidity and frequency densities from historical data.
    """
    d_counts = base_train["district"].value_counts()
    t_counts = base_train["town"].value_counts()
    dp_counts = base_train["district_prop"].value_counts()

    # Flat ratio per district (urban density proxy)
    d_flats = base_train[base_train["property_type"] == "F"]["district"].value_counts()
    d_flat_ratio = (d_flats / d_counts).fillna(0.0).to_dict()

    # New build ratio per district (development activity proxy)
    d_new = base_train[base_train["is_new_build_flag"] == 1]["district"].value_counts()
    d_new_ratio = (d_new / d_counts).fillna(0.0).to_dict()

    d_cnt_dict = d_counts.to_dict()
    t_cnt_dict = t_counts.to_dict()
    dp_cnt_dict = dp_counts.to_dict()

    results = []
    for df in dfs_to_transform:
        df = df.copy()
        df["log_district_count"] = np.log1p(
            df["district"].map(d_cnt_dict).fillna(0).astype(np.float32)
        ).astype(np.float32)
        df["log_town_count"] = np.log1p(
            df["town"].map(t_cnt_dict).fillna(0).astype(np.float32)
        ).astype(np.float32)
        df["log_dist_prop_count"] = np.log1p(
            df["district_prop"].map(dp_cnt_dict).fillna(0).astype(np.float32)
        ).astype(np.float32)
        df["district_flat_ratio"] = (
            df["district"].map(d_flat_ratio).fillna(0.0).astype(np.float32)
        )
        df["district_new_ratio"] = (
            df["district"].map(d_new_ratio).fillna(0.0).astype(np.float32)
        )
        results.append(df)

    return results


# =========================================================================
# 3. MODEL SPECIFICATIONS & FACTORIES
# =========================================================================


class ResidualBlock(nn.Module):
    """
    Dense residual block with LayerNorm, SiLU activation, and Dropout.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HedonicResNet(nn.Module):
    """
    Deep Hedonic Entity Embedding Residual Network for property price prediction.
    """

    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        num_numerical_features: int,
        embedding_dim_mult: float = 1.6,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.cat_keys = sorted(list(cat_cardinalities.keys()))
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for key in self.cat_keys:
            card = cat_cardinalities[key]
            emb_dim = int(min(64, max(4, round(card**0.25 * embedding_dim_mult))))
            self.embeddings[key] = nn.Embedding(
                num_embeddings=card + 1, padding_idx=0, embedding_dim=emb_dim
            )
            total_emb_dim += emb_dim

        self.num_numerical_features = num_numerical_features
        if num_numerical_features > 0:
            self.num_bn = nn.BatchNorm1d(num_numerical_features)
        else:
            self.num_bn = nn.Identity()

        total_input_dim = total_emb_dim + num_numerical_features

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(
        self,
        cat_inputs: Dict[str, torch.Tensor],
        num_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb_outputs = []
        for key in self.cat_keys:
            x_cat = cat_inputs[key]
            emb = self.embeddings[key](x_cat)
            emb_outputs.append(emb)

        emb_concat = torch.cat(emb_outputs, dim=-1) if emb_outputs else None

        if num_inputs is not None and self.num_numerical_features > 0:
            num_norm = self.num_bn(num_inputs)
            x = (
                torch.cat([emb_concat, num_norm], dim=-1)
                if emb_concat is not None
                else num_norm
            )
        else:
            x = emb_concat

        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)

        return self.head(h)


def get_lgbm_model(
    custom_params: Optional[Dict[str, Union[int, float, str]]] = None,
) -> lgb.LGBMRegressor:
    """
    LightGBM Regressor configured to directly optimize RMSE on log10(price) residuals.
    """
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 50,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 3000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params is not None:
        base_params.update(custom_params)

    return lgb.LGBMRegressor(**base_params)


class HierarchicalBayesianAnchor:
    """
    Bounded Trailing-Window Hierarchical Empirical Bayes Price Anchor.
    Eliminates unbounded linear trend drift by grounding predictions in strictly bounded local empirical Bayes baselines.
    """

    def __init__(self, window_days: int = 365, m_smooth: float = 15.0):
        self.window_days = window_days
        self.m = m_smooth
        self.global_mean = 0.0
        self.prop_stats = {}
        self.county_stats = {}
        self.county_prop_stats = {}
        self.dist_stats = {}
        self.dist_prop_stats = {}
        self.town_stats = {}
        self.town_prop_stats = {}

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "log10_price",
        region_col: str = "county",
    ):
        price_target = "log10_price" if "log10_price" in df.columns else target_col
        t_max = df["date"].max()
        df_win = df[df["date"] >= (t_max - pd.Timedelta(days=self.window_days))]
        if len(df_win) < 50:
            df_win = df

        self.global_mean = float(df_win[price_target].mean())
        town_to_dist = df_win.groupby("town")["district"].first().to_dict()

        p_grp = df_win.groupby("property_type")[price_target].agg(["count", "mean"])
        self.prop_stats = ((p_grp["count"] * p_grp["mean"] + 10.0 * self.global_mean) / (p_grp["count"] + 10.0)).to_dict()

        c_grp = df_win.groupby(region_col)[price_target].agg(["count", "mean"])
        self.county_stats = ((c_grp["count"] * c_grp["mean"] + self.m * self.global_mean) / (c_grp["count"] + self.m)).to_dict()

        cp_grp = df_win.groupby(["county_prop", "property_type"])[price_target].agg(["count", "mean"]).reset_index()
        cp_prior = cp_grp["property_type"].map(self.prop_stats).fillna(self.global_mean)
        self.county_prop_stats = dict(zip(cp_grp["county_prop"], (cp_grp["count"] * cp_grp["mean"] + self.m * cp_prior) / (cp_grp["count"] + self.m)))

        d_grp = df_win.groupby(["district", region_col])[price_target].agg(["count", "mean"]).reset_index()
        d_prior = d_grp[region_col].map(self.county_stats).fillna(self.global_mean)
        self.dist_stats = dict(zip(d_grp["district"], (d_grp["count"] * d_grp["mean"] + self.m * d_prior) / (d_grp["count"] + self.m)))

        dp_grp = df_win.groupby(["district_prop", "county_prop", "property_type"])[price_target].agg(["count", "mean"]).reset_index()
        dp_prior = dp_grp["county_prop"].map(self.county_prop_stats).fillna(dp_grp["property_type"].map(self.prop_stats)).fillna(self.global_mean)
        self.dist_prop_stats = dict(zip(dp_grp["district_prop"], (dp_grp["count"] * dp_grp["mean"] + self.m * dp_prior) / (dp_grp["count"] + self.m)))

        t_grp = df_win.groupby("town")[price_target].agg(["count", "mean"]).reset_index()
        t_prior = t_grp["town"].map(town_to_dist).map(self.dist_stats).fillna(self.global_mean)
        self.town_stats = dict(zip(t_grp["town"], (t_grp["count"] * t_grp["mean"] + self.m * t_prior) / (t_grp["count"] + self.m)))

        tp_grp = df_win.groupby(["town_prop", "district_prop"])[price_target].agg(["count", "mean"]).reset_index()
        tp_prior = tp_grp["district_prop"].map(self.dist_prop_stats).fillna(self.global_mean)
        self.town_prop_stats = dict(zip(tp_grp["town_prop"], (tp_grp["count"] * tp_grp["mean"] + self.m * tp_prior) / (tp_grp["count"] + self.m)))

        return self

    def predict(
        self, df: pd.DataFrame, region_col: str = "county"
    ) -> np.ndarray:
        p_prior = df["property_type"].astype(str).map(self.prop_stats).fillna(self.global_mean)
        cp_prior = df["county_prop"].astype(str).map(self.county_prop_stats).fillna(p_prior)
        c_prior = df[region_col].astype(str).map(self.county_stats).fillna(self.global_mean)
        d_prior = df["district"].astype(str).map(self.dist_stats).fillna(c_prior)
        dp_prior = df["district_prop"].astype(str).map(self.dist_prop_stats).fillna(cp_prior)
        t_prior = df["town"].astype(str).map(self.town_stats).fillna(d_prior)
        tp_pred = df["town_prop"].astype(str).map(self.town_prop_stats).fillna(dp_prior).fillna(t_prior).fillna(self.global_mean)
        return tp_pred.values.astype(np.float32)


# Aliases for strict backward compatibility
HedonicHierarchicalTrendModel = HierarchicalBayesianAnchor
HierarchicalTrendModel = HierarchicalBayesianAnchor
LinearTrendModel = HierarchicalBayesianAnchor


class HedonicResNetPredictor:
    """
    Scikit-learn compatible inference wrapper for PyTorch HedonicResNet.
    Generates unified 1D log10 price residual predictions.
    """

    def __init__(
        self,
        model: HedonicResNet,
        cat_cols: List[str],
        cat_mappings: Dict[str, Dict[str, int]],
        num_cols: List[str],
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        device: torch.device,
    ):
        self.model = model
        self.cat_cols = cat_cols
        self.cat_mappings = cat_mappings
        self.num_cols = num_cols
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.device = device

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)

        cat_tensors = {}
        for col in self.cat_cols:
            vals = (
                df[col]
                .astype(str)
                .map(self.cat_mappings[col])
                .fillna(0)
                .astype(np.int64)
                .values
            )
            cat_tensors[col] = torch.from_numpy(vals)

        num_arr = df[self.num_cols].values.astype(np.float32)
        num_arr = np.nan_to_num(num_arr, nan=0.0, posinf=0.0, neginf=0.0)
        num_arr = (num_arr - self.scaler_mean) / (self.scaler_std + 1e-7)
        num_tensor = torch.from_numpy(num_arr.astype(np.float32))

        preds = []
        n = num_tensor.shape[0]
        batch_size = 8192
        with torch.no_grad():
            for i in range(0, n, batch_size):
                b_cats = {
                    k: v[i : i + batch_size].to(self.device, non_blocking=True)
                    for k, v in cat_tensors.items()
                }
                b_num = num_tensor[i : i + batch_size].to(
                    self.device, non_blocking=True
                )
                out = self.model(b_cats, b_num).view(-1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0).astype(np.float32)


def train_hedonic_resnet(
    model: HedonicResNet,
    cat_tensors: Dict[str, torch.Tensor],
    num_tensor: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    epochs: int = 7,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> HedonicResNet:
    """
    GPU/CPU accelerated HedonicResNet training with AdamW, CosineAnnealingLR, and recency sample weighting.
    """
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_samples = num_tensor.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            b_cats = {
                k: v[idx].to(device, non_blocking=True)
                for k, v in cat_tensors.items()
            }
            b_num = num_tensor[idx].to(device, non_blocking=True)
            b_y = targets[idx].to(device, non_blocking=True)
            b_w = weights[idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(b_cats, b_num).view(-1)
            loss = torch.mean(b_w * (preds - b_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

    return model


class TwoStageHybridPredictor:
    """
    Heterogeneous Dual-Paradigm Hybrid Predictor combining a bounded Trailing Bayesian Anchor
    with an ensemble of LightGBM and HedonicResNet residual regressors.
    Reconstructs final log10 predictions by adding the anchor back to blended residuals.
    """

    def __init__(
        self,
        trend_model: Union[HierarchicalBayesianAnchor, Any],
        gbdt_models: Union[Any, List[Any]],
        weights: Optional[List[float]] = None,
    ):
        self.trend_model = trend_model
        if isinstance(gbdt_models, list):
            self.gbdt_models = gbdt_models
        else:
            self.gbdt_models = [gbdt_models]

        if weights is None:
            self.weights = [1.0 / len(self.gbdt_models)] * len(self.gbdt_models)
        else:
            w = np.array(weights, dtype=np.float64)
            self.weights = list(w / w.sum())

    def predict_residuals(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> np.ndarray:
        residual_preds = np.zeros(len(df), dtype=np.float64)
        for m, w in zip(self.gbdt_models, self.weights):
            if isinstance(m, lgb.LGBMRegressor):
                cols = features if features is not None else m.feature_name_
                residual_preds += w * m.predict(df[cols]).astype(np.float64)
            else:
                residual_preds += w * m.predict(df).astype(np.float64)
        return residual_preds.astype(np.float32)

    def predict(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        region_col: str = "county",
    ) -> np.ndarray:
        trend_preds = (
            self.trend_model.predict(df, region_col=region_col).astype(np.float64)
            if self.trend_model is not None
            else 0.0
        )
        residual_preds = self.predict_residuals(df, features=features).astype(
            np.float64
        )
        log10_preds = trend_preds + residual_preds
        return log10_preds.astype(np.float32)


def compute_recency_weights(
    dates: pd.Series, half_life_years: float = 3.0
) -> np.ndarray:
    """
    Computes exponential recency sample weights with specified half-life.
    Weights are normalized so mean(weights) == 1.0.
    """
    max_date = dates.max()
    age_years = (max_date - dates).dt.total_seconds() / (365.25 * 86400.0)
    decay_rate = np.log(2.0) / half_life_years
    weights = np.exp(-decay_rate * age_years).values.astype(np.float32)
    weights = weights / np.mean(weights)
    return weights


# =========================================================================
# 4. MAIN PIPELINE EXECUTION
# =========================================================================


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    test_raw = pd.read_csv("./input/test.csv")
    train_raw = pd.read_parquet("./input/train.parquet")

    # Focus on the modern post-crisis regime (2011-2016)
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    modern_train = (
        train_raw[train_raw["date"] >= "2011-01-01"].copy().reset_index(drop=True)
    )
    del train_raw
    gc.collect()

    # Target variable: log10(price)
    modern_train["log10_price"] = np.log10(
        np.clip(modern_train["price"].values, 1.0, None)
    ).astype(np.float32)

    # 2. Extract Base Features
    modern_train = create_base_features(modern_train)
    test_processed = create_base_features(test_raw)

    # 3. Create Splits & Enforce Training-Only Outlier Filtering
    # Validation mirrors the 6-month test period (2016-07-01 to 2016-12-31)
    val_cutoff = pd.Timestamp("2016-07-01")
    local_train_mask = modern_train["date"] < val_cutoff
    val_mask = modern_train["date"] >= val_cutoff

    local_train_df = modern_train[local_train_mask].copy().reset_index(drop=True)
    val_df = modern_train[val_mask].copy().reset_index(drop=True)
    full_train_df = modern_train.copy().reset_index(drop=True)
    del modern_train
    gc.collect()

    # Filter extreme non-market outliers (< £5,000 or > £25,000,000) strictly from training partitions
    local_train_df = local_train_df[
        (local_train_df["price"] >= 5000) & (local_train_df["price"] <= 25000000)
    ].reset_index(drop=True)
    full_train_df = full_train_df[
        (full_train_df["price"] >= 5000) & (full_train_df["price"] <= 25000000)
    ].reset_index(drop=True)
    # val_df is strictly left untouched and unfiltered for honest competition-aligned evaluation

    # 4. Trailing Bayesian Price Anchor and Stationary Residual Engineering
    # Fit bounded multi-level trailing-window empirical Bayes price anchors
    local_anchor_model = HierarchicalBayesianAnchor(window_days=365, m_smooth=15.0)
    local_anchor_model.fit(local_train_df, target_col="log10_price")
    local_train_df["anchor_pred"] = local_anchor_model.predict(local_train_df)
    local_train_df["residual"] = (
        local_train_df["log10_price"] - local_train_df["anchor_pred"]
    ).astype(np.float32)

    val_df["anchor_pred"] = local_anchor_model.predict(val_df)
    val_df["residual"] = (val_df["log10_price"] - val_df["anchor_pred"]).astype(
        np.float32
    )

    # Local validation split encodings on stationary residuals (strictly leak-free)
    local_train_oof_te = compute_oof_target_encoding(
        local_train_df, n_splits=5, m_smooth=15.0, target_col="residual"
    )
    for col in local_train_oof_te.columns:
        local_train_df[col] = local_train_oof_te[col]

    local_encoder = HierarchicalBayesianTargetEncoder(m_smooth=15.0)
    local_encoder.fit(local_train_df, target_col="residual")
    val_te = local_encoder.transform(val_df)
    for col in val_te.columns:
        val_df[col] = val_te[col]

    # Full production anchor fitting and stationary residual retraining encodings
    prod_anchor_model = HierarchicalBayesianAnchor(window_days=365, m_smooth=15.0)
    prod_anchor_model.fit(full_train_df, target_col="log10_price")
    full_train_df["anchor_pred"] = prod_anchor_model.predict(full_train_df)
    full_train_df["residual"] = (
        full_train_df["log10_price"] - full_train_df["anchor_pred"]
    ).astype(np.float32)

    test_processed["anchor_pred"] = prod_anchor_model.predict(test_processed)

    full_train_oof_te = compute_oof_target_encoding(
        full_train_df, n_splits=5, m_smooth=15.0, target_col="residual"
    )
    for col in full_train_oof_te.columns:
        full_train_df[col] = full_train_oof_te[col]

    prod_encoder = HierarchicalBayesianTargetEncoder(m_smooth=15.0)
    prod_encoder.fit(full_train_df, target_col="residual")
    test_te = prod_encoder.transform(test_processed)
    for col in test_te.columns:
        test_processed[col] = test_te[col]

    # 5. Frequency & Spatial Liquidity Features
    local_train_df, val_df = compute_frequency_and_spatial_features(
        local_train_df, [local_train_df.copy(), val_df.copy()]
    )
    full_train_df, test_processed = compute_frequency_and_spatial_features(
        full_train_df, [full_train_df.copy(), test_processed.copy()]
    )

    numeric_features = [
        "month",
        "day",
        "dayofweek",
        "is_friday",
        "quarter",
        "dayofyear",
        "sin_month",
        "cos_month",
        "is_new_build_flag",
        "is_standard_sale",
        "tenure_code",
        "property_type_code",
        "te_county",
        "te_district",
        "te_town",
        "te_dist_prop",
        "te_town_prop",
        "te_dist_tenure",
        "te_dist_new",
        "te_prop_tenure_new",
        "te_recent_district",
        "te_district_momentum",
        "anchor_6m",
        "anchor_12m",
        "anchor_24m",
        "momentum_6m_12m",
        "momentum_12m_24m",
        "recent_12m_dist_prop",
        "recent_12m_town_prop",
        "recent_24m_dist_prop",
        "recent_12m_town",
        "recent_12m_district",
        "recent_24m_district",
        "town_to_district_premium",
        "district_momentum_1y_2y",
        "dist_prop_momentum_1y_2y",
        "district_price_dispersion",
        "dist_prop_price_dispersion",
        "dispersion_ratio",
        "log_district_count",
        "log_town_count",
        "log_dist_prop_count",
        "district_flat_ratio",
        "district_new_ratio",
    ]

    categorical_features = [
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
        "town",
        "district",
        "county",
    ]

    features = numeric_features + categorical_features

    # Align categorical encodings across all partitions
    for cat in categorical_features:
        all_cats = sorted(
            list(
                set(local_train_df[cat].dropna().astype(str).unique())
                | set(val_df[cat].dropna().astype(str).unique())
                | set(full_train_df[cat].dropna().astype(str).unique())
                | set(test_processed[cat].dropna().astype(str).unique())
            )
        )
        local_train_df[cat] = pd.Categorical(
            local_train_df[cat].astype(str), categories=all_cats
        )
        val_df[cat] = pd.Categorical(val_df[cat].astype(str), categories=all_cats)
        full_train_df[cat] = pd.Categorical(
            full_train_df[cat].astype(str), categories=all_cats
        )
        test_processed[cat] = pd.Categorical(
            test_processed[cat].astype(str), categories=all_cats
        )

    # PyTorch Entity Embedding Setup
    nn_cat_cols = ["town", "district", "county", "tenure", "property_type"]
    nn_cat_cardinalities = {}
    nn_cat_mappings = {}
    for col in nn_cat_cols:
        unique_vals = sorted(
            list(
                set(full_train_df[col].dropna().astype(str).unique())
                | set(test_processed[col].dropna().astype(str).unique())
            )
        )
        nn_cat_mappings[col] = {
            val: idx + 1 for idx, val in enumerate(unique_vals)
        }
        nn_cat_cardinalities[col] = len(unique_vals)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare PyTorch Tensors for Validation Split
    scaler_mean = (
        local_train_df[numeric_features].mean().values.astype(np.float32)
    )
    scaler_std = local_train_df[numeric_features].std().values.astype(np.float32)
    scaler_std[scaler_std < 1e-6] = 1.0

    val_tr_num = local_train_df[numeric_features].values.astype(np.float32)
    val_tr_num = np.nan_to_num(val_tr_num, nan=0.0, posinf=0.0, neginf=0.0)
    val_tr_num_scaled = (val_tr_num - scaler_mean) / (scaler_std + 1e-7)
    val_tr_num_tensor = torch.from_numpy(val_tr_num_scaled.astype(np.float32))

    val_tr_cat_tensors = {}
    for col in nn_cat_cols:
        idx_vals = (
            local_train_df[col]
            .astype(str)
            .map(nn_cat_mappings[col])
            .fillna(0)
            .astype(np.int64)
            .values
        )
        val_tr_cat_tensors[col] = torch.from_numpy(idx_vals)

    train_weights = compute_recency_weights(
        local_train_df["date"], half_life_years=2.5
    )
    val_tr_y_tensor = torch.from_numpy(
        local_train_df["residual"].values.astype(np.float32)
    )
    val_tr_w_tensor = torch.from_numpy(train_weights.astype(np.float32))

    # 6. Local Validation Training: Heterogeneous Dual-Paradigm Architecture
    X_train = local_train_df[features]
    y_train = local_train_df["residual"].values
    X_val = val_df[features]
    y_val = val_df["residual"].values

    # Model 1: Deeper trees with fine-grained split capacity
    cfg1 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 50,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 3000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    val_model1 = get_lgbm_model(cfg1)
    val_model1.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    best_iter1 = (
        val_model1.best_iteration_
        if hasattr(val_model1, "best_iteration_")
        and val_model1.best_iteration_ is not None
        and val_model1.best_iteration_ > 0
        else 1500
    )

    # Model 2: Moderately shallow trees with regularized subsampling
    cfg2 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 35,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.60,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_estimators": 3000,
        "random_state": 2024,
        "n_jobs": -1,
        "verbose": -1,
    }
    val_model2 = get_lgbm_model(cfg2)
    val_model2.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    best_iter2 = (
        val_model2.best_iteration_
        if hasattr(val_model2, "best_iteration_")
        and val_model2.best_iteration_ is not None
        and val_model2.best_iteration_ > 0
        else 1500
    )

    # Model 3: PyTorch HedonicResNet with continuous spatial entity embeddings
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    val_nn_model = HedonicResNet(
        cat_cardinalities=nn_cat_cardinalities,
        num_numerical_features=len(numeric_features),
        embedding_dim_mult=1.6,
        hidden_dim=256,
        num_res_blocks=3,
        dropout_rate=0.2,
    )
    val_nn_model = train_hedonic_resnet(
        val_nn_model,
        val_tr_cat_tensors,
        val_tr_num_tensor,
        val_tr_y_tensor,
        val_tr_w_tensor,
        epochs=7,
        batch_size=4096,
        lr=1e-3,
        device=device,
    )
    val_nn_predictor = HedonicResNetPredictor(
        model=val_nn_model,
        cat_cols=nn_cat_cols,
        cat_mappings=nn_cat_mappings,
        num_cols=numeric_features,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        device=device,
    )

    # Out-of-Time Validation Residual Predictions
    val_lgb1_res = val_model1.predict(X_val).astype(np.float64)
    val_lgb2_res = val_model2.predict(X_val).astype(np.float64)
    val_nn_res = val_nn_predictor.predict(val_df).astype(np.float64)

    y_val_log10 = val_df["log10_price"].values
    val_anchor = local_anchor_model.predict(val_df).astype(np.float64)

    # Optimize Ensemble Blend Weights on Out-of-Time Validation Split
    def blend_objective(w):
        w1, w2, w3 = w
        b_res = w1 * val_lgb1_res + w2 * val_lgb2_res + w3 * val_nn_res
        b_log10 = val_anchor + b_res
        return compute_log10_rmse(y_val_log10, b_log10, is_log10_scale=True)

    opt_res = minimize(
        blend_objective,
        x0=[0.45, 0.40, 0.15],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    if opt_res.success and np.sum(opt_res.x) > 0:
        optimal_weights = list(opt_res.x / np.sum(opt_res.x))
    else:
        optimal_weights = [0.45, 0.40, 0.15]

    # Reconstruct predictions via two-stage hybrid predictor
    hybrid_val = TwoStageHybridPredictor(
        local_anchor_model,
        [val_model1, val_model2, val_nn_predictor],
        weights=optimal_weights,
    )
    val_preds_log10 = hybrid_val.predict(val_df, features)
    val_preds_raw = np.power(10.0, val_preds_log10)
    val_preds_raw = np.clip(val_preds_raw, 1.0, None)

    val_score = compute_log10_rmse(
        val_df["price"].values, val_preds_raw, is_log10_scale=False
    )

    val_model1.booster_.save_model("./working/lgbm_val_model1.txt")
    val_model2.booster_.save_model("./working/lgbm_val_model2.txt")
    del (
        X_train,
        y_train,
        X_val,
        y_val,
        val_model1,
        val_model2,
        val_nn_model,
        val_nn_predictor,
        hybrid_val,
        val_tr_cat_tensors,
        val_tr_num_tensor,
    )
    gc.collect()

    # 7. Production Retraining on Full Modern Dataset
    X_full = full_train_df[features]
    y_full = full_train_df["residual"].values
    full_weights = compute_recency_weights(
        full_train_df["date"], half_life_years=2.5
    )

    prod_cfg1 = cfg1.copy()
    prod_cfg1["n_estimators"] = max(100, int(best_iter1 * 1.1))
    prod_model1 = get_lgbm_model(prod_cfg1)
    prod_model1.fit(
        X_full,
        y_full,
        sample_weight=full_weights,
        categorical_feature=categorical_features,
        callbacks=[lgb.log_evaluation(period=200)],
    )

    prod_cfg2 = cfg2.copy()
    prod_cfg2["n_estimators"] = max(100, int(best_iter2 * 1.1))
    prod_model2 = get_lgbm_model(prod_cfg2)
    prod_model2.fit(
        X_full,
        y_full,
        sample_weight=full_weights,
        categorical_feature=categorical_features,
        callbacks=[lgb.log_evaluation(period=200)],
    )

    # Retrain HedonicResNet on full modern dataset
    prod_scaler_mean = (
        full_train_df[numeric_features].mean().values.astype(np.float32)
    )
    prod_scaler_std = (
        full_train_df[numeric_features].std().values.astype(np.float32)
    )
    prod_scaler_std[prod_scaler_std < 1e-6] = 1.0

    full_tr_num = full_train_df[numeric_features].values.astype(np.float32)
    full_tr_num = np.nan_to_num(full_tr_num, nan=0.0, posinf=0.0, neginf=0.0)
    full_tr_num_scaled = (full_tr_num - prod_scaler_mean) / (
        prod_scaler_std + 1e-7
    )
    full_tr_num_tensor = torch.from_numpy(full_tr_num_scaled.astype(np.float32))

    full_tr_cat_tensors = {}
    for col in nn_cat_cols:
        idx_vals = (
            full_train_df[col]
            .astype(str)
            .map(nn_cat_mappings[col])
            .fillna(0)
            .astype(np.int64)
            .values
        )
        full_tr_cat_tensors[col] = torch.from_numpy(idx_vals)

    full_tr_y_tensor = torch.from_numpy(
        full_train_df["residual"].values.astype(np.float32)
    )
    full_tr_w_tensor = torch.from_numpy(full_weights.astype(np.float32))

    prod_nn_model = HedonicResNet(
        cat_cardinalities=nn_cat_cardinalities,
        num_numerical_features=len(numeric_features),
        embedding_dim_mult=1.6,
        hidden_dim=256,
        num_res_blocks=3,
        dropout_rate=0.2,
    )
    prod_nn_model = train_hedonic_resnet(
        prod_nn_model,
        full_tr_cat_tensors,
        full_tr_num_tensor,
        full_tr_y_tensor,
        full_tr_w_tensor,
        epochs=7,
        batch_size=4096,
        lr=1e-3,
        device=device,
    )
    prod_nn_predictor = HedonicResNetPredictor(
        model=prod_nn_model,
        cat_cols=nn_cat_cols,
        cat_mappings=nn_cat_mappings,
        num_cols=numeric_features,
        scaler_mean=prod_scaler_mean,
        scaler_std=prod_scaler_std,
        device=device,
    )

    prod_model1.booster_.save_model("./working/lgbm_prod_model1.txt")
    prod_model2.booster_.save_model("./working/lgbm_prod_model2.txt")

    hybrid_prod = TwoStageHybridPredictor(
        prod_anchor_model,
        [prod_model1, prod_model2, prod_nn_predictor],
        weights=optimal_weights,
    )
    test_preds_log10 = hybrid_prod.predict(test_processed, features)
    test_preds_raw = np.power(10.0, test_preds_log10)
    test_preds_raw = np.clip(test_preds_raw, 1.0, None)

    del (
        X_full,
        y_full,
        full_tr_cat_tensors,
        full_tr_num_tensor,
        prod_model1,
        prod_model2,
        prod_nn_model,
        prod_nn_predictor,
        hybrid_prod,
    )
    gc.collect()

    # 8. Format and Save Submission
    sample_sub = pd.read_csv("./input/sample_submission.csv")
    sub_df = pd.DataFrame({"id": test_processed["id"], "price": test_preds_raw})
    final_submission = sample_sub[["id"]].merge(sub_df, on="id", how="left")

    if final_submission["price"].isnull().any():
        final_submission["price"] = final_submission["price"].fillna(
            np.median(test_preds_raw)
        )

    final_submission.to_csv("./submission/submission.csv", index=False)

    # Audits
    assert os.path.exists("./submission/submission.csv")
    assert len(final_submission) == len(sample_sub)
    assert list(final_submission.columns) == ["id", "price"]
    assert not final_submission["price"].isnull().any()

    print(f"Final Validation Score: {val_score}")


if __name__ == "__main__":
    main()