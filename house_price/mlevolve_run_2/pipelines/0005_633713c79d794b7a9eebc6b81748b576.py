import gc
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # Temporal features
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["is_friday"] = (df["dayofweek"] == 4).astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(np.int16)

    # Continuous time trend in years from baseline (2011-01-01) for macro extrapolation
    baseline_date = pd.Timestamp("2011-01-01")
    df["time_trend_years"] = ((df["date"] - baseline_date).dt.days / 365.25).astype(
        np.float32
    )

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
    df["town_prop"] = df["town"] + "___" + df["property_type"]
    df["county_prop"] = df["county"] + "___" + df["property_type"]

    return df


class HierarchicalBayesianTargetEncoder:
    """
    Leak-free Hierarchical Empirical Bayes Target Encoder with multi-level shrinkage on detrended residuals:
    Level 0: Global prior
    Level 1: County (shrunk to Global)
    Level 2: District (shrunk to County)
    Level 3: Town (shrunk to District)
    Level 4: District x Property Type (shrunk to District)
    Level 5: Town x Property Type (shrunk to Town / District x Property Type)
    Level 6: District x Tenure (shrunk to District)
    Level 7: Recent District momentum (last 1.5 years vs prior)
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
        self.recent_dist_stats = {}

    def fit(self, df: pd.DataFrame, target_col: str = "residual"):
        y = df[target_col].values
        self.global_mean = float(np.mean(y))

        # Level 1: County
        c_grp = df.groupby("county")[target_col].agg(["count", "mean"])
        c_shrunk = (c_grp["count"] * c_grp["mean"] + self.m * self.global_mean) / (
            c_grp["count"] + self.m
        )
        self.county_stats = c_shrunk.to_dict()

        # Level 2: District
        d_grp = (
            df.groupby(["district", "county"])[target_col]
            .agg(["count", "mean"])
            .reset_index()
        )
        d_prior = d_grp["county"].map(self.county_stats).fillna(self.global_mean)
        d_grp["d_shrunk"] = (d_grp["count"] * d_grp["mean"] + self.m * d_prior) / (
            d_grp["count"] + self.m
        )
        self.district_stats = dict(zip(d_grp["district"], d_grp["d_shrunk"]))

        # Level 3: Town (shrunk to District)
        town_to_dist = df.groupby("town")["district"].first().to_dict()
        t_grp = df.groupby("town")[target_col].agg(["count", "mean"]).reset_index()
        t_prior = (
            t_grp["town"]
            .map(town_to_dist)
            .map(self.district_stats)
            .fillna(self.global_mean)
        )
        t_grp["t_shrunk"] = (t_grp["count"] * t_grp["mean"] + self.m * t_prior) / (
            t_grp["count"] + self.m
        )
        self.town_stats = dict(zip(t_grp["town"], t_grp["t_shrunk"]))

        # Level 4: District x Property Type
        dp_grp = (
            df.groupby(["district_prop", "district"])[target_col]
            .agg(["count", "mean"])
            .reset_index()
        )
        dp_prior = dp_grp["district"].map(self.district_stats).fillna(self.global_mean)
        dp_grp["dp_shrunk"] = (dp_grp["count"] * dp_grp["mean"] + self.m * dp_prior) / (
            dp_grp["count"] + self.m
        )
        self.dist_prop_stats = dict(zip(dp_grp["district_prop"], dp_grp["dp_shrunk"]))

        # Level 5: Town x Property Type
        tp_grp = (
            df.groupby(["town_prop", "district_prop", "town"])[target_col]
            .agg(["count", "mean"])
            .reset_index()
        )
        tp_prior = (
            tp_grp["town"]
            .map(self.town_stats)
            .fillna(
                tp_grp["district_prop"]
                .map(self.dist_prop_stats)
                .fillna(self.global_mean)
            )
        )
        tp_grp["tp_shrunk"] = (tp_grp["count"] * tp_grp["mean"] + self.m * tp_prior) / (
            tp_grp["count"] + self.m
        )
        self.town_prop_stats = dict(zip(tp_grp["town_prop"], tp_grp["tp_shrunk"]))

        # Level 6: District x Tenure
        dt_grp = (
            df.groupby(["district", "prop_tenure"])[target_col]
            .agg(["count", "mean"])
            .reset_index()
        )
        dt_key = dt_grp["district"] + "___" + dt_grp["prop_tenure"]
        dt_prior = dt_grp["district"].map(self.district_stats).fillna(self.global_mean)
        dt_shrunk = (dt_grp["count"] * dt_grp["mean"] + self.m * dt_prior) / (
            dt_grp["count"] + self.m
        )
        self.dist_tenure_stats = dict(zip(dt_key, dt_shrunk))

        # Level 7: Recent momentum (last 1.5 years vs prior)
        recent_mask = df["date"] >= (df["date"].max() - pd.Timedelta(days=540))
        if recent_mask.sum() > 1000:
            rec_grp = (
                df[recent_mask]
                .groupby("district")[target_col]
                .agg(["count", "mean"])
                .reset_index()
            )
            rec_prior = (
                rec_grp["district"].map(self.district_stats).fillna(self.global_mean)
            )
            rec_shrunk = (rec_grp["count"] * rec_grp["mean"] + self.m * rec_prior) / (
                rec_grp["count"] + self.m
            )
            self.recent_dist_stats = dict(zip(rec_grp["district"], rec_shrunk))
        else:
            self.recent_dist_stats = self.district_stats.copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # County TE
        out["te_county"] = (
            df["county"]
            .map(self.county_stats)
            .fillna(self.global_mean)
            .astype(np.float32)
        )

        # District TE
        out["te_district"] = (
            df["district"]
            .map(self.district_stats)
            .fillna(out["te_county"])
            .astype(np.float32)
        )

        # Town TE
        out["te_town"] = (
            df["town"]
            .map(self.town_stats)
            .fillna(out["te_district"])
            .astype(np.float32)
        )

        # District x Property TE
        out["te_dist_prop"] = (
            df["district_prop"]
            .map(self.dist_prop_stats)
            .fillna(out["te_district"])
            .astype(np.float32)
        )

        # Town x Property TE
        out["te_town_prop"] = (
            df["town_prop"]
            .map(self.town_prop_stats)
            .fillna(out["te_town"])
            .astype(np.float32)
        )

        # District x Tenure TE
        dt_key = df["district"] + "___" + df["prop_tenure"]
        out["te_dist_tenure"] = (
            dt_key.map(self.dist_tenure_stats)
            .fillna(out["te_district"])
            .astype(np.float32)
        )

        # Recent District momentum
        out["te_recent_district"] = (
            df["district"]
            .map(self.recent_dist_stats)
            .fillna(out["te_district"])
            .astype(np.float32)
        )
        out["te_district_momentum"] = (
            out["te_recent_district"] - out["te_district"]
        ).astype(np.float32)

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
        "te_recent_district",
        "te_district_momentum",
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
        "learning_rate": 0.04,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
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


class LinearTrendModel:
    """
    Parametric linear time-trend extrapolator.
    Fits macro and regional slope parameters on time_trend_years against log10 price
    to capture baseline temporal market appreciation over time.
    """

    def __init__(self, m_smooth: float = 100.0):
        self.m_smooth = m_smooth
        self.macro_intercept = 0.0
        self.macro_slope = 0.0
        self.regional_slopes = {}

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "log10_price",
        region_col: str = "county",
    ):
        t = df["time_trend_years"].values.astype(np.float64)
        y = df[target_col].values.astype(np.float64)

        t_mean = np.mean(t)
        y_mean = np.mean(y)

        var_t = np.var(t)
        cov_ty = np.mean((t - t_mean) * (y - y_mean))
        self.macro_slope = float(cov_ty / (var_t + 1e-9))
        self.macro_intercept = float(y_mean - self.macro_slope * t_mean)

        # Regional slopes shrunk towards macro slope
        reg_groups = df.groupby(region_col)
        self.regional_slopes = {}
        for reg_name, grp in reg_groups:
            n = len(grp)
            t_reg = grp["time_trend_years"].values.astype(np.float64)
            y_reg = grp[target_col].values.astype(np.float64)
            var_reg = np.var(t_reg)
            if var_reg > 1e-6:
                cov_reg = np.mean((t_reg - np.mean(t_reg)) * (y_reg - np.mean(y_reg)))
                raw_slope = cov_reg / var_reg
            else:
                raw_slope = self.macro_slope

            shrunk_slope = (n * raw_slope + self.m_smooth * self.macro_slope) / (
                n + self.m_smooth
            )
            self.regional_slopes[reg_name] = float(shrunk_slope)

        return self

    def predict(
        self, df: pd.DataFrame, region_col: str = "county"
    ) -> np.ndarray:
        t = df["time_trend_years"].values.astype(np.float64)
        regions = df[region_col].astype(str).values
        slopes = np.array(
            [self.regional_slopes.get(r, self.macro_slope) for r in regions],
            dtype=np.float64,
        )
        trend = self.macro_intercept + slopes * t
        return trend.astype(np.float32)


class TwoStageHybridPredictor:
    """
    Two-stage hybrid architecture combining a parametric linear time-trend extrapolator
    with a LightGBM hedonic residual regressor.
    Reconstructs final log10 predictions by adding the extrapolated trend back to the GBDT predictions.
    """

    def __init__(self, trend_model: LinearTrendModel, gbdt_model: lgb.LGBMRegressor):
        self.trend_model = trend_model
        self.gbdt_model = gbdt_model

    def predict(
        self,
        df: pd.DataFrame,
        features: List[str],
        region_col: str = "county",
    ) -> np.ndarray:
        trend_preds = self.trend_model.predict(df, region_col=region_col)
        residual_preds = self.gbdt_model.predict(df[features])
        log10_preds = (trend_preds + residual_preds).astype(np.float64)
        return log10_preds


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

    # 3. Create Splits
    # Validation mirrors the 6-month test period (2016-07-01 to 2016-12-31)
    val_cutoff = pd.Timestamp("2016-07-01")
    local_train_mask = modern_train["date"] < val_cutoff
    val_mask = modern_train["date"] >= val_cutoff

    local_train_df = modern_train[local_train_mask].copy().reset_index(drop=True)
    val_df = modern_train[val_mask].copy().reset_index(drop=True)
    full_train_df = modern_train.copy().reset_index(drop=True)
    del modern_train
    gc.collect()

    # 4. Detrending and Bayesian Target Encoding on Stationary Residuals
    # Fit temporal trend first to obtain stationary log10 price residuals
    local_trend_model = LinearTrendModel(m_smooth=100.0)
    local_trend_model.fit(local_train_df, target_col="log10_price")
    local_train_df["trend_pred"] = local_trend_model.predict(local_train_df)
    local_train_df["residual"] = (
        local_train_df["log10_price"] - local_train_df["trend_pred"]
    ).astype(np.float32)

    val_df["trend_pred"] = local_trend_model.predict(val_df)
    val_df["residual"] = (val_df["log10_price"] - val_df["trend_pred"]).astype(
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

    # Full production detrending and retraining encodings
    prod_trend_model = LinearTrendModel(m_smooth=100.0)
    prod_trend_model.fit(full_train_df, target_col="log10_price")
    full_train_df["trend_pred"] = prod_trend_model.predict(full_train_df)
    full_train_df["residual"] = (
        full_train_df["log10_price"] - full_train_df["trend_pred"]
    ).astype(np.float32)

    test_processed["trend_pred"] = prod_trend_model.predict(test_processed)

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
        "time_trend_years",
        "year",
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
        "te_recent_district",
        "te_district_momentum",
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

    # 6. Local Validation Training with Recency Weighting
    X_train = local_train_df[features]
    y_train = local_train_df["residual"].values
    X_val = val_df[features]
    y_val = val_df["residual"].values

    train_weights = compute_recency_weights(
        local_train_df["date"], half_life_years=3.0
    )

    val_model = get_lgbm_model()
    val_model.fit(
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

    best_iter = (
        val_model.best_iteration_
        if hasattr(val_model, "best_iteration_") and val_model.best_iteration_ > 0
        else 1500
    )

    # Reconstruct predictions via two-stage hybrid predictor
    hybrid_val = TwoStageHybridPredictor(local_trend_model, val_model)
    val_preds_log10 = hybrid_val.predict(val_df, features)
    val_preds_raw = np.power(10.0, val_preds_log10)
    val_preds_raw = np.clip(val_preds_raw, 1.0, None)

    val_score = compute_log10_rmse(
        val_df["price"].values, val_preds_raw, is_log10_scale=False
    )

    val_model.booster_.save_model("./working/lgbm_val_model.txt")
    del X_train, y_train, X_val, y_val, val_model, hybrid_val
    gc.collect()

    # 7. Production Retraining & Test Inference
    X_full = full_train_df[features]
    y_full = full_train_df["residual"].values
    X_test = test_processed[features]

    full_weights = compute_recency_weights(
        full_train_df["date"], half_life_years=3.0
    )

    n_prod_estimators = max(100, int(best_iter * 1.1))
    prod_model = get_lgbm_model(custom_params={"n_estimators": n_prod_estimators})
    prod_model.fit(
        X_full,
        y_full,
        sample_weight=full_weights,
        categorical_feature=categorical_features,
        callbacks=[lgb.log_evaluation(period=200)],
    )

    prod_model.booster_.save_model("./working/lgbm_prod_model.txt")

    hybrid_prod = TwoStageHybridPredictor(prod_trend_model, prod_model)
    test_preds_log10 = hybrid_prod.predict(test_processed, features)
    test_preds_raw = np.power(10.0, test_preds_log10)
    test_preds_raw = np.clip(test_preds_raw, 1.0, None)

    del X_full, y_full, X_test, prod_model, hybrid_prod
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