import gc
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42):
    """Ensure determinism across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directory_structure():
    """Ensure required output directories exist."""
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)


def clean_text_series(s: pd.Series) -> pd.Series:
    """Normalize string categorical fields."""
    return s.fillna("UNKNOWN").astype(str).str.strip().str.upper()


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive continuous trend, weekly cadence, and cyclical seasonality features."""
    dt = pd.to_datetime(df["date"])
    df["year"] = dt.dt.year.astype(np.int16)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["is_friday"] = (df["dayofweek"] == 4).astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    base_date = pd.Timestamp("2012-01-01")
    df["time_trend"] = ((dt - base_date).dt.total_seconds() / (365.25 * 86400)).astype(
        np.float32
    )

    dayofyear = dt.dt.dayofyear.astype(np.float32)
    df["sin_doy"] = np.sin(2.0 * np.pi * dayofyear / 365.25).astype(np.float32)
    df["cos_doy"] = np.cos(2.0 * np.pi * dayofyear / 365.25).astype(np.float32)
    df["sin_month"] = np.sin(2.0 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2.0 * np.pi * df["month"] / 12.0).astype(np.float32)
    return df


def create_interaction_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Construct compound spatial and property attribute interaction hierarchies."""
    df["town_district"] = df["town"] + "__" + df["district"]
    df["district_county"] = df["district"] + "__" + df["county"]
    df["dist_prop_type"] = df["district"] + "__" + df["property_type"]
    df["town_prop_type"] = df["town"] + "__" + df["property_type"]
    df["type_new"] = df["property_type"] + "_" + df["is_new_build"]
    df["type_tenure"] = df["property_type"] + "_" + df["tenure"]
    df["type_cat"] = df["property_type"] + "_" + df["sale_category"]
    return df


def compute_hierarchical_target_encodings(
    train_ref_df: pd.DataFrame,
    target_col: str = "target",
    m_county: float = 50.0,
    m_dist: float = 30.0,
    m_town: float = 20.0,
    m_dist_type: float = 15.0,
    m_town_type: float = 10.0,
    m_sale_cat: float = 20.0,
    m_type_new: float = 20.0,
) -> Dict[str, Any]:
    """Fit hierarchical empirical Bayes target encodings on stationarized price residuals."""
    if "year" in train_ref_df.columns and "month" in train_ref_df.columns:
        macro_trend = (
            train_ref_df.groupby(["year", "month"])[target_col]
            .transform("median")
            .values
        )
    else:
        macro_trend = float(train_ref_df[target_col].median())

    residuals = (train_ref_df[target_col].values - macro_trend).astype(np.float64)
    weights = np.ones(len(train_ref_df), dtype=np.float64)
    weighted_y = residuals

    total_weight = float(np.sum(weights))
    global_mean = float(np.sum(weighted_y) / (total_weight + 1e-8))

    agg_df = pd.DataFrame(
        {
            "county": train_ref_df["county"].values,
            "district": train_ref_df["district"].values,
            "town": train_ref_df["town"].values,
            "dist_prop_type": train_ref_df["dist_prop_type"].values,
            "town_prop_type": train_ref_df["town_prop_type"].values,
            "sale_category": train_ref_df["sale_category"].values,
            "type_new": train_ref_df["type_new"].values,
            "w": weights,
            "wy": weighted_y,
        }
    )

    # County
    c_stats = agg_df.groupby("county", observed=False)[["w", "wy"]].sum()
    county_enc = (
        (c_stats["wy"] + m_county * global_mean) / (c_stats["w"] + m_county)
    ).to_dict()

    # District
    d_stats = (
        agg_df.groupby(["district", "county"], observed=False)[["w", "wy"]]
        .sum()
        .reset_index()
    )
    d_stats["county_prior"] = d_stats["county"].map(county_enc).fillna(global_mean)
    d_stats["enc"] = (d_stats["wy"] + m_dist * d_stats["county_prior"]) / (
        d_stats["w"] + m_dist
    )
    dist_enc = dict(zip(d_stats["district"], d_stats["enc"]))

    # Town
    t_stats = (
        agg_df.groupby(["town", "district"], observed=False)[["w", "wy"]]
        .sum()
        .reset_index()
    )
    t_stats["dist_prior"] = t_stats["district"].map(dist_enc).fillna(global_mean)
    t_stats["enc"] = (t_stats["wy"] + m_town * t_stats["dist_prior"]) / (
        t_stats["w"] + m_town
    )
    town_enc = dict(zip(t_stats["town"], t_stats["enc"]))

    # District x Property Type
    dt_stats = (
        agg_df.groupby(["dist_prop_type", "district"], observed=False)[["w", "wy"]]
        .sum()
        .reset_index()
    )
    dt_stats["dist_prior"] = dt_stats["district"].map(dist_enc).fillna(global_mean)
    dt_stats["enc"] = (dt_stats["wy"] + m_dist_type * dt_stats["dist_prior"]) / (
        dt_stats["w"] + m_dist_type
    )
    dt_enc = dict(zip(dt_stats["dist_prop_type"], dt_stats["enc"]))

    # Town x Property Type
    tt_stats = (
        agg_df.groupby(["town_prop_type", "dist_prop_type"], observed=False)[
            ["w", "wy"]
        ]
        .sum()
        .reset_index()
    )
    tt_stats["dt_prior"] = tt_stats["dist_prop_type"].map(dt_enc).fillna(global_mean)
    tt_stats["enc"] = (tt_stats["wy"] + m_town_type * tt_stats["dt_prior"]) / (
        tt_stats["w"] + m_town_type
    )
    tt_enc = dict(zip(tt_stats["town_prop_type"], tt_stats["enc"]))

    # Sale Category
    sc_stats = agg_df.groupby("sale_category", observed=False)[["w", "wy"]].sum()
    sale_cat_enc = (
        (sc_stats["wy"] + m_sale_cat * global_mean) / (sc_stats["w"] + m_sale_cat)
    ).to_dict()

    # Property Type x New Build
    tn_stats = agg_df.groupby("type_new", observed=False)[["w", "wy"]].sum()
    type_new_enc = (
        (tn_stats["wy"] + m_type_new * global_mean) / (tn_stats["w"] + m_type_new)
    ).to_dict()

    return {
        "global_mean": global_mean,
        "county_enc": county_enc,
        "dist_enc": dist_enc,
        "town_enc": town_enc,
        "dt_enc": dt_enc,
        "tt_enc": tt_enc,
        "sale_cat_enc": sale_cat_enc,
        "type_new_enc": type_new_enc,
    }


def apply_target_encodings(df: pd.DataFrame, encodings: Dict[str, Any]) -> pd.DataFrame:
    """Apply fitted hierarchical target encodings with fallback to parent levels."""
    gm = encodings["global_mean"]
    df["te_county"] = (
        df["county"].map(encodings["county_enc"]).fillna(gm).astype(np.float32)
    )
    df["te_district"] = (
        df["district"]
        .map(encodings["dist_enc"])
        .fillna(df["te_county"])
        .astype(np.float32)
    )
    df["te_town"] = (
        df["town"]
        .map(encodings["town_enc"])
        .fillna(df["te_district"])
        .astype(np.float32)
    )
    df["te_dist_prop"] = (
        df["dist_prop_type"]
        .map(encodings["dt_enc"])
        .fillna(df["te_district"])
        .astype(np.float32)
    )
    df["te_town_prop"] = (
        df["town_prop_type"]
        .map(encodings["tt_enc"])
        .fillna(df["te_dist_prop"])
        .astype(np.float32)
    )
    df["te_sale_cat"] = (
        df["sale_category"]
        .map(encodings.get("sale_cat_enc", {}))
        .fillna(gm)
        .astype(np.float32)
    )
    df["te_type_new"] = (
        df["type_new"]
        .map(encodings.get("type_new_enc", {}))
        .fillna(gm)
        .astype(np.float32)
    )
    return df


def add_oof_target_encodings(
    train_df: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Compute strictly leakage-free Out-Of-Fold hierarchical target encodings on training data."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    feature_cols = [
        "te_county",
        "te_district",
        "te_town",
        "te_dist_prop",
        "te_town_prop",
        "te_sale_cat",
        "te_type_new",
    ]
    oof_arrays = {
        col: np.zeros(len(train_df), dtype=np.float32) for col in feature_cols
    }

    for train_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        encs = compute_hierarchical_target_encodings(tr_fold)
        val_fold_transformed = apply_target_encodings(val_fold.copy(), encs)
        for col in feature_cols:
            oof_arrays[col][val_idx] = val_fold_transformed[col].values

    for col in feature_cols:
        train_df[col] = oof_arrays[col]

    return train_df


def compute_district_momentum(train_ref_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate district-level trailing short-term (1.5-year) vs older log10 price growth velocity with shrinkage."""
    t_max_date = train_ref_df["date"].max()
    cutoff_date = t_max_date - pd.DateOffset(months=18)
    recent_mask = train_ref_df["date"] >= cutoff_date
    older_mask = train_ref_df["date"] < cutoff_date

    mean_recent = train_ref_df[recent_mask].groupby("district")["target"].mean()
    count_recent = train_ref_df[recent_mask].groupby("district")["target"].count()
    mean_older = train_ref_df[older_mask].groupby("district")["target"].mean()
    count_older = train_ref_df[older_mask].groupby("district")["target"].count()

    raw_diff = mean_recent - mean_older
    global_mom = float(
        train_ref_df[recent_mask]["target"].mean()
        - train_ref_df[older_mask]["target"].mean()
    )

    eff_n = (count_recent * count_older) / (count_recent + count_older + 1e-6)
    shrunk_diff = (eff_n * raw_diff + 15.0 * global_mom) / (eff_n + 15.0)

    momentum = shrunk_diff.dropna().to_dict()
    return {"momentum_dict": momentum, "global_momentum": global_mom}


def calculate_embedding_dim(cardinality: int) -> int:
    """Heuristic rule for tabular categorical embedding dimensionality."""
    return max(4, min(64, int(1.6 * (cardinality**0.56))))


class TabularResidualBlock(nn.Module):
    """Dense residual block with LayerNorm, SiLU, and Dropout."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.act = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act(self.linear1(out))
        out = self.dropout1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class SemiParametricTabResNet(nn.Module):
    """Semi-Parametric Tabular Residual Network with linear macro-trend extrapolation bypass."""

    def __init__(
        self,
        categorical_cardinalities: Dict[str, int],
        numeric_feature_dim: int,
        time_trend_idx: int = 0,
        d_model: int = 256,
        d_hidden: int = 512,
        num_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.time_trend_idx = time_trend_idx
        self.cat_keys = list(categorical_cardinalities.keys())

        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in categorical_cardinalities.items():
            emb_dim = calculate_embedding_dim(card)
            self.embeddings[col] = nn.Embedding(
                num_embeddings=card + 1, embedding_dim=emb_dim
            )
            total_emb_dim += emb_dim

        self.emb_dropout = nn.Dropout(dropout)

        self.trend_bypass = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.trend_bypass.weight, 0.05)
        nn.init.constant_(self.trend_bypass.bias, 5.0)

        self.dist_cat_idx = (
            self.cat_keys.index("district") if "district" in self.cat_keys else None
        )
        self.county_cat_idx = (
            self.cat_keys.index("county") if "county" in self.cat_keys else None
        )
        self.prop_cat_idx = (
            self.cat_keys.index("property_type")
            if "property_type" in self.cat_keys
            else None
        )

        dist_card = categorical_cardinalities.get("district", 512)
        county_card = categorical_cardinalities.get("county", 128)
        prop_card = categorical_cardinalities.get("property_type", 16)

        self.trend_bypass_dist_slope = nn.Embedding(dist_card + 1, 1)
        self.trend_bypass_dist_intercept = nn.Embedding(dist_card + 1, 1)
        self.trend_bypass_county_slope = nn.Embedding(county_card + 1, 1)
        self.trend_bypass_county_intercept = nn.Embedding(county_card + 1, 1)
        self.trend_bypass_prop_slope = nn.Embedding(prop_card + 1, 1)
        self.trend_bypass_prop_intercept = nn.Embedding(prop_card + 1, 1)

        nn.init.zeros_(self.trend_bypass_dist_slope.weight)
        nn.init.zeros_(self.trend_bypass_dist_intercept.weight)
        nn.init.zeros_(self.trend_bypass_county_slope.weight)
        nn.init.zeros_(self.trend_bypass_county_intercept.weight)
        nn.init.zeros_(self.trend_bypass_prop_slope.weight)
        nn.init.zeros_(self.trend_bypass_prop_intercept.weight)

        input_dim = total_emb_dim + numeric_feature_dim - 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(
                    d_model=d_model, d_hidden=d_hidden, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.head_norm = nn.LayerNorm(d_model)
        self.non_linear_head = nn.Linear(d_model, 1)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        emb_list = []
        for i, col in enumerate(self.cat_keys):
            col_tokens = x_cat[:, i]
            emb_list.append(self.embeddings[col](col_tokens))

        x_emb = torch.cat(emb_list, dim=-1)
        x_emb = self.emb_dropout(x_emb)

        t = x_num[:, self.time_trend_idx : self.time_trend_idx + 1]
        if self.time_trend_idx == 0:
            x_num_stationary = x_num[:, 1:]
        elif self.time_trend_idx == x_num.shape[1] - 1:
            x_num_stationary = x_num[:, :-1]
        else:
            x_num_stationary = torch.cat(
                [x_num[:, : self.time_trend_idx], x_num[:, self.time_trend_idx + 1 :]],
                dim=-1,
            )

        x_combined = torch.cat([x_emb, x_num_stationary], dim=-1)
        h = self.input_proj(x_combined)

        for block in self.res_blocks:
            h = block(h)

        h = self.head_norm(h)
        res_pred = self.non_linear_head(h)

        slope = self.trend_bypass.weight
        intercept = self.trend_bypass.bias

        if self.dist_cat_idx is not None:
            dist_tok = x_cat[:, self.dist_cat_idx]
            slope = slope + self.trend_bypass_dist_slope(dist_tok)
            intercept = intercept + self.trend_bypass_dist_intercept(dist_tok)

        if self.county_cat_idx is not None:
            county_tok = x_cat[:, self.county_cat_idx]
            slope = slope + self.trend_bypass_county_slope(county_tok)
            intercept = intercept + self.trend_bypass_county_intercept(county_tok)

        if self.prop_cat_idx is not None:
            prop_tok = x_cat[:, self.prop_cat_idx]
            slope = slope + self.trend_bypass_prop_slope(prop_tok)
            intercept = intercept + self.trend_bypass_prop_intercept(prop_tok)

        trend_pred = slope * t + intercept

        return (trend_pred + res_pred).squeeze(-1)


class RobustCompetitionLoss(nn.Module):
    """Competition RMSE loss with Huber softening for extreme outliers."""

    def __init__(self, delta: float = 1.5, eps: float = 1e-8):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred.view(-1) - y_true.view(-1)
        abs_diff = torch.abs(diff)
        huber_sq = torch.where(
            abs_diff <= self.delta,
            diff**2,
            2.0 * self.delta * abs_diff - self.delta**2,
        )
        return torch.sqrt(torch.mean(huber_sq) + self.eps)


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    total_steps: int = 10000,
) -> Tuple[torch.optim.Optimizer, Any]:
    """AdamW optimizer with parameter-group specific learning rates."""
    embedding_params = []
    trend_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "trend" in name:
            trend_params.append(param)
        elif "embeddings" in name:
            embedding_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr, "weight_decay": weight_decay},
        {"params": embedding_params, "lr": lr * 1.5, "weight_decay": 1e-5},
        {"params": trend_params, "lr": lr * 0.2, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.01
    )
    return optimizer, scheduler


class TabularDataset(Dataset):
    """Memory-efficient PyTorch Dataset for tabular categorical, numeric, and target tensors."""

    def __init__(
        self,
        cat_data: np.ndarray,
        num_data: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ):
        self.cat_data = torch.from_numpy(cat_data).long()
        self.num_data = torch.from_numpy(num_data).float()
        self.targets = (
            torch.from_numpy(targets).float() if targets is not None else None
        )

    def __len__(self) -> int:
        return len(self.cat_data)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return self.cat_data[idx], self.num_data[idx], self.targets[idx]
        return self.cat_data[idx], self.num_data[idx]


def prepare_tabular_matrices(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    cardinalities: Dict[str, int],
    target_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sanitize, clamp categorical tokens to embedding vocabularies, and format arrays."""
    cat_arrays = []
    for col in cat_cols:
        vals = df[col].values.astype(np.int64)
        card = cardinalities[col]
        sanitized_vals = np.where((vals < 0) | (vals >= card), card, vals)
        cat_arrays.append(sanitized_vals[:, None])

    x_cat = np.concatenate(cat_arrays, axis=1)
    x_num = np.nan_to_num(
        df[num_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    targets = (
        df[target_col].values.astype(np.float32)
        if target_col and target_col in df.columns
        else None
    )
    return x_cat, x_num, targets


def compute_competition_metric(
    pred_log10: np.ndarray, true_prices: np.ndarray
) -> float:
    """Compute official competition RMSE on log10(price). Predictions below 1 clipped to 1."""
    pred_prices = np.clip(10.0**pred_log10, 1.0, None)
    log10_pred = np.log10(pred_prices)
    log10_true = np.log10(np.clip(true_prices, 1.0, None))
    return float(np.sqrt(np.mean((log10_pred - log10_true) ** 2)))


def main():
    seed_everything(42)
    create_directory_structure()

    # Load raw datasets
    train_full_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_csv("./input/test.csv")

    # Standardize date types to datetime64 across both datasets
    train_full_raw["date"] = pd.to_datetime(train_full_raw["date"])
    test_raw["date"] = pd.to_datetime(test_raw["date"])

    # Focus on contemporary post-recession regime (2012-01-01 to 2016-12-31) and filter administrative outliers
    train_recent = (
        train_full_raw[
            (train_full_raw["date"] >= pd.Timestamp("2012-01-01"))
            & (train_full_raw["price"] >= 3000)
            & (train_full_raw["price"] <= 25000000)
        ]
        .copy()
        .reset_index(drop=True)
    )
    del train_full_raw
    gc.collect()

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
        train_recent[col] = clean_text_series(train_recent[col])
        test_raw[col] = clean_text_series(test_raw[col])

    # Official target: log10(price), clipped to 1.0 per task specification
    train_recent["target"] = np.log10(
        np.clip(train_recent["price"].values, 1.0, None)
    ).astype(np.float32)

    # Feature engineering
    train_recent = extract_temporal_features(train_recent)
    test_df = extract_temporal_features(test_raw.copy())

    train_recent = create_interaction_keys(train_recent)
    test_df = create_interaction_keys(test_df)

    # Out-of-time validation partition (Train: 2012-01 to 2016-06, Val: 2016-07 to 2016-12)
    val_cutoff = pd.Timestamp("2016-07-01")
    tr_split_mask = train_recent["date"] < val_cutoff
    val_split_mask = train_recent["date"] >= val_cutoff

    train_split = train_recent[tr_split_mask].copy().reset_index(drop=True)
    val_split = train_recent[val_split_mask].copy().reset_index(drop=True)

    # Validation pipeline encodings
    val_encodings = compute_hierarchical_target_encodings(train_split)
    val_momentum = compute_district_momentum(train_split)

    train_split = add_oof_target_encodings(train_split)
    val_split = apply_target_encodings(val_split, val_encodings)

    train_split["dist_momentum"] = (
        train_split["district"]
        .map(val_momentum["momentum_dict"])
        .fillna(val_momentum["global_momentum"])
        .astype(np.float32)
    )
    val_split["dist_momentum"] = (
        val_split["district"]
        .map(val_momentum["momentum_dict"])
        .fillna(val_momentum["global_momentum"])
        .astype(np.float32)
    )

    for f_col in ["district", "town", "town_district", "dist_prop_type"]:
        freq_map = train_split[f_col].value_counts().to_dict()
        train_split[f"{f_col}_freq"] = (
            train_split[f_col].map(freq_map).fillna(0).astype(np.int32)
        )
        val_split[f"{f_col}_freq"] = (
            val_split[f_col].map(freq_map).fillna(0).astype(np.int32)
        )

    # Full test pipeline encodings
    full_encodings = compute_hierarchical_target_encodings(train_recent)
    full_momentum = compute_district_momentum(train_recent)

    test_processed = apply_target_encodings(test_df, full_encodings)
    test_processed["dist_momentum"] = (
        test_processed["district"]
        .map(full_momentum["momentum_dict"])
        .fillna(full_momentum["global_momentum"])
        .astype(np.float32)
    )

    for f_col in ["district", "town", "town_district", "dist_prop_type"]:
        freq_map = train_recent[f_col].value_counts().to_dict()
        test_processed[f"{f_col}_freq"] = (
            test_processed[f_col].map(freq_map).fillna(0).astype(np.int32)
        )

    # Full train dataset processed for post-validation fine-tuning
    full_train_processed = apply_target_encodings(train_recent.copy(), full_encodings)
    full_train_processed["dist_momentum"] = (
        full_train_processed["district"]
        .map(full_momentum["momentum_dict"])
        .fillna(full_momentum["global_momentum"])
        .astype(np.float32)
    )
    for f_col in ["district", "town", "town_district", "dist_prop_type"]:
        freq_map = train_recent[f_col].value_counts().to_dict()
        full_train_processed[f"{f_col}_freq"] = (
            full_train_processed[f_col].map(freq_map).fillna(0).astype(np.int32)
        )

    # Integer encode nominal categoricals
    encoding_cats = [
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
        "county",
        "district",
        "town",
        "type_new",
        "type_tenure",
        "type_cat",
        "dist_prop_type",
    ]
    for col in encoding_cats:
        cat_type = pd.CategoricalDtype(categories=train_recent[col].unique())
        train_split[col] = train_split[col].astype(cat_type).cat.codes.astype(np.int32)
        val_split[col] = val_split[col].astype(cat_type).cat.codes.astype(np.int32)
        test_processed[col] = (
            test_processed[col].astype(cat_type).cat.codes.astype(np.int32)
        )
        full_train_processed[col] = (
            full_train_processed[col].astype(cat_type).cat.codes.astype(np.int32)
        )

    numeric_features = [
        "time_trend",
        "month",
        "day",
        "dayofweek",
        "is_friday",
        "is_weekend",
        "sin_doy",
        "cos_doy",
        "sin_month",
        "cos_month",
        "te_county",
        "te_district",
        "te_town",
        "te_dist_prop",
        "te_town_prop",
        "te_sale_cat",
        "te_type_new",
        "dist_momentum",
        "district_freq",
        "town_freq",
        "town_district_freq",
        "dist_prop_type_freq",
    ]

    # Cardinalities
    cardinalities = {}
    for col in encoding_cats:
        max_val = max(
            int(train_split[col].max()),
            int(val_split[col].max()),
            int(test_processed[col].max()),
        )
        cardinalities[col] = max(max_val + 1, 16)

    # Build matrix representations
    x_train_cat, x_train_num, y_train = prepare_tabular_matrices(
        train_split, encoding_cats, numeric_features, cardinalities, "target"
    )
    x_val_cat, x_val_num, y_val = prepare_tabular_matrices(
        val_split, encoding_cats, numeric_features, cardinalities, "target"
    )
    x_test_cat, x_test_num, _ = prepare_tabular_matrices(
        test_processed, encoding_cats, numeric_features, cardinalities, target_col=None
    )
    x_full_cat, x_full_num, y_full = prepare_tabular_matrices(
        full_train_processed, encoding_cats, numeric_features, cardinalities, "target"
    )

    batch_size = 4096
    num_workers = 4

    train_dataset = TabularDataset(x_train_cat, x_train_num, y_train)
    val_dataset = TabularDataset(x_val_cat, x_val_num, y_val)
    test_dataset = TabularDataset(x_test_cat, x_test_num)
    full_train_dataset = TabularDataset(x_full_cat, x_full_num, y_full)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_trend_idx = (
        numeric_features.index("time_trend") if "time_trend" in numeric_features else 0
    )

    model = SemiParametricTabResNet(
        categorical_cardinalities=cardinalities,
        numeric_feature_dim=len(numeric_features),
        time_trend_idx=time_trend_idx,
        d_model=256,
        d_hidden=512,
        num_blocks=3,
        dropout=0.15,
    )
    model.to(device)

    criterion = nn.MSELoss()

    max_epochs = 12
    total_steps = max_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=1e-3, weight_decay=1e-4, total_steps=total_steps
    )

    best_val_score = float("inf")
    patience = 3
    patience_counter = 0
    best_model_path = "./working/best_model.pt"

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for x_cat_b, x_num_b, y_b in train_loader:
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(x_cat_b, x_num_b)
            loss = criterion(preds, y_b)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * len(y_b)
            total_samples += len(y_b)

        train_loss = running_loss / total_samples

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_cat_b, x_num_b, _ in val_loader:
                x_cat_b = x_cat_b.to(device, non_blocking=True)
                x_num_b = x_num_b.to(device, non_blocking=True)
                preds = model(x_cat_b, x_num_b)
                val_preds_list.append(preds.cpu().numpy())

        val_preds = np.concatenate(val_preds_list, axis=0)
        val_rmse = compute_competition_metric(val_preds, val_split["price"].values)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Log10-RMSE: {val_rmse:.4f} - LR: {current_lr:.6f}"
        )

        if val_rmse < best_val_score:
            best_val_score = val_rmse
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Compute Final Hold-out Validation Score
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    final_val_preds_list = []
    with torch.no_grad():
        for x_cat_b, x_num_b, _ in val_loader:
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            preds = model(x_cat_b, x_num_b)
            final_val_preds_list.append(preds.cpu().numpy())

    final_val_preds = np.concatenate(final_val_preds_list, axis=0)
    final_score = compute_competition_metric(final_val_preds, val_split["price"].values)

    print(f"Final Validation Score: {final_score}")

    # Post-validation 1-epoch fine-tuning on full 2012-2016 data with decayed lr (1e-4)
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.train()
    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for x_cat_b, x_num_b, y_b in full_train_loader:
        x_cat_b = x_cat_b.to(device, non_blocking=True)
        x_num_b = x_num_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)

        ft_optimizer.zero_grad()
        preds = model(x_cat_b, x_num_b)
        loss = criterion(preds, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        ft_optimizer.step()

    model.eval()

    # Test Inference and Submission
    test_preds_list = []
    with torch.no_grad():
        for x_cat_b, x_num_b in test_loader:
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            preds = model(x_cat_b, x_num_b)
            test_preds_list.append(preds.cpu().numpy())

    test_preds = np.concatenate(test_preds_list, axis=0)
    test_raw_prices = np.clip(10.0**test_preds, 1.0, None)
    submission_prices = np.maximum(1, np.round(test_raw_prices).astype(np.int64))

    submission = pd.DataFrame(
        {"id": test_processed["id"].values, "price": submission_prices}
    )

    sample_sub = pd.read_csv("./input/sample_submission.csv")
    submission = sample_sub[["id"]].merge(submission, on="id", how="left")
    submission["price"] = submission["price"].astype(np.int64)

    assert len(submission) == len(
        sample_sub
    ), f"Row count mismatch: expected {len(sample_sub)}, got {len(submission)}"
    assert list(submission.columns) == [
        "id",
        "price",
    ], f"Invalid columns: {submission.columns}"
    assert (
        submission["id"].values == sample_sub["id"].values
    ).all(), "ID alignment mismatch"
    assert not submission["price"].isnull().any(), "Found NaNs in predictions"
    assert (submission["price"] >= 1).all(), "Found invalid non-positive predictions"

    submission.to_csv("./submission/submission.csv", index=False)


if __name__ == "__main__":
    main()
