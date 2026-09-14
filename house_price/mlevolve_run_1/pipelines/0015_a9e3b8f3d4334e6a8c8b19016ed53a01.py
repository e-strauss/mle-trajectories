import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ==============================================================================
# 1. Custom Loss Functions Aligned with Official Metric
# ==============================================================================
class Log10RMSELoss(nn.Module):
    """Calculates the exact official competition metric:

    RMSE on log10(price) predictions vs ground truth targets, with support
    for optional recency sample weighting.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        diff_sq = torch.square(y_pred.view(-1) - y_true.view(-1))
        if sample_weight is not None:
            w = sample_weight.view(-1)
            mse = torch.sum(w * diff_sq) / (torch.sum(w) + self.eps)
        else:
            mse = torch.mean(diff_sq)
        return torch.sqrt(mse + self.eps)


# ==============================================================================
# 2. Tabular Residual Network Architecture with Linear Trend Highway
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm, SiLU activation, and Dropout

    regularization to prevent overfitting on dense tabular interactions.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.30):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout1(self.act1(self.norm1(self.linear1(x))))
        out = self.dropout2(self.norm2(self.linear2(out)))
        out = self.act2(out + residual)
        return out


class PropertyPriceNet(nn.Module):
    """High-performance Tabular Neural Network combining:

    1. Entity embeddings for categorical features.
    2. Batch-normalized dense projection of engineered spatial & momentum features.
    3. Deep Residual MLP backbone capturing multi-way non-linear interactions.
    4. Spatially Conditioned Trend Highway enabling region-specific linear extrapolation.
    """

    def __init__(
        self,
        num_dense_features: int,
        cat_cardinalities: dict,
        linear_feature_indices: list,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
        dropout_rate: float = 0.30,
    ):
        super().__init__()
        self.linear_feature_indices = linear_feature_indices
        self.time_idx = linear_feature_indices[0]
        self.anchor_indices = linear_feature_indices[1:]

        # Categorical Entity Embeddings
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = int(min(64, max(4, round(card**0.5 * 1.5))))
            self.embeddings[col] = nn.Embedding(
                num_embeddings=card + 1, embedding_dim=emb_dim, padding_idx=0
            )
            total_emb_dim += emb_dim

        # Continuous Features Normalization & Projection
        self.dense_norm = nn.BatchNorm1d(num_dense_features)
        total_input_dim = num_dense_features + total_emb_dim

        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        # Residual MLP Backbone with increased regularization
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_residual_blocks)
            ]
        )

        # Non-linear Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Spatially Conditioned Trend Highway (time, anchors, and explicit cross-interactions)
        num_linear_feats = len(linear_feature_indices) + len(self.anchor_indices)
        self.linear_highway = nn.Linear(num_linear_feats, 1, bias=False)

    def forward(
        self, x_dense: torch.Tensor, x_cat: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        emb_tensors = []
        for col, emb_layer in self.embeddings.items():
            codes = x_cat[col]
            safe_codes = torch.where(
                codes < 0,
                torch.zeros_like(codes),
                torch.clamp(codes + 1, 0, emb_layer.num_embeddings - 1),
            )
            emb_tensors.append(emb_layer(safe_codes))

        cat_repr = torch.cat(emb_tensors, dim=-1)
        dense_repr = self.dense_norm(x_dense)

        features = torch.cat([dense_repr, cat_repr], dim=-1)
        h = self.input_proj(features)
        for block in self.res_blocks:
            h = block(h)
        nonlinear_pred = self.head(h)

        base_linear = x_dense[:, self.linear_feature_indices]
        time_term = x_dense[:, self.time_idx : self.time_idx + 1]
        anchors = x_dense[:, self.anchor_indices]
        time_interactions = time_term * anchors
        linear_inputs = torch.cat([base_linear, time_interactions], dim=-1)
        linear_trend = self.linear_highway(linear_inputs)

        out = nonlinear_pred + linear_trend
        return out.squeeze(-1)


def get_model_and_optimizer(
    num_dense: int,
    cat_cards: dict,
    linear_indices: list,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = PropertyPriceNet(
        num_dense_features=num_dense,
        cat_cardinalities=cat_cards,
        linear_feature_indices=linear_indices,
        hidden_dim=256,
        num_residual_blocks=3,
        dropout_rate=0.30,
    ).to(device)

    criterion = Log10RMSELoss()

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)

    return model, criterion, optimizer, scheduler


# ==============================================================================
# 3. Main Data Processing, Feature Engineering & Training Pipeline
# ==============================================================================
def main():
    print("Starting data processing and feature engineering pipeline...")
    input_dir = "./input"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load Data
    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.csv")

    print(f"Loading training parquet from {train_path}...")
    df_train_raw = pd.read_parquet(train_path)
    print(f"Loaded train: {len(df_train_raw):,} rows.")

    print(f"Loading test csv from {test_path}...")
    df_test_raw = pd.read_csv(test_path)
    print(f"Loaded test: {len(df_test_raw):,} rows.")

    # 2. Data Cleaning & Target Transformation
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
        df_train_raw[col] = (
            df_train_raw[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        )
        df_test_raw[col] = (
            df_test_raw[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        )

    df_train_raw["date"] = pd.to_datetime(df_train_raw["date"])
    df_test_raw["date"] = pd.to_datetime(df_test_raw["date"])

    # Clean extreme training anomalies
    valid_price_mask = (df_train_raw["price"] >= 1000) & (
        df_train_raw["price"] <= 60000000
    )
    df_train_clean = df_train_raw[valid_price_mask].copy()
    del df_train_raw
    gc.collect()

    df_train_clean["log_price"] = np.log10(
        np.clip(df_train_clean["price"].values, 1.0, None)
    )

    # 3. Temporal Out-of-Time Split
    val_cutoff = pd.Timestamp("2016-07-01")
    train_min_date = pd.Timestamp("2010-01-01")

    val_mask = df_train_clean["date"] >= val_cutoff
    train_mask = (df_train_clean["date"] < val_cutoff) & (
        df_train_clean["date"] >= train_min_date
    )

    df_train = df_train_clean[train_mask].copy().reset_index(drop=True)
    df_val = df_train_clean[val_mask].copy().reset_index(drop=True)
    df_test = df_test_raw.copy().reset_index(drop=True)

    del df_train_clean
    gc.collect()

    print(f"Active training set (2010-01-01 to 2016-06-30): {len(df_train):,} rows.")
    print(f"Validation set (2016-07-01 to 2016-12-31): {len(df_val):,} rows.")
    print(f"Test set (2017-01-01 to 2017-06-29): {len(df_test):,} rows.")

    # 4. Feature Engineering
    ref_date = pd.Timestamp("2010-01-01")

    def extract_calendar_features(df):
        date_series = df["date"]
        df["year"] = date_series.dt.year.astype(np.int16)
        df["month"] = date_series.dt.month.astype(np.int8)
        df["day"] = date_series.dt.day.astype(np.int8)
        df["dayofweek"] = date_series.dt.dayofweek.astype(np.int8)
        df["dayofyear"] = date_series.dt.dayofyear.astype(np.int16)
        df["quarter"] = date_series.dt.quarter.astype(np.int8)

        df["time_elapsed_years"] = ((date_series - ref_date).dt.days / 365.25).astype(
            np.float32
        )
        df["is_friday"] = (df["dayofweek"] == 4).astype(np.int8)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
        df["is_month_end"] = date_series.dt.is_month_end.astype(np.int8)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)
        df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25).astype(
            np.float32
        )
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25).astype(
            np.float32
        )
        return df

    def extract_interaction_keys(df):
        df["district_county"] = df["district"] + "__" + df["county"]
        df["town_district"] = df["town"] + "__" + df["district"]
        df["district_type"] = df["district"] + "__" + df["property_type"]
        df["town_type"] = df["town"] + "__" + df["property_type"]
        df["type_tenure"] = df["property_type"] + "__" + df["tenure"]
        df["type_new"] = df["property_type"] + "__" + df["is_new_build"]
        df["type_cat"] = df["property_type"] + "__" + df["sale_category"]
        return df

    print("Extracting calendar and composite interaction features...")
    df_train = extract_calendar_features(df_train)
    df_val = extract_calendar_features(df_val)
    df_test = extract_calendar_features(df_test)

    df_train = extract_interaction_keys(df_train)
    df_val = extract_interaction_keys(df_val)
    df_test = extract_interaction_keys(df_test)

    # 5. Frequency & Proportion Encodings
    print("Computing frequency and structural housing proportions...")
    district_counts = df_train["district"].value_counts()
    for df in [df_train, df_val, df_test]:
        df["district_log_freq"] = np.log10(
            df["district"].map(district_counts).fillna(1.0).values + 1.0
        ).astype(np.float32)

    district_totals = df_train.groupby("district").size()
    for ptype in ["D", "S", "T", "F"]:
        ptype_counts = (
            df_train[df_train["property_type"] == ptype].groupby("district").size()
        )
        ptype_share = (ptype_counts / district_totals).to_dict()
        for df in [df_train, df_val, df_test]:
            df[f"district_share_type_{ptype}"] = (
                df["district"].map(ptype_share).fillna(0.0).astype(np.float32)
            )

    lease_counts = df_train[df_train["tenure"] == "L"].groupby("district").size()
    lease_share = (lease_counts / district_totals).to_dict()
    new_counts = df_train[df_train["is_new_build"] == "Y"].groupby("district").size()
    new_share = (new_counts / district_totals).to_dict()

    for df in [df_train, df_val, df_test]:
        df["district_leasehold_share"] = (
            df["district"].map(lease_share).fillna(0.0).astype(np.float32)
        )
        df["district_newbuild_share"] = (
            df["district"].map(new_share).fillna(0.0).astype(np.float32)
        )

    # 6. Parametric Linear Macro and Regional Detrending
    print("Fitting hierarchical empirical Bayes linear macro, county, and district trends...")
    t_train = df_train["time_elapsed_years"].values.astype(np.float64)
    y_train = df_train["log_price"].values.astype(np.float64)

    t_mean_nat = float(np.mean(t_train))
    y_mean_nat = float(np.mean(y_train))
    var_t_nat = float(np.mean((t_train - t_mean_nat) ** 2))
    cov_ty_nat = float(np.mean((t_train - t_mean_nat) * (y_train - y_mean_nat)))
    slope_nat = cov_ty_nat / (var_t_nat + 1e-9)
    intercept_nat = y_mean_nat - slope_nat * t_mean_nat

    print(
        f"National Parametric Trend: slope={slope_nat:.5f} log10(price)/year, intercept={intercept_nat:.5f}"
    )

    # County Empirical Bayes Trend Regression
    df_train["_t"] = df_train["time_elapsed_years"].astype(np.float64)
    df_train["_y"] = df_train["log_price"].astype(np.float64)
    df_train["_t2"] = df_train["_t"] * df_train["_t"]
    df_train["_ty"] = df_train["_t"] * df_train["_y"]

    c_stats = df_train.groupby("county").agg(
        n=("_y", "count"),
        sum_t=("_t", "sum"),
        sum_y=("_y", "sum"),
        sum_t2=("_t2", "sum"),
        sum_ty=("_ty", "sum"),
    )

    c_n = c_stats["n"].values
    c_mean_t = c_stats["sum_t"].values / c_n
    c_mean_y = c_stats["sum_y"].values / c_n
    c_var_t = (c_stats["sum_t2"].values / c_n) - (c_mean_t**2)
    c_cov_ty = (c_stats["sum_ty"].values / c_n) - (c_mean_t * c_mean_y)

    c_slope_raw = np.where(
        c_var_t > 1e-4, c_cov_ty / (c_var_t + 1e-9), slope_nat
    )
    c_intercept_raw = c_mean_y - c_slope_raw * c_mean_t

    m_county = 100.0
    c_weight = (c_n / (c_n + m_county)).astype(np.float64)
    c_slope_smooth = c_weight * c_slope_raw + (1.0 - c_weight) * slope_nat
    c_intercept_smooth = (
        c_weight * c_intercept_raw + (1.0 - c_weight) * intercept_nat
    )

    county_slope_map = dict(zip(c_stats.index, c_slope_smooth))
    county_intercept_map = dict(zip(c_stats.index, c_intercept_smooth))

    # District Empirical Bayes Trend Regression smoothed towards County
    d_stats = df_train.groupby("district").agg(
        n=("_y", "count"),
        sum_t=("_t", "sum"),
        sum_y=("_y", "sum"),
        sum_t2=("_t2", "sum"),
        sum_ty=("_ty", "sum"),
    )

    district_to_county = df_train.groupby("district")["county"].first().to_dict()

    d_n = d_stats["n"].values
    d_mean_t = d_stats["sum_t"].values / d_n
    d_mean_y = d_stats["sum_y"].values / d_n
    d_var_t = (d_stats["sum_t2"].values / d_n) - (d_mean_t**2)
    d_cov_ty = (d_stats["sum_ty"].values / d_n) - (d_mean_t * d_mean_y)

    d_county_slope_prior = np.array(
        [
            county_slope_map.get(district_to_county.get(d, ""), slope_nat)
            for d in d_stats.index
        ],
        dtype=np.float64,
    )
    d_county_intercept_prior = np.array(
        [
            county_intercept_map.get(
                district_to_county.get(d, ""), intercept_nat
            )
            for d in d_stats.index
        ],
        dtype=np.float64,
    )

    d_slope_raw = np.where(
        d_var_t > 1e-4, d_cov_ty / (d_var_t + 1e-9), d_county_slope_prior
    )
    d_intercept_raw = d_mean_y - d_slope_raw * d_mean_t

    m_district = 50.0
    d_weight = (d_n / (d_n + m_district)).astype(np.float64)
    d_slope_smooth = (
        d_weight * d_slope_raw + (1.0 - d_weight) * d_county_slope_prior
    )
    d_intercept_smooth = (
        d_weight * d_intercept_raw + (1.0 - d_weight) * d_county_intercept_prior
    )

    district_slope_map = dict(zip(d_stats.index, d_slope_smooth))
    district_intercept_map = dict(zip(d_stats.index, d_intercept_smooth))

    df_train.drop(columns=["_t", "_y", "_t2", "_ty"], inplace=True)

    def compute_trend(df):
        t_vals = df["time_elapsed_years"].values
        c_slope = df["county"].map(county_slope_map).fillna(slope_nat)
        c_intercept = (
            df["county"].map(county_intercept_map).fillna(intercept_nat)
        )

        d_slope = df["district"].map(district_slope_map).fillna(c_slope).values
        d_intercept = (
            df["district"].map(district_intercept_map).fillna(c_intercept).values
        )

        trend = d_intercept + d_slope * t_vals
        return trend.astype(np.float32)

    df_train["y_trend"] = compute_trend(df_train)
    df_train["y_residual"] = (
        df_train["log_price"] - df_train["y_trend"]
    ).astype(np.float32)

    df_val["y_trend"] = compute_trend(df_val)
    df_val["y_residual"] = (df_val["log_price"] - df_val["y_trend"]).astype(
        np.float32
    )

    df_test["y_trend"] = compute_trend(df_test)

    # Compute exponential recency sample weights
    t_val = float((val_cutoff - ref_date).days / 365.25)
    df_train["sample_weight"] = np.exp(
        0.20 * (df_train["time_elapsed_years"] - t_val)
    ).astype(np.float32)
    df_train["weighted_res"] = (
        df_train["sample_weight"] * df_train["y_residual"]
    ).astype(np.float32)
    global_res_mean = float(
        df_train["weighted_res"].sum() / df_train["sample_weight"].sum()
    )

    # 7. Leak-Free Hierarchical Bayesian Target Encodings on Stationary Residuals
    print(
        "Constructing Hierarchical Bayesian target encodings on stationary residuals..."
    )
    dist_type_to_dist = {
        k: k.split("__")[0] for k in df_train["district_type"].unique()
    }
    town_type_to_dist_type = (
        df_train.groupby("town_type")["district_type"].first().to_dict()
    )

    def compute_bayesian_residual_dict(
        sub_df, group_col, prior_dict, weight_m, default_prior=0.0
    ):
        grouped = sub_df.groupby(group_col)[
            ["sample_weight", "weighted_res"]
        ].sum()
        sum_w = grouped["sample_weight"].values
        sum_wy = grouped["weighted_res"].values
        if isinstance(prior_dict, dict) and len(prior_dict) > 0:
            priors = np.array(
                [
                    prior_dict.get(k, default_prior)
                    if prior_dict.get(k, default_prior) is not None
                    and not np.isnan(prior_dict.get(k, default_prior))
                    else default_prior
                    for k in grouped.index
                ],
                dtype=np.float32,
            )
        else:
            priors = np.full(len(grouped), default_prior, dtype=np.float32)
        smoothed = (sum_wy + weight_m * priors) / (sum_w + weight_m)
        return dict(zip(grouped.index, smoothed))

    county_te_map = compute_bayesian_residual_dict(
        df_train, "county", {}, 50.0, global_res_mean
    )
    district_priors = {
        d: county_te_map.get(c, global_res_mean)
        for d, c in district_to_county.items()
    }
    district_te_map = compute_bayesian_residual_dict(
        df_train, "district", district_priors, 30.0, global_res_mean
    )
    dist_type_priors = {
        dt: district_te_map.get(d, global_res_mean)
        for dt, d in dist_type_to_dist.items()
    }
    dist_type_te_map = compute_bayesian_residual_dict(
        df_train, "district_type", dist_type_priors, 20.0, global_res_mean
    )
    town_type_priors = {
        tt: dist_type_te_map.get(dt, global_res_mean)
        for tt, dt in town_type_to_dist_type.items()
    }
    town_type_te_map = compute_bayesian_residual_dict(
        df_train, "town_type", town_type_priors, 15.0, global_res_mean
    )

    prop_type_te_map = compute_bayesian_residual_dict(
        df_train, "property_type", {}, 10.0, global_res_mean
    )
    type_new_te_map = compute_bayesian_residual_dict(
        df_train, "type_new", {}, 10.0, global_res_mean
    )
    type_tenure_te_map = compute_bayesian_residual_dict(
        df_train, "type_tenure", {}, 10.0, global_res_mean
    )

    print(
        "Applying 5-fold Out-Of-Fold residual target encoding to training data..."
    )
    n_splits = 5
    fold_indices = np.random.randint(0, n_splits, size=len(df_train))

    oof_county = np.zeros(len(df_train), dtype=np.float32)
    oof_district = np.zeros(len(df_train), dtype=np.float32)
    oof_district_type = np.zeros(len(df_train), dtype=np.float32)
    oof_town_type = np.zeros(len(df_train), dtype=np.float32)

    for fold in range(n_splits):
        idx_train_fold = fold_indices != fold
        idx_val_fold = fold_indices == fold
        sub_train = df_train.iloc[idx_train_fold]
        sub_global_mean = float(
            sub_train["weighted_res"].sum()
            / sub_train["sample_weight"].sum()
        )

        sub_c_map = compute_bayesian_residual_dict(
            sub_train, "county", {}, 50.0, sub_global_mean
        )
        sub_d_priors = {
            d: sub_c_map.get(c, sub_global_mean)
            for d, c in district_to_county.items()
        }
        sub_d_map = compute_bayesian_residual_dict(
            sub_train, "district", sub_d_priors, 30.0, sub_global_mean
        )
        sub_dt_priors = {
            dt: sub_d_map.get(d, sub_global_mean)
            for dt, d in dist_type_to_dist.items()
        }
        sub_dt_map = compute_bayesian_residual_dict(
            sub_train, "district_type", sub_dt_priors, 20.0, sub_global_mean
        )
        sub_tt_priors = {
            tt: sub_dt_map.get(dt, sub_global_mean)
            for tt, dt in town_type_to_dist_type.items()
        }
        sub_tt_map = compute_bayesian_residual_dict(
            sub_train, "town_type", sub_tt_priors, 15.0, sub_global_mean
        )

        val_sub = df_train.iloc[idx_val_fold]
        oof_c_series = val_sub["county"].map(sub_c_map).fillna(sub_global_mean)
        oof_d_series = val_sub["district"].map(sub_d_map).fillna(oof_c_series)
        oof_dt_series = (
            val_sub["district_type"].map(sub_dt_map).fillna(oof_d_series)
        )
        oof_tt_series = (
            val_sub["town_type"].map(sub_tt_map).fillna(oof_dt_series)
        )

        oof_county[idx_val_fold] = oof_c_series.values
        oof_district[idx_val_fold] = oof_d_series.values
        oof_district_type[idx_val_fold] = oof_dt_series.values
        oof_town_type[idx_val_fold] = oof_tt_series.values

    df_train["te_county"] = oof_county
    df_train["te_district"] = oof_district
    df_train["te_district_type"] = oof_district_type
    df_train["te_town_type"] = oof_town_type
    df_train["te_property_type"] = (
        df_train["property_type"]
        .map(prop_type_te_map)
        .fillna(global_res_mean)
        .astype(np.float32)
    )
    df_train["te_type_new"] = (
        df_train["type_new"]
        .map(type_new_te_map)
        .fillna(global_res_mean)
        .astype(np.float32)
    )
    df_train["te_type_tenure"] = (
        df_train["type_tenure"]
        .map(type_tenure_te_map)
        .fillna(global_res_mean)
        .astype(np.float32)
    )

    for df in [df_val, df_test]:
        df["te_county"] = (
            df["county"]
            .map(county_te_map)
            .fillna(global_res_mean)
            .astype(np.float32)
        )
        df["te_district"] = (
            df["district"]
            .map(district_te_map)
            .fillna(df["te_county"])
            .astype(np.float32)
        )
        df["te_district_type"] = (
            df["district_type"]
            .map(dist_type_te_map)
            .fillna(df["te_district"])
            .astype(np.float32)
        )
        df["te_town_type"] = (
            df["town_type"]
            .map(town_type_te_map)
            .fillna(df["te_district_type"])
            .astype(np.float32)
        )
        df["te_property_type"] = (
            df["property_type"]
            .map(prop_type_te_map)
            .fillna(global_res_mean)
            .astype(np.float32)
        )
        df["te_type_new"] = (
            df["type_new"]
            .map(type_new_te_map)
            .fillna(global_res_mean)
            .astype(np.float32)
        )
        df["te_type_tenure"] = (
            df["type_tenure"]
            .map(type_tenure_te_map)
            .fillna(global_res_mean)
            .astype(np.float32)
        )

    # 8. Categorical Encodings for LightGBM and Neural Network
    print("Encoding categorical indices and native categories...")
    encoding_cols = [
        "property_type",
        "is_new_build",
        "tenure",
        "sale_category",
        "county",
        "district",
        "town",
        "district_type",
        "town_type",
    ]
    cat_feature_names = [f"{col}_code" for col in encoding_cols]

    for col in encoding_cols:
        categories = df_train[col].value_counts().index.tolist()
        cat_to_id = {val: idx for idx, val in enumerate(categories)}
        code_col = f"{col}_code"
        df_train[code_col] = (
            df_train[col].map(cat_to_id).fillna(-1).astype(np.int32)
        )
        df_val[code_col] = df_val[col].map(cat_to_id).fillna(-1).astype(np.int32)
        df_test[code_col] = (
            df_test[col].map(cat_to_id).fillna(-1).astype(np.int32)
        )

        cat_dtype = pd.CategoricalDtype(categories=categories)
        df_train[col] = df_train[col].astype(cat_dtype)
        df_val[col] = df_val[col].astype(cat_dtype)
        df_test[col] = df_test[col].astype(cat_dtype)

    # 9. Feature Schema Definition
    numeric_features = [
        "time_elapsed_years",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "quarter",
        "is_friday",
        "is_weekend",
        "is_month_end",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "district_log_freq",
        "district_leasehold_share",
        "district_newbuild_share",
        "district_share_type_D",
        "district_share_type_S",
        "district_share_type_T",
        "district_share_type_F",
        "te_county",
        "te_district",
        "te_district_type",
        "te_town_type",
        "te_property_type",
        "te_type_new",
        "te_type_tenure",
    ]

    train_output_cols = (
        ["id", "log_price", "price", "y_trend", "y_residual", "sample_weight"]
        + numeric_features
        + encoding_cols
        + cat_feature_names
    )
    val_output_cols = (
        ["id", "log_price", "price", "y_trend", "y_residual"]
        + numeric_features
        + encoding_cols
        + cat_feature_names
    )
    test_output_cols = (
        ["id", "y_trend"]
        + numeric_features
        + encoding_cols
        + cat_feature_names
    )

    df_train_out = df_train[train_output_cols].copy()
    df_val_out = df_val[val_output_cols].copy()
    df_test_out = df_test[test_output_cols].copy()

    df_train_out[numeric_features] = df_train_out[numeric_features].fillna(0.0)
    df_val_out[numeric_features] = df_val_out[numeric_features].fillna(0.0)
    df_test_out[numeric_features] = df_test_out[numeric_features].fillna(0.0)

    del df_train, df_val, df_test
    gc.collect()

    assert (
        len(df_test_out) == 375098
    ), f"Expected 375098 test rows, got {len(df_test_out)}"
    assert (
        df_test_out["id"] == df_test_raw["id"]
    ).all(), "Test IDs must match exactly"

    # 10. LightGBM Model Definition & Training on Detrended Residuals
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.04,
        "min_child_samples": 50,
        "cat_smooth": 25.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    lgb_features = numeric_features + encoding_cols
    print(
        f"Instantiating LightGBM datasets with {len(lgb_features)} features..."
    )
    lgb_train_data = lgb.Dataset(
        df_train_out[lgb_features],
        label=df_train_out["y_residual"].values,
        weight=df_train_out["sample_weight"].values,
        categorical_feature=encoding_cols,
        free_raw_data=False,
    )
    lgb_val_data = lgb.Dataset(
        df_val_out[lgb_features],
        label=df_val_out["y_residual"].values,
        categorical_feature=encoding_cols,
        reference=lgb_train_data,
        free_raw_data=False,
    )

    print("Training LightGBM on detrended residuals with recency weighting...")
    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=1000,
        valid_sets=[lgb_val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    lgb_checkpoint_path = os.path.join(working_dir, "best_lgb_residual_model.txt")
    lgb_model.save_model(lgb_checkpoint_path)
    print(f"LightGBM model checkpoint saved to {lgb_checkpoint_path}")

    # 11. PropertyPriceNet Instantiation for Stationary Residual Modeling
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear_indices = [
        numeric_features.index("time_elapsed_years"),
        numeric_features.index("te_county"),
        numeric_features.index("te_district"),
        numeric_features.index("te_district_type"),
        numeric_features.index("te_town_type"),
        numeric_features.index("te_property_type"),
    ]
    cat_cards = {
        col: max(int(df_train_out[col].max() + 10), 16)
        for col in cat_feature_names
    }

    model, criterion, optimizer, scheduler = get_model_and_optimizer(
        num_dense=len(numeric_features),
        cat_cards=cat_cards,
        linear_indices=linear_indices,
        lr=1e-3,
        weight_decay=1e-3,
        device=device,
    )

    # 12. Prepare Torch Tensors and DataLoaders on Residuals
    print("Preparing TensorDataLoaders for PropertyPriceNet...")
    train_dense = torch.from_numpy(
        df_train_out[numeric_features].values.astype(np.float32, copy=False)
    )
    train_cat = torch.from_numpy(
        df_train_out[cat_feature_names].values.astype(np.int64, copy=False)
    )
    train_y = torch.from_numpy(
        df_train_out["y_residual"].values.astype(np.float32, copy=False)
    )

    val_dense = torch.from_numpy(
        df_val_out[numeric_features].values.astype(np.float32, copy=False)
    )
    val_cat = torch.from_numpy(
        df_val_out[cat_feature_names].values.astype(np.int64, copy=False)
    )
    val_y = torch.from_numpy(
        df_val_out["y_residual"].values.astype(np.float32, copy=False)
    )

    test_dense = torch.from_numpy(
        df_test_out[numeric_features].values.astype(np.float32, copy=False)
    )
    test_cat = torch.from_numpy(
        df_test_out[cat_feature_names].values.astype(np.int64, copy=False)
    )

    batch_size = 8192
    train_dataset = TensorDataset(train_dense, train_cat, train_y)
    val_dataset = TensorDataset(val_dense, val_cat, val_y)
    test_dataset = TensorDataset(test_dense, test_cat)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 13. PropertyPriceNet Training Loop
    best_model_path = os.path.join(working_dir, "best_property_net.pt")
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    max_epochs = 8

    print(
        f"Starting PropertyPriceNet training on device: {device} for maximum of {max_epochs} epochs..."
    )

    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for b_dense, b_cat, b_y in train_loader:
            b_dense = b_dense.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            cat_dict = {
                col: b_cat[:, i].to(device, non_blocking=True)
                for i, col in enumerate(cat_feature_names)
            }

            # Exponential recency sample weight: exp(0.20 * (time_elapsed_years - t_val))
            b_weights = torch.exp(0.20 * (b_dense[:, 0] - t_val))

            optimizer.zero_grad(set_to_none=True)
            preds = model(b_dense, cat_dict)
            loss = criterion(preds, b_y, sample_weight=b_weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        train_loss = running_train_loss / max(1, num_train_batches)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for b_dense, b_cat, b_y in val_loader:
                b_dense = b_dense.to(device, non_blocking=True)
                b_y = b_y.to(device, non_blocking=True)
                cat_dict = {
                    col: b_cat[:, i].to(device, non_blocking=True)
                    for i, col in enumerate(cat_feature_names)
                }

                val_preds = model(b_dense, cat_dict)
                val_batch_loss = criterion(val_preds, b_y)
                running_val_loss += val_batch_loss.item()
                num_val_batches += 1

        val_loss = running_val_loss / max(1, num_val_batches)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | Train RMSE: {train_loss:.5f} | Val RMSE: {val_loss:.5f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} with best Val RMSE: {best_val_loss:.5f}"
                )
                break

    # 14. Validation Inference & Ensemble Blending
    print("Generating validation residual predictions...")
    lgb_val_res = lgb_model.predict(df_val_out[lgb_features])

    print(
        f"Loading best checkpoint from {best_model_path} for neural net evaluation..."
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for b_dense, b_cat, _ in val_loader:
            b_dense = b_dense.to(device, non_blocking=True)
            cat_dict = {
                col: b_cat[:, i].to(device, non_blocking=True)
                for i, col in enumerate(cat_feature_names)
            }
            batch_out = model(b_dense, cat_dict)
            val_preds_list.append(batch_out.cpu().numpy())

    nn_val_res = np.concatenate(val_preds_list, axis=0)

    # Blended validation residual: 0.70 * LightGBM + 0.30 * PropertyPriceNet
    val_blended_res = 0.70 * lgb_val_res + 0.30 * nn_val_res
    y_val_trend = df_val_out["y_trend"].values
    y_val_pred = y_val_trend + val_blended_res

    val_preds_price = np.power(10.0, y_val_pred)
    val_preds_clipped = np.clip(val_preds_price, 1.0, None)
    val_true_price_clipped = np.clip(df_val_out["price"].values, 1.0, None)

    val_rmse = float(
        np.sqrt(
            np.mean(
                (np.log10(val_preds_clipped) - np.log10(val_true_price_clipped))
                ** 2
            )
        )
    )

    # Diagnostics on individual components
    lgb_val_price = np.clip(
        np.power(10.0, y_val_trend + lgb_val_res), 1.0, None
    )
    lgb_val_rmse = float(
        np.sqrt(
            np.mean(
                (np.log10(lgb_val_price) - np.log10(val_true_price_clipped)) ** 2
            )
        )
    )
    print(f"LightGBM Alone Validation RMSE on log10(price): {lgb_val_rmse:.6f}")

    nn_val_price = np.clip(np.power(10.0, y_val_trend + nn_val_res), 1.0, None)
    nn_val_rmse = float(
        np.sqrt(
            np.mean(
                (np.log10(nn_val_price) - np.log10(val_true_price_clipped)) ** 2
            )
        )
    )
    print(
        f"PropertyPriceNet Alone Validation RMSE on log10(price): {nn_val_rmse:.6f}"
    )

    print(f"Hold-out Out-Of-Time Validation RMSE on log10(price): {val_rmse:.6f}")

    # 15. Test Set Inference & Submission Generation
    print("Generating predictions on test set...")
    lgb_test_res = lgb_model.predict(df_test_out[lgb_features])

    test_preds_list = []
    with torch.no_grad():
        for b_dense, b_cat in test_loader:
            b_dense = b_dense.to(device, non_blocking=True)
            cat_dict = {
                col: b_cat[:, i].to(device, non_blocking=True)
                for i, col in enumerate(cat_feature_names)
            }
            test_out = model(b_dense, cat_dict)
            test_preds_list.append(test_out.cpu().numpy())

    nn_test_res = np.concatenate(test_preds_list, axis=0)

    test_blended_res = 0.70 * lgb_test_res + 0.30 * nn_test_res
    y_test_trend = df_test_out["y_trend"].values
    y_test_pred = y_test_trend + test_blended_res

    test_preds_price = np.power(10.0, y_test_pred)
    test_preds_price = np.clip(test_preds_price, 1.0, None)

    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {
            "id": df_test_out["id"].values,
            "price": np.clip(np.round(test_preds_price), 1.0, None).astype(
                np.int64
            ),
        }
    )
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission successfully saved to {submission_path}")

    # Verification assertions
    assert os.path.exists(submission_path), "Submission file was not written!"
    assert (
        len(submission_df) == 375098
    ), f"Expected 375098 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == [
        "id",
        "price",
    ], f"Unexpected columns: {submission_df.columns}"
    assert not submission_df["price"].isna().any(), "Submission contains NaN values"
    assert not np.isinf(
        submission_df["price"]
    ).any(), "Submission contains Inf values"
    assert (
        submission_df["price"] > 0
    ).all(), "Submission contains non-positive price values"

    print(f"Final Validation Score: {val_rmse}")


if __name__ == "__main__":
    main()
