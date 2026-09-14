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

    RMSE on log10(price) predictions vs ground truth targets.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred.view(-1) - y_true.view(-1)
        mse = torch.mean(torch.square(diff))
        return torch.sqrt(mse + self.eps)


# ==============================================================================
# 2. Tabular Residual Network Architecture with Linear Trend Highway
# ==============================================================================
class TabularResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm, SiLU activation, and Dropout

    regularization to prevent overfitting on dense tabular interactions.
    """

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.20):
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
    2. Batch-normalized dense projection of engineered spatial & momentum
    features.
    3. Deep Residual MLP backbone capturing multi-way non-linear interactions.
    4. Wide linear trend highway enabling monotonic out-of-time price
    extrapolation.
    """

    def __init__(
        self,
        num_dense_features: int,
        cat_cardinalities: dict,
        linear_feature_indices: list,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
        dropout_rate: float = 0.20,
    ):
        super().__init__()
        self.linear_feature_indices = linear_feature_indices

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

        # Residual MLP Backbone
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate * 0.8)
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

        # Linear Trend Highway
        num_linear_feats = len(linear_feature_indices)
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

        linear_inputs = x_dense[:, self.linear_feature_indices]
        linear_trend = self.linear_highway(linear_inputs)

        out = nonlinear_pred + linear_trend
        return out.squeeze(-1)


def get_model_and_optimizer(
    num_dense: int,
    cat_cards: dict,
    linear_indices: list,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = PropertyPriceNet(
        num_dense_features=num_dense,
        cat_cardinalities=cat_cards,
        linear_feature_indices=linear_indices,
        hidden_dim=256,
        num_residual_blocks=3,
        dropout_rate=0.20,
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

    # 6. Recent District Price Momentum
    print("Calculating trailing district price momentum features...")
    one_year_prior = val_cutoff - pd.DateOffset(years=1)
    two_years_prior = val_cutoff - pd.DateOffset(years=2)

    recent_1y_mean = (
        df_train[df_train["date"] >= one_year_prior]
        .groupby("district")["log_price"]
        .mean()
        .to_dict()
    )
    recent_2y_mean = (
        df_train[df_train["date"] >= two_years_prior]
        .groupby("district")["log_price"]
        .mean()
        .to_dict()
    )
    all_time_mean = df_train.groupby("district")["log_price"].mean().to_dict()
    global_mean = df_train["log_price"].mean()

    for df in [df_train, df_val, df_test]:
        df["district_recent_1y_mean"] = (
            df["district"].map(recent_1y_mean).fillna(global_mean).astype(np.float32)
        )
        df["district_recent_2y_mean"] = (
            df["district"].map(recent_2y_mean).fillna(global_mean).astype(np.float32)
        )
        df["district_all_mean"] = (
            df["district"].map(all_time_mean).fillna(global_mean).astype(np.float32)
        )
        df["district_momentum_1y"] = (
            df["district_recent_1y_mean"] - df["district_all_mean"]
        ).astype(np.float32)

    # 7. Hierarchical Empirical Bayes Target Encodings
    print("Constructing Hierarchical Bayesian target encodings...")

    def compute_bayesian_te(group_col, prior_dict, weight_m, train_df):
        grouped = train_df.groupby(group_col)["log_price"].agg(["count", "mean"])
        grouped["prior"] = grouped.index.map(prior_dict).fillna(global_mean)
        grouped["smoothed"] = (
            grouped["count"] * grouped["mean"] + weight_m * grouped["prior"]
        ) / (grouped["count"] + weight_m)
        return grouped["smoothed"].to_dict()

    county_stats = df_train.groupby("county")["log_price"].agg(["count", "mean"])
    county_stats["smoothed"] = (
        county_stats["count"] * county_stats["mean"] + 50.0 * global_mean
    ) / (county_stats["count"] + 50.0)
    county_te_map = county_stats["smoothed"].to_dict()

    district_to_county = df_train.groupby("district")["county"].first().to_dict()
    district_priors = {
        d: county_te_map.get(c, global_mean) for d, c in district_to_county.items()
    }
    district_te_map = compute_bayesian_te(
        "district", district_priors, weight_m=30.0, train_df=df_train
    )

    dist_type_to_dist = {
        k: k.split("__")[0] for k in df_train["district_type"].unique()
    }
    dist_type_priors = {
        dt: district_te_map.get(d, global_mean) for dt, d in dist_type_to_dist.items()
    }
    dist_type_te_map = compute_bayesian_te(
        "district_type", dist_type_priors, weight_m=20.0, train_df=df_train
    )

    town_type_to_dist_type = (
        df_train.groupby("town_type")["district_type"].first().to_dict()
    )
    town_type_priors = {
        tt: dist_type_te_map.get(dt, global_mean)
        for tt, dt in town_type_to_dist_type.items()
    }
    town_type_te_map = compute_bayesian_te(
        "town_type", town_type_priors, weight_m=15.0, train_df=df_train
    )

    prop_type_stats = df_train.groupby("property_type")["log_price"].agg(
        ["count", "mean"]
    )
    prop_type_te_map = (
        (prop_type_stats["count"] * prop_type_stats["mean"] + 10.0 * global_mean)
        / (prop_type_stats["count"] + 10.0)
    ).to_dict()

    type_new_stats = df_train.groupby("type_new")["log_price"].agg(["count", "mean"])
    type_new_te_map = (
        (type_new_stats["count"] * type_new_stats["mean"] + 10.0 * global_mean)
        / (type_new_stats["count"] + 10.0)
    ).to_dict()

    type_tenure_stats = df_train.groupby("type_tenure")["log_price"].agg(
        ["count", "mean"]
    )
    type_tenure_te_map = (
        (type_tenure_stats["count"] * type_tenure_stats["mean"] + 10.0 * global_mean)
        / (type_tenure_stats["count"] + 10.0)
    ).to_dict()

    # Out-of-fold target encoding for df_train
    print("Applying Out-Of-Fold target encoding to training data...")
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

        sub_c_stats = sub_train.groupby("county")["log_price"].agg(["count", "mean"])
        sub_c_map = (
            (sub_c_stats["count"] * sub_c_stats["mean"] + 50.0 * global_mean)
            / (sub_c_stats["count"] + 50.0)
        ).to_dict()

        sub_d_priors = {
            d: sub_c_map.get(c, global_mean) for d, c in district_to_county.items()
        }
        sub_d_map = compute_bayesian_te(
            "district", sub_d_priors, weight_m=30.0, train_df=sub_train
        )

        sub_dt_priors = {
            dt: sub_d_map.get(d, global_mean) for dt, d in dist_type_to_dist.items()
        }
        sub_dt_map = compute_bayesian_te(
            "district_type", sub_dt_priors, weight_m=20.0, train_df=sub_train
        )

        sub_tt_priors = {
            tt: sub_dt_map.get(dt, global_mean)
            for tt, dt in town_type_to_dist_type.items()
        }
        sub_tt_map = compute_bayesian_te(
            "town_type", sub_tt_priors, weight_m=15.0, train_df=sub_train
        )

        val_sub = df_train.iloc[idx_val_fold]
        oof_county[idx_val_fold] = (
            val_sub["county"].map(sub_c_map).fillna(global_mean).values
        )
        oof_district[idx_val_fold] = (
            val_sub["district"].map(sub_d_map).fillna(global_mean).values
        )
        oof_district_type[idx_val_fold] = (
            val_sub["district_type"].map(sub_dt_map).fillna(global_mean).values
        )
        oof_town_type[idx_val_fold] = (
            val_sub["town_type"].map(sub_tt_map).fillna(global_mean).values
        )

    df_train["te_county"] = oof_county
    df_train["te_district"] = oof_district
    df_train["te_district_type"] = oof_district_type
    df_train["te_town_type"] = oof_town_type
    df_train["te_property_type"] = (
        df_train["property_type"]
        .map(prop_type_te_map)
        .fillna(global_mean)
        .astype(np.float32)
    )
    df_train["te_type_new"] = (
        df_train["type_new"].map(type_new_te_map).fillna(global_mean).astype(np.float32)
    )
    df_train["te_type_tenure"] = (
        df_train["type_tenure"]
        .map(type_tenure_te_map)
        .fillna(global_mean)
        .astype(np.float32)
    )

    for df in [df_val, df_test]:
        df["te_county"] = (
            df["county"].map(county_te_map).fillna(global_mean).astype(np.float32)
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
            .fillna(global_mean)
            .astype(np.float32)
        )
        df["te_type_new"] = (
            df["type_new"].map(type_new_te_map).fillna(global_mean).astype(np.float32)
        )
        df["te_type_tenure"] = (
            df["type_tenure"]
            .map(type_tenure_te_map)
            .fillna(global_mean)
            .astype(np.float32)
        )

    # 8. Categorical Label Encoding
    print("Encoding categorical indices...")
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
        df_train[code_col] = df_train[col].map(cat_to_id).fillna(-1).astype(np.int32)
        df_val[code_col] = df_val[col].map(cat_to_id).fillna(-1).astype(np.int32)
        df_test[code_col] = df_test[col].map(cat_to_id).fillna(-1).astype(np.int32)

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
        "district_recent_1y_mean",
        "district_recent_2y_mean",
        "district_all_mean",
        "district_momentum_1y",
        "te_county",
        "te_district",
        "te_district_type",
        "te_town_type",
        "te_property_type",
        "te_type_new",
        "te_type_tenure",
    ]

    train_output_cols = (
        ["id", "log_price", "price"] + numeric_features + cat_feature_names
    )
    val_output_cols = (
        ["id", "log_price", "price"] + numeric_features + cat_feature_names
    )
    test_output_cols = ["id"] + numeric_features + cat_feature_names

    df_train_out = df_train[train_output_cols].copy()
    df_val_out = df_val[val_output_cols].copy()
    df_test_out = df_test[test_output_cols].copy()

    del df_train, df_val, df_test
    gc.collect()

    assert (
        len(df_test_out) == 375098
    ), f"Expected 375098 test rows, got {len(df_test_out)}"
    assert (df_test_out["id"] == df_test_raw["id"]).all(), "Test IDs must match exactly"

    # 10. Model Instantiation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear_indices = [
        0,
        25,
        26,
        27,
        28,
    ]  # time_elapsed_years, te_county, te_district, te_district_type, te_town_type
    cat_cards = {
        col: max(int(df_train_out[col].max() + 10), 16) for col in cat_feature_names
    }

    model, criterion, optimizer, scheduler = get_model_and_optimizer(
        num_dense=len(numeric_features),
        cat_cards=cat_cards,
        linear_indices=linear_indices,
        device=device,
    )

    # 11. Prepare Torch Tensors and DataLoaders
    print("Preparing TensorDataLoaders...")
    train_dense = torch.from_numpy(
        df_train_out[numeric_features].values.astype(np.float32, copy=False)
    )
    train_cat = torch.from_numpy(
        df_train_out[cat_feature_names].values.astype(np.int64, copy=False)
    )
    train_y = torch.from_numpy(
        df_train_out["log_price"].values.astype(np.float32, copy=False)
    )

    val_dense = torch.from_numpy(
        df_val_out[numeric_features].values.astype(np.float32, copy=False)
    )
    val_cat = torch.from_numpy(
        df_val_out[cat_feature_names].values.astype(np.int64, copy=False)
    )
    val_y = torch.from_numpy(
        df_val_out["log_price"].values.astype(np.float32, copy=False)
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

    # 12. Model Training Loop
    best_model_path = os.path.join(working_dir, "best_property_net.pt")
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    max_epochs = 12

    print(
        f"Starting training on device: {device} for maximum of {max_epochs} epochs..."
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

            optimizer.zero_grad(set_to_none=True)
            preds = model(b_dense, cat_dict)
            loss = criterion(preds, b_y)

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

    # 13. Inference on Validation Set
    print(f"Loading best checkpoint from {best_model_path} for evaluation...")
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

    val_preds_log10 = np.concatenate(val_preds_list, axis=0)
    val_preds_price = np.power(10.0, val_preds_log10)

    val_preds_clipped = np.clip(val_preds_price, 1.0, None)
    val_true_price_clipped = np.clip(df_val_out["price"].values, 1.0, None)

    val_rmse = float(
        np.sqrt(
            np.mean(
                (np.log10(val_preds_clipped) - np.log10(val_true_price_clipped)) ** 2
            )
        )
    )
    print(f"Hold-out Out-Of-Time Validation RMSE on log10(price): {val_rmse:.6f}")

    # 14. Test Set Inference & Submission Generation
    print("Generating predictions on test set...")
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

    test_preds_log10 = np.concatenate(test_preds_list, axis=0)
    test_preds_price = np.power(10.0, test_preds_log10)
    test_preds_price = np.clip(test_preds_price, 1.0, None)

    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {"id": df_test_out["id"].values, "price": test_preds_price}
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
    assert not np.isinf(submission_df["price"]).any(), "Submission contains Inf values"
    assert (
        submission_df["price"] > 0
    ).all(), "Submission contains non-positive price values"

    print(f"Final Validation Score: {val_rmse}")


if __name__ == "__main__":
    main()
