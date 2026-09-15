import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
import torch
import torch.nn as nn
import xgboost as xgb


# =============================================================================
# Model Architectures and Builder Functions
# =============================================================================
class TaxiResidualBlock(nn.Module):
    """Residual block for tabular features with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.ln1(self.fc1(x))))
        out = self.drop2(self.act2(self.ln2(self.fc2(out))))
        return out + residual


class TaxiResNet(nn.Module):
    """Deep Tabular Residual Network tailored for NYC Taxi Fare regression."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
        init_bias: float = 11.35,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [TaxiResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        final_layer = self.head[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_layer.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss faithful to the competition evaluation metric."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


def build_neural_model(
    in_features: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    dropout: float = 0.15,
    init_bias: float = 11.35,
    device: Optional[str] = None,
) -> TaxiResNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TaxiResNet(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        init_bias=init_bias,
    )
    return model.to(device)


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    total_steps: int = 10000,
    eta_min: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min
    )
    return optimizer, scheduler


def build_lgbm_regressor(params: Optional[Dict[str, Any]] = None) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.08,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_jobs": 48,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def build_xgboost_regressor(
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBRegressor:
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": 2000,
        "learning_rate": 0.08,
        "max_depth": 9,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "early_stopping_rounds": 40,
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)


# =============================================================================
# Main Pipeline Execution
# =============================================================================
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Phase 1: Ingesting raw NYC Taxi trip data...")
    NYC_BOUNDS = {
        "lon_min": -74.50,
        "lon_max": -72.80,
        "lat_min": 40.40,
        "lat_max": 41.90,
        "fare_min": 2.50,
        "fare_max": 400.00,
        "pass_min": 1,
        "pass_max": 6,
    }

    test_pl = pl.read_csv("./input/test.csv")
    test_keys = test_pl["key"].to_list()
    print(f"Loaded {len(test_pl)} test instances.")

    train_clean_filter = (
        (pl.col("fare_amount") >= NYC_BOUNDS["fare_min"])
        & (pl.col("fare_amount") <= NYC_BOUNDS["fare_max"])
        & (pl.col("pickup_longitude") >= NYC_BOUNDS["lon_min"])
        & (pl.col("pickup_longitude") <= NYC_BOUNDS["lon_max"])
        & (pl.col("pickup_latitude") >= NYC_BOUNDS["lat_min"])
        & (pl.col("pickup_latitude") <= NYC_BOUNDS["lat_max"])
        & (pl.col("dropoff_longitude") >= NYC_BOUNDS["lon_min"])
        & (pl.col("dropoff_longitude") <= NYC_BOUNDS["lon_max"])
        & (pl.col("dropoff_latitude") >= NYC_BOUNDS["lat_min"])
        & (pl.col("dropoff_latitude") <= NYC_BOUNDS["lat_max"])
        & (pl.col("passenger_count") >= NYC_BOUNDS["pass_min"])
        & (pl.col("passenger_count") <= NYC_BOUNDS["pass_max"])
    )

    print("Streaming and filtering training data...")
    train_raw = (
        pl.scan_csv("./input/train.csv")
        .filter(train_clean_filter)
        .head(10_100_000)
        .collect()
    )
    print(f"Collected {len(train_raw):,} cleaned trip records.")

    val_size = 100_000
    train_size = len(train_raw) - val_size

    train_raw = train_raw.sample(fraction=1.0, shuffle=True, seed=42)
    train_split = train_raw.slice(0, train_size)
    val_split = train_raw.slice(train_size, val_size)
    print(
        f"Strict isolation: Train = {len(train_split):,} rows | Validation = {len(val_split):,} rows"
    )

    def build_features(df: pl.DataFrame, is_test: bool = False) -> pl.DataFrame:
        p_lat = df["pickup_latitude"].to_numpy().astype(np.float64)
        p_lon = df["pickup_longitude"].to_numpy().astype(np.float64)
        d_lat = df["dropoff_latitude"].to_numpy().astype(np.float64)
        d_lon = df["dropoff_longitude"].to_numpy().astype(np.float64)

        if "passenger_count" in df.columns:
            pass_count = np.clip(df["passenger_count"].to_numpy().astype(np.int8), 1, 6)
        else:
            pass_count = np.ones(len(df), dtype=np.int8)

        dt_col = (
            df["pickup_datetime"].str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
        year = dt_col.dt.year().to_numpy().astype(np.int16)
        month = dt_col.dt.month().to_numpy().astype(np.int8)
        day = dt_col.dt.day().to_numpy().astype(np.int8)
        hour = dt_col.dt.hour().to_numpy().astype(np.int8)
        minute = dt_col.dt.minute().to_numpy().astype(np.int8)
        dayofweek = (dt_col.dt.weekday() - 1).to_numpy().astype(np.int8)

        lat_diff = d_lat - p_lat
        lon_diff = d_lon - p_lon
        abs_lat_diff = np.abs(lat_diff)
        abs_lon_diff = np.abs(lon_diff)

        KM_PER_LAT = 111.03
        KM_PER_LON = 84.14

        euclidean_km = np.sqrt(
            (lat_diff * KM_PER_LAT) ** 2 + (lon_diff * KM_PER_LON) ** 2
        )
        manhattan_km = abs_lat_diff * KM_PER_LAT + abs_lon_diff * KM_PER_LON

        rad_p_lat = np.radians(p_lat)
        rad_p_lon = np.radians(p_lon)
        rad_d_lat = np.radians(d_lat)
        rad_d_lon = np.radians(d_lon)

        dlat_rad = rad_d_lat - rad_p_lat
        dlon_rad = rad_d_lon - rad_p_lon
        hav_a = (
            np.sin(dlat_rad / 2.0) ** 2
            + np.cos(rad_p_lat) * np.cos(rad_d_lat) * np.sin(dlon_rad / 2.0) ** 2
        )
        hav_a = np.clip(hav_a, 0.0, 1.0)
        haversine_km = 2.0 * 6371.0088 * np.arcsin(np.sqrt(hav_a))

        theta = np.radians(-29.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_dx = lon_diff * cos_t - lat_diff * sin_t
        rot_dy = lon_diff * sin_t + lat_diff * cos_t
        rot_manhattan_km = np.abs(rot_dy) * KM_PER_LAT + np.abs(rot_dx) * KM_PER_LON

        bearing_y = np.sin(dlon_rad) * np.cos(rad_d_lat)
        bearing_x = np.cos(rad_p_lat) * np.sin(rad_d_lat) - np.sin(rad_p_lat) * np.cos(
            rad_d_lat
        ) * np.cos(dlon_rad)
        bearing = (np.degrees(np.arctan2(bearing_y, bearing_x)) + 360.0) % 360.0
        bearing_sin = np.sin(np.radians(bearing))
        bearing_cos = np.cos(np.radians(bearing))

        mid_lat = (p_lat + d_lat) / 2.0
        mid_lon = (p_lon + d_lon) / 2.0

        hubs = {
            "jfk": (40.6413, -73.7781),
            "lga": (40.7769, -73.8740),
            "ewr": (40.6895, -74.1745),
            "midtown": (40.7580, -73.9855),
            "fidi": (40.7075, -74.0090),
            "gct": (40.7527, -73.9772),
        }

        def hub_distance(lat, lon, hub_lat, hub_lon):
            dlat = (lat - hub_lat) * KM_PER_LAT
            dlon = (lon - hub_lon) * KM_PER_LON
            return np.sqrt(dlat**2 + dlon**2)

        hub_features = {}
        for name, (h_lat, h_lon) in hubs.items():
            dist_p = hub_distance(p_lat, p_lon, h_lat, h_lon)
            dist_d = hub_distance(d_lat, d_lon, h_lat, h_lon)
            hub_features[f"pickup_dist_{name}"] = dist_p.astype(np.float32)
            hub_features[f"dropoff_dist_{name}"] = dist_d.astype(np.float32)
            if name in ["jfk", "lga", "ewr"]:
                min_hub_dist = np.minimum(dist_p, dist_d)
                hub_features[f"min_dist_{name}"] = min_hub_dist.astype(np.float32)
                hub_features[f"is_{name}"] = (min_hub_dist < 2.5).astype(np.int8)

        is_airport = (
            (hub_features["is_jfk"] == 1)
            | (hub_features["is_lga"] == 1)
            | (hub_features["is_ewr"] == 1)
        ).astype(np.int8)

        hour_fraction = hour.astype(np.float32) + minute.astype(np.float32) / 60.0
        hour_sin = np.sin(2.0 * np.pi * hour_fraction / 24.0).astype(np.float32)
        hour_cos = np.cos(2.0 * np.pi * hour_fraction / 24.0).astype(np.float32)
        month_sin = np.sin(2.0 * np.pi * (month - 1) / 12.0).astype(np.float32)
        month_cos = np.cos(2.0 * np.pi * (month - 1) / 12.0).astype(np.float32)
        dow_sin = np.sin(2.0 * np.pi * dayofweek / 7.0).astype(np.float32)
        dow_cos = np.cos(2.0 * np.pi * dayofweek / 7.0).astype(np.float32)

        is_weekend = (dayofweek >= 5).astype(np.int8)
        year_fraction = (
            year.astype(np.float32)
            + (month.astype(np.float32) - 1.0) / 12.0
            + (day.astype(np.float32) - 1.0) / 365.25
        ).astype(np.float32)

        is_rush_hour = ((dayofweek < 5) & (hour >= 16) & (hour < 20)).astype(np.int8)
        is_overnight = ((hour >= 20) | (hour < 6)).astype(np.int8)
        is_post_sep_2012 = (
            (year > 2012)
            | ((year == 2012) & ((month > 9) | ((month == 9) & (day >= 4))))
        ).astype(np.int8)

        tortuosity = (haversine_km / (manhattan_km + 1e-4)).astype(np.float32)

        feature_dict = {
            "pickup_longitude": p_lon.astype(np.float32),
            "pickup_latitude": p_lat.astype(np.float32),
            "dropoff_longitude": d_lon.astype(np.float32),
            "dropoff_latitude": d_lat.astype(np.float32),
            "lat_diff": lat_diff.astype(np.float32),
            "lon_diff": lon_diff.astype(np.float32),
            "abs_lat_diff": abs_lat_diff.astype(np.float32),
            "abs_lon_diff": abs_lon_diff.astype(np.float32),
            "euclidean_km": euclidean_km.astype(np.float32),
            "manhattan_km": manhattan_km.astype(np.float32),
            "rot_manhattan_km": rot_manhattan_km.astype(np.float32),
            "haversine_km": haversine_km.astype(np.float32),
            "tortuosity": tortuosity,
            "bearing": bearing.astype(np.float32),
            "bearing_sin": bearing_sin.astype(np.float32),
            "bearing_cos": bearing_cos.astype(np.float32),
            "mid_latitude": mid_lat.astype(np.float32),
            "mid_longitude": mid_lon.astype(np.float32),
            "is_airport": is_airport,
            "year": year,
            "month": month,
            "day": day,
            "dayofweek": dayofweek,
            "hour": hour,
            "minute": minute,
            "hour_fraction": hour_fraction,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "is_weekend": is_weekend,
            "year_fraction": year_fraction,
            "is_rush_hour": is_rush_hour,
            "is_overnight": is_overnight,
            "is_post_sep_2012": is_post_sep_2012,
            "passenger_count": pass_count,
            "is_solo": (pass_count == 1).astype(np.int8),
            "is_group": (pass_count >= 5).astype(np.int8),
        }
        feature_dict.update(hub_features)

        for k, v in feature_dict.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
                feature_dict[k] = np.nan_to_num(
                    v, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)

        out_df = pl.DataFrame(feature_dict)

        if not is_test:
            out_df = out_df.with_columns(
                pl.Series(
                    "fare_amount", df["fare_amount"].to_numpy().astype(np.float32)
                )
            )
        else:
            out_df = out_df.with_columns(pl.Series("key", df["key"].to_list()))

        return out_df

    print("Extracting domain features for Train split...")
    train_features = build_features(train_split, is_test=False)

    print("Extracting domain features for Validation split...")
    val_features = build_features(val_split, is_test=False)

    print("Extracting domain features for Test split...")
    test_features = build_features(test_pl, is_test=True)

    feature_cols = [c for c in train_features.columns if c != "fare_amount"]
    train_stats = {}
    for col in feature_cols:
        col_arr = train_features[col].to_numpy()
        train_stats[col] = {
            "mean": float(np.mean(col_arr)),
            "std": float(np.std(col_arr)) if np.std(col_arr) > 1e-6 else 1.0,
        }

    meta = {
        "feature_cols": feature_cols,
        "target_col": "fare_amount",
        "train_rows": len(train_features),
        "val_rows": len(val_features),
        "test_rows": len(test_features),
        "train_stats": train_stats,
    }
    with open("./working/feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Persisting parquet files to ./working/...")
    train_features.write_parquet(
        "./working/train_features.parquet", compression="snappy"
    )
    val_features.write_parquet("./working/val_features.parquet", compression="snappy")
    test_features.write_parquet("./working/test_features.parquet", compression="snappy")

    # Arrays for training and evaluation
    X_train = train_features.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_features["fare_amount"].to_numpy().astype(np.float32)
    X_val = val_features.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_features["fare_amount"].to_numpy().astype(np.float32)
    X_test = test_features.select(feature_cols).to_numpy().astype(np.float32)

    train_mean = float(np.mean(y_train))
    baseline_val_rmse = float(np.sqrt(np.mean((y_val - train_mean) ** 2)))
    print(
        f"Data prepared: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )
    print(
        f"Baseline Validation RMSE (Mean Predictor ${train_mean:.2f}): {baseline_val_rmse:.4f}"
    )

    # -------------------------------------------------------------------------
    # Architecture Verification
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target compute device: {device}")
    test_model = build_neural_model(
        in_features=len(feature_cols),
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15,
        init_bias=train_mean,
        device=device,
    )
    total_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"TaxiResNet compiled: {total_params:,} trainable parameters.")

    test_opt, _ = build_optimizer_and_scheduler(
        test_model, lr=1e-3, weight_decay=1e-4, total_steps=100
    )
    test_x = torch.randn(32, len(feature_cols), device=device)
    test_y = torch.full((32,), train_mean, device=device)
    test_model.train()
    test_opt.zero_grad()
    t_loss = nn.MSELoss()(test_model(test_x), test_y)
    t_loss.backward()
    test_opt.step()
    assert not torch.isnan(t_loss).any(), "NaN detected in architecture check"
    print("Neural architecture forward/backward verification passed.")
    del test_model, test_opt, test_x, test_y

    # -------------------------------------------------------------------------
    # Model 1: LightGBM Regressor
    # -------------------------------------------------------------------------
    print("Fitting Model 1: LightGBM Regressor on 10M rows...")
    lgb_model = build_lgbm_regressor(
        {
            "n_estimators": 2000,
            "learning_rate": 0.08,
            "num_leaves": 127,
            "n_jobs": 48,
            "random_state": 42,
        }
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    lgb_val_preds = lgb_model.predict(X_val)
    lgb_val_rmse = float(np.sqrt(np.mean((lgb_val_preds - y_val) ** 2)))
    lgb_test_preds = lgb_model.predict(X_test)
    print(f"LightGBM Holdout Validation RMSE: {lgb_val_rmse:.4f}")
    lgb_model.booster_.save_model("./working/best_lgb_model.txt")

    # -------------------------------------------------------------------------
    # Model 2: GPU-Accelerated XGBoost Regressor
    # -------------------------------------------------------------------------
    print("Fitting Model 2: GPU-Accelerated XGBoost Regressor on 10M rows...")
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    xgb_model = build_xgboost_regressor(
        {
            "tree_method": "hist",
            "device": xgb_device,
            "n_estimators": 2000,
            "learning_rate": 0.08,
            "max_depth": 9,
            "early_stopping_rounds": 40,
            "random_state": 42,
        }
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_preds = xgb_model.predict(X_val)
    xgb_val_rmse = float(np.sqrt(np.mean((xgb_val_preds - y_val) ** 2)))
    xgb_test_preds = xgb_model.predict(X_test)
    print(f"XGBoost Holdout Validation RMSE: {xgb_val_rmse:.4f}")
    xgb_model.save_model("./working/best_xgb_model.json")

    # -------------------------------------------------------------------------
    # Model 3: Deep Tabular Residual Network (TaxiResNet)
    # -------------------------------------------------------------------------
    print("Fitting Model 3: Deep Tabular Residual Network (TaxiResNet) on GPU...")
    means = np.array(
        [train_stats[col]["mean"] for col in feature_cols], dtype=np.float32
    )
    stds = np.array(
        [
            train_stats[col]["std"] if train_stats[col]["std"] > 1e-6 else 1.0
            for col in feature_cols
        ],
        dtype=np.float32,
    )

    X_train_norm = (X_train - means) / stds
    X_val_norm = (X_val - means) / stds
    X_test_norm = (X_test - means) / stds

    nn_model = build_neural_model(
        in_features=len(feature_cols),
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15,
        init_bias=train_mean,
        device=device,
    )

    criterion = RMSELoss()
    total_epochs = 4
    batch_size = 16384
    num_samples = len(X_train_norm)
    total_steps = (num_samples // batch_size + 1) * total_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(
        nn_model, lr=1e-3, weight_decay=1e-4, total_steps=total_steps
    )

    X_train_t = torch.from_numpy(X_train_norm).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val_norm).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_nn_val_rmse = float("inf")
    best_nn_state = None

    for epoch in range(1, total_epochs + 1):
        nn_model.train()
        perm = torch.randperm(num_samples, device=device)
        running_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1

        nn_model.eval()
        with torch.no_grad():
            val_preds_t = nn_model(X_val_t)
            val_loss = criterion(val_preds_t, y_val_t).item()

        if val_loss < best_nn_val_rmse:
            best_nn_val_rmse = val_loss
            best_nn_state = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{total_epochs} - Train RMSE: {running_loss / num_batches:.4f} | Val RMSE: {val_loss:.4f}"
        )

    if best_nn_state is not None:
        nn_model.load_state_dict(best_nn_state)
    nn_model.eval()

    with torch.no_grad():
        nn_val_preds = nn_model(X_val_t).cpu().numpy()
        X_test_t = torch.from_numpy(X_test_norm).to(device)
        nn_test_preds = nn_model(X_test_t).cpu().numpy()

    nn_val_rmse = float(np.sqrt(np.mean((nn_val_preds - y_val) ** 2)))
    print(f"TaxiResNet Holdout Validation RMSE: {nn_val_rmse:.4f}")
    torch.save(best_nn_state, "./working/best_taxi_resnet.pt")

    # -------------------------------------------------------------------------
    # Hybrid Ensembling & Floor Post-Processing
    # -------------------------------------------------------------------------
    print("Optimizing hybrid ensemble weights on holdout validation split...")

    def loss_func(w):
        weights = np.maximum(w, 0.0)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(3) / 3.0
        blend = (
            weights[0] * lgb_val_preds
            + weights[1] * xgb_val_preds
            + weights[2] * nn_val_preds
        )
        blend = np.clip(blend, 2.50, None)
        return np.sqrt(np.mean((blend - y_val) ** 2))

    res = minimize(loss_func, [0.45, 0.45, 0.10], method="Nelder-Mead", tol=1e-5)
    opt_w = np.maximum(res.x, 0.0)
    opt_w = opt_w / np.sum(opt_w)
    print(
        f"Optimal Weights: LightGBM={opt_w[0]:.3f} | XGBoost={opt_w[1]:.3f} | TaxiResNet={opt_w[2]:.3f}"
    )

    final_val_preds = (
        opt_w[0] * lgb_val_preds + opt_w[1] * xgb_val_preds + opt_w[2] * nn_val_preds
    )
    final_val_preds = np.clip(final_val_preds, 2.50, None)
    final_val_rmse = float(np.sqrt(np.mean((final_val_preds - y_val) ** 2)))
    print(f"Final Ensembled Holdout Validation RMSE: {final_val_rmse:.6f}")

    final_test_preds = (
        opt_w[0] * lgb_test_preds + opt_w[1] * xgb_test_preds + opt_w[2] * nn_test_preds
    )
    final_test_preds = np.clip(final_test_preds, 2.50, None)

    # -------------------------------------------------------------------------
    # Submission Generation & Validation
    # -------------------------------------------------------------------------
    print("Writing predictions to ./submission/submission.csv...")
    submission_df = pd.DataFrame(
        {
            "key": test_keys,
            "fare_amount": np.round(final_test_preds.astype(np.float64), 2),
        }
    )
    submission_path = "./submission/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    assert os.path.exists(submission_path), "Submission file missing"
    assert len(submission_df) == 9914, f"Expected 9914 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == ["key", "fare_amount"], "Column mismatch"
    assert submission_df["fare_amount"].isna().sum() == 0, "NaN detected in predictions"
    assert (
        submission_df["fare_amount"] >= 2.50
    ).all(), "Predictions violate minimum fare floor"
    print(
        f"Verified submission successfully persisted ({len(submission_df)} rows) to {submission_path}."
    )

    print(f"Final Validation Score: {final_val_rmse:.6f}")


if __name__ == "__main__":
    main()
