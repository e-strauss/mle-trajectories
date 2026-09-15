import gc
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import copy
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# =============================================================================
# Model Architectures and Builder Functions
# =============================================================================
def build_lgbm_regressor(params: Optional[Dict[str, Any]] = None) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.08,
        "num_leaves": 511,
        "max_depth": 11,
        "max_bin": 512,
        "min_child_samples": 150,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def build_xgboost_regressor(
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBRegressor:
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": xgb_device,
        "max_bin": 512,
        "n_estimators": 2000,
        "learning_rate": 0.08,
        "max_depth": 11,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "early_stopping_rounds": 40,
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = self.act2(out + residual)
        return out


class SpatialResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.in_bn = nn.BatchNorm1d(hidden_dim)
        self.in_act = nn.Mish()

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_act(self.in_bn(self.in_proj(self.input_bn(x))))
        for block in self.blocks:
            h = block(h)
        out = self.out_head(h)
        return out.squeeze(-1)


def train_spatial_resnet(
    model: SpatialResNet,
    X_train: np.ndarray,
    delta_train: np.ndarray,
    X_val: np.ndarray,
    delta_val: np.ndarray,
    statutory_base_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 4,
    batch_size: int = 16384,
    lr: float = 2e-3,
    device: str = "cuda",
) -> SpatialResNet:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * int(math.ceil(len(X_train) / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps), eta_min=1e-5
    )
    criterion = nn.MSELoss()
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(delta_train),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    best_val_rmse = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(batch_x)
                loss = criterion(preds, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        val_delta_pred = predict_spatial_resnet(model, X_val, device)
        val_fare_pred = np.maximum(statutory_base_val + val_delta_pred, 2.50)
        val_rmse = float(np.sqrt(np.mean((val_fare_pred - y_val) ** 2)))
        print(
            f"  Epoch {epoch + 1}/{epochs} | Train Residual MSE: {avg_loss:.4f} | Val Fare RMSE: {val_rmse:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def predict_spatial_resnet(
    model: SpatialResNet,
    X: np.ndarray,
    device: str,
    batch_size: int = 32768,
) -> np.ndarray:
    model.eval()
    preds = []
    n = len(X)
    use_amp = device == "cuda"
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(batch)
            preds.append(pred.cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


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

    # Physics-based coordinate-fare consistency checks
    dlat_km = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * 111.03
    dlon_km = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * 84.14
    dist_km = (dlat_km * dlat_km + dlon_km * dlon_km).sqrt()

    # Airport hub proximity checks for non-airport filtering
    p_dist_jfk = (((pl.col("pickup_latitude") - 40.6413) * 111.03)**2 + ((pl.col("pickup_longitude") - (-73.7781)) * 84.14)**2).sqrt()
    d_dist_jfk = (((pl.col("dropoff_latitude") - 40.6413) * 111.03)**2 + ((pl.col("dropoff_longitude") - (-73.7781)) * 84.14)**2).sqrt()
    p_dist_lga = (((pl.col("pickup_latitude") - 40.7769) * 111.03)**2 + ((pl.col("pickup_longitude") - (-73.8740)) * 84.14)**2).sqrt()
    d_dist_lga = (((pl.col("dropoff_latitude") - 40.7769) * 111.03)**2 + ((pl.col("dropoff_longitude") - (-73.8740)) * 84.14)**2).sqrt()
    p_dist_ewr = (((pl.col("pickup_latitude") - 40.6895) * 111.03)**2 + ((pl.col("pickup_longitude") - (-74.1745)) * 84.14)**2).sqrt()
    d_dist_ewr = (((pl.col("dropoff_latitude") - 40.6895) * 111.03)**2 + ((pl.col("dropoff_longitude") - (-74.1745)) * 84.14)**2).sqrt()

    is_airport_vicinity = (
        (p_dist_jfk < 3.0) | (d_dist_jfk < 3.0) |
        (p_dist_lga < 3.0) | (d_dist_lga < 3.0) |
        (p_dist_ewr < 3.0) | (d_dist_ewr < 3.0)
    )

    valid_distance_fare = ~((dist_km > 2.0) & (pl.col("fare_amount") < 3.00))
    valid_short_fare = ~((dist_km < 0.2) & (pl.col("fare_amount") > 30.00) & (~is_airport_vicinity))

    physics_filter = (
        train_clean_filter
        & valid_distance_fare
        & valid_short_fare
    )

    print("Streaming and filtering training data with physics-consistency guards...")
    train_raw = (
        pl.scan_csv("./input/train.csv")
        .filter(physics_filter)
        .collect()
    )
    print(f"Collected {len(train_raw):,} cleaned trip records.")

    val_size = 100_000
    target_train_size = 32_000_000
    total_needed = target_train_size + val_size

    if len(train_raw) > total_needed:
        train_raw = train_raw.sample(n=total_needed, shuffle=True, seed=42)
    else:
        train_raw = train_raw.sample(fraction=1.0, shuffle=True, seed=42)

    train_size = len(train_raw) - val_size
    train_split = train_raw.slice(0, train_size)
    val_split = train_raw.slice(train_size, val_size)
    del train_raw
    gc.collect()
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

        # Metric-space 29-degree Manhattan grid rotation
        dy_km = lat_diff * KM_PER_LAT
        dx_km = lon_diff * KM_PER_LON

        theta_29 = np.radians(29.0)
        cos_29 = np.cos(theta_29)
        sin_29 = np.sin(theta_29)

        rot_avenue_vec = dy_km * cos_29 + dx_km * sin_29
        rot_crosstown_vec = -dy_km * sin_29 + dx_km * cos_29
        rot_avenue_km = np.abs(rot_avenue_vec).astype(np.float32)
        rot_crosstown_km = np.abs(rot_crosstown_vec).astype(np.float32)
        rot_manhattan_km = (rot_avenue_km + rot_crosstown_km).astype(np.float32)

        # 29-degree Manhattan grid street orientation angle via arctan2
        grid_orientation_angle = np.arctan2(rot_avenue_vec, rot_crosstown_vec).astype(np.float32)
        grid_angle_sin = np.sin(grid_orientation_angle).astype(np.float32)
        grid_angle_cos = np.cos(grid_orientation_angle).astype(np.float32)

        # Directly exposed 29-degree rotated coordinates
        theta = np.radians(-29.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_p_lon = (p_lon * cos_t - p_lat * sin_t).astype(np.float32)
        rot_p_lat = (p_lon * sin_t + p_lat * cos_t).astype(np.float32)
        rot_d_lon = (d_lon * cos_t - d_lat * sin_t).astype(np.float32)
        rot_d_lat = (d_lon * sin_t + d_lat * cos_t).astype(np.float32)

        # 0.005-degree micro-neighborhood coordinate bins (~500m resolution)
        p_lat_bin = np.floor((p_lat - 40.40) / 0.005).astype(np.float32)
        p_lon_bin = np.floor((p_lon - (-74.50)) / 0.005).astype(np.float32)
        d_lat_bin = np.floor((d_lat - 40.40) / 0.005).astype(np.float32)
        d_lon_bin = np.floor((d_lon - (-74.50)) / 0.005).astype(np.float32)
        p_micro_bin = (p_lat_bin * 1000.0 + p_lon_bin).astype(np.float32)
        d_micro_bin = (d_lat_bin * 1000.0 + d_lon_bin).astype(np.float32)

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

        # 5 major transit hubs
        transit_landmarks = {
            "penn": (40.7505, -73.9934),
            "port_auth": (40.7570, -73.9903),
            "times_sq": (40.7580, -73.9855),
            "wtc": (40.7128, -74.0134),
            "barclays": (40.6826, -73.9754),
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

        for name, (th_lat, th_lon) in transit_landmarks.items():
            dist_p = hub_distance(p_lat, p_lon, th_lat, th_lon)
            dist_d = hub_distance(d_lat, d_lon, th_lat, th_lon)
            hub_features[f"pickup_dist_{name}"] = dist_p.astype(np.float32)
            hub_features[f"dropoff_dist_{name}"] = dist_d.astype(np.float32)
            hub_features[f"min_dist_{name}"] = np.minimum(dist_p, dist_d).astype(np.float32)

        is_airport = (
            (hub_features["is_jfk"] == 1)
            | (hub_features["is_lga"] == 1)
            | (hub_features["is_ewr"] == 1)
        ).astype(np.int8)

        p_jfk = hub_features["pickup_dist_jfk"] < 3.0
        d_jfk = hub_features["dropoff_dist_jfk"] < 3.0
        p_manhattan = (p_lat >= 40.70) & (p_lat <= 40.85) & (p_lon >= -74.02) & (p_lon <= -73.93)
        d_manhattan = (d_lat >= 40.70) & (d_lat <= 40.85) & (d_lon >= -74.02) & (d_lon <= -73.93)
        is_manhattan_to_jfk = ((p_manhattan & d_jfk) | (d_manhattan & p_jfk)).astype(np.int8)

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

        # NYC borough spatial bounding indicators
        is_pickup_manhattan = ((p_lat >= 40.70) & (p_lat <= 40.88) & (p_lon >= -74.03) & (p_lon <= -73.90)).astype(np.int8)
        is_dropoff_manhattan = ((d_lat >= 40.70) & (d_lat <= 40.88) & (d_lon >= -74.03) & (d_lon <= -73.90)).astype(np.int8)
        is_pickup_brooklyn = ((p_lat >= 40.57) & (p_lat <= 40.74) & (p_lon >= -74.05) & (p_lon <= -73.85)).astype(np.int8)
        is_dropoff_brooklyn = ((d_lat >= 40.57) & (d_lat <= 40.74) & (d_lon >= -74.05) & (d_lon <= -73.85)).astype(np.int8)
        is_pickup_queens = ((p_lat >= 40.54) & (p_lat <= 40.80) & (p_lon >= -73.97) & (p_lon <= -73.70)).astype(np.int8)
        is_dropoff_queens = ((d_lat >= 40.54) & (d_lat <= 40.80) & (d_lon >= -73.97) & (d_lon <= -73.70)).astype(np.int8)
        is_pickup_bronx = ((p_lat >= 40.79) & (p_lat <= 40.92) & (p_lon >= -73.93) & (p_lon <= -73.77)).astype(np.int8)
        is_dropoff_bronx = ((d_lat >= 40.79) & (d_lat <= 40.92) & (d_lon >= -73.93) & (d_lon <= -73.77)).astype(np.int8)
        is_pickup_staten_island = ((p_lat >= 40.49) & (p_lat <= 40.65) & (p_lon >= -74.26) & (p_lon <= -74.05)).astype(np.int8)
        is_dropoff_staten_island = ((d_lat >= 40.49) & (d_lat <= 40.65) & (d_lon >= -74.26) & (d_lon <= -74.05)).astype(np.int8)
        is_pickup_newark = ((p_lat >= 40.65) & (p_lat <= 40.78) & (p_lon >= -74.25) & (p_lon <= -74.05)).astype(np.int8)
        is_dropoff_newark = ((d_lat >= 40.65) & (d_lat <= 40.78) & (d_lon >= -74.25) & (d_lon <= -74.05)).astype(np.int8)

        # River crossing flags
        p_bk_qn = (is_pickup_brooklyn == 1) | (is_pickup_queens == 1)
        d_bk_qn = (is_dropoff_brooklyn == 1) | (is_dropoff_queens == 1)
        is_east_river_crossing = (
            ((is_pickup_manhattan == 1) & d_bk_qn) | (p_bk_qn & (is_dropoff_manhattan == 1))
        ).astype(np.int8)
        is_hudson_river_crossing = (
            ((is_pickup_manhattan == 1) & (is_dropoff_newark == 1))
            | ((is_pickup_newark == 1) & (is_dropoff_manhattan == 1))
        ).astype(np.int8)

        # 36-state borough origin-destination (OD) transition feature (6 pickup x 6 dropoff zones)
        p_borough = np.zeros(len(df), dtype=np.int32)
        p_borough = np.where(is_pickup_brooklyn == 1, 1, p_borough)
        p_borough = np.where(is_pickup_queens == 1, 2, p_borough)
        p_borough = np.where(is_pickup_bronx == 1, 3, p_borough)
        p_borough = np.where(is_pickup_staten_island == 1, 4, p_borough)
        p_borough = np.where(is_pickup_newark == 1, 5, p_borough)
        p_borough = np.where(is_pickup_manhattan == 1, 0, p_borough)

        d_borough = np.zeros(len(df), dtype=np.int32)
        d_borough = np.where(is_dropoff_brooklyn == 1, 1, d_borough)
        d_borough = np.where(is_dropoff_queens == 1, 2, d_borough)
        d_borough = np.where(is_dropoff_bronx == 1, 3, d_borough)
        d_borough = np.where(is_dropoff_staten_island == 1, 4, d_borough)
        d_borough = np.where(is_dropoff_newark == 1, 5, d_borough)
        d_borough = np.where(is_dropoff_manhattan == 1, 0, d_borough)

        borough_od = (p_borough * 6 + d_borough).astype(np.float32)

        # Newark out-of-city surcharge flag
        is_newark_surcharge = (
            (is_pickup_newark == 1)
            | (is_dropoff_newark == 1)
            | (hub_features["is_ewr"] == 1)
        ).astype(np.int8)

        # Statutory pre-November 2009 MTA tax indicator ($0.50 surcharge introduced Nov 1, 2009)
        is_pre_nov_2009_mta = (
            (year < 2009) | ((year == 2009) & (month < 11))
        ).astype(np.int8)

        # Statutory JFK flat-fare interaction ($45 pre-Sept 2012 vs $52 post-Sept 2012)
        is_jfk_flat_pre_sep_2012 = (is_manhattan_to_jfk & (1 - is_post_sep_2012)).astype(np.int8)
        is_jfk_flat_post_sep_2012 = (is_manhattan_to_jfk & is_post_sep_2012).astype(np.int8)

        hav_post_sep_2012 = (haversine_km * is_post_sep_2012).astype(np.float32)
        rot_manhattan_post_sep_2012 = (rot_manhattan_km * is_post_sep_2012).astype(np.float32)
        is_rush_post_2012 = (is_rush_hour * is_post_sep_2012).astype(np.int8)
        is_overnight_post_2012 = (is_overnight * is_post_sep_2012).astype(np.int8)
        rush_haversine = (haversine_km * is_rush_hour).astype(np.float32)
        overnight_haversine = (haversine_km * is_overnight).astype(np.float32)

        # Avenue vs crosstown distance interactions with peak congestion indicators
        avenue_rush = (rot_avenue_km * is_rush_hour).astype(np.float32)
        crosstown_rush = (rot_crosstown_km * is_rush_hour).astype(np.float32)
        avenue_overnight = (rot_avenue_km * is_overnight).astype(np.float32)
        crosstown_overnight = (rot_crosstown_km * is_overnight).astype(np.float32)

        # Analytical NYC TLC Statutory Base Fare calculation with JFK Flat Fare & Newark surcharges
        rot_manhattan_miles = rot_manhattan_km / 1.609344
        mileage_rate = np.where(is_post_sep_2012 == 1, 2.50, 2.00)
        mta_tax = np.where(is_pre_nov_2009_mta == 1, 0.0, 0.50)
        overnight_surcharge = np.where(is_overnight == 1, 0.50, 0.0)
        rush_hour_surcharge = np.where(is_rush_hour == 1, 1.00, 0.0)
        standard_base_fare = (
            2.50
            + mileage_rate * rot_manhattan_miles
            + mta_tax
            + overnight_surcharge
            + rush_hour_surcharge
        )
        # JFK flat fare: $45 pre-Sept 2012, $52 post-Sept 2012, with $4.50 post-Sep 2012 rush surcharge
        jfk_rush_surcharge = np.where(
            (is_post_sep_2012 == 1) & (is_rush_hour == 1),
            4.50,
            rush_hour_surcharge,
        )
        jfk_flat_fare = (
            np.where(is_post_sep_2012 == 1, 52.00, 45.00)
            + mta_tax
            + overnight_surcharge
            + jfk_rush_surcharge
        )
        # Newark flat surcharges: $15.00 pre-2012, $17.50 post-2012
        newark_surcharge_amount = np.where(
            is_newark_surcharge == 1,
            np.where(is_post_sep_2012 == 1, 17.50, 15.00),
            0.0,
        )
        statutory_base_fare = np.where(
            is_manhattan_to_jfk == 1,
            jfk_flat_fare,
            standard_base_fare + newark_surcharge_amount,
        ).astype(np.float32)

        # Tolled facility proximities distinct from toll-free East River bridges
        toll_facilities = {
            "queens_midtown": (40.7445, -73.9630),
            "brooklyn_battery": (40.6975, -74.0130),
            "rfk_triborough": (40.7798, -73.9214),
            "lincoln_tunnel": (40.7610, -74.0040),
            "holland_tunnel": (40.7265, -74.0155),
            "verrazzano": (40.6066, -74.0447),
        }
        toll_features = {}
        for name, (fac_lat, fac_lon) in toll_facilities.items():
            dist_p = hub_distance(p_lat, p_lon, fac_lat, fac_lon)
            dist_d = hub_distance(d_lat, d_lon, fac_lat, fac_lon)
            min_dist = np.minimum(dist_p, dist_d)
            detour = dist_p + dist_d - euclidean_km
            toll_features[f"toll_dist_p_{name}"] = dist_p.astype(np.float32)
            toll_features[f"toll_dist_d_{name}"] = dist_d.astype(np.float32)
            toll_features[f"toll_min_dist_{name}"] = min_dist.astype(np.float32)
            toll_features[f"toll_detour_{name}"] = detour.astype(np.float32)
            toll_features[f"is_near_toll_{name}"] = (min_dist < 1.5).astype(np.int8)

        min_toll_dist = np.minimum.reduce([toll_features[f"toll_min_dist_{name}"] for name in toll_facilities])
        toll_features["min_toll_facility_dist"] = min_toll_dist.astype(np.float32)

        free_bridges = {
            "brooklyn_bridge": (40.7061, -73.9969),
            "manhattan_bridge": (40.7075, -73.9908),
            "williamsburg_bridge": (40.7135, -73.9724),
            "queensboro_bridge": (40.7569, -73.9546),
        }
        free_bridge_dists = []
        for name, (br_lat, br_lon) in free_bridges.items():
            dist_p = hub_distance(p_lat, p_lon, br_lat, br_lon)
            dist_d = hub_distance(d_lat, d_lon, br_lat, br_lon)
            min_dist = np.minimum(dist_p, dist_d)
            toll_features[f"free_bridge_min_dist_{name}"] = min_dist.astype(np.float32)
            free_bridge_dists.append(min_dist)
        toll_features["min_free_bridge_dist"] = np.minimum.reduce(free_bridge_dists).astype(np.float32)
        toll_features["toll_vs_free_diff"] = (toll_features["min_toll_facility_dist"] - toll_features["min_free_bridge_dist"]).astype(np.float32)

        tortuosity = (haversine_km / (manhattan_km + 1e-4)).astype(np.float32)

        # Metric-projected rotated coordinates relative to NYC center (40.75, -73.98)
        p_y_km = (p_lat - 40.75) * KM_PER_LAT
        p_x_km = (p_lon - (-73.98)) * KM_PER_LON
        d_y_km = (d_lat - 40.75) * KM_PER_LAT
        d_x_km = (d_lon - (-73.98)) * KM_PER_LON

        rot_p_u_km = (p_y_km * cos_29 + p_x_km * sin_29).astype(np.float32)
        rot_p_v_km = (-p_y_km * sin_29 + p_x_km * cos_29).astype(np.float32)
        rot_d_u_km = (d_y_km * cos_29 + d_x_km * sin_29).astype(np.float32)
        rot_d_v_km = (-d_y_km * sin_29 + d_x_km * cos_29).astype(np.float32)
        rot_u_diff_km = (rot_d_u_km - rot_p_u_km).astype(np.float32)
        rot_v_diff_km = (rot_d_v_km - rot_p_v_km).astype(np.float32)

        # Multi-frequency sinusoidal spatial coordinate embeddings
        p_lat_c = (p_lat - 40.75).astype(np.float32)
        p_lon_c = (p_lon - (-73.98)).astype(np.float32)
        d_lat_c = (d_lat - 40.75).astype(np.float32)
        d_lon_c = (d_lon - (-73.98)).astype(np.float32)

        spatial_sinusoids = {}
        for freq in [5.0, 20.0, 100.0, 500.0]:
            f_str = str(int(freq))
            spatial_sinusoids[f"sin_p_lat_{f_str}"] = np.sin(freq * p_lat_c).astype(np.float32)
            spatial_sinusoids[f"cos_p_lat_{f_str}"] = np.cos(freq * p_lat_c).astype(np.float32)
            spatial_sinusoids[f"sin_p_lon_{f_str}"] = np.sin(freq * p_lon_c).astype(np.float32)
            spatial_sinusoids[f"cos_p_lon_{f_str}"] = np.cos(freq * p_lon_c).astype(np.float32)
            spatial_sinusoids[f"sin_d_lat_{f_str}"] = np.sin(freq * d_lat_c).astype(np.float32)
            spatial_sinusoids[f"cos_d_lat_{f_str}"] = np.cos(freq * d_lat_c).astype(np.float32)
            spatial_sinusoids[f"sin_d_lon_{f_str}"] = np.sin(freq * d_lon_c).astype(np.float32)
            spatial_sinusoids[f"cos_d_lon_{f_str}"] = np.cos(freq * d_lon_c).astype(np.float32)

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
            "rot_avenue_km": rot_avenue_km,
            "rot_crosstown_km": rot_crosstown_km,
            "rot_p_lon": rot_p_lon,
            "rot_p_lat": rot_p_lat,
            "rot_d_lon": rot_d_lon,
            "rot_d_lat": rot_d_lat,
            "rot_p_u_km": rot_p_u_km,
            "rot_p_v_km": rot_p_v_km,
            "rot_d_u_km": rot_d_u_km,
            "rot_d_v_km": rot_d_v_km,
            "rot_u_diff_km": rot_u_diff_km,
            "rot_v_diff_km": rot_v_diff_km,
            "grid_orientation_angle": grid_orientation_angle,
            "grid_angle_sin": grid_angle_sin,
            "grid_angle_cos": grid_angle_cos,
            "p_lat_bin": p_lat_bin,
            "p_lon_bin": p_lon_bin,
            "d_lat_bin": d_lat_bin,
            "d_lon_bin": d_lon_bin,
            "p_micro_bin": p_micro_bin,
            "d_micro_bin": d_micro_bin,
            "statutory_base_fare": statutory_base_fare,
            "newark_surcharge_amount": newark_surcharge_amount.astype(np.float32),
            "jfk_rush_surcharge": jfk_rush_surcharge.astype(np.float32),
            "avenue_rush": avenue_rush,
            "crosstown_rush": crosstown_rush,
            "avenue_overnight": avenue_overnight,
            "crosstown_overnight": crosstown_overnight,
            "haversine_km": haversine_km.astype(np.float32),
            "hav_post_sep_2012": hav_post_sep_2012,
            "rot_manhattan_post_sep_2012": rot_manhattan_post_sep_2012,
            "tortuosity": tortuosity,
            "bearing": bearing.astype(np.float32),
            "bearing_sin": bearing_sin.astype(np.float32),
            "bearing_cos": bearing_cos.astype(np.float32),
            "mid_latitude": mid_lat.astype(np.float32),
            "mid_longitude": mid_lon.astype(np.float32),
            "is_airport": is_airport,
            "is_manhattan_to_jfk": is_manhattan_to_jfk,
            "is_pickup_manhattan": is_pickup_manhattan,
            "is_dropoff_manhattan": is_dropoff_manhattan,
            "is_pickup_brooklyn": is_pickup_brooklyn,
            "is_dropoff_brooklyn": is_dropoff_brooklyn,
            "is_pickup_queens": is_pickup_queens,
            "is_dropoff_queens": is_dropoff_queens,
            "is_pickup_bronx": is_pickup_bronx,
            "is_dropoff_bronx": is_dropoff_bronx,
            "is_pickup_staten_island": is_pickup_staten_island,
            "is_dropoff_staten_island": is_dropoff_staten_island,
            "is_pickup_newark": is_pickup_newark,
            "is_dropoff_newark": is_dropoff_newark,
            "is_east_river_crossing": is_east_river_crossing,
            "is_hudson_river_crossing": is_hudson_river_crossing,
            "is_newark_surcharge": is_newark_surcharge,
            "is_pre_nov_2009_mta": is_pre_nov_2009_mta,
            "is_jfk_flat_pre_sep_2012": is_jfk_flat_pre_sep_2012,
            "is_jfk_flat_post_sep_2012": is_jfk_flat_post_sep_2012,
            "borough_od": borough_od,
            "year": year.astype(np.float32),
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
            "is_rush_post_2012": is_rush_post_2012,
            "is_overnight_post_2012": is_overnight_post_2012,
            "rush_haversine": rush_haversine,
            "overnight_haversine": overnight_haversine,
            "passenger_count": pass_count,
            "is_solo": (pass_count == 1).astype(np.int8),
            "is_group": (pass_count >= 5).astype(np.int8),
        }
        feature_dict.update(hub_features)
        feature_dict.update(toll_features)
        feature_dict.update(spatial_sinusoids)

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
    del train_split
    gc.collect()

    print("Extracting domain features for Validation split...")
    val_features = build_features(val_split, is_test=False)
    del val_split
    gc.collect()

    print("Extracting domain features for Test split...")
    test_features = build_features(test_pl, is_test=True)
    del test_pl
    gc.collect()

    feature_cols = [c for c in train_features.columns if c != "fare_amount"]
    meta = {
        "feature_cols": feature_cols,
        "target_col": "fare_amount",
        "train_rows": len(train_features),
        "val_rows": len(val_features),
        "test_rows": len(test_features),
    }
    with open("./working/feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Statutory base fare extraction and residual Delta calculation
    statutory_base_train = train_features["statutory_base_fare"].to_numpy().astype(np.float32)
    statutory_base_val = val_features["statutory_base_fare"].to_numpy().astype(np.float32)
    statutory_base_test = test_features["statutory_base_fare"].to_numpy().astype(np.float32)

    y_train = train_features["fare_amount"].to_numpy().astype(np.float32)
    y_val = val_features["fare_amount"].to_numpy().astype(np.float32)

    delta_train = y_train - statutory_base_train
    delta_val = y_val - statutory_base_val

    # Feature matrices
    X_train = train_features.select(feature_cols).to_numpy().astype(np.float32)
    del train_features
    gc.collect()

    X_val = val_features.select(feature_cols).to_numpy().astype(np.float32)
    val_features.write_parquet("./working/val_features.parquet", compression="snappy")
    del val_features
    gc.collect()

    X_test = test_features.select(feature_cols).to_numpy().astype(np.float32)
    test_features.write_parquet("./working/test_features.parquet", compression="snappy")
    del test_features
    gc.collect()

    statutory_val_rmse = float(np.sqrt(np.mean((np.maximum(statutory_base_val, 2.50) - y_val) ** 2)))
    print(
        f"Data prepared: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )
    print(
        f"Statutory Baseline Holdout Fare RMSE (Zero Residual Baseline): {statutory_val_rmse:.4f}"
    )

    # -------------------------------------------------------------------------
    # Model 1: High-Capacity LightGBM Regressor (num_leaves=511, max_depth=11)
    # -------------------------------------------------------------------------
    print(f"Fitting Model 1: High-Capacity LightGBM Regressor on {len(X_train):,} rows...")
    lgb_model = build_lgbm_regressor(
        {
            "num_leaves": 511,
            "max_depth": 11,
            "n_estimators": 2000,
            "learning_rate": 0.08,
            "max_bin": 512,
            "min_child_samples": 150,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 5.0,
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(X_train, delta_train, eval_set=[(X_val, delta_val)], callbacks=callbacks)
    lgb_val_delta = lgb_model.predict(X_val).astype(np.float32)
    lgb_val_preds = np.maximum(statutory_base_val + lgb_val_delta, 2.50)
    lgb_val_rmse = float(np.sqrt(np.mean((lgb_val_preds - y_val) ** 2)))
    lgb_test_delta = lgb_model.predict(X_test).astype(np.float32)
    print(f"LightGBM Holdout Validation Fare RMSE: {lgb_val_rmse:.4f}")
    lgb_model.booster_.save_model("./working/best_lgb_model.txt")
    del lgb_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 2: GPU-Accelerated XGBoost Regressor (max_depth=11)
    # -------------------------------------------------------------------------
    print(f"Fitting Model 2: GPU-Accelerated XGBoost Regressor on {len(X_train):,} rows...")
    xgb_model = build_xgboost_regressor(
        {
            "max_depth": 11,
            "n_estimators": 2000,
            "learning_rate": 0.08,
            "max_bin": 512,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_lambda": 5.0,
            "early_stopping_rounds": 40,
            "random_state": 42,
        }
    )
    xgb_model.fit(X_train, delta_train, eval_set=[(X_val, delta_val)], verbose=False)
    xgb_val_delta = xgb_model.predict(X_val).astype(np.float32)
    xgb_val_preds = np.maximum(statutory_base_val + xgb_val_delta, 2.50)
    xgb_val_rmse = float(np.sqrt(np.mean((xgb_val_preds - y_val) ** 2)))
    xgb_test_delta = xgb_model.predict(X_test).astype(np.float32)
    print(f"XGBoost Holdout Validation Fare RMSE: {xgb_val_rmse:.4f}")
    xgb_model.save_model("./working/best_xgb_model.json")
    del xgb_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 3: Deep GPU PyTorch Spatial-ResNet
    # -------------------------------------------------------------------------
    print(f"Fitting Model 3: Deep GPU PyTorch Spatial-ResNet on {len(X_train):,} rows...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spatial_resnet = SpatialResNet(
        in_features=X_train.shape[1],
        hidden_dim=256,
        num_blocks=3,
        dropout=0.05,
    ).to(device)

    spatial_resnet = train_spatial_resnet(
        model=spatial_resnet,
        X_train=X_train,
        delta_train=delta_train,
        X_val=X_val,
        delta_val=delta_val,
        statutory_base_val=statutory_base_val,
        y_val=y_val,
        epochs=4,
        batch_size=16384,
        lr=2e-3,
        device=device,
    )
    torch.save(spatial_resnet.state_dict(), "./working/best_spatial_resnet.pt")
    nn_val_delta = predict_spatial_resnet(spatial_resnet, X_val, device)
    nn_val_preds = np.maximum(statutory_base_val + nn_val_delta, 2.50)
    nn_val_rmse = float(np.sqrt(np.mean((nn_val_preds - y_val) ** 2)))
    nn_test_delta = predict_spatial_resnet(spatial_resnet, X_test, device)
    print(f"Spatial-ResNet Holdout Validation Fare RMSE: {nn_val_rmse:.4f}")

    del spatial_resnet
    del X_train, delta_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Tri-Model Ensembling & Simplex Optimization
    # -------------------------------------------------------------------------
    print("Optimizing tri-model ensemble weights on holdout validation split...")

    def loss_func(w):
        weights = np.maximum(w, 0.0)
        s = np.sum(weights)
        if s > 0:
            weights = weights / s
        else:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        blend_delta = (
            weights[0] * lgb_val_delta
            + weights[1] * xgb_val_delta
            + weights[2] * nn_val_delta
        )
        blend_fare = np.maximum(statutory_base_val + blend_delta, 2.50)
        return float(np.sqrt(np.mean((blend_fare - y_val) ** 2)))

    best_loss = float("inf")
    opt_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_points = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.3, 0.2],
        [0.6, 0.2, 0.2],
        [0.4, 0.4, 0.2],
        [0.4, 0.3, 0.3],
        [0.5, 0.25, 0.25],
        [0.7, 0.15, 0.15],
        [0.45, 0.45, 0.10],
    ]
    for w_init in init_points:
        res = minimize(
            loss_func,
            w_init,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            tol=1e-6,
        )
        if res.fun < best_loss:
            best_loss = res.fun
            opt_w = res.x

    # Fine-grained grid search verification across the 2D simplex manifold
    step = 0.02
    grid_vals = np.arange(0.0, 1.0 + step / 2, step)
    for w0 in grid_vals:
        for w1 in np.arange(0.0, 1.0 - w0 + step / 2, step):
            w2 = max(0.0, 1.0 - w0 - w1)
            candidate_w = np.array([w0, w1, w2])
            val_loss = loss_func(candidate_w)
            if val_loss < best_loss:
                best_loss = val_loss
                opt_w = candidate_w

    opt_w = np.maximum(opt_w, 0.0)
    opt_w = opt_w / np.sum(opt_w)
    print(
        f"Optimal Tri-Model Simplex Weights: LightGBM={opt_w[0]:.4f} | XGBoost={opt_w[1]:.4f} | Spatial-ResNet={opt_w[2]:.4f}"
    )

    final_val_delta = (
        opt_w[0] * lgb_val_delta
        + opt_w[1] * xgb_val_delta
        + opt_w[2] * nn_val_delta
    )
    final_val_preds = np.maximum(statutory_base_val + final_val_delta, 2.50)
    final_val_rmse = float(np.sqrt(np.mean((final_val_preds - y_val) ** 2)))
    print(f"Final Ensembled Holdout Validation RMSE: {final_val_rmse:.6f}")

    final_test_delta = (
        opt_w[0] * lgb_test_delta
        + opt_w[1] * xgb_test_delta
        + opt_w[2] * nn_test_delta
    )
    final_test_preds = np.maximum(statutory_base_test + final_test_delta, 2.50)

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
