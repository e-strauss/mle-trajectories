import gc
import json
import math
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. Model Architectures & Configurations (Step 2: model_design)
# =========================================================================
class ResidualBlock(nn.Module):
    """Residual block with LayerNorm, SiLU activations, and Dropout for tabular regularization."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.act(self.ln2(self.fc2(out)))
        return x + out


class TaxiFareResidualNN(nn.Module):
    """High-Capacity Deep Tabular Residual Network for NYC Taxi Fare Prediction.

    Incorporates input projection, stacked residual blocks, feature
    regularization,
    and a domain-constrained positive fare output head (NYC min fare >= $2.50).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        self.input_act = nn.SiLU()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_act(self.input_ln(self.input_layer(x)))
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        fare = F.relu(out) + 2.50
        return fare.squeeze(-1)


class RMSELoss(nn.Module):
    """Exact Root Mean Squared Error (RMSE) Loss Function."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 20000,
    eta_min: float = 1e-6,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )
    return optimizer, scheduler


def get_lgbm_regressor_config():
    """Returns optimal LightGBM configuration aligned to RMSE minimization on tabular data."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 512,
        "max_depth": -1,
        "learning_rate": 0.04,
        "n_estimators": 5000,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 100,
        "n_jobs": 64,
        "random_state": 42,
        "verbose": -1,
    }


def get_xgboost_regressor_config():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 10,
        "learning_rate": 0.05,
        "n_estimators": 4000,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 50,
        "random_state": 42,
    }


def get_catboost_regressor_config():
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "GPU",
        "depth": 8,
        "learning_rate": 0.05,
        "iterations": 4000,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
    }


# =========================================================================
# 2. Domain Feature Extraction (Step 1: data_processing_and_feature_engineering)
# =========================================================================
def extract_features(df: pl.DataFrame, is_test: bool = False):
    p_lat = df["pickup_latitude"].to_numpy().astype(np.float32)
    p_lon = df["pickup_longitude"].to_numpy().astype(np.float32)
    d_lat = df["dropoff_latitude"].to_numpy().astype(np.float32)
    d_lon = df["dropoff_longitude"].to_numpy().astype(np.float32)
    pass_cnt = df["passenger_count"].to_numpy().astype(np.float32)

    # Datetime component extraction
    dt_series = (
        df["pickup_datetime"]
        .str.slice(0, 19)
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
    )
    year = dt_series.dt.year().to_numpy().astype(np.int16)
    month = dt_series.dt.month().to_numpy().astype(np.int8)
    day = dt_series.dt.day().to_numpy().astype(np.int8)
    hour = dt_series.dt.hour().to_numpy().astype(np.int8)
    minute = dt_series.dt.minute().to_numpy().astype(np.int8)

    try:
        weekday = (dt_series.dt.weekday() - 1).to_numpy().astype(np.int8)
        dayofyear = dt_series.dt.ordinal_day().to_numpy().astype(np.int16)
    except Exception:
        weekday = (
            (year + year // 4 - year // 100 + year // 400 + month * 2 + day) % 7
        ).astype(np.int8)
        dayofyear = (
            (month.astype(np.float32) - 1.0) * 30.4375 + day.astype(np.float32)
        ).astype(np.int16)

    # Coordinate Deltas
    lat_diff = d_lat - p_lat
    lon_diff = d_lon - p_lon
    abs_lat_diff = np.abs(lat_diff)
    abs_lon_diff = np.abs(lon_diff)

    # Haversine Distance (in km)
    R = 6371.0
    lat1_r = np.radians(p_lat)
    lat2_r = np.radians(d_lat)
    dlat_r = np.radians(lat_diff)
    dlon_r = np.radians(lon_diff)
    a = np.sin(dlat_r * 0.5) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * (
        np.sin(dlon_r * 0.5) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    haversine_dist = (2.0 * R * np.arcsin(np.sqrt(a))).astype(np.float32)

    # Street Grid L1 & Rotated Manhattan Distances
    mean_lat_r = np.radians((p_lat + d_lat) * 0.5)
    dx_km = (lon_diff * 111.03 * np.cos(mean_lat_r)).astype(np.float32)
    dy_km = (lat_diff * 111.03).astype(np.float32)
    manhattan_dist = np.abs(dx_km) + np.abs(dy_km)
    euclidean_dist = np.sqrt(dx_km**2 + dy_km**2)

    theta = -0.5061455
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    rot_x = dx_km * cos_t - dy_km * sin_t
    rot_y = dx_km * sin_t + dy_km * cos_t
    rot_manhattan_dist = (np.abs(rot_x) + np.abs(rot_y)).astype(np.float32)

    # Street-aligned Rotated Coordinates (Manhattan street grid angle -28.98 deg / -0.5058 rad)
    theta_grid = -0.5058
    cos_grid, sin_grid = float(np.cos(theta_grid)), float(np.sin(theta_grid))
    rot_pickup_x = (p_lon * cos_grid - p_lat * sin_grid).astype(np.float32)
    rot_pickup_y = (p_lon * sin_grid + p_lat * cos_grid).astype(np.float32)
    rot_dropoff_x = (d_lon * cos_grid - d_lat * sin_grid).astype(np.float32)
    rot_dropoff_y = (d_lon * sin_grid + d_lat * cos_grid).astype(np.float32)

    # Bearing Angle
    y_b = np.sin(dlon_r) * np.cos(lat2_r)
    x_b = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(
        dlon_r
    )
    bearing = np.arctan2(y_b, x_b).astype(np.float32)

    # Major NYC Hubs and Airport Distances
    def hub_distances(hub_lat, hub_lon):
        h_lat_r = math.radians(hub_lat)
        dp_lat = np.radians(hub_lat - p_lat)
        dp_lon = np.radians(hub_lon - p_lon)
        ap = np.sin(dp_lat * 0.5) ** 2 + np.cos(lat1_r) * math.cos(h_lat_r) * (
            np.sin(dp_lon * 0.5) ** 2
        )
        dist_p = (2.0 * R * np.arcsin(np.sqrt(np.clip(ap, 0.0, 1.0)))).astype(
            np.float32
        )

        dd_lat = np.radians(hub_lat - d_lat)
        dd_lon = np.radians(hub_lon - d_lon)
        ad = np.sin(dd_lat * 0.5) ** 2 + np.cos(lat2_r) * math.cos(h_lat_r) * (
            np.sin(dd_lon * 0.5) ** 2
        )
        dist_d = (2.0 * R * np.arcsin(np.sqrt(np.clip(ad, 0.0, 1.0)))).astype(
            np.float32
        )
        return dist_p, dist_d, np.minimum(dist_p, dist_d)

    p_jfk, d_jfk, min_jfk = hub_distances(40.6413, -73.7781)
    p_lga, d_lga, min_lga = hub_distances(40.7769, -73.8740)
    p_ewr, d_ewr, min_ewr = hub_distances(40.6895, -74.1745)
    p_mid, d_mid, min_mid = hub_distances(40.7580, -73.9855)
    p_low, d_low, min_low = hub_distances(40.7075, -74.0090)

    # Transit Hub Distances
    p_gct, d_gct, min_gct = hub_distances(40.7527, -73.9772)
    p_penn, d_penn, min_penn = hub_distances(40.7505, -73.9935)
    p_fidi, d_fidi, min_fidi = hub_distances(40.7075, -74.0090)

    is_jfk = (min_jfk < 2.5).astype(np.float32)
    is_lga = (min_lga < 2.0).astype(np.float32)
    is_ewr = (min_ewr < 2.5).astype(np.float32)

    center_lat = ((p_lat + d_lat) * 0.5).astype(np.float32)
    center_lon = ((p_lon + d_lon) * 0.5).astype(np.float32)
    dc_lat = np.radians(40.7580 - center_lat)
    dc_lon = np.radians(-73.9855 - center_lon)
    ac = np.sin(dc_lat * 0.5) ** 2 + np.cos(np.radians(center_lat)) * math.cos(
        math.radians(40.7580)
    ) * (np.sin(dc_lon * 0.5) ** 2)
    center_midtown_dist = (2.0 * R * np.arcsin(np.sqrt(np.clip(ac, 0.0, 1.0)))).astype(
        np.float32
    )

    # Proximities to Major Toll Crossings
    p_qmt, d_qmt, min_qmt = hub_distances(40.7445, -73.9650)
    p_lincoln, d_lincoln, min_lincoln = hub_distances(40.7600, -74.0050)
    p_holland, d_holland, min_holland = hub_distances(40.7275, -74.0150)
    min_hudson_tunnel = np.minimum(min_lincoln, min_holland)
    p_rfk, d_rfk, min_rfk = hub_distances(40.7800, -73.9250)
    p_verrazzano, d_verrazzano, min_verrazzano = hub_distances(40.6066, -74.0447)

    # Route Circuity and Aspect Ratios
    circuity_ratio = (manhattan_dist / np.maximum(haversine_dist, 0.001)).astype(np.float32)
    rot_circuity_ratio = (rot_manhattan_dist / np.maximum(haversine_dist, 0.001)).astype(np.float32)
    coord_aspect_ratio = (abs_lat_diff / np.maximum(abs_lon_diff, 0.0001)).astype(np.float32)

    # River Crossing Indicators
    # East River separates Manhattan (west) from Queens & Brooklyn (east)
    p_is_west_er = p_lon < (p_lat - 114.70)
    d_is_west_er = d_lon < (d_lat - 114.70)
    cross_east_river = (
        (p_is_west_er != d_is_west_er)
        & (p_lat >= 40.57)
        & (p_lat <= 40.88)
        & (d_lat >= 40.57)
        & (d_lat <= 40.88)
    ).astype(np.float32)

    # Hudson River separates Manhattan/NYC (east) from New Jersey (west)
    cross_hudson_river = (
        ((p_lon < -74.03) & (d_lon > -74.01))
        | ((d_lon < -74.03) & (p_lon > -74.01))
    ).astype(np.float32)

    haversine_per_pass = (haversine_dist / np.maximum(pass_cnt, 1.0)).astype(np.float32)
    hour_float = hour.astype(np.float32) + minute.astype(np.float32) / 60.0
    is_weekend = (weekday >= 5).astype(np.float32)
    is_rush_hour = ((weekday < 5) & (hour >= 16) & (hour < 20)).astype(np.float32)
    is_overnight = ((hour >= 20) | (hour < 6)).astype(np.float32)

    post_2012_rate_hike = (
        (year > 2012) | ((year == 2012) & ((month > 9) | ((month == 9) & (day >= 4))))
    ).astype(np.float32)
    time_elapsed_years = (
        (year.astype(np.float32) - 2009.0)
        + (month.astype(np.float32) - 1.0) / 12.0
        + (day.astype(np.float32) - 1.0) / 365.25
    ).astype(np.float32)

    # Spatio-Temporal Manhattan Congestion Interactions
    p_is_man = (p_lat >= 40.70) & (p_lat <= 40.88) & (p_lon >= -74.025) & (p_lon <= -73.93)
    d_is_man = (d_lat >= 40.70) & (d_lat <= 40.88) & (d_lon >= -74.025) & (d_lon <= -73.93)
    both_in_manhattan = (p_is_man & d_is_man).astype(np.float32)
    either_in_manhattan = (p_is_man | d_is_man).astype(np.float32)
    manhattan_rush_hour = (both_in_manhattan * is_rush_hour).astype(np.float32)
    manhattan_overnight = (both_in_manhattan * is_overnight).astype(np.float32)
    manhattan_weekday = (both_in_manhattan * (1.0 - is_weekend)).astype(np.float32)
    congestion_slow_penalty = (both_in_manhattan * is_rush_hour * manhattan_dist * 0.5).astype(np.float32)

    # TLC Statutory Surcharges & Airport Tolls
    ewr_surcharge = np.where(is_ewr > 0.5, 17.50, 0.0).astype(np.float32)
    airport_toll = np.where(is_ewr > 0.5, 13.0, np.where((is_jfk > 0.5) | (is_lga > 0.5), 5.5, 0.0)).astype(np.float32)

    # TLC Statutory Meter Fare Baseline
    dist_miles = (haversine_dist * 0.62137119).astype(np.float32)
    mileage_rate = np.where(post_2012_rate_hike > 0.5, 2.50, 2.00).astype(np.float32)
    post_mta = (
        (year > 2009) | ((year == 2009) & (month >= 11))
    ).astype(np.float32)
    mta_surcharge = 0.50 * post_mta
    overnight_surcharge = 0.50 * is_overnight
    peak_surcharge = 1.00 * is_rush_hour

    is_jfk_flat = (
        (p_is_man & (d_jfk < 2.5)) | (d_is_man & (p_jfk < 2.5))
    )
    jfk_flat_rate = np.where(post_2012_rate_hike > 0.5, 52.0, 45.0).astype(np.float32)
    metered_fare = 2.50 + mileage_rate * dist_miles + mta_surcharge + overnight_surcharge + peak_surcharge + ewr_surcharge + airport_toll
    statutory_base_fare = np.where(
        is_jfk_flat,
        jfk_flat_rate + mta_surcharge,
        metered_fare,
    ).astype(np.float32)
    statutory_base_fare = np.maximum(statutory_base_fare, 2.50).astype(np.float32)

    hour_sin = np.sin(2.0 * np.pi * hour_float / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour_float / 24.0).astype(np.float32)
    dow_sin = np.sin(2.0 * np.pi * weekday.astype(np.float32) / 7.0).astype(np.float32)
    dow_cos = np.cos(2.0 * np.pi * weekday.astype(np.float32) / 7.0).astype(np.float32)
    doy_sin = np.sin(2.0 * np.pi * dayofyear.astype(np.float32) / 365.25).astype(
        np.float32
    )
    doy_cos = np.cos(2.0 * np.pi * dayofyear.astype(np.float32) / 365.25).astype(
        np.float32
    )

    feat_dict = {
        "pickup_longitude": p_lon,
        "pickup_latitude": p_lat,
        "dropoff_longitude": d_lon,
        "dropoff_latitude": d_lat,
        "passenger_count": pass_cnt,
        "abs_lat_diff": abs_lat_diff,
        "abs_lon_diff": abs_lon_diff,
        "lat_diff": lat_diff,
        "lon_diff": lon_diff,
        "rot_pickup_x": rot_pickup_x,
        "rot_pickup_y": rot_pickup_y,
        "rot_dropoff_x": rot_dropoff_x,
        "rot_dropoff_y": rot_dropoff_y,
        "haversine_dist": haversine_dist,
        "manhattan_dist": manhattan_dist,
        "euclidean_dist": euclidean_dist,
        "rot_manhattan_dist": rot_manhattan_dist,
        "bearing": bearing,
        "pickup_jfk_dist": p_jfk,
        "dropoff_jfk_dist": d_jfk,
        "min_jfk_dist": min_jfk,
        "is_jfk_trip": is_jfk,
        "pickup_lga_dist": p_lga,
        "dropoff_lga_dist": d_lga,
        "min_lga_dist": min_lga,
        "is_lga_trip": is_lga,
        "pickup_ewr_dist": p_ewr,
        "dropoff_ewr_dist": d_ewr,
        "min_ewr_dist": min_ewr,
        "is_ewr_trip": is_ewr,
        "pickup_midtown_dist": p_mid,
        "dropoff_midtown_dist": d_mid,
        "min_midtown_dist": min_mid,
        "pickup_lower_man_dist": p_low,
        "dropoff_lower_man_dist": d_low,
        "min_lower_man_dist": min_low,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "center_midtown_dist": center_midtown_dist,
        "haversine_per_passenger": haversine_per_pass,
        "year": year,
        "month": month,
        "day": day,
        "dayofweek": weekday,
        "hour": hour,
        "minute": minute,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_overnight": is_overnight,
        "statutory_base_fare": statutory_base_fare,
        "circuity_ratio": circuity_ratio,
        "rot_circuity_ratio": rot_circuity_ratio,
        "coord_aspect_ratio": coord_aspect_ratio,
        "both_in_manhattan": both_in_manhattan,
        "either_in_manhattan": either_in_manhattan,
        "manhattan_rush_hour": manhattan_rush_hour,
        "manhattan_overnight": manhattan_overnight,
        "manhattan_weekday": manhattan_weekday,
        "congestion_slow_penalty": congestion_slow_penalty,
        "ewr_surcharge": ewr_surcharge,
        "airport_toll": airport_toll,
        "pickup_gct_dist": p_gct,
        "dropoff_gct_dist": d_gct,
        "min_gct_dist": min_gct,
        "pickup_penn_dist": p_penn,
        "dropoff_penn_dist": d_penn,
        "min_penn_dist": min_penn,
        "pickup_fidi_dist": p_fidi,
        "dropoff_fidi_dist": d_fidi,
        "min_fidi_dist": min_fidi,
        "cross_east_river": cross_east_river,
        "cross_hudson_river": cross_hudson_river,
        "pickup_qmt_dist": p_qmt,
        "dropoff_qmt_dist": d_qmt,
        "min_qmt_dist": min_qmt,
        "pickup_lincoln_dist": p_lincoln,
        "dropoff_lincoln_dist": d_lincoln,
        "min_lincoln_dist": min_lincoln,
        "pickup_holland_dist": p_holland,
        "dropoff_holland_dist": d_holland,
        "min_holland_dist": min_holland,
        "min_hudson_tunnel_dist": min_hudson_tunnel,
        "pickup_rfk_dist": p_rfk,
        "dropoff_rfk_dist": d_rfk,
        "min_rfk_dist": min_rfk,
        "pickup_verrazzano_dist": p_verrazzano,
        "dropoff_verrazzano_dist": d_verrazzano,
        "min_verrazzano_dist": min_verrazzano,
        "post_2012_rate_hike": post_2012_rate_hike,
        "time_elapsed_years": time_elapsed_years,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dayofweek_sin": dow_sin,
        "dayofweek_cos": dow_cos,
        "dayofyear_sin": doy_sin,
        "dayofyear_cos": doy_cos,
    }

    if is_test:
        feat_dict["key"] = df["key"].to_numpy()
    else:
        feat_dict["fare_amount"] = df["fare_amount"].to_numpy().astype(np.float32)

    return feat_dict


# =========================================================================
# 3. Main Pipeline Execution
# =========================================================================
def main():
    working_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    # NYC Geographic and Fare Bounding Constants
    LON_MIN, LON_MAX = -74.50, -72.80
    LAT_MIN, LAT_MAX = 40.40, 41.90
    FARE_MIN, FARE_MAX = 2.50, 400.0
    PASS_MIN, PASS_MAX = 1, 6

    # 1. High-Performance Ingestion and Cleaning of Train Data
    train_dtypes = {
        "fare_amount": pl.Float32,
        "pickup_datetime": pl.String,
        "pickup_longitude": pl.Float32,
        "pickup_latitude": pl.Float32,
        "dropoff_longitude": pl.Float32,
        "dropoff_latitude": pl.Float32,
        "passenger_count": pl.Int16,
    }
    use_cols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    df_train_raw = pl.read_csv(
        train_path,
        columns=use_cols,
        schema_overrides=train_dtypes,
        rechunk=True,
    )

    df_clean = df_train_raw.filter(
        (pl.col("fare_amount") >= FARE_MIN)
        & (pl.col("fare_amount") <= FARE_MAX)
        & (pl.col("pickup_longitude") >= LON_MIN)
        & (pl.col("pickup_longitude") <= LON_MAX)
        & (pl.col("pickup_latitude") >= LAT_MIN)
        & (pl.col("pickup_latitude") <= LAT_MAX)
        & (pl.col("dropoff_longitude") >= LON_MIN)
        & (pl.col("dropoff_longitude") <= LON_MAX)
        & (pl.col("dropoff_latitude") >= LAT_MIN)
        & (pl.col("dropoff_latitude") <= LAT_MAX)
        & (pl.col("passenger_count") >= PASS_MIN)
        & (pl.col("passenger_count") <= PASS_MAX)
        & pl.col("pickup_datetime").is_not_null()
        & (pl.col("pickup_datetime").str.len_bytes() >= 19)
    )
    del df_train_raw
    gc.collect()

    # 2. Leakage-Free Validation Split: Random 100k holdout matching competition split
    n_total = len(df_clean)
    val_size = 100_000
    rng = np.random.RandomState(42)
    val_mask = np.zeros(n_total, dtype=bool)
    val_indices = rng.choice(n_total, size=val_size, replace=False)
    val_mask[val_indices] = True

    df_clean = df_clean.with_columns(pl.Series("is_val", val_mask))
    df_val = df_clean.filter(pl.col("is_val")).drop("is_val")
    df_train = df_clean.filter(~pl.col("is_val")).drop("is_val")
    del df_clean
    gc.collect()

    # Apply physical plausibility filtering exclusively on training partition
    lat1_tr = df_train["pickup_latitude"].to_numpy()
    lon1_tr = df_train["pickup_longitude"].to_numpy()
    lat2_tr = df_train["dropoff_latitude"].to_numpy()
    lon2_tr = df_train["dropoff_longitude"].to_numpy()
    fares_tr = df_train["fare_amount"].to_numpy()

    dlat_rad = np.radians(lat2_tr - lat1_tr)
    dlon_rad = np.radians(lon2_tr - lon1_tr)
    a_tr = np.sin(dlat_rad * 0.5) ** 2 + np.cos(np.radians(lat1_tr)) * np.cos(np.radians(lat2_tr)) * (np.sin(dlon_rad * 0.5) ** 2)
    a_tr = np.clip(a_tr, 0.0, 1.0)
    h_dist_tr = 2.0 * 6371.0 * np.arcsin(np.sqrt(a_tr))

    is_short_expensive = (h_dist_tr < 0.08) & (fares_tr > 10.0)
    is_long_cheap = (h_dist_tr > 10.0) & (fares_tr < 3.50)
    is_extreme_velocity = (fares_tr < (2.50 + h_dist_tr * (30.0 / 130.0))) & (h_dist_tr > 5.0)
    valid_train_mask = ~(is_short_expensive | is_long_cheap | is_extreme_velocity)

    df_train = df_train.filter(pl.Series("valid_train", valid_train_mask))
    del lat1_tr, lon1_tr, lat2_tr, lon2_tr, fares_tr, dlat_rad, dlon_rad, a_tr, h_dist_tr
    del is_short_expensive, is_long_cheap, is_extreme_velocity, valid_train_mask
    gc.collect()

    # 3. Process and Persist Validation Data
    val_features = extract_features(df_val, is_test=False)
    table_val = pa.Table.from_pydict(val_features)
    pq.write_table(
        table_val, os.path.join(working_dir, "val.parquet"), compression="snappy"
    )
    y_val = val_features["fare_amount"]
    del df_val, val_features
    gc.collect()

    # 4. Process and Persist Training Data (20M Partition)
    df_train_20m = df_train.slice(0, 20_000_000)
    del df_train
    gc.collect()

    train_features = extract_features(df_train_20m, is_test=False)
    del df_train_20m
    gc.collect()

    table_train = pa.Table.from_pydict(train_features)
    feature_cols = [
        c for c in table_train.column_names if c not in ("fare_amount", "key")
    ]
    with open(os.path.join(working_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    pq.write_table(
        table_train,
        os.path.join(working_dir, "train_20m.parquet"),
        compression="snappy",
    )
    del train_features, table_train
    gc.collect()

    # 5. Process and Persist Test Data
    df_test = pl.read_csv(test_path)
    test_features = extract_features(df_test, is_test=True)
    table_test = pa.Table.from_pydict(test_features)
    pq.write_table(
        table_test, os.path.join(working_dir, "test.parquet"), compression="snappy"
    )
    del df_test, test_features, table_test
    gc.collect()

    # 6. Model Design Verification Smoke Test (Step 2)
    input_dim = len(feature_cols)
    model_nn = TaxiFareResidualNN(
        input_dim=input_dim, hidden_dim=256, num_blocks=4, dropout_rate=0.15
    )
    criterion = RMSELoss()
    optimizer, scheduler = build_optimizer_and_scheduler(
        model_nn, lr=1e-3, weight_decay=1e-4
    )

    batch_size = 32
    dummy_input = torch.randn(batch_size, input_dim)
    dummy_target = torch.full((batch_size,), 12.50)
    model_nn.train()
    optimizer.zero_grad()
    predictions = model_nn(dummy_input)
    loss = criterion(predictions, dummy_target)
    loss.backward()
    optimizer.step()
    scheduler.step()

    assert predictions.shape == (batch_size,)
    assert torch.all(predictions >= 2.50)
    assert not torch.isnan(loss)
    del model_nn, criterion, optimizer, scheduler, dummy_input, dummy_target
    gc.collect()

    # 7. Model Training & Evaluation (Step 3: training_evaluation)
    df_val_load = pl.read_parquet(os.path.join(working_dir, "val.parquet"))
    X_val = df_val_load.select(feature_cols).to_numpy()
    y_val_actual = df_val_load["fare_amount"].to_numpy().astype(np.float32)
    del df_val_load
    gc.collect()

    df_train_load = pl.read_parquet(os.path.join(working_dir, "train_20m.parquet"))
    X_train = df_train_load.select(feature_cols).to_numpy()
    y_train_actual = df_train_load["fare_amount"].to_numpy().astype(np.float32)
    del df_train_load
    gc.collect()

    lgbm_config = get_lgbm_regressor_config()
    model = lgb.LGBMRegressor(**lgbm_config)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    model.fit(
        X_train,
        y_train_actual,
        eval_set=[(X_val, y_val_actual)],
        eval_names=["val"],
        callbacks=callbacks,
    )

    del X_train, y_train_actual
    gc.collect()

    model_save_path = os.path.join(working_dir, "lgb_fare_regressor.txt")
    model.booster_.save_model(model_save_path)

    raw_val_preds = model.predict(X_val)
    val_preds = np.clip(raw_val_preds, 2.50, None)
    val_rmse = float(np.sqrt(np.mean((val_preds - y_val_actual) ** 2)))

    del X_val, y_val_actual, raw_val_preds
    gc.collect()

    # 8. Test Data Ingestion & Full Sample Inference
    df_test_load = pl.read_parquet(os.path.join(working_dir, "test.parquet"))
    test_keys = df_test_load["key"].to_numpy()
    X_test = df_test_load.select(feature_cols).to_numpy()
    del df_test_load
    gc.collect()

    raw_test_preds = model.predict(X_test)
    test_preds = np.clip(raw_test_preds, 2.50, None)

    # 9. Generate and Validate Submission File
    submission_path = os.path.join(submission_dir, "submission.csv")
    df_sub = pd.DataFrame({"key": test_keys, "fare_amount": test_preds})

    assert len(df_sub) == 9914, f"Expected 9,914 test rows, but found {len(df_sub)}"
    assert not df_sub["fare_amount"].isna().any(), "Submission contains NaN predictions"
    assert not np.isinf(
        df_sub["fare_amount"]
    ).any(), "Submission contains Infinite predictions"
    assert list(df_sub.columns) == [
        "key",
        "fare_amount",
    ], f"Incorrect submission columns: {list(df_sub.columns)}"

    df_sub.to_csv(submission_path, index=False)

    # Print official validation metric
    print(f"Final Validation Score: {val_rmse:.5f}")


if __name__ == "__main__":
    main()
