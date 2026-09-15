import gc
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
import torch
import xgboost as xgb


# =============================================================================
# Model Architectures and Builder Functions
# =============================================================================
def build_catboost_regressor(
    params: Optional[Dict[str, Any]] = None,
) -> CatBoostRegressor:
    cb_task = "GPU" if torch.cuda.is_available() else "CPU"
    default_params = {
        "iterations": 2500,
        "learning_rate": 0.08,
        "depth": 9,
        "l2_leaf_reg": 5.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": cb_task,
        "early_stopping_rounds": 40,
        "random_seed": 42,
        "verbose": 250,
    }
    if params:
        default_params.update(params)
    return CatBoostRegressor(**default_params)


def build_lgbm_regressor(params: Optional[Dict[str, Any]] = None) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.08,
        "num_leaves": 511,
        "max_depth": 11,
        "min_child_samples": 100,
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
        .head(25_100_000)
        .collect()
    )
    print(f"Collected {len(train_raw):,} cleaned trip records.")

    val_size = 100_000
    train_size = len(train_raw) - val_size

    train_raw = train_raw.sample(fraction=1.0, shuffle=True, seed=42)
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

        theta = np.radians(-29.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_dx = lon_diff * cos_t - lat_diff * sin_t
        rot_dy = lon_diff * sin_t + lat_diff * cos_t
        rot_manhattan_km = np.abs(rot_dy) * KM_PER_LAT + np.abs(rot_dx) * KM_PER_LON
        rot_avenue_km = (np.abs(rot_dy) * KM_PER_LAT).astype(np.float32)
        rot_crosstown_km = (np.abs(rot_dx) * KM_PER_LON).astype(np.float32)

        # Directly exposed 29-degree rotated coordinates
        rot_p_lon = (p_lon * cos_t - p_lat * sin_t).astype(np.float32)
        rot_p_lat = (p_lon * sin_t + p_lat * cos_t).astype(np.float32)
        rot_d_lon = (d_lon * cos_t - d_lat * sin_t).astype(np.float32)
        rot_d_lat = (d_lon * sin_t + d_lat * cos_t).astype(np.float32)

        # Multi-frequency spatial Fourier projections on pickup & dropoff coordinates
        fourier_features = {}
        for freq in [10.0, 50.0, 100.0]:
            f_int = int(freq)
            fourier_features[f"sin_p_lat_{f_int}"] = np.sin(2.0 * np.pi * freq * p_lat).astype(np.float32)
            fourier_features[f"cos_p_lat_{f_int}"] = np.cos(2.0 * np.pi * freq * p_lat).astype(np.float32)
            fourier_features[f"sin_p_lon_{f_int}"] = np.sin(2.0 * np.pi * freq * p_lon).astype(np.float32)
            fourier_features[f"cos_p_lon_{f_int}"] = np.cos(2.0 * np.pi * freq * p_lon).astype(np.float32)
            fourier_features[f"sin_d_lat_{f_int}"] = np.sin(2.0 * np.pi * freq * d_lat).astype(np.float32)
            fourier_features[f"cos_d_lat_{f_int}"] = np.cos(2.0 * np.pi * freq * d_lat).astype(np.float32)
            fourier_features[f"sin_d_lon_{f_int}"] = np.sin(2.0 * np.pi * freq * d_lon).astype(np.float32)
            fourier_features[f"cos_d_lon_{f_int}"] = np.cos(2.0 * np.pi * freq * d_lon).astype(np.float32)

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

        # Analytical NYC TLC Statutory Base Fare calculation
        rot_manhattan_miles = rot_manhattan_km / 1.609344
        mileage_rate = np.where(is_post_sep_2012 == 1, 2.50, 2.00)
        mta_tax = np.where(is_pre_nov_2009_mta == 1, 0.0, 0.50)
        overnight_surcharge = np.where(is_overnight == 1, 0.50, 0.0)
        rush_hour_surcharge = np.where(is_rush_hour == 1, 1.00, 0.0)
        statutory_base_fare = (
            2.50
            + mileage_rate * rot_manhattan_miles
            + mta_tax
            + overnight_surcharge
            + rush_hour_surcharge
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
            "statutory_base_fare": statutory_base_fare,
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
        feature_dict.update(fourier_features)
        feature_dict.update(toll_features)

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

    # Arrays for training and evaluation
    X_train = train_features.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_features["fare_amount"].to_numpy().astype(np.float32)
    del train_features
    gc.collect()

    X_val = val_features.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_features["fare_amount"].to_numpy().astype(np.float32)
    val_features.write_parquet("./working/val_features.parquet", compression="snappy")
    del val_features
    gc.collect()

    X_test = test_features.select(feature_cols).to_numpy().astype(np.float32)
    test_features.write_parquet("./working/test_features.parquet", compression="snappy")
    del test_features
    gc.collect()

    train_mean = float(np.mean(y_train))
    baseline_val_rmse = float(np.sqrt(np.mean((y_val - train_mean) ** 2)))
    print(
        f"Data prepared: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )
    print(
        f"Baseline Validation RMSE (Mean Predictor ${train_mean:.2f}): {baseline_val_rmse:.4f}"
    )

    # -------------------------------------------------------------------------
    # Model 1: LightGBM Regressor
    # -------------------------------------------------------------------------
    print("Fitting Model 1: LightGBM Regressor on 25M rows...")
    lgb_model = build_lgbm_regressor(
        {
            "n_estimators": 2000,
            "learning_rate": 0.08,
            "num_leaves": 511,
            "max_depth": 11,
            "min_child_samples": 100,
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
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    lgb_val_preds = lgb_model.predict(X_val)
    lgb_val_rmse = float(np.sqrt(np.mean((lgb_val_preds - y_val) ** 2)))
    lgb_test_preds = lgb_model.predict(X_test)
    print(f"LightGBM Holdout Validation RMSE: {lgb_val_rmse:.4f}")
    lgb_model.booster_.save_model("./working/best_lgb_model.txt")
    del lgb_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 2: GPU-Accelerated XGBoost Regressor
    # -------------------------------------------------------------------------
    print("Fitting Model 2: GPU-Accelerated XGBoost Regressor on 25M rows...")
    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    xgb_model = build_xgboost_regressor(
        {
            "tree_method": "hist",
            "device": xgb_device,
            "n_estimators": 2000,
            "learning_rate": 0.08,
            "max_depth": 11,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "reg_lambda": 5.0,
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
    del xgb_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Model 3: GPU-Accelerated CatBoost Regressor
    # -------------------------------------------------------------------------
    print("Fitting Model 3: GPU-Accelerated CatBoost Regressor on 25M rows...")
    cb_task_type = "GPU" if torch.cuda.is_available() else "CPU"
    cb_model = build_catboost_regressor(
        {
            "iterations": 2500,
            "learning_rate": 0.08,
            "depth": 9,
            "l2_leaf_reg": 5.0,
            "task_type": cb_task_type,
            "early_stopping_rounds": 40,
            "random_seed": 42,
            "verbose": 250,
        }
    )
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=250,
    )
    cb_val_preds = cb_model.predict(X_val)
    cb_val_rmse = float(np.sqrt(np.mean((cb_val_preds - y_val) ** 2)))
    cb_test_preds = cb_model.predict(X_test)
    print(f"CatBoost Holdout Validation RMSE: {cb_val_rmse:.4f}")
    cb_model.save_model("./working/best_cb_model.cbm")
    del cb_model
    del X_train, y_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Hybrid Ensembling & Floor Post-Processing
    # -------------------------------------------------------------------------
    print("Optimizing hybrid ensemble weights on holdout validation split...")

    def loss_func(w):
        weights = np.maximum(w, 0.0)
        s = np.sum(weights)
        if s > 0:
            weights = weights / s
        else:
            weights = np.ones(3) / 3.0
        blend = (
            weights[0] * lgb_val_preds
            + weights[1] * xgb_val_preds
            + weights[2] * cb_val_preds
        )
        blend = np.clip(blend, 2.50, None)
        return float(np.sqrt(np.mean((blend - y_val) ** 2)))

    # Bounded SLSQP optimization constrained to the 3D unit simplex
    best_loss = float("inf")
    opt_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_points = [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2],
        [0.6, 0.2, 0.2],
        [0.34, 0.33, 0.33],
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

    # Fine-grained grid search verification across the 3D unit simplex
    step = 0.02
    w0_vals = np.arange(0.0, 1.0 + 1e-5, step)
    for w0 in w0_vals:
        w1_max = 1.0 - w0
        w1_vals = np.arange(0.0, w1_max + 1e-5, step)
        for w1 in w1_vals:
            w2 = max(0.0, 1.0 - w0 - w1)
            candidate_w = np.array([w0, w1, w2])
            val_loss = loss_func(candidate_w)
            if val_loss < best_loss:
                best_loss = val_loss
                opt_w = candidate_w

    opt_w = np.maximum(opt_w, 0.0)
    opt_w = opt_w / np.sum(opt_w)
    print(
        f"Optimal Simplex Weights: LightGBM={opt_w[0]:.4f} | XGBoost={opt_w[1]:.4f} | CatBoost={opt_w[2]:.4f}"
    )

    final_val_preds = (
        opt_w[0] * lgb_val_preds + opt_w[1] * xgb_val_preds + opt_w[2] * cb_val_preds
    )
    final_val_preds = np.clip(final_val_preds, 2.50, None)
    final_val_rmse = float(np.sqrt(np.mean((final_val_preds - y_val) ** 2)))
    print(f"Final Ensembled Holdout Validation RMSE: {final_val_rmse:.6f}")

    final_test_preds = (
        opt_w[0] * lgb_test_preds + opt_w[1] * xgb_test_preds + opt_w[2] * cb_test_preds
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
