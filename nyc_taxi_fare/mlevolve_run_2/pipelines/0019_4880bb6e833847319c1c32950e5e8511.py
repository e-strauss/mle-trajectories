import os
import gc
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from scipy.optimize import nnls

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:

    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# -------------------------------------------------------------------------
# Reproducibility Setup
# -------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)


# -------------------------------------------------------------------------
# Step 1: Spatial, Temporal & Regulatory Feature Engineering
# -------------------------------------------------------------------------
def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized computation of geodesic distances, rotated Manhattan grid projections,
    tortuosity, compass bearings, and multi-hub landmark proximities.
    """
    p_lat = df["pickup_latitude"].to_numpy(dtype=np.float64)
    p_lon = df["pickup_longitude"].to_numpy(dtype=np.float64)
    d_lat = df["dropoff_latitude"].to_numpy(dtype=np.float64)
    d_lon = df["dropoff_longitude"].to_numpy(dtype=np.float64)

    # Coordinate differences
    dlat = d_lat - p_lat
    dlon = d_lon - p_lon
    df["dlat"] = dlat.astype(np.float32)
    df["dlon"] = dlon.astype(np.float32)
    df["abs_dlat"] = np.abs(dlat).astype(np.float32)
    df["abs_dlon"] = np.abs(dlon).astype(np.float32)

    # Haversine distance in kilometers
    r = 6371.0088  # Mean Earth radius in km
    phi1 = np.radians(p_lat)
    phi2 = np.radians(d_lat)
    dphi = np.radians(dlat)
    dlambda = np.radians(dlon)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    haversine_km = 2.0 * r * np.arcsin(np.sqrt(a))
    df["haversine_km"] = haversine_km.astype(np.float32)

    # Rotated Manhattan Street Grid (angled ~28.9 degrees clockwise from North)
    theta = 0.5044
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Metric conversion: ~111.03 km per degree lat, ~84.35 km per degree lon at 40.75 deg N
    dx_km = dlon * 84.35
    dy_km = dlat * 111.03

    rot_x = dx_km * cos_theta - dy_km * sin_theta
    rot_y = dx_km * sin_theta + dy_km * cos_theta
    manhattan_rot_km = np.abs(rot_x) + np.abs(rot_y)
    df["manhattan_rot_km"] = manhattan_rot_km.astype(np.float32)
    df["euclidean_km"] = np.sqrt(dx_km**2 + dy_km**2).astype(np.float32)

    # Rotated Manhattan coordinate projections along the 28.9-degree street grid
    p_x = (p_lon - (-73.98)) * 84.35
    p_y = (p_lat - 40.75) * 111.03
    df["p_rot_x"] = (p_x * cos_theta - p_y * sin_theta).astype(np.float32)
    df["p_rot_y"] = (p_x * sin_theta + p_y * cos_theta).astype(np.float32)

    d_x = (d_lon - (-73.98)) * 84.35
    d_y = (d_lat - 40.75) * 111.03
    df["d_rot_x"] = (d_x * cos_theta - d_y * sin_theta).astype(np.float32)
    df["d_rot_y"] = (d_x * sin_theta + d_y * cos_theta).astype(np.float32)

    df["rot_dlon"] = (df["d_rot_x"] - df["p_rot_x"]).astype(np.float32)
    df["rot_dlat"] = (df["d_rot_y"] - df["p_rot_y"]).astype(np.float32)
    df["abs_rot_dlon"] = np.abs(df["rot_dlon"]).astype(np.float32)
    df["abs_rot_dlat"] = np.abs(df["rot_dlat"]).astype(np.float32)

    # Tortuosity ratio: Manhattan detour ratio relative to straight-line distance
    df["tortuosity"] = (df["manhattan_rot_km"] / (df["haversine_km"] + 0.01)).astype(
        np.float32
    )

    # Compass Bearing
    y_bear = np.sin(dlambda) * np.cos(phi2)
    x_bear = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = np.arctan2(y_bear, x_bear)
    df["sin_bearing"] = np.sin(bearing).astype(np.float32)
    df["cos_bearing"] = np.cos(bearing).astype(np.float32)

    # Multi-Hub Proximities (Airports & Transit Centers)
    landmarks = {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
        "midtown": (40.7580, -73.9855),
        "downtown": (40.7075, -74.0090),
    }

    def dist_to_point(lat_arr, lon_arr, point_lat, point_lon):
        p_phi = np.radians(lat_arr)
        lm_phi = math.radians(point_lat)
        dp = np.radians(lat_arr - point_lat)
        dl = np.radians(lon_arr - point_lon)
        val = (
            np.sin(dp / 2.0) ** 2
            + np.cos(p_phi) * math.cos(lm_phi) * np.sin(dl / 2.0) ** 2
        )
        val = np.clip(val, 0.0, 1.0)
        return 2.0 * r * np.arcsin(np.sqrt(val))

    for name, (l_lat, l_lon) in landmarks.items():
        p_d = dist_to_point(p_lat, p_lon, l_lat, l_lon)
        d_d = dist_to_point(d_lat, d_lon, l_lat, l_lon)
        df[f"p_dist_{name}"] = p_d.astype(np.float32)
        df[f"d_dist_{name}"] = d_d.astype(np.float32)

    # Minimum airport proximity across major regional airports
    df["min_airport_p_dist"] = np.minimum(
        df["p_dist_jfk"], np.minimum(df["p_dist_lga"], df["p_dist_ewr"])
    ).astype(np.float32)
    df["min_airport_d_dist"] = np.minimum(
        df["d_dist_jfk"], np.minimum(df["d_dist_lga"], df["d_dist_ewr"])
    ).astype(np.float32)

    # Toll crossing gateway proximities (Lincoln, Holland, Queens-Midtown, RFK, Verrazzano)
    toll_gateways = {
        "lincoln": (40.7610, -74.0020),
        "holland": (40.7260, -74.0100),
        "qmt": (40.7450, -73.9700),
        "rfk": (40.7800, -73.9250),
        "verrazzano": (40.6060, -74.0450),
    }

    for name, (g_lat, g_lon) in toll_gateways.items():
        p_toll_d = dist_to_point(p_lat, p_lon, g_lat, g_lon)
        d_toll_d = dist_to_point(d_lat, d_lon, g_lat, g_lon)
        df[f"p_dist_{name}"] = p_toll_d.astype(np.float32)
        df[f"d_dist_{name}"] = d_toll_d.astype(np.float32)

    df["min_toll_p_dist"] = np.minimum.reduce(
        [df[f"p_dist_{name}"] for name in toll_gateways]
    ).astype(np.float32)
    df["min_toll_d_dist"] = np.minimum.reduce(
        [df[f"d_dist_{name}"] for name in toll_gateways]
    ).astype(np.float32)

    # JFK Flat-Fare Corridor Indicator (JFK <-> Manhattan)
    is_p_jfk = df["p_dist_jfk"] < 3.0
    is_d_jfk = df["d_dist_jfk"] < 3.0
    is_p_manh = (
        (p_lat >= 40.70) & (p_lat <= 40.86) & (p_lon >= -74.02) & (p_lon <= -73.93)
    )
    is_d_manh = (
        (d_lat >= 40.70) & (d_lat <= 40.86) & (d_lon >= -74.02) & (d_lon <= -73.93)
    )
    df["is_jfk_manhattan"] = ((is_p_jfk & is_d_manh) | (is_d_jfk & is_p_manh)).astype(
        np.float32
    )

    # Topological Borough Classification
    def classify_borough(lat_arr, lon_arr):
        east_river_boundary = -73.975 + (lat_arr - 40.71) * 0.50
        is_manh = (
            (lat_arr >= 40.700)
            & (lat_arr <= 40.880)
            & (lon_arr >= -74.025)
            & (lon_arr <= east_river_boundary)
            & (lon_arr <= -73.910)
        )
        is_bronx = (
            (lat_arr >= 40.785)
            & (lat_arr <= 40.920)
            & (lon_arr >= -73.930)
            & (lon_arr <= -73.760)
            & (~is_manh)
        )
        is_si = (
            (lat_arr >= 40.495)
            & (lat_arr <= 40.655)
            & (lon_arr >= -74.260)
            & (lon_arr <= -74.050)
        )
        is_newark = (
            (lat_arr >= 40.650)
            & (lat_arr <= 40.760)
            & (lon_arr >= -74.260)
            & (lon_arr <= -74.050)
            & (~is_si)
        )
        is_bk_lat = lat_arr <= 40.730
        is_bk_lon = lon_arr <= -73.850
        is_bk = (
            (lat_arr >= 40.570)
            & (lat_arr <= 40.740)
            & (lon_arr >= -74.045)
            & (lon_arr <= -73.830)
            & (~is_manh)
            & (~is_si)
            & (is_bk_lat | is_bk_lon)
        )
        is_qn = (
            (lat_arr >= 40.540)
            & (lat_arr <= 40.800)
            & (lon_arr >= -73.960)
            & (lon_arr <= -73.700)
            & (~is_manh)
            & (~is_bronx)
            & (~is_bk)
            & (~is_si)
        )
        borough = np.zeros(len(lat_arr), dtype=np.float32)
        borough[is_manh] = 1.0
        borough[is_qn] = 2.0
        borough[is_bk] = 3.0
        borough[is_bronx] = 4.0
        borough[is_si] = 5.0
        borough[is_newark] = 6.0
        return borough, is_manh, is_qn, is_bk, is_bronx, is_si, is_newark

    p_borough, p_is_manh, p_is_qn, p_is_bk, p_is_bronx, _, p_is_newark = classify_borough(p_lat, p_lon)
    d_borough, d_is_manh, d_is_qn, d_is_bk, d_is_bronx, _, d_is_newark = classify_borough(d_lat, d_lon)

    df["p_borough"] = p_borough
    df["d_borough"] = d_borough
    df["is_borough_transition"] = (p_borough != d_borough).astype(np.float32)

    # True water-barrier crossing flags
    df["crosses_east_river"] = (
        ((p_is_manh | p_is_bronx) & (d_is_qn | d_is_bk))
        | ((d_is_manh | d_is_bronx) & (p_is_qn | p_is_bk))
    ).astype(np.float32)

    df["crosses_harlem_river"] = (
        (p_is_manh & d_is_bronx) | (d_is_manh & p_is_bronx)
    ).astype(np.float32)

    df["crosses_hudson_river"] = (
        ((p_is_newark | (p_lon < -74.03)) & (d_lon >= -74.02))
        | ((d_is_newark | (d_lon < -74.03)) & (p_lon >= -74.02))
    ).astype(np.float32)

    # Newark Airport Interstate Surcharge Indicator
    df["is_ewr_trip"] = ((df["p_dist_ewr"] < 3.5) | (df["d_dist_ewr"] < 3.5)).astype(
        np.float32
    )

    # High-density midtown zone indicator and continuous congestion distance
    is_midtown_zone = ((df["p_dist_midtown"] < 2.5) | (df["d_dist_midtown"] < 2.5)).astype(
        np.float32
    )
    df["is_midtown_zone"] = is_midtown_zone
    df["haversine_midtown"] = (df["haversine_km"] * is_midtown_zone).astype(np.float32)

    min_midtown_dist = np.minimum(df["p_dist_midtown"], df["d_dist_midtown"])
    df["midtown_congestion_dist"] = (
        df["manhattan_rot_km"] / (1.0 + min_midtown_dist)
    ).astype(np.float32)

    # Official TLC Newark Airport Interstate Surcharge ($17.50)
    df["newark_surcharge"] = (df["is_ewr_trip"] * 17.50).astype(np.float32)

    # Discretize pickup and dropoff coordinates into uniform 0.005-degree bins (OD grid tokens)
    lat_bins = 280  # (41.80 - 40.40) / 0.005
    lon_bins = 320  # (-72.80 - (-74.40)) / 0.005

    p_lat_bin = np.clip(((p_lat - 40.40) / 0.005).astype(np.int32), 0, lat_bins - 1)
    p_lon_bin = np.clip(((p_lon - (-74.40)) / 0.005).astype(np.int32), 0, lon_bins - 1)
    df["p_cell"] = (p_lat_bin * lon_bins + p_lon_bin).astype(np.float32)

    d_lat_bin = np.clip(((d_lat - 40.40) / 0.005).astype(np.int32), 0, lat_bins - 1)
    d_lon_bin = np.clip(((d_lon - (-74.40)) / 0.005).astype(np.int32), 0, lon_bins - 1)
    df["d_cell"] = (d_lat_bin * lon_bins + d_lon_bin).astype(np.float32)

    # Origin-Destination joint interaction token
    od_token = (
        (df["p_cell"].to_numpy(dtype=np.int64) * 95003 + df["d_cell"].to_numpy(dtype=np.int64))
        % 100000
    ).astype(np.float32)
    df["od_token"] = od_token

    # Cyclical coordinate sinusoidal encodings across spatial frequency bands
    for scale, freq in [("high", 20.0), ("med", 5.0), ("low", 1.0)]:
        df[f"sin_plat_{scale}"] = np.sin(p_lat * (freq * np.pi)).astype(np.float32)
        df[f"cos_plat_{scale}"] = np.cos(p_lat * (freq * np.pi)).astype(np.float32)
        df[f"sin_plon_{scale}"] = np.sin(p_lon * (freq * np.pi)).astype(np.float32)
        df[f"cos_plon_{scale}"] = np.cos(p_lon * (freq * np.pi)).astype(np.float32)
        df[f"sin_dlat_{scale}"] = np.sin(d_lat * (freq * np.pi)).astype(np.float32)
        df[f"cos_dlat_{scale}"] = np.cos(d_lat * (freq * np.pi)).astype(np.float32)
        df[f"sin_dlon_{scale}"] = np.sin(d_lon * (freq * np.pi)).astype(np.float32)
        df[f"cos_dlon_{scale}"] = np.cos(d_lon * (freq * np.pi)).astype(np.float32)

    return df


def compute_temporal_and_regulatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized extraction of temporal cycles and official NYC TLC regulatory tariff components.
    """
    dt_series = pd.to_datetime(
        df["pickup_datetime"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
    )

    years = dt_series.dt.year.to_numpy(dtype=np.int16)
    months = dt_series.dt.month.to_numpy(dtype=np.int8)
    days = dt_series.dt.day.to_numpy(dtype=np.int8)
    hours = dt_series.dt.hour.to_numpy(dtype=np.int8)
    dayofweeks = dt_series.dt.dayofweek.to_numpy(dtype=np.int8)

    df["year"] = years
    df["month"] = months
    df["hour"] = hours
    df["dayofweek"] = dayofweeks
    df["is_weekend"] = (dayofweeks >= 5).astype(np.float32)

    # Continuous timeline progress (years elapsed since Jan 1, 2009)
    df["timeline_progress"] = (
        (years - 2009) + (months - 1) / 12.0 + (days - 1) / 365.25 + hours / 8766.0
    ).astype(np.float32)

    # Cyclical encodings
    df["sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
    df["sin_dow"] = np.sin(2.0 * np.pi * dayofweeks / 7.0).astype(np.float32)
    df["cos_dow"] = np.cos(2.0 * np.pi * dayofweeks / 7.0).astype(np.float32)
    df["sin_month"] = np.sin(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2.0 * np.pi * (months - 1) / 12.0).astype(np.float32)

    # TLC Official Rate Hike (September 4, 2012: mileage rate increased from $2.00 to $2.50 / mile)
    is_post_sept_2012 = (years > 2012) | (
        (years == 2012) & ((months > 9) | ((months == 9) & (days >= 4)))
    )
    df["is_post_sept_2012"] = is_post_sept_2012.astype(np.float32)

    # TLC Tariff Multiplicative Interaction Terms
    df["haversine_post_2012"] = (df["haversine_km"] * df["is_post_sept_2012"]).astype(
        np.float32
    )
    df["manhattan_rot_post_2012"] = (
        df["manhattan_rot_km"] * df["is_post_sept_2012"]
    ).astype(np.float32)
    df["jfk_flat_rate"] = (
        df["is_jfk_manhattan"] * np.where(is_post_sept_2012, 52.0, 45.0)
    ).astype(np.float32)

    # TLC Regulatory Surcharges:
    # Rush-hour: $1.00 weekdays 16:00 to 20:00
    is_rush_hour = ((dayofweeks < 5) & (hours >= 16) & (hours < 20)).astype(np.float32)
    df["rush_hour_surcharge"] = is_rush_hour

    # Distance-congestion multiplicative interaction (metered waiting time during peak rush hour)
    df["haversine_rush_hour"] = (df["haversine_km"] * is_rush_hour).astype(np.float32)

    # Official TLC JFK rush-hour surcharge ($4.50 on flat-rate trips during peak weekday hours)
    df["jfk_rush_surcharge"] = (
        df["is_jfk_manhattan"] * is_rush_hour * 4.50
    ).astype(np.float32)

    # Overnight: $0.50 all days 20:00 to 06:00
    is_overnight = (hours >= 20) | (hours < 6)
    df["overnight_surcharge"] = is_overnight.astype(np.float32) * 0.5

    # TLC MTA State Tax ($0.50 enacted Nov 1, 2009)
    is_post_mta = (years > 2009) | ((years == 2009) & (months >= 11))
    df["mta_tax"] = np.where(is_post_mta, 0.50, 0.0).astype(np.float32)

    # Theoretical Base Fare Prior (Rectilinear road distance + Statutory MTA Tax)
    road_miles = df["manhattan_rot_km"] * 0.621371
    meter_rate_per_mile = np.where(is_post_sept_2012, 2.50, 2.00)

    standard_fare = (
        2.50
        + (road_miles * meter_rate_per_mile)
        + df["rush_hour_surcharge"]
        + df["overnight_surcharge"]
        + df["newark_surcharge"]
        + df["mta_tax"]
    )

    jfk_fare = (
        np.where(is_post_sept_2012, 52.0, 45.0)
        + df["jfk_rush_surcharge"]
        + df["mta_tax"]
    )

    df["theoretical_fare"] = np.where(
        df["is_jfk_manhattan"] == 1.0,
        jfk_fare,
        standard_fare
    ).astype(np.float32)

    # Passenger count feature
    df["passenger_count"] = df["passenger_count"].astype(np.float32)
    df["is_solo_passenger"] = (df["passenger_count"] == 1.0).astype(np.float32)

    # Free memory of timestamp string column
    del df["pickup_datetime"]

    return df


def load_and_preprocess_data():
    """
    Ingests high-quality domain-filtered training samples, performs leak-free splitting,
    engineers features across splits, and prepares test features.
    """
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    use_cols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    n_sample_rows = 15_000_000
    dtypes = {
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "float32",
    }
    train_df = pd.read_csv(
        train_path, nrows=n_sample_rows, usecols=use_cols, dtype=dtypes
    )

    # Domain cleaning within geographic and regulatory bounds
    mask = (
        (train_df["fare_amount"] >= 2.50)
        & (train_df["fare_amount"] <= 400.0)
        & (train_df["pickup_longitude"] >= -74.45)
        & (train_df["pickup_longitude"] <= -72.85)
        & (train_df["pickup_latitude"] >= 40.40)
        & (train_df["pickup_latitude"] <= 41.80)
        & (train_df["dropoff_longitude"] >= -74.45)
        & (train_df["dropoff_longitude"] <= -72.85)
        & (train_df["dropoff_latitude"] >= 40.40)
        & (train_df["dropoff_latitude"] <= 41.80)
        & (train_df["passenger_count"] >= 1)
        & (train_df["passenger_count"] <= 6)
    )
    train_df = train_df[mask].reset_index(drop=True)

    # Load Test Data
    test_df = pd.read_csv(test_path)

    # Leak-free Train/Validation Split (95% Train, 5% Holdout)
    n_total = len(train_df)
    n_val = int(n_total * 0.05)
    rng = np.random.RandomState(42)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]

    df_train = train_df.iloc[train_indices].reset_index(drop=True)
    df_val = train_df.iloc[val_indices].reset_index(drop=True)
    del train_df
    gc.collect()

    # Engineer Features independently across splits
    df_train = compute_spatial_features(df_train)
    df_train = compute_temporal_and_regulatory_features(df_train)

    # Prune severe distance-fare sensor recording anomalies in the training partition
    train_clean_mask = ~(
        ((df_train["haversine_km"] < 0.08) & (df_train["fare_amount"] > 12.00))
        | ((df_train["haversine_km"] > 15.0) & (df_train["fare_amount"] < 6.00))
    )
    df_train = df_train[train_clean_mask].reset_index(drop=True)

    df_val = compute_spatial_features(df_val)
    df_val = compute_temporal_and_regulatory_features(df_val)

    test_df = compute_spatial_features(test_df)
    test_df = compute_temporal_and_regulatory_features(test_df)

    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "dlat",
        "dlon",
        "abs_dlat",
        "abs_dlon",
        "haversine_km",
        "manhattan_rot_km",
        "euclidean_km",
        "p_rot_x",
        "p_rot_y",
        "d_rot_x",
        "d_rot_y",
        "rot_dlon",
        "rot_dlat",
        "abs_rot_dlon",
        "abs_rot_dlat",
        "tortuosity",
        "sin_bearing",
        "cos_bearing",
        "p_dist_jfk",
        "d_dist_jfk",
        "p_dist_lga",
        "d_dist_lga",
        "p_dist_ewr",
        "d_dist_ewr",
        "p_dist_midtown",
        "d_dist_midtown",
        "p_dist_downtown",
        "d_dist_downtown",
        "min_airport_p_dist",
        "min_airport_d_dist",
        "p_dist_lincoln",
        "d_dist_lincoln",
        "p_dist_holland",
        "d_dist_holland",
        "p_dist_qmt",
        "d_dist_qmt",
        "p_dist_rfk",
        "d_dist_rfk",
        "p_dist_verrazzano",
        "d_dist_verrazzano",
        "min_toll_p_dist",
        "min_toll_d_dist",
        "p_borough",
        "d_borough",
        "is_borough_transition",
        "crosses_east_river",
        "crosses_harlem_river",
        "crosses_hudson_river",
        "is_jfk_manhattan",
        "is_ewr_trip",
        "is_midtown_zone",
        "haversine_midtown",
        "midtown_congestion_dist",
        "newark_surcharge",
        "year",
        "month",
        "hour",
        "dayofweek",
        "is_weekend",
        "timeline_progress",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "sin_month",
        "cos_month",
        "is_post_sept_2012",
        "haversine_post_2012",
        "manhattan_rot_post_2012",
        "jfk_flat_rate",
        "rush_hour_surcharge",
        "haversine_rush_hour",
        "jfk_rush_surcharge",
        "overnight_surcharge",
        "mta_tax",
        "theoretical_fare",
        "passenger_count",
        "is_solo_passenger",
        "p_cell",
        "d_cell",
        "od_token",
        "sin_plat_high",
        "cos_plat_high",
        "sin_plon_high",
        "cos_plon_high",
        "sin_dlat_high",
        "cos_dlat_high",
        "sin_dlon_high",
        "cos_dlon_high",
        "sin_plat_med",
        "cos_plat_med",
        "sin_plon_med",
        "cos_plon_med",
        "sin_dlat_med",
        "cos_dlat_med",
        "sin_dlon_med",
        "cos_dlon_med",
        "sin_plat_low",
        "cos_plat_low",
        "sin_plon_low",
        "cos_plon_low",
        "sin_dlat_low",
        "cos_dlat_low",
        "sin_dlon_low",
        "cos_dlon_low",
    ]

    X_train = df_train[feature_cols]
    # Centered regulatory meter residual target
    y_train = (df_train["fare_amount"] - df_train["theoretical_fare"]).to_numpy(
        dtype=np.float32
    )

    X_val = df_val[feature_cols]
    y_val = df_val["fare_amount"].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols]
    test_keys = test_df["key"].to_numpy()

    del df_train, df_val
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, test_keys


# -------------------------------------------------------------------------
# Step 2: Deep Spatial Residual Neural Network (Spatial-ResNet)
# -------------------------------------------------------------------------
class ResDenseBlock(nn.Module):
    """
    Residual dense block with LayerNorm, GELU, and Dropout.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return res + out


class SpatialResNet(nn.Module):
    """
    Deep Spatial Residual Neural Network with learnable OD grid embeddings,
    continuous tabular feature projection, and residual dense blocks.
    """

    def __init__(
        self,
        num_grid_cells: int = 95000,
        num_od_tokens: int = 100000,
        grid_embed_dim: int = 32,
        od_embed_dim: int = 32,
        num_cont_features: int = 105,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_embedding = nn.Embedding(num_grid_cells, grid_embed_dim)
        self.od_embedding = nn.Embedding(num_od_tokens, od_embed_dim)

        total_embed_dim = grid_embed_dim * 2 + od_embed_dim
        self.embed_proj = nn.Linear(total_embed_dim, hidden_dim)
        self.cont_proj = nn.Linear(num_cont_features, hidden_dim)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResDenseBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        p_cells: torch.Tensor,
        d_cells: torch.Tensor,
        od_tokens: torch.Tensor,
        cont_feats: torch.Tensor,
    ) -> torch.Tensor:
        p_emb = self.grid_embedding(p_cells)
        d_emb = self.grid_embedding(d_cells)
        od_emb = self.od_embedding(od_tokens)
        spatial_emb = torch.cat([p_emb, d_emb, od_emb], dim=-1)

        h_spatial = self.embed_proj(spatial_emb)
        h_cont = self.cont_proj(cont_feats)

        x = self.input_norm(h_spatial + h_cont)
        for block in self.res_blocks:
            x = block(x)

        return self.out_head(self.out_norm(x)).squeeze(-1)


class SpatialResNetRegressor:
    """
    Scikit-learn compatible estimator wrapper for SpatialResNet optimizing residual RMSE.
    """

    def __init__(
        self,
        batch_size: int = 8192,
        epochs: int = 3,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mean = None
        self.std = None
        self.grid_cols = ["p_cell", "d_cell", "od_token"]
        self.cont_cols = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, eval_set=None):
        self.cont_cols = [c for c in X_train.columns if c not in self.grid_cols]

        # Standardize continuous features
        train_cont_raw = X_train[self.cont_cols].to_numpy(dtype=np.float32)
        self.mean = np.mean(train_cont_raw, axis=0)
        self.std = np.std(train_cont_raw, axis=0) + 1e-5
        train_cont = (train_cont_raw - self.mean) / self.std

        train_p_cell = np.clip(
            X_train["p_cell"].to_numpy(dtype=np.int64), 0, 94999
        )
        train_d_cell = np.clip(
            X_train["d_cell"].to_numpy(dtype=np.int64), 0, 94999
        )
        train_od_token = np.clip(
            X_train["od_token"].to_numpy(dtype=np.int64), 0, 99999
        )

        train_dataset = TensorDataset(
            torch.from_numpy(train_p_cell),
            torch.from_numpy(train_d_cell),
            torch.from_numpy(train_od_token),
            torch.from_numpy(train_cont),
            torch.from_numpy(y_train.astype(np.float32)),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        # Build model
        self.model = SpatialResNet(
            num_grid_cells=95000,
            num_od_tokens=100000,
            grid_embed_dim=32,
            od_embed_dim=32,
            num_cont_features=len(self.cont_cols),
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = len(train_loader) * self.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=1e-5
        )
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # Prepare validation data if provided
        val_loader = None
        if eval_set is not None:
            X_val, y_val_res = eval_set[0]
            val_cont = (
                X_val[self.cont_cols].to_numpy(dtype=np.float32) - self.mean
            ) / self.std
            val_p_cell = np.clip(
                X_val["p_cell"].to_numpy(dtype=np.int64), 0, 94999
            )
            val_d_cell = np.clip(
                X_val["d_cell"].to_numpy(dtype=np.int64), 0, 94999
            )
            val_od_token = np.clip(
                X_val["od_token"].to_numpy(dtype=np.int64), 0, 99999
            )
            val_dataset = TensorDataset(
                torch.from_numpy(val_p_cell),
                torch.from_numpy(val_d_cell),
                torch.from_numpy(val_od_token),
                torch.from_numpy(val_cont),
                torch.from_numpy(y_val_res.astype(np.float32)),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for p_c, d_c, od_t, cont, target in train_loader:
                p_c = p_c.to(self.device, non_blocking=True)
                d_c = d_c.to(self.device, non_blocking=True)
                od_t = od_t.to(self.device, non_blocking=True)
                cont = cont.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    preds = self.model(p_c, d_c, od_t, cont)
                    loss = criterion(preds, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if val_loader is not None:
                self.model.eval()
                val_mse = 0.0
                total_val_samples = 0
                with torch.no_grad():
                    for p_c, d_c, od_t, cont, target in val_loader:
                        p_c = p_c.to(self.device, non_blocking=True)
                        d_c = d_c.to(self.device, non_blocking=True)
                        od_t = od_t.to(self.device, non_blocking=True)
                        cont = cont.to(self.device, non_blocking=True)
                        target = target.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            v_preds = self.model(p_c, d_c, od_t, cont)
                            val_mse += ((v_preds - target) ** 2).sum().item()
                            total_val_samples += len(target)

                val_rmse = math.sqrt(val_mse / max(total_val_samples, 1))
                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        del train_dataset, train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        p_cells = np.clip(X["p_cell"].to_numpy(dtype=np.int64), 0, 94999)
        d_cells = np.clip(X["d_cell"].to_numpy(dtype=np.int64), 0, 94999)
        od_tokens = np.clip(X["od_token"].to_numpy(dtype=np.int64), 0, 99999)
        cont = (X[self.cont_cols].to_numpy(dtype=np.float32) - self.mean) / self.std

        dataset = TensorDataset(
            torch.from_numpy(p_cells),
            torch.from_numpy(d_cells),
            torch.from_numpy(od_tokens),
            torch.from_numpy(cont),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        all_preds = []
        with torch.no_grad():
            for p_c, d_c, od_t, c in loader:
                p_c = p_c.to(self.device, non_blocking=True)
                d_c = d_c.to(self.device, non_blocking=True)
                od_t = od_t.to(self.device, non_blocking=True)
                c = c.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = self.model(p_c, d_c, od_t, c)
                all_preds.append(out.cpu().numpy())

        return np.concatenate(all_preds).astype(np.float32)


# -------------------------------------------------------------------------
# Step 3: Training, Evaluation & Optimal Blending
# -------------------------------------------------------------------------
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, test_keys = load_and_preprocess_data()

    print(
        f"Dataset shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    # Extract deterministic theoretical fare baselines
    theoretical_val = X_val["theoretical_fare"].to_numpy(dtype=np.float32)
    theoretical_test = X_test["theoretical_fare"].to_numpy(dtype=np.float32)

    # Calculate validation meter residual target
    y_val_res = (y_val - theoretical_val).astype(np.float32)

    # Train Model 1: GPU-Accelerated Deep Spatial Residual Neural Network (Spatial-ResNet)
    print("Training GPU-Accelerated Spatial-ResNet on meter residuals...")
    spatial_resnet = SpatialResNetRegressor(
        batch_size=8192,
        epochs=3,
        lr=2e-3,
        weight_decay=1e-4,
        hidden_dim=256,
        num_blocks=3,
        dropout=0.1,
    )
    spatial_resnet.fit(X_train, y_train, eval_set=[(X_val, y_val_res)])

    val_res_nn = spatial_resnet.predict(X_val)
    test_res_nn = spatial_resnet.predict(X_test)

    val_preds_nn = np.clip(theoretical_val + val_res_nn, 2.50, None)
    test_preds_nn = np.clip(theoretical_test + test_res_nn, 2.50, None)
    val_rmse_nn = root_mean_squared_error(y_val, val_preds_nn)
    print(f"Spatial-ResNet Val RMSE: {val_rmse_nn:.4f}")

    # Train Model 2: GPU-Histogram XGBoost Regressor on meter residuals
    print("Training GPU-Histogram XGBoost Regressor on meter residuals...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.06,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=30,
        reg_lambda=2.0,
        reg_alpha=0.5,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val_res)],
        verbose=False,
    )

    val_res_xgb = xgb_model.predict(X_val)
    test_res_xgb = xgb_model.predict(X_test)

    val_preds_xgb = np.clip(theoretical_val + val_res_xgb, 2.50, None)
    test_preds_xgb = np.clip(theoretical_test + test_res_xgb, 2.50, None)
    val_rmse_xgb = root_mean_squared_error(y_val, val_preds_xgb)
    print(f"GPU GBDT Val RMSE: {val_rmse_xgb:.4f}")

    # Solve optimal non-negative least-squares blend weights on reconstructed validation fares
    print("Solving optimal non-negative least-squares blend weights...")
    val_matrix = np.column_stack([val_preds_nn, val_preds_xgb])
    weights, _ = nnls(val_matrix, y_val)
    weight_sum = np.sum(weights)
    if weight_sum > 1e-8:
        weights = weights / weight_sum
    else:
        weights = np.array([0.5, 0.5], dtype=np.float64)

    val_preds = weights[0] * val_preds_nn + weights[1] * val_preds_xgb
    test_preds = weights[0] * test_preds_nn + weights[1] * test_preds_xgb

    # Enforce statutory minimum TLC fare ($2.50)
    val_preds = np.clip(val_preds, 2.50, None)
    test_preds = np.clip(test_preds, 2.50, None)

    val_rmse = root_mean_squared_error(y_val, val_preds)
    print(
        f"Optimal Blend Weights (Spatial-ResNet: {weights[0]:.4f}, GPU GBDT: {weights[1]:.4f}) | Blended Val RMSE: {val_rmse:.4f}"
    )

    # Export Submission and Verify Format
    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {"key": test_keys, "fare_amount": test_preds.astype(np.float64)}
    )
    submission_df.to_csv(submission_path, index=False)

    assert len(submission_df) == 9914, f"Expected 9914 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == [
        "key",
        "fare_amount",
    ], f"Invalid submission columns: {list(submission_df.columns)}"
    assert (
        not submission_df.isnull().values.any()
    ), "Null values found in final submission"
    assert (
        submission_df["fare_amount"] >= 2.50
    ).all(), "Fares below $2.50 legal minimum found"

    print(
        f"Submission saved successfully to {submission_path} with {len(submission_df)} rows."
    )
    print(f"Final Validation Score: {val_rmse:.6f}")


if __name__ == "__main__":
    main()
