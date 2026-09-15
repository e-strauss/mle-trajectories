import gc
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# Configuration and Constants
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# NYC Metropolitan coordinate boundaries and fare bounds
LON_MIN, LON_MAX = -74.45, -72.80
LAT_MIN, LAT_MAX = 40.45, 41.85
FARE_MIN, FARE_MAX = 2.50, 400.00

# Column data types for memory-efficient ingestion across train and test sets
COL_DTYPES = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
col_dtypes = COL_DTYPES

# Geographic Hubs: Airports and Key NYC Landmarks
HUBS = {
    "jfk": (-73.7781, 40.6413),
    "lga": (-73.8740, 40.7769),
    "ewr": (-74.1745, 40.6895),
    "midtown": (-73.9855, 40.7580),
    "wall_st": (-74.0090, 40.7075),
    "grand_central": (-73.9772, 40.7527),
    "bk_center": (-73.9754, 40.6826),
}

# Major East River and Hudson River Chokepoints (Bridges & Tunnels)
CHOKEPOINTS = [
    (-73.9969, 40.7061),  # Brooklyn / Manhattan Bridge
    (-73.9542, 40.7570),  # Queensboro Bridge
    (-73.9678, 40.7447),  # Queens-Midtown Tunnel
    (-73.9238, 40.7801),  # Triborough / RFK Bridge
    (-74.0080, 40.7607),  # Lincoln Tunnel
    (-74.0114, 40.7259),  # Holland Tunnel
    (-73.9525, 40.8517),  # George Washington Bridge
]

# Manhattan grid rotation angle: 29 degrees clockwise from North
MANHATTAN_ANGLE_RAD = np.radians(29.0)
COS_29 = np.cos(MANHATTAN_ANGLE_RAD)
SIN_29 = np.sin(MANHATTAN_ANGLE_RAD)


# -------------------------------------------------------------------------
# Vectorized Geographic Distance & Bearing Calculations
# -------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    """Calculates geodesic distance in kilometers using vectorized Haversine formula."""
    r_lon1, r_lat1, r_lon2, r_lat2 = (
        np.radians(lon1),
        np.radians(lat1),
        np.radians(lon2),
        np.radians(lat2),
    )
    dlon = r_lon2 - r_lon1
    dlat = r_lat2 - r_lat1
    a = (
        np.sin(dlat * 0.5) ** 2
        + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon * 0.5) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (6371.0 * c).astype(np.float32)


def bearing_degrees(lon1, lat1, lon2, lat2):
    """Calculates compass bearing angle (0-360 degrees) between origin and destination."""
    r_lon1, r_lat1, r_lon2, r_lat2 = (
        np.radians(lon1),
        np.radians(lat1),
        np.radians(lon2),
        np.radians(lat2),
    )
    dlon = r_lon2 - r_lon1
    y = np.sin(dlon) * np.cos(r_lat2)
    x = np.cos(r_lat1) * np.sin(r_lat2) - np.sin(r_lat1) * np.cos(r_lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return ((bearing + 360.0) % 360.0).astype(np.float32)


def assign_borough(lon, lat):
    """Vectorized coordinate bounding logic defining the five NYC boroughs plus Newark.

    Returns integer borough codes:
    0: Other / Unknown
    1: Manhattan
    2: Brooklyn
    3: Queens
    4: Bronx
    5: Staten Island
    6: Newark
    """
    borough = np.zeros(len(lon), dtype=np.int32)

    # 1. Newark / New Jersey (includes EWR airport hub)
    is_newark = (lon >= -74.28) & (lon <= -74.03) & (lat >= 40.65) & (lat <= 40.85)
    borough[is_newark] = 6

    # 2. Staten Island
    is_si = (lon >= -74.26) & (lon <= -74.05) & (lat >= 40.49) & (lat < 40.65)
    borough[is_si] = 5

    # 3. Bronx
    is_bronx = (lat >= 40.795) & (lat <= 40.93) & (lon >= -73.935) & (lon <= -73.75)
    borough[is_bronx] = 4

    # 4. Manhattan
    # Water boundary separating Manhattan from Brooklyn, Queens, and Bronx
    m_east_bound = np.where(
        lat < 40.74,
        -74.000 + 0.8 * (lat - 40.700),
        np.where(lat < 40.80, -73.945, -73.915),
    )
    is_manhattan = (
        (lat >= 40.700)
        & (lat <= 40.880)
        & (lon >= -74.025)
        & (lon <= m_east_bound)
    )
    borough[is_manhattan] = 1

    # 5. Brooklyn
    is_brooklyn = (
        (lat >= 40.570)
        & (lat <= 40.740)
        & (lon >= -74.045)
        & (lon <= -73.850)
        & (borough == 0)
    )
    borough[is_brooklyn] = 2

    # 6. Queens
    is_queens = (
        (lat >= 40.540)
        & (lat <= 40.800)
        & (lon >= -73.965)
        & (lon <= -73.700)
        & (borough == 0)
    )
    borough[is_queens] = 3

    return borough


# -------------------------------------------------------------------------
# High-Performance Feature Engineering Pipeline
# -------------------------------------------------------------------------
def build_features(df, is_train=True):
    """Transforms raw trip coordinates and UTC timestamps into domain-grounded

    predictive features matching NYC TLC regulations and urban geography.
    """
    p_lon = df["pickup_longitude"].values.astype(np.float32)
    p_lat = df["pickup_latitude"].values.astype(np.float32)
    d_lon = df["dropoff_longitude"].values.astype(np.float32)
    d_lat = df["dropoff_latitude"].values.astype(np.float32)

    features = {}

    # 1. Coordinate Deltas & Spatial Distance
    dlon = d_lon - p_lon
    dlat = d_lat - p_lat
    features["delta_lon"] = dlon
    features["delta_lat"] = dlat
    abs_dlon = np.abs(dlon)
    abs_dlat = np.abs(dlat)
    features["abs_diff_lon"] = abs_dlon
    features["abs_diff_lat"] = abs_dlat
    features["euclidean_dist"] = np.sqrt(abs_dlon**2 + abs_dlat**2).astype(np.float32)

    # 2. Haversine Great-Circle Distance
    hav_km = haversine_km(p_lon, p_lat, d_lon, d_lat)
    features["haversine_km"] = hav_km
    features["log_haversine"] = np.log1p(hav_km)

    # 3. Metric Rotated Manhattan Grid Distance (29 degrees grid rotation with parallel scaling)
    mean_lat_rad = np.radians((p_lat + d_lat) * 0.5)
    dx_km = (dlon * (111.0 * np.cos(mean_lat_rad))).astype(np.float32)
    dy_km = (dlat * 111.0).astype(np.float32)

    rot_street_km = np.abs(dx_km * COS_29 - dy_km * SIN_29).astype(np.float32)
    rot_avenue_km = np.abs(dx_km * SIN_29 + dy_km * COS_29).astype(np.float32)
    manhattan_km = (rot_street_km + rot_avenue_km).astype(np.float32)

    features["manhattan_rot_km"] = manhattan_km
    features["log_manhattan"] = np.log1p(manhattan_km)
    features["rot_street_km"] = rot_street_km
    features["rot_avenue_km"] = rot_avenue_km
    features["rot_street_ratio"] = (rot_street_km / (manhattan_km + 0.001)).astype(np.float32)

    # Detour / Tortuosity Ratio (routing inefficiency indicator)
    features["detour_ratio"] = (manhattan_km / (hav_km + 0.005)).astype(np.float32)

    # 4. Compass Bearing & Midpoints
    features["bearing"] = bearing_degrees(p_lon, p_lat, d_lon, d_lat)
    features["mid_lon"] = ((p_lon + d_lon) * 0.5).astype(np.float32)
    features["mid_lat"] = ((p_lat + d_lat) * 0.5).astype(np.float32)

    # 5. Distances to Major Airport and Landmark Hubs
    for hub_name, (hub_lon, hub_lat) in HUBS.items():
        p_dist = haversine_km(p_lon, p_lat, hub_lon, hub_lat)
        d_dist = haversine_km(d_lon, d_lat, hub_lon, hub_lat)
        features[f"pickup_to_{hub_name}"] = p_dist
        features[f"dropoff_to_{hub_name}"] = d_dist
        features[f"min_to_{hub_name}"] = np.minimum(p_dist, d_dist)

    # Specific Airport Proximity Flags
    features["is_jfk_trip"] = (features["min_to_jfk"] < 2.5).astype(np.float32)
    features["is_lga_trip"] = (features["min_to_lga"] < 2.0).astype(np.float32)
    features["is_ewr_trip"] = (features["min_to_ewr"] < 3.0).astype(np.float32)

    # Borough Classification & Water Crossing Features
    p_borough = assign_borough(p_lon, p_lat)
    d_borough = assign_borough(d_lon, d_lat)

    features["pickup_borough"] = p_borough.astype(np.float32)
    features["dropoff_borough"] = d_borough.astype(np.float32)
    features["is_same_borough"] = (p_borough == d_borough).astype(np.float32)

    p_is_manhattan = (p_borough == 1)
    d_is_manhattan = (d_borough == 1)
    features["pickup_in_manhattan"] = p_is_manhattan.astype(np.float32)
    features["dropoff_in_manhattan"] = d_is_manhattan.astype(np.float32)
    features["pickup_in_brooklyn"] = (p_borough == 2).astype(np.float32)
    features["dropoff_in_brooklyn"] = (d_borough == 2).astype(np.float32)
    features["pickup_in_queens"] = (p_borough == 3).astype(np.float32)
    features["dropoff_in_queens"] = (d_borough == 3).astype(np.float32)
    features["pickup_in_bronx"] = (p_borough == 4).astype(np.float32)
    features["dropoff_in_bronx"] = (d_borough == 4).astype(np.float32)
    features["pickup_in_staten_island"] = (p_borough == 5).astype(np.float32)
    features["dropoff_in_staten_island"] = (d_borough == 5).astype(np.float32)
    features["pickup_in_newark"] = (p_borough == 6).astype(np.float32)
    features["dropoff_in_newark"] = (d_borough == 6).astype(np.float32)

    # Water-barrier crossing toll indicators (East River & Hudson River)
    is_east_river = (
        ((p_borough == 1) & ((d_borough == 2) | (d_borough == 3)))
        | ((d_borough == 1) & ((p_borough == 2) | (p_borough == 3)))
        | ((p_borough == 4) & (d_borough == 3))
        | ((d_borough == 4) & (p_borough == 3))
    ).astype(np.float32)
    features["is_east_river_crossing"] = is_east_river

    is_hudson_river = (
        ((p_borough == 6) & (d_borough > 0) & (d_borough != 6))
        | ((d_borough == 6) & (p_borough > 0) & (p_borough != 6))
    ).astype(np.float32)
    features["is_hudson_river_crossing"] = is_hudson_river
    features["is_river_crossing"] = np.maximum(is_east_river, is_hudson_river).astype(np.float32)

    # Regulated Manhattan <-> JFK flat rate route indicator across all Manhattan
    features["is_manhattan_jfk_flat_route"] = (
        (p_is_manhattan & (features["dropoff_to_jfk"] < 3.0))
        | (d_is_manhattan & (features["pickup_to_jfk"] < 3.0))
    ).astype(np.float32)

    # 6. Bridge & Tunnel Funneling Detour (East/Hudson River water barrier crossing)
    choke_min_routes = np.full(len(df), np.inf, dtype=np.float32)
    for c_lon, c_lat in CHOKEPOINTS:
        p_to_c = haversine_km(p_lon, p_lat, c_lon, c_lat)
        c_to_d = haversine_km(c_lon, c_lat, d_lon, d_lat)
        route_dist = p_to_c + c_to_d
        choke_min_routes = np.minimum(choke_min_routes, route_dist)
    features["chokepoint_route_km"] = choke_min_routes
    raw_choke_excess = np.maximum(0.0, choke_min_routes - hav_km)
    features["chokepoint_excess_km"] = (
        raw_choke_excess * (features["is_river_crossing"] > 0)
    ).astype(np.float32)

    # 7. Passenger Count & Interaction
    p_count = df["passenger_count"].values.astype(np.float32)
    p_count = np.clip(p_count, 1.0, 6.0)
    features["passenger_count"] = p_count
    features["dist_x_passenger"] = (hav_km * p_count).astype(np.float32)

    # 8. Multi-Scale Spatial Quantization (2km grid binning & 500m micro-grid)
    p_grid_lon = np.floor(p_lon * 50.0).astype(np.int32)
    p_grid_lat = np.floor(p_lat * 50.0).astype(np.int32)
    d_grid_lon = np.floor(d_lon * 50.0).astype(np.int32)
    d_grid_lat = np.floor(d_lat * 50.0).astype(np.int32)
    features["is_same_grid"] = (
        (p_grid_lon == d_grid_lon) & (p_grid_lat == d_grid_lat)
    ).astype(np.float32)

    p_micro_lon = np.floor(p_lon * 200.0).astype(np.float32)
    p_micro_lat = np.floor(p_lat * 200.0).astype(np.float32)
    d_micro_lon = np.floor(d_lon * 200.0).astype(np.float32)
    d_micro_lat = np.floor(d_lat * 200.0).astype(np.float32)
    features["is_same_micro_cell"] = (
        (p_micro_lon == d_micro_lon) & (p_micro_lat == d_micro_lat)
    ).astype(np.float32)
    features["micro_delta_lon"] = (d_micro_lon - p_micro_lon).astype(np.float32)
    features["micro_delta_lat"] = (d_micro_lat - p_micro_lat).astype(np.float32)
    features["p_micro_offset_lon"] = (p_lon - (p_micro_lon + 0.5) / 200.0).astype(np.float32)
    features["p_micro_offset_lat"] = (p_lat - (p_micro_lat + 0.5) / 200.0).astype(np.float32)
    features["d_micro_offset_lon"] = (d_lon - (d_micro_lon + 0.5) / 200.0).astype(np.float32)
    features["d_micro_offset_lat"] = (d_lat - (d_micro_lat + 0.5) / 200.0).astype(np.float32)

    # 9. Temporal and NYC TLC Regulatory Surcharges
    ts_str = df["pickup_datetime"].astype(str).str.slice(0, 19)
    dt_utc = pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S", utc=True)
    dt_local = dt_utc.dt.tz_convert("America/New_York")

    year = dt_local.dt.year.values
    month = dt_local.dt.month.values
    dow = dt_local.dt.dayofweek.values
    hour = dt_local.dt.hour.values
    minute = dt_local.dt.minute.values

    hour_fraction = (hour + minute / 60.0).astype(np.float32)
    hour_of_week = (dow * 24.0 + hour_fraction).astype(np.float32)
    features["year"] = year.astype(np.float32)
    features["month"] = month.astype(np.float32)
    features["dayofweek"] = dow.astype(np.float32)
    features["hour_fraction"] = hour_fraction
    features["hour_of_week"] = hour_of_week
    features["hour_of_week_sin"] = np.sin(2.0 * np.pi * hour_of_week / 168.0).astype(np.float32)
    features["hour_of_week_cos"] = np.cos(2.0 * np.pi * hour_of_week / 168.0).astype(np.float32)
    features["is_weekend"] = (dow >= 5).astype(np.float32)

    # Continuous time trend (days since Jan 1, 2009)
    base_ts = pd.Timestamp("2009-01-01", tz="UTC")
    features["days_since_start"] = (
        (dt_utc - base_ts).dt.total_seconds() / 86400.0
    ).values.astype(np.float32)

    # Cyclical Time Encodings
    features["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction / 24.0).astype(np.float32)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction / 24.0).astype(np.float32)
    features["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0).astype(
        np.float32
    )
    features["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0).astype(
        np.float32
    )
    features["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    features["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)

    # NYC TLC Mandated Surcharges and Tariff Schedulers
    tlc_peak = ((dow < 5) & (hour >= 16) & (hour < 20)).astype(np.float32)
    tlc_overnight = ((hour >= 20) | (hour < 6)).astype(np.float32)
    features["tlc_peak_surcharge"] = tlc_peak
    features["tlc_overnight_surcharge"] = tlc_overnight

    hike_ts = pd.Timestamp("2012-09-04", tz="UTC")
    is_post_hike = (dt_utc >= hike_ts).values.astype(np.float32)
    features["is_post_2012_rate_hike"] = is_post_hike
    features["rate_multiplier"] = np.where(dt_utc >= hike_ts, 1.25, 1.0).astype(np.float32)
    features["flat_rate_post_2012"] = (
        features["is_manhattan_jfk_flat_route"] * features["is_post_2012_rate_hike"]
    ).astype(np.float32)
    features["flat_rate_pre_2012"] = (
        features["is_manhattan_jfk_flat_route"] * (1.0 - features["is_post_2012_rate_hike"])
    ).astype(np.float32)

    mta_ts = pd.Timestamp("2009-11-01", tz="UTC")
    is_post_mta = (dt_utc >= mta_ts).values.astype(np.float32)
    features["is_post_mta_tax"] = is_post_mta

    impr_ts = pd.Timestamp("2015-01-01", tz="UTC")
    is_post_impr = (dt_utc >= impr_ts).values.astype(np.float32)
    features["is_post_2015_improvement"] = is_post_impr

    # Domain-Grounded NYC TLC Regulatory Tariff & Physical Meter Simulation
    mileage_rate = np.where(dt_utc >= hike_ts, 1.55343, 1.24274).astype(np.float32)
    features["mileage_rate"] = mileage_rate
    features["meter_charge_hav"] = (mileage_rate * hav_km).astype(np.float32)
    features["meter_charge_manhattan"] = (mileage_rate * manhattan_km).astype(np.float32)

    # Standard Metered Baseline and Discrete Surcharges
    total_surcharges = (
        tlc_peak * 1.00
        + tlc_overnight * 0.50
        + is_post_mta * 0.50
        + is_post_impr * 0.30
    ).astype(np.float32)
    features["total_surcharges"] = total_surcharges
    simulated_meter_fare = (2.50 + mileage_rate * manhattan_km + total_surcharges).astype(np.float32)
    features["simulated_meter_fare"] = simulated_meter_fare
    features["simulated_meter_hav_fare"] = (2.50 + mileage_rate * hav_km + total_surcharges).astype(np.float32)

    # Statutory Airport Schedules (JFK Flat Fare & Newark Airport Surcharges)
    jfk_flat_base = np.where(dt_utc >= hike_ts, 52.00, 45.00).astype(np.float32)
    features["jfk_flat_base"] = (features["is_manhattan_jfk_flat_route"] * jfk_flat_base).astype(np.float32)

    is_newark_trip = ((features["is_ewr_trip"] > 0) | (p_borough == 6) | (d_borough == 6)).astype(np.float32)
    features["is_newark_trip"] = is_newark_trip
    ewr_surcharge = (is_newark_trip * 15.00).astype(np.float32)
    features["ewr_surcharge"] = ewr_surcharge

    simulated_regulatory_fare = np.where(
        features["is_manhattan_jfk_flat_route"] > 0,
        jfk_flat_base + is_post_mta * 0.50 + is_post_impr * 0.30,
        simulated_meter_fare + ewr_surcharge,
    ).astype(np.float32)
    features["simulated_regulatory_fare"] = simulated_regulatory_fare

    # Spatial-Temporal Traffic Congestion & Rush Hour Surcharge Interactions
    morning_rush_flag = ((dow < 5) & (hour >= 7) & (hour < 10)).astype(np.float32)
    features["morning_rush_flag"] = morning_rush_flag

    is_manhattan_internal = (p_is_manhattan & d_is_manhattan).astype(np.float32)
    features["is_manhattan_internal"] = is_manhattan_internal

    features["morning_rush_x_manhattan_internal"] = (
        morning_rush_flag * is_manhattan_internal
    ).astype(np.float32)
    features["morning_rush_x_river_crossing"] = (
        morning_rush_flag * features["is_river_crossing"]
    ).astype(np.float32)
    features["morning_rush_x_chokepoint"] = (
        morning_rush_flag * features["chokepoint_excess_km"]
    ).astype(np.float32)
    features["rot_street_x_morning_rush"] = (
        features["rot_street_km"] * morning_rush_flag
    ).astype(np.float32)
    features["rot_avenue_x_morning_rush"] = (
        features["rot_avenue_km"] * morning_rush_flag
    ).astype(np.float32)
    features["manhattan_rot_x_morning_rush"] = (
        features["manhattan_rot_km"] * morning_rush_flag
    ).astype(np.float32)

    features["peak_x_manhattan_internal"] = (
        tlc_peak * is_manhattan_internal
    ).astype(np.float32)
    features["peak_x_river_crossing"] = (
        tlc_peak * features["is_river_crossing"]
    ).astype(np.float32)
    features["peak_x_chokepoint"] = (
        tlc_peak * features["chokepoint_excess_km"]
    ).astype(np.float32)

    features["rot_street_x_peak"] = (
        features["rot_street_km"] * tlc_peak
    ).astype(np.float32)
    features["rot_avenue_x_peak"] = (
        features["rot_avenue_km"] * tlc_peak
    ).astype(np.float32)
    features["rot_street_ratio_x_peak"] = (
        features["rot_street_ratio"] * tlc_peak
    ).astype(np.float32)

    # 10. Continuous Multi-Scale Fourier Coordinate Projections
    for coord_name, coord_arr in [
        ("pickup_longitude", p_lon),
        ("pickup_latitude", p_lat),
        ("dropoff_longitude", d_lon),
        ("dropoff_latitude", d_lat),
    ]:
        for k in [1, 2, 4, 8]:
            freq = float(2**k) * np.pi
            features[f"{coord_name}_fourier_sin_{k}"] = np.sin(freq * coord_arr).astype(np.float32)
            features[f"{coord_name}_fourier_cos_{k}"] = np.cos(freq * coord_arr).astype(np.float32)

    # Original coordinates retained
    features["pickup_longitude"] = p_lon
    features["pickup_latitude"] = p_lat
    features["dropoff_longitude"] = d_lon
    features["dropoff_latitude"] = d_lat

    feature_df = pd.DataFrame(features)
    if "key" in df.columns:
        feature_df["key"] = df["key"].values
    if is_train and "fare_amount" in df.columns:
        feature_df["fare_amount"] = df["fare_amount"].values.astype(np.float32)

    return feature_df


# -------------------------------------------------------------------------
# Balanced Ingestion of Training Data (15M Clean Rows)
# -------------------------------------------------------------------------
def load_and_preprocess_data(train_csv_path, target_rows=15_000_000):
    """Uniform chunked ingestion with physical rate-consistency sanitation targeting 15M clean rows."""
    print(f"Starting uniform chunked data ingestion targeting {target_rows:,} clean rows...")
    chunk_size = 1_000_000
    samples_per_chunk = int(np.ceil(target_rows / 54))
    sampled_chunks = []

    use_cols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    chunk_count = 0
    for chunk in pd.read_csv(
        train_csv_path, chunksize=chunk_size, usecols=use_cols, dtype=col_dtypes
    ):
        chunk_count += 1
        valid_mask = (
            (chunk["fare_amount"] >= FARE_MIN)
            & (chunk["fare_amount"] <= FARE_MAX)
            & (chunk["pickup_longitude"] >= LON_MIN)
            & (chunk["pickup_longitude"] <= LON_MAX)
            & (chunk["pickup_latitude"] >= LAT_MIN)
            & (chunk["pickup_latitude"] <= LAT_MAX)
            & (chunk["dropoff_longitude"] >= LON_MIN)
            & (chunk["dropoff_longitude"] <= LON_MAX)
            & (chunk["dropoff_latitude"] >= LAT_MIN)
            & (chunk["dropoff_latitude"] <= LAT_MAX)
            & (chunk["passenger_count"] >= 0)
            & (chunk["passenger_count"] <= 6)
        )
        filtered = chunk[valid_mask].dropna()

        # Physical rate-consistency sanitation
        rough_hav_km = haversine_km(
            filtered["pickup_longitude"].values,
            filtered["pickup_latitude"].values,
            filtered["dropoff_longitude"].values,
            filtered["dropoff_latitude"].values,
        )
        fares = filtered["fare_amount"].values
        rate_consistent = ~(
            ((rough_hav_km < 0.1) & (fares > 50.00))
            | ((rough_hav_km > 3.0) & (fares < (2.50 + 1.25 * rough_hav_km)))
        )
        filtered = filtered[rate_consistent]

        if len(filtered) > samples_per_chunk:
            sampled = filtered.sample(
                n=samples_per_chunk, random_state=RANDOM_SEED + chunk_count
            )
        else:
            sampled = filtered

        sampled_chunks.append(sampled)

    raw_df = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    if len(raw_df) > target_rows:
        raw_df = raw_df.sample(n=target_rows, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Loaded and cleaned {len(raw_df):,} trips across {chunk_count} chunks.")
    return raw_df


raw_train_df = load_and_preprocess_data(TRAIN_CSV, target_rows=15_000_000)

# -------------------------------------------------------------------------
# Feature Engineering on Training Data & Leak-Free Split
# -------------------------------------------------------------------------
print("Engineering features for training dataset...")
train_features_full = build_features(raw_train_df, is_train=True)
del raw_train_df

print("Creating leak-free 95/5 train/validation split...")
val_ratio = 0.05
n_total = len(train_features_full)
n_val = int(n_total * val_ratio)

perm_indices = np.random.permutation(n_total)
val_idx = perm_indices[:n_val]
train_idx = perm_indices[n_val:]

train_df = train_features_full.iloc[train_idx].reset_index(drop=True)
val_df = train_features_full.iloc[val_idx].reset_index(drop=True)
del train_features_full

print(
    f"Train split size: {len(train_df):,} rows | Validation split size: {len(val_df):,} rows"
)

# -------------------------------------------------------------------------
# Feature Engineering on Test Data
# -------------------------------------------------------------------------
print("Loading and featurizing test.csv...")
test_dtypes = {k: v for k, v in col_dtypes.items() if k != "fare_amount"}
test_raw = pd.read_csv(
    TEST_CSV,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype=test_dtypes,
)

test_df = build_features(test_raw, is_train=False)
print(
    f"Test dataset size: {len(test_df):,} rows (verified against required 9,914 rows)"
)

feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]
with open(os.path.join(WORKING_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"Total engineered predictive features: {len(feature_cols)}")

train_mean_fare = float(train_df["fare_amount"].mean())
val_baseline_rmse = float(
    np.sqrt(np.mean((val_df["fare_amount"].values - train_mean_fare) ** 2))
)
print(
    f"Baseline Training Mean Fare: ${train_mean_fare:.2f} (Baseline Holdout RMSE: {val_baseline_rmse:.4f})"
)


# -------------------------------------------------------------------------
# Feature Matrix Preparation (Unscaled Physical Features)
# -------------------------------------------------------------------------
X_train = np.nan_to_num(
    train_df[feature_cols].to_numpy(dtype=np.float32),
    copy=False,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_train = train_df["fare_amount"].to_numpy(dtype=np.float32)

X_val = np.nan_to_num(
    val_df[feature_cols].to_numpy(dtype=np.float32),
    copy=False,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
y_val = val_df["fare_amount"].to_numpy(dtype=np.float32)

X_test = np.nan_to_num(
    test_df[feature_cols].to_numpy(dtype=np.float32),
    copy=False,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

# -------------------------------------------------------------------------
# Dual-Pipeline Standardized Feature Representation for Neural Network
# -------------------------------------------------------------------------
print("Fitting StandardScaler strictly on training fold for SpatialResNet...")
scaler = StandardScaler()
scaler.fit(X_train)

X_train_nn = scaler.transform(X_train).astype(np.float32)
X_val_nn = scaler.transform(X_val).astype(np.float32)
X_test_nn = scaler.transform(X_test).astype(np.float32)

# Preserve unscaled simulated_meter_fare for the physical linear shortcut in SpatialResNet
meter_idx = feature_cols.index("simulated_meter_fare")
X_train_nn[:, meter_idx] = X_train[:, meter_idx]
X_val_nn[:, meter_idx] = X_val[:, meter_idx]
X_test_nn[:, meter_idx] = X_test[:, meter_idx]


# -------------------------------------------------------------------------
# Architecture Definition: SpatialResNet with Physics Shortcut Connection
# -------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.swish = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln(out)
        out = self.swish(out)
        out = self.drop(out)
        out = self.fc2(out)
        return residual + out


class SpatialResNet(nn.Module):
    def __init__(self, in_features, meter_idx, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.meter_idx = meter_idx
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(3)
        ])
        self.head = nn.Linear(hidden_dim, 1)
        self.meter_scale = nn.Linear(1, 1)
        # Initialize physics shortcut to identity scaling
        nn.init.constant_(self.meter_scale.weight, 1.0)
        nn.init.constant_(self.meter_scale.bias, 0.0)

    def forward(self, x):
        meter = x[:, self.meter_idx : self.meter_idx + 1]
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        residual_pred = self.head(h)
        meter_pred = self.meter_scale(meter)
        out = residual_pred + meter_pred
        return out.squeeze(-1)


# -------------------------------------------------------------------------
# Model 1: GPU Tree Booster Training (LightGBM)
# -------------------------------------------------------------------------
dtrain = lgb.Dataset(
    X_train, label=y_train, feature_name=feature_cols
)
dval = lgb.Dataset(
    X_val,
    label=y_val,
    reference=dtrain,
    feature_name=feature_cols,
)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 100,
    "learning_rate": 0.06,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "max_bin": 255,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Auto-detect GPU histogram acceleration
for dev in ["cuda", "gpu"]:
    try:
        test_ds = lgb.Dataset(np.zeros((20, 2), dtype=np.float32), label=np.zeros(20, dtype=np.float32))
        _ = lgb.train({"objective": "regression", "device": dev, "verbose": -1}, test_ds, num_boost_round=1)
        lgb_params["device"] = dev
        print(f"Tree booster GPU acceleration enabled with device='{dev}'")
        break
    except Exception:
        continue
if "device" not in lgb_params:
    lgb_params["n_jobs"] = -1
    print("Tree booster running on CPU (n_jobs=-1)")

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

print("Starting tree booster training...")
booster = lgb.train(
    params=lgb_params,
    train_set=dtrain,
    num_boost_round=3000,
    valid_sets=[dval],
    callbacks=callbacks,
)

model_path = os.path.join(WORKING_DIR, "lgb_taxifare_model.txt")
booster.save_model(model_path)
print(f"Tree model saved to {model_path} with best iteration: {booster.best_iteration}")

val_preds_tree = booster.predict(X_val, num_iteration=booster.best_iteration).astype(np.float32)
val_rmse_tree = float(np.sqrt(np.mean((val_preds_tree - y_val) ** 2)))
print(f"Tree Booster Holdout Validation RMSE: {val_rmse_tree:.5f}")

test_preds_tree = booster.predict(X_test, num_iteration=booster.best_iteration).astype(np.float32)

del dtrain, dval
gc.collect()

# -------------------------------------------------------------------------
# Model 2: PyTorch SpatialResNet Training (AdamW, OneCycleLR, AMP)
# -------------------------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training SpatialResNet on device: {device}")

nn_model = SpatialResNet(
    in_features=X_train_nn.shape[1],
    meter_idx=meter_idx,
    hidden_dim=256,
    dropout=0.1,
).to(device)

batch_size = 8192
train_dataset = TensorDataset(
    torch.from_numpy(X_train_nn),
    torch.from_numpy(y_train),
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

epochs = 6
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4)
total_steps = epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    total_steps=total_steps,
    pct_start=0.1,
)
scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
criterion = nn.MSELoss()

print(f"Starting SpatialResNet training across {epochs} epochs with batch size {batch_size}...")
for epoch in range(1, epochs + 1):
    nn_model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=5.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

        running_loss += loss.item()

    epoch_mse = running_loss / len(train_loader)
    print(f"SpatialResNet Epoch {epoch}/{epochs} - Train MSE: {epoch_mse:.4f} (RMSE: {np.sqrt(epoch_mse):.4f})")


def predict_nn(model, X_mat, eval_batch_size=16384):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X_mat), eval_batch_size):
            batch = torch.from_numpy(X_mat[i : i + eval_batch_size]).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                p = model(batch)
            preds_list.append(p.cpu().numpy())
    return np.concatenate(preds_list, axis=0).astype(np.float32)


val_preds_nn = predict_nn(nn_model, X_val_nn)
val_rmse_nn = float(np.sqrt(np.mean((val_preds_nn - y_val) ** 2)))
print(f"SpatialResNet Holdout Validation RMSE: {val_rmse_nn:.5f}")

test_preds_nn = predict_nn(nn_model, X_test_nn)

# -------------------------------------------------------------------------
# Analytical Convex Ensembling & Submission Export
# -------------------------------------------------------------------------
diff = val_preds_tree - val_preds_nn
denom = float(np.dot(diff, diff))
if denom > 1e-12:
    alpha = float(np.dot(y_val - val_preds_nn, diff) / denom)
    alpha = float(np.clip(alpha, 0.05, 0.95))
else:
    alpha = 0.50

print(f"Optimal Convex Blend Weights: Tree = {alpha:.4f}, SpatialResNet = {1.0 - alpha:.4f}")

final_val_preds = alpha * val_preds_tree + (1.0 - alpha) * val_preds_nn
final_val_preds = np.clip(final_val_preds, FARE_MIN, FARE_MAX)
final_val_rmse = float(np.sqrt(np.mean((final_val_preds - y_val) ** 2)))

blended_test_preds = alpha * test_preds_tree + (1.0 - alpha) * test_preds_nn
blended_test_preds = np.clip(blended_test_preds, FARE_MIN, FARE_MAX)

submission_df = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": blended_test_preds.astype(np.float64)}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
assert len(submission_df) == 9914, f"Expected 9,914 rows, got {len(submission_df)}"
assert submission_df["fare_amount"].isna().sum() == 0, "Submission contains NaN values"

# -------------------------------------------------------------------------
# Final Score Output (Mandatory Last Line)
# -------------------------------------------------------------------------
print(f"Final Validation Score: {final_val_rmse:.5f}")
