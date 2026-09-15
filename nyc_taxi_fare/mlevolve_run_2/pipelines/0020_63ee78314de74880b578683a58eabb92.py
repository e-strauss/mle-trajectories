import gc
import json
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

N_ROWS_TO_LOAD = 30_000_000
RANDOM_STATE = 42

# NYC Bounding Box (covers all 5 boroughs, airports, Westchester & Long Island)
BB_MIN_LAT = 40.40
BB_MAX_LAT = 41.85
BB_MIN_LON = -74.40
BB_MAX_LON = -72.80

# Key NYC Landmarks & Airports (lat, lon)
LANDMARKS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
    "midtown": (40.7580, -73.9855),
    "lower_manhattan": (40.7071, -74.0090),
    "downtown_brooklyn": (40.6928, -73.9872),
}

# Major NYC Toll Plazas & River Crossings (lat, lon)
TOLL_PLAZAS = {
    "queens_midtown_tunnel": (40.7445, -73.9654),
    "brooklyn_battery_tunnel": (40.6978, -74.0135),
    "rfk_triborough_bridge": (40.7801, -73.9247),
    "lincoln_tunnel": (40.7587, -74.0086),
    "holland_tunnel": (40.7258, -74.0145),
    "gwb": (40.8517, -73.9527),
    "verrazzano_bridge": (40.6066, -74.0447),
}

# Toll-free East River Bridges (lat, lon)
FREE_BRIDGES = {
    "brooklyn_bridge": (40.7061, -73.9969),
    "manhattan_bridge": (40.7075, -73.9906),
    "williamsburg_bridge": (40.7135, -73.9724),
    "queensboro_bridge": (40.7570, -73.9542),
}

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Core Geospatial & Temporal Feature Engineering Functions
# -----------------------------------------------------------------------------
def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate Haversine great-circle distance in kilometers."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return (r * c).astype(np.float32)


def bearing_np(lat1, lon1, lat2, lon2):
    """Calculate forward azimuth / compass bearing in radians."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return np.arctan2(y, x).astype(np.float32)


def assign_borough_zone(lat, lon, to_jfk_km, to_lga_km, to_ewr_km):
    """Discretize coordinates into 9 discrete NYC metropolitan zones."""
    zone = np.full(lat.shape, 8, dtype=np.int8)  # 8: Other / Suburbs

    # 1. Brooklyn
    mask_bk = (lat >= 40.57) & (lat <= 40.74) & (lon >= -74.05) & (lon <= -73.85)
    zone[mask_bk] = 1

    # 2. Queens
    mask_qn = (lat >= 40.55) & (lat <= 40.80) & (lon >= -73.96) & (lon <= -73.70)
    zone[mask_qn] = 2

    # 3. Staten Island
    mask_si = (lat >= 40.50) & (lat <= 40.66) & (lon <= -74.05)
    zone[mask_si] = 4

    # 4. Bronx
    mask_bx = (lat >= 40.79) & (lon >= -73.93)
    zone[mask_bx] = 3

    # 5. Manhattan
    mask_mh = (lat >= 40.70) & (lat <= 40.88) & (lon >= -74.025) & (lon <= -73.91)
    zone[mask_mh] = 0

    # 6. EWR / New Jersey
    mask_ewr = (to_ewr_km < 3.5) | (lon < -74.12)
    zone[mask_ewr] = 7

    # 7. LGA
    mask_lga = to_lga_km < 2.5
    zone[mask_lga] = 6

    # 8. JFK
    mask_jfk = to_jfk_km < 3.0
    zone[mask_jfk] = 5

    return zone


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive rich domain-specific geospatial, temporal, and interaction features using dictionary accumulation."""
    feats = {}

    p_lat = df["pickup_latitude"].values.astype(np.float32)
    p_lon = df["pickup_longitude"].values.astype(np.float32)
    d_lat = df["dropoff_latitude"].values.astype(np.float32)
    d_lon = df["dropoff_longitude"].values.astype(np.float32)

    # 1. Coordinate differences & midpoints
    delta_lat = d_lat - p_lat
    delta_lon = d_lon - p_lon
    abs_delta_lat = np.abs(delta_lat)
    abs_delta_lon = np.abs(delta_lon)
    mid_lat = (p_lat + d_lat) / 2.0
    mid_lon = (p_lon + d_lon) / 2.0

    feats["pickup_latitude"] = p_lat
    feats["pickup_longitude"] = p_lon
    feats["dropoff_latitude"] = d_lat
    feats["dropoff_longitude"] = d_lon
    feats["delta_latitude"] = delta_lat
    feats["delta_longitude"] = delta_lon
    feats["abs_delta_latitude"] = abs_delta_lat
    feats["abs_delta_longitude"] = abs_delta_lon
    feats["mid_latitude"] = mid_lat
    feats["mid_longitude"] = mid_lon

    # Spatial coordinate rounding (2 and 3 decimal places)
    feats["pickup_latitude_round2"] = np.round(p_lat, 2).astype(np.float32)
    feats["pickup_longitude_round2"] = np.round(p_lon, 2).astype(np.float32)
    feats["dropoff_latitude_round2"] = np.round(d_lat, 2).astype(np.float32)
    feats["dropoff_longitude_round2"] = np.round(d_lon, 2).astype(np.float32)
    feats["pickup_latitude_round3"] = np.round(p_lat, 3).astype(np.float32)
    feats["pickup_longitude_round3"] = np.round(p_lon, 3).astype(np.float32)
    feats["dropoff_latitude_round3"] = np.round(d_lat, 3).astype(np.float32)
    feats["dropoff_longitude_round3"] = np.round(d_lon, 3).astype(np.float32)

    # 2. Distance metrics
    feats["haversine_km"] = haversine_np(p_lat, p_lon, d_lat, d_lon)
    feats["bearing"] = bearing_np(p_lat, p_lon, d_lat, d_lon)

    # Manhattan distance orthogonal in km
    lat_dist_km = abs_delta_lat * 111.139
    lon_dist_km = abs_delta_lon * 111.139 * np.cos(np.radians(mid_lat))
    feats["manhattan_km"] = (lat_dist_km + lon_dist_km).astype(np.float32)

    # Rotated Manhattan distance along NYC street grid (~29 degrees clockwise from North)
    grid_angle = np.radians(29.0)
    rot_dlat = lat_dist_km * np.cos(grid_angle) - lon_dist_km * np.sin(grid_angle)
    rot_dlon = lat_dist_km * np.sin(grid_angle) + lon_dist_km * np.cos(grid_angle)
    feats["rotated_manhattan_km"] = (np.abs(rot_dlat) + np.abs(rot_dlon)).astype(
        np.float32
    )
    # Decomposed avenue and street distance vectors
    feats["rotated_avenue_km"] = np.abs(rot_dlat).astype(np.float32)
    feats["rotated_street_km"] = np.abs(rot_dlon).astype(np.float32)
    feats["street_to_avenue_ratio"] = (rot_dlon / (rot_dlat + 1e-4)).astype(np.float32)
    feats["avenue_to_street_ratio"] = (rot_dlat / (rot_dlon + 1e-4)).astype(np.float32)

    # Non-linear distance transforms & tortuosity ratio
    feats["log_haversine_km"] = np.log1p(feats["haversine_km"]).astype(np.float32)
    feats["log_manhattan_km"] = np.log1p(feats["manhattan_km"]).astype(np.float32)
    feats["log_rotated_manhattan_km"] = np.log1p(feats["rotated_manhattan_km"]).astype(np.float32)
    feats["sqrt_haversine_km"] = np.sqrt(feats["haversine_km"]).astype(np.float32)
    feats["sqrt_manhattan_km"] = np.sqrt(feats["manhattan_km"]).astype(np.float32)
    feats["sqrt_rotated_manhattan_km"] = np.sqrt(feats["rotated_manhattan_km"]).astype(np.float32)
    feats["tortuosity_ratio"] = (feats["manhattan_km"] / (feats["haversine_km"] + 1e-4)).astype(np.float32)
    feats["rotated_tortuosity_ratio"] = (feats["rotated_manhattan_km"] / (feats["haversine_km"] + 1e-4)).astype(np.float32)

    # 3. Airport and Landmark Distances
    for name, (l_lat, l_lon) in LANDMARKS.items():
        feats[f"p_to_{name}_km"] = haversine_np(p_lat, p_lon, l_lat, l_lon)
        feats[f"d_to_{name}_km"] = haversine_np(d_lat, d_lon, l_lat, l_lon)

    # Airport Proximity Indicators
    feats["is_jfk_trip"] = (
        (feats["p_to_jfk_km"] < 2.5) | (feats["d_to_jfk_km"] < 2.5)
    ).astype(np.int8)
    feats["is_ewr_trip"] = (
        (feats["p_to_ewr_km"] < 2.5) | (feats["d_to_ewr_km"] < 2.5)
    ).astype(np.int8)
    feats["is_to_ewr"] = (feats["d_to_ewr_km"] < 2.5).astype(np.int8)
    feats["is_from_ewr"] = (feats["p_to_ewr_km"] < 2.5).astype(np.int8)
    feats["is_lga_trip"] = (
        (feats["p_to_lga_km"] < 2.5) | (feats["d_to_lga_km"] < 2.5)
    ).astype(np.int8)

    # Directional JFK-to-Manhattan Flat-Rate Flags & Intra-Manhattan Core Flags
    p_in_manhattan = (p_lat >= 40.70) & (p_lat <= 40.88) & (p_lon >= -74.02) & (p_lon <= -73.93)
    d_in_manhattan = (d_lat >= 40.70) & (d_lat <= 40.88) & (d_lon >= -74.02) & (d_lon <= -73.93)
    feats["is_jfk_to_manhattan"] = ((feats["p_to_jfk_km"] < 2.5) & d_in_manhattan).astype(np.int8)
    feats["is_manhattan_to_jfk"] = (p_in_manhattan & (feats["d_to_jfk_km"] < 2.5)).astype(np.int8)
    feats["is_jfk_flat_rate"] = (feats["is_jfk_to_manhattan"] | feats["is_manhattan_to_jfk"]).astype(np.int8)
    feats["is_intra_manhattan"] = (p_in_manhattan & d_in_manhattan).astype(np.int8)
    feats["intra_manhattan_dist"] = (feats["haversine_km"] * feats["is_intra_manhattan"]).astype(np.float32)

    # Toll Plaza Proximity and Crossing Distances
    for t_name, (t_lat, t_lon) in TOLL_PLAZAS.items():
        p_t_dist = haversine_np(p_lat, p_lon, t_lat, t_lon)
        d_t_dist = haversine_np(d_lat, d_lon, t_lat, t_lon)
        min_t_dist = np.minimum(p_t_dist, d_t_dist)
        feats[f"p_to_{t_name}_km"] = p_t_dist
        feats[f"d_to_{t_name}_km"] = d_t_dist
        feats[f"min_to_{t_name}_km"] = min_t_dist
        feats[f"is_near_{t_name}"] = (min_t_dist < 1.5).astype(np.float32)

    # Proximity to Toll-free East River Bridges
    for fb_name, (fb_lat, fb_lon) in FREE_BRIDGES.items():
        p_fb_dist = haversine_np(p_lat, p_lon, fb_lat, fb_lon)
        d_fb_dist = haversine_np(d_lat, d_lon, fb_lat, fb_lon)
        feats[f"p_to_{fb_name}_km"] = p_fb_dist
        feats[f"d_to_{fb_name}_km"] = d_fb_dist
        feats[f"min_to_{fb_name}_km"] = np.minimum(p_fb_dist, d_fb_dist)

    feats["min_to_any_free_bridge_km"] = np.minimum(
        np.minimum(feats["min_to_brooklyn_bridge_km"], feats["min_to_manhattan_bridge_km"]),
        np.minimum(feats["min_to_williamsburg_bridge_km"], feats["min_to_queensboro_bridge_km"]),
    )

    # East River and Hudson River Water Crossing Indicators
    p_manhattan_core = (p_lat >= 40.70) & (p_lat <= 40.85) & (p_lon <= -73.965)
    d_outer_borough = (d_lat >= 40.58) & (d_lat <= 40.80) & (d_lon >= -73.945)
    p_outer_borough = (p_lat >= 40.58) & (p_lat <= 40.80) & (p_lon >= -73.945)
    d_manhattan_core = (d_lat >= 40.70) & (d_lat <= 40.85) & (d_lon <= -73.965)

    feats["is_cross_east_river"] = (
        (p_manhattan_core & d_outer_borough) | (p_outer_borough & d_manhattan_core)
    ).astype(np.float32)

    feats["is_cross_hudson"] = (
        ((p_lon <= -74.03) & (d_lon >= -74.01)) | ((p_lon >= -74.01) & (d_lon <= -74.03))
    ).astype(np.float32)

    feats["is_water_crossing"] = (
        (feats["is_cross_east_river"] > 0) | (feats["is_cross_hudson"] > 0)
    ).astype(np.float32)

    # 4. 9-Zone Borough Encodings and OD Interaction Pairs
    p_zone = assign_borough_zone(
        p_lat, p_lon, feats["p_to_jfk_km"], feats["p_to_lga_km"], feats["p_to_ewr_km"]
    )
    d_zone = assign_borough_zone(
        d_lat, d_lon, feats["d_to_jfk_km"], feats["d_to_lga_km"], feats["d_to_ewr_km"]
    )
    feats["pickup_borough_zone"] = p_zone.astype(np.float32)
    feats["dropoff_borough_zone"] = d_zone.astype(np.float32)
    feats["od_borough_pair"] = (p_zone * 9 + d_zone).astype(np.float32)

    # 5. Explicit Bridge/Tunnel Corridor Encodings & Estimated Statutory Tolls
    p_near_qmt = feats["p_to_queens_midtown_tunnel_km"] < 3.5
    d_near_qmt = feats["d_to_queens_midtown_tunnel_km"] < 3.5
    is_qmt = (feats["is_cross_east_river"] > 0) & (
        (p_near_qmt & ((d_zone == 2) | (d_zone == 5) | (d_zone == 6)))
        | (d_near_qmt & ((p_zone == 2) | (p_zone == 5) | (p_zone == 6)))
    )

    p_near_bbt = feats["p_to_brooklyn_battery_tunnel_km"] < 3.0
    d_near_bbt = feats["d_to_brooklyn_battery_tunnel_km"] < 3.0
    is_bbt = (feats["is_cross_east_river"] > 0) & (
        (p_near_bbt & (d_zone == 1)) | (d_near_bbt & (p_zone == 1))
    )

    p_near_rfk = feats["p_to_rfk_triborough_bridge_km"] < 3.5
    d_near_rfk = feats["d_to_rfk_triborough_bridge_km"] < 3.5
    is_rfk = (
        (p_near_rfk | d_near_rfk)
        & ((p_zone == 3) | (d_zone == 3) | (p_zone == 2) | (d_zone == 2))
        & (p_zone != d_zone)
    )

    is_lincoln = (
        ((p_in_manhattan & ((d_zone == 7) | (d_lon < -74.02)))
         | (d_in_manhattan & ((p_zone == 7) | (p_lon < -74.02))))
        & (feats["min_to_lincoln_tunnel_km"] < 4.0)
    )

    is_holland = (
        ((p_in_manhattan & ((d_zone == 7) | (d_lon < -74.02)))
         | (d_in_manhattan & ((p_zone == 7) | (p_lon < -74.02))))
        & (feats["min_to_holland_tunnel_km"] < 3.5)
        & (~is_lincoln)
    )

    is_verrazzano = (
        ((p_zone == 1) & (d_zone == 4))
        | ((p_zone == 4) & (d_zone == 1))
        | ((feats["min_to_verrazzano_bridge_km"] < 3.5) & ((p_lat < 40.65) | (d_lat < 40.65)))
    )

    is_toll_free = (feats["is_cross_east_river"] > 0) & (~is_qmt) & (~is_bbt) & (~is_rfk)

    feats["is_qmt_corridor"] = is_qmt.astype(np.float32)
    feats["is_qmt_to_queens"] = (is_qmt & p_in_manhattan).astype(np.float32)
    feats["is_qmt_to_manhattan"] = (is_qmt & d_in_manhattan).astype(np.float32)

    feats["is_bbt_corridor"] = is_bbt.astype(np.float32)
    feats["is_bbt_to_brooklyn"] = (is_bbt & p_in_manhattan).astype(np.float32)
    feats["is_bbt_to_manhattan"] = (is_bbt & d_in_manhattan).astype(np.float32)

    feats["is_rfk_corridor"] = is_rfk.astype(np.float32)
    feats["is_rfk_to_queens"] = (is_rfk & ((d_zone == 2) | (d_zone == 6))).astype(np.float32)
    feats["is_rfk_from_queens"] = (is_rfk & ((p_zone == 2) | (p_zone == 6))).astype(np.float32)

    feats["is_lincoln_corridor"] = is_lincoln.astype(np.float32)
    feats["is_lincoln_to_nj"] = (is_lincoln & p_in_manhattan).astype(np.float32)
    feats["is_lincoln_to_ny"] = (is_lincoln & d_in_manhattan).astype(np.float32)

    feats["is_holland_corridor"] = is_holland.astype(np.float32)
    feats["is_holland_to_nj"] = (is_holland & p_in_manhattan).astype(np.float32)
    feats["is_holland_to_ny"] = (is_holland & d_in_manhattan).astype(np.float32)

    feats["is_verrazzano_corridor"] = is_verrazzano.astype(np.float32)
    feats["is_verrazzano_to_si"] = (is_verrazzano & (d_zone == 4)).astype(np.float32)
    feats["is_verrazzano_to_bk"] = (is_verrazzano & (d_zone == 1)).astype(np.float32)

    feats["is_toll_free_cross"] = is_toll_free.astype(np.float32)

    # Statutory toll pricing lookup ($5.00-$16.00 step-function)
    est_toll = np.zeros(len(p_lat), dtype=np.float32)
    est_toll = np.where(is_verrazzano, 15.0, est_toll)
    est_toll = np.where(is_lincoln | is_holland, 12.5, est_toll)
    est_toll = np.where(is_qmt | is_bbt | is_rfk, 6.5, est_toll)
    feats["estimated_toll"] = est_toll.astype(np.float32)
    feats["is_tolled_crossing"] = (est_toll > 0.0).astype(np.float32)

    # 6. Fine-grained Spatial Micro-Clusters via 0.01-degree Coordinate Discretization
    p_lat_bin = np.floor((np.clip(p_lat, BB_MIN_LAT, BB_MAX_LAT) - BB_MIN_LAT) / 0.01).astype(np.int32)
    p_lon_bin = np.floor((np.clip(p_lon, BB_MIN_LON, BB_MAX_LON) - BB_MIN_LON) / 0.01).astype(np.int32)
    d_lat_bin = np.floor((np.clip(d_lat, BB_MIN_LAT, BB_MAX_LAT) - BB_MIN_LAT) / 0.01).astype(np.int32)
    d_lon_bin = np.floor((np.clip(d_lon, BB_MIN_LON, BB_MAX_LON) - BB_MIN_LON) / 0.01).astype(np.int32)

    p_micro = (p_lat_bin * 170 + p_lon_bin).astype(np.int32)
    d_micro = (d_lat_bin * 170 + d_lon_bin).astype(np.int32)
    feats["pickup_micro_cluster"] = p_micro.astype(np.float32)
    feats["dropoff_micro_cluster"] = d_micro.astype(np.float32)
    feats["od_micro_cluster"] = ((p_micro.astype(np.int64) * 30000 + d_micro.astype(np.int64)) % 1000003).astype(np.float32)

    # 7. Temporal Features
    dt = pd.to_datetime(
        df["pickup_datetime"].astype(str).str.slice(0, 19),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    year = dt.dt.year.fillna(2012).values.astype(np.float32)
    month = dt.dt.month.fillna(6).values.astype(np.float32)
    day = dt.dt.day.fillna(15).values.astype(np.float32)
    dayofweek = dt.dt.dayofweek.fillna(2).values.astype(np.float32)
    hour = dt.dt.hour.fillna(12).values.astype(np.float32)
    minute = dt.dt.minute.fillna(0).values.astype(np.float32)
    hour_cont = (hour + minute / 60.0).astype(np.float32)
    dayofyear = dt.dt.dayofyear.fillna(180).values.astype(np.float32)

    feats["year"] = year
    feats["month"] = month
    feats["day"] = day
    feats["dayofweek"] = dayofweek
    feats["hour"] = hour
    feats["is_weekend"] = (dayofweek >= 5).astype(np.float32)

    feats["time_elapsed_years"] = (
        (year - 2009.0) + (dayofyear - 1.0 + hour_cont / 24.0) / 365.25
    ).astype(np.float32)
    feats["week_hour_continuous"] = (dayofweek * 24.0 + hour_cont).astype(np.float32)

    # Cyclical encodings
    feats["hour_sin"] = np.sin(2 * np.pi * hour_cont / 24.0).astype(np.float32)
    feats["hour_cos"] = np.cos(2 * np.pi * hour_cont / 24.0).astype(np.float32)
    feats["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    feats["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    feats["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    feats["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)

    # 8. NYC TLC Regulatory Rules & Step Functions
    is_rush_hour = ((dayofweek < 5) & (hour >= 16) & (hour < 20)).astype(np.float32)
    is_overnight = ((hour >= 20) | (hour < 6)).astype(np.float32)
    is_post_2012_hike = ((year > 2012) | ((year == 2012) & (month >= 9))).astype(np.float32)

    feats["is_rush_hour"] = is_rush_hour
    feats["is_overnight"] = is_overnight
    feats["is_post_2012_hike"] = is_post_2012_hike
    feats["intra_manhattan_rush"] = (feats["is_intra_manhattan"] * is_rush_hour).astype(np.float32)
    feats["jfk_flat_fare_prior"] = np.where(
        feats["is_jfk_flat_rate"] == 1,
        np.where(is_post_2012_hike == 1, 52.0, 45.0),
        0.0,
    ).astype(np.float32)

    # 9. Passenger Counts & Interactions
    p_cnt = df["passenger_count"].fillna(1).clip(1, 6).values.astype(np.float32)
    feats["passenger_count"] = p_cnt
    feats["is_solo_passenger"] = (p_cnt == 1.0).astype(np.float32)

    # Interactions with distance
    feats["dist_x_rush"] = (feats["haversine_km"] * is_rush_hour).astype(np.float32)
    feats["dist_x_hike"] = (feats["haversine_km"] * is_post_2012_hike).astype(np.float32)
    feats["dist_x_overnight"] = (feats["haversine_km"] * is_overnight).astype(np.float32)

    # 10. Deterministic NYC TLC Tariff Meter Baseline Calculations
    km_to_miles = np.float32(0.62137119)
    dist_hav_miles = feats["haversine_km"] * km_to_miles
    dist_man_miles = feats["manhattan_km"] * km_to_miles

    is_post_sep4_2012 = (
        (year > 2012)
        | ((year == 2012) & ((month > 9) | ((month == 9) & (day >= 4))))
    )
    mileage_rate = np.where(is_post_sep4_2012, 2.50, 2.00).astype(np.float32)

    mta_surcharge = np.where(
        (year > 2009) | ((year == 2009) & (month >= 11)), 0.50, 0.0
    ).astype(np.float32)
    imp_surcharge = np.where(year >= 2015, 0.30, 0.0).astype(np.float32)
    rush_surcharge = np.where(is_rush_hour == 1.0, 1.00, 0.0).astype(np.float32)
    overnight_surcharge = np.where(is_overnight == 1.0, 0.50, 0.0).astype(np.float32)
    ewr_surcharge = np.where(feats["is_to_ewr"] == 1.0, 17.50, 0.0).astype(np.float32)

    fixed_meter_drop = (
        2.50
        + mta_surcharge
        + imp_surcharge
        + rush_surcharge
        + overnight_surcharge
        + ewr_surcharge
    ).astype(np.float32)

    feats["tlc_fare_haversine"] = (fixed_meter_drop + dist_hav_miles * mileage_rate).astype(
        np.float32
    )
    feats["tlc_fare_manhattan"] = (fixed_meter_drop + dist_man_miles * mileage_rate).astype(
        np.float32
    )

    return pd.DataFrame(feats, index=df.index)


def get_od_micro_pair(p_lat: np.ndarray, p_lon: np.ndarray, d_lat: np.ndarray, d_lon: np.ndarray) -> np.ndarray:
    """Discretize pickup/dropoff coordinates into 0.01-degree grid buckets to form OD interaction pairs."""
    p_lat_bin = np.floor((np.clip(p_lat, BB_MIN_LAT, BB_MAX_LAT) - BB_MIN_LAT) / 0.01).astype(np.int32)
    p_lon_bin = np.floor((np.clip(p_lon, BB_MIN_LON, BB_MAX_LON) - BB_MIN_LON) / 0.01).astype(np.int32)
    d_lat_bin = np.floor((np.clip(d_lat, BB_MIN_LAT, BB_MAX_LAT) - BB_MIN_LAT) / 0.01).astype(np.int32)
    d_lon_bin = np.floor((np.clip(d_lon, BB_MIN_LON, BB_MAX_LON) - BB_MIN_LON) / 0.01).astype(np.int32)
    p_micro = (p_lat_bin * 170 + p_lon_bin).astype(np.int32)
    d_micro = (d_lat_bin * 170 + d_lon_bin).astype(np.int32)
    return p_micro.astype(np.int64) * 30000 + d_micro.astype(np.int64)


def compute_oof_od_statistics(
    train_od: np.ndarray,
    y_train: np.ndarray,
    dist_train: np.ndarray,
    val_od: np.ndarray,
    test_od: np.ndarray,
    n_splits: int = 5,
    smoothing: float = 20.0,
    random_state: int = RANDOM_STATE,
):
    """Compute leak-free out-of-fold target statistics (mean fare and fare-per-distance per OD micro-zone pair)."""
    y_rate = (y_train / (dist_train + 0.1)).astype(np.float32)
    global_fare_mean = float(np.mean(y_train))
    global_rate_mean = float(np.mean(y_rate))

    train_oof_fare = np.full(len(y_train), global_fare_mean, dtype=np.float32)
    train_oof_rate = np.full(len(y_train), global_rate_mean, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, fold_idx in kf.split(train_od):
        f_od = train_od[train_idx]
        f_y = y_train[train_idx]
        f_rate = y_rate[train_idx]

        df_fold = pd.DataFrame({"od": f_od, "y": f_y, "rate": f_rate})
        grp = df_fold.groupby("od").agg(
            sum_y=("y", "sum"),
            sum_rate=("rate", "sum"),
            count=("y", "count"),
        )

        c = grp["count"].values
        smoothed_y = (grp["sum_y"].values + smoothing * global_fare_mean) / (c + smoothing)
        smoothed_rate = (grp["sum_rate"].values + smoothing * global_rate_mean) / (c + smoothing)

        map_y = dict(zip(grp.index.values, smoothed_y.astype(np.float32)))
        map_rate = dict(zip(grp.index.values, smoothed_rate.astype(np.float32)))

        out_od = train_od[fold_idx]
        train_oof_fare[fold_idx] = pd.Series(out_od).map(map_y).fillna(global_fare_mean).values.astype(np.float32)
        train_oof_rate[fold_idx] = pd.Series(out_od).map(map_rate).fillna(global_rate_mean).values.astype(np.float32)

    # Full training set aggregations strictly mapped to validation and test
    df_all = pd.DataFrame({"od": train_od, "y": y_train, "rate": y_rate})
    grp_all = df_all.groupby("od").agg(
        sum_y=("y", "sum"),
        sum_rate=("rate", "sum"),
        count=("y", "count"),
    )
    c_all = grp_all["count"].values
    smoothed_y_all = (grp_all["sum_y"].values + smoothing * global_fare_mean) / (c_all + smoothing)
    smoothed_rate_all = (grp_all["sum_rate"].values + smoothing * global_rate_mean) / (c_all + smoothing)

    map_y_all = dict(zip(grp_all.index.values, smoothed_y_all.astype(np.float32)))
    map_rate_all = dict(zip(grp_all.index.values, smoothed_rate_all.astype(np.float32)))

    val_fare = pd.Series(val_od).map(map_y_all).fillna(global_fare_mean).values.astype(np.float32)
    val_rate = pd.Series(val_od).map(map_rate_all).fillna(global_rate_mean).values.astype(np.float32)

    test_fare = pd.Series(test_od).map(map_y_all).fillna(global_fare_mean).values.astype(np.float32)
    test_rate = pd.Series(test_od).map(map_rate_all).fillna(global_rate_mean).values.astype(np.float32)

    del df_all, grp_all, y_rate
    gc.collect()

    return (train_oof_fare, train_oof_rate), (val_fare, val_rate), (test_fare, test_rate)


# -----------------------------------------------------------------------------
# 1. Data Loading & Quality Filtering
# -----------------------------------------------------------------------------
train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

raw_train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=N_ROWS_TO_LOAD,
    usecols=train_usecols,
    dtype={
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
)

# Domain-faithful cleaning rules
clean_mask = (
    raw_train_df["fare_amount"].notna()
    & (raw_train_df["fare_amount"] >= 2.50)
    & (raw_train_df["fare_amount"] <= 500.0)
    & (raw_train_df["pickup_longitude"] >= BB_MIN_LON)
    & (raw_train_df["pickup_longitude"] <= BB_MAX_LON)
    & (raw_train_df["dropoff_longitude"] >= BB_MIN_LON)
    & (raw_train_df["dropoff_longitude"] <= BB_MAX_LON)
    & (raw_train_df["pickup_latitude"] >= BB_MIN_LAT)
    & (raw_train_df["pickup_latitude"] <= BB_MAX_LAT)
    & (raw_train_df["dropoff_latitude"] >= BB_MIN_LAT)
    & (raw_train_df["dropoff_latitude"] <= BB_MAX_LAT)
    & (raw_train_df["passenger_count"] >= 1)
    & (raw_train_df["passenger_count"] <= 6)
)

clean_train_df = raw_train_df[clean_mask].reset_index(drop=True)
del raw_train_df
gc.collect()

# -----------------------------------------------------------------------------
# 2. Train/Validation Split (Leak-Free Holdout)
# -----------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
n_total = len(clean_train_df)
indices = rng.permutation(n_total)
n_val = int(0.05 * n_total)

val_idx = indices[:n_val]
train_idx = indices[n_val:]

train_split = clean_train_df.iloc[train_idx].reset_index(drop=True)
val_split = clean_train_df.iloc[val_idx].reset_index(drop=True)
del clean_train_df
gc.collect()

# -----------------------------------------------------------------------------
# 3. Feature Extraction on Train, Val, and Test Sets
# -----------------------------------------------------------------------------
test_df = pd.read_csv(TEST_PATH)
test_keys = test_df["key"].values

X_train = extract_features(train_split)
y_train = train_split["fare_amount"].values.astype(np.float32)

X_val = extract_features(val_split)
y_val = val_split["fare_amount"].values.astype(np.float32)

X_test = extract_features(test_df)

# Calculate discrete OD micro-pair identifiers for target statistics
train_od = get_od_micro_pair(
    train_split["pickup_latitude"].values,
    train_split["pickup_longitude"].values,
    train_split["dropoff_latitude"].values,
    train_split["dropoff_longitude"].values,
)
val_od = get_od_micro_pair(
    val_split["pickup_latitude"].values,
    val_split["pickup_longitude"].values,
    val_split["dropoff_latitude"].values,
    val_split["dropoff_longitude"].values,
)
test_od = get_od_micro_pair(
    test_df["pickup_latitude"].values,
    test_df["pickup_longitude"].values,
    test_df["dropoff_latitude"].values,
    test_df["dropoff_longitude"].values,
)

# Compute leak-free smoothed out-of-fold target statistics strictly on train folds
(tr_oof_fare, tr_oof_rate), (v_fare, v_rate), (te_fare, te_rate) = compute_oof_od_statistics(
    train_od=train_od,
    y_train=y_train,
    dist_train=X_train["haversine_km"].values,
    val_od=val_od,
    test_od=test_od,
    n_splits=5,
    smoothing=20.0,
    random_state=RANDOM_STATE,
)

X_train["od_oof_mean_fare"] = tr_oof_fare
X_train["od_oof_mean_fare_per_km"] = tr_oof_rate

X_val["od_oof_mean_fare"] = v_fare
X_val["od_oof_mean_fare_per_km"] = v_rate

X_test["od_oof_mean_fare"] = te_fare
X_test["od_oof_mean_fare_per_km"] = te_rate

feature_cols = list(X_train.columns)
X_test = X_test[feature_cols]

# Persist processed datasets & metadata
train_parquet_path = os.path.join(WORKING_DIR, "train_features.parquet")
val_parquet_path = os.path.join(WORKING_DIR, "val_features.parquet")
test_parquet_path = os.path.join(WORKING_DIR, "test_features.parquet")

X_train_to_save = X_train.assign(fare_amount=y_train)
X_train_to_save.to_parquet(train_parquet_path, index=False)
del X_train_to_save

X_val_to_save = X_val.assign(fare_amount=y_val)
X_val_to_save.to_parquet(val_parquet_path, index=False)
del X_val_to_save

X_test_to_save = X_test.copy()
X_test_to_save["key"] = test_keys
X_test_to_save.to_parquet(test_parquet_path, index=False)
del X_test_to_save

metadata = {
    "feature_columns": feature_cols,
    "target_column": "fare_amount",
    "train_samples": len(X_train),
    "val_samples": len(X_val),
    "test_samples": len(X_test),
}
with open(os.path.join(WORKING_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

del train_split, val_split, train_od, val_od, test_od
gc.collect()

# -----------------------------------------------------------------------------
# 4. Heterogeneous Dual-Engine Model Architecture & Sequential Training
# -----------------------------------------------------------------------------
# Model 1: GPU-accelerated depth-wise XGBoost with L2 regularization
xgb_params = {
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 10,
    "learning_rate": 0.06,
    "reg_lambda": 2.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "n_estimators": 2500,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
}

try:
    xgb_model = xgb.XGBRegressor(**xgb_params)
except Exception:
    xgb_params.pop("device", None)
    xgb_params["tree_method"] = "gpu_hist"
    xgb_model = xgb.XGBRegressor(**xgb_params)

xgb_model.fit(
    X_train[feature_cols],
    y_train,
    eval_set=[(X_val[feature_cols], y_val)],
    verbose=100,
)

# Model 2: High-capacity leaf-wise LightGBM on CPU
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3500,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)

training_callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=200),
]

lgb_model.fit(
    X_train[feature_cols],
    y_train,
    eval_set=[(X_val[feature_cols], y_val)],
    callbacks=training_callbacks,
)

# -----------------------------------------------------------------------------
# 5. NNLS Ensembling, Validation Evaluation & Model Persistence
# -----------------------------------------------------------------------------
val_xgb_preds = xgb_model.predict(X_val[feature_cols]).astype(np.float32)
val_lgb_preds = lgb_model.predict(X_val[feature_cols]).astype(np.float32)

val_xgb_rmse = float(root_mean_squared_error(y_val, val_xgb_preds))
val_lgb_rmse = float(root_mean_squared_error(y_val, val_lgb_preds))
print(f"XGBoost Validation RMSE: {val_xgb_rmse:.5f}")
print(f"LightGBM Validation RMSE: {val_lgb_rmse:.5f}")

# Analytical Non-Negative Least Squares (NNLS) convex blend
A_val = np.column_stack([val_xgb_preds, val_lgb_preds])
weights, _ = nnls(A_val, y_val)
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)
else:
    weights = np.array([0.5, 0.5], dtype=np.float32)

print(f"Optimal Ensemble Weights -> XGBoost: {weights[0]:.4f}, LightGBM: {weights[1]:.4f}")

val_blend_preds = A_val @ weights
val_rmse = float(root_mean_squared_error(y_val, val_blend_preds))

# Save models and ensemble artifacts
joblib.dump(xgb_model, os.path.join(WORKING_DIR, "best_xgb_model.joblib"))
joblib.dump(lgb_model, os.path.join(WORKING_DIR, "best_lgbm_model.joblib"))
with open(os.path.join(WORKING_DIR, "ensemble_weights.json"), "w") as f:
    json.dump({"xgb_weight": float(weights[0]), "lgb_weight": float(weights[1])}, f, indent=2)

# -----------------------------------------------------------------------------
# 6. Blended Test Inference & Submission Generation
# -----------------------------------------------------------------------------
test_xgb_preds = xgb_model.predict(X_test[feature_cols]).astype(np.float32)
test_lgb_preds = lgb_model.predict(X_test[feature_cols]).astype(np.float32)

A_test = np.column_stack([test_xgb_preds, test_lgb_preds])
test_raw_preds = A_test @ weights
test_preds = np.clip(test_raw_preds, 2.50, 500.0)

submission_df = pd.DataFrame(
    {
        "key": test_keys,
        "fare_amount": test_preds.astype(np.float64),
    }
)
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Score: {val_rmse:.5f}")
