import gc
import math
import os
import time
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

# -----------------------------------------------------------------------------
# Configuration & Domain Constants
# -----------------------------------------------------------------------------
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Training sample budget: 40M rows provides massive statistical power while
# ensuring rapid execution within the available compute limits
N_ROWS_TO_LOAD = 40_000_000
VAL_SIZE = 250_000
RANDOM_SEED = 42

# Manhattan Street Grid: 29.0 degrees clockwise from true North
MANHATTAN_ANGLE_RAD = 29.0 * np.pi / 180.0
COS_MANHATTAN = np.cos(MANHATTAN_ANGLE_RAD)
SIN_MANHATTAN = np.sin(MANHATTAN_ANGLE_RAD)
REF_LON, REF_LAT = -73.9850, 40.7550
KM_PER_LAT = 111.03
KM_PER_LON = 84.14

# NYC Landmark, Airport & Toll Facility Coordinates (lon, lat)
AIRPORTS = {
    "jfk": (-73.7781, 40.6413),
    "lga": (-73.8740, 40.7769),
    "ewr": (-74.1745, 40.6895),
}
LANDMARKS = {
    "midtown": (-73.9850, 40.7550),
    "times_sq": (-73.9855, 40.7580),
    "grand_central": (-73.9772, 40.7527),
    "penn_station": (-73.9935, 40.7505),
    "lower_manhattan": (-74.0090, 40.7070),
    "brooklyn_downtown": (-73.9780, 40.6830),
}
TOLL_FACILITIES = {
    "queens_midtown": (-73.9680, 40.7445),
    "rfk_bridge": (-73.9242, 40.7800),
    "lincoln_tunnel": (-74.0110, 40.7620),
    "holland_tunnel": (-74.0210, 40.7270),
    "gwb": (-73.9525, 40.8517),
    "verrazzano": (-74.0450, 40.6066),
    "brooklyn_battery": (-74.0135, 40.7000),
}

# Global fitted MiniBatchKMeans instance for spatial clustering
KMEANS_MODEL = None

# NYC Coordinate filter bounds
MIN_LON, MAX_LON = -74.45, -72.80
MIN_LAT, MAX_LAT = 40.45, 41.85
MIN_FARE, MAX_FARE = 2.50, 200.00
MIN_PASSENGERS, MAX_PASSENGERS = 1, 6


# -----------------------------------------------------------------------------
# Feature Engineering Functions
# -----------------------------------------------------------------------------
def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized Haversine distance in kilometers."""
    r_km = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * r_km * np.arcsin(np.sqrt(a))


def bearing_np(lon1, lat1, lon2, lat2):
    """Vectorized compass bearing in degrees (0 - 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Computes spatial, rotational Manhattan grid, TLC tariff, and temporal features."""
    # Robustly parse ISO datetime string (first 19 chars: YYYY-MM-DD HH:MM:SS)
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.slice(0, 19)
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .alias("dt")
    )

    # Calendar and time components
    df = df.with_columns(
        [
            pl.col("dt").dt.year().cast(pl.Int16).alias("year"),
            pl.col("dt").dt.month().cast(pl.Int8).alias("month"),
            pl.col("dt").dt.day().cast(pl.Int8).alias("day"),
            pl.col("dt").dt.weekday().cast(pl.Int8).alias("dayofweek"),
            pl.col("dt").dt.hour().cast(pl.Int8).alias("hour"),
            pl.col("dt").dt.minute().cast(pl.Int8).alias("minute"),
        ]
    )

    # Fractional hour and continuous elapsed days since 2009-01-01
    df = df.with_columns(
        [
            (
                pl.col("hour").cast(pl.Float32)
                + pl.col("minute").cast(pl.Float32) / 60.0
            ).alias("hour_float"),
            (
                (pl.col("year").cast(pl.Float32) - 2009.0) * 365.25
                + (pl.col("month").cast(pl.Float32) - 1.0) * 30.4375
                + pl.col("day").cast(pl.Float32)
            ).alias("days_elapsed"),
        ]
    )

    # Cyclical encodings
    pi2 = 2.0 * np.pi
    df = df.with_columns(
        [
            (pl.col("hour_float") * (pi2 / 24.0))
            .sin()
            .cast(pl.Float32)
            .alias("sin_hour"),
            (pl.col("hour_float") * (pi2 / 24.0))
            .cos()
            .cast(pl.Float32)
            .alias("cos_hour"),
            (pl.col("dayofweek").cast(pl.Float32) * (pi2 / 7.0))
            .sin()
            .cast(pl.Float32)
            .alias("sin_dow"),
            (pl.col("dayofweek").cast(pl.Float32) * (pi2 / 7.0))
            .cos()
            .cast(pl.Float32)
            .alias("cos_dow"),
            ((pl.col("month").cast(pl.Float32) - 1.0) * (pi2 / 12.0))
            .sin()
            .cast(pl.Float32)
            .alias("sin_month"),
            ((pl.col("month").cast(pl.Float32) - 1.0) * (pi2 / 12.0))
            .cos()
            .cast(pl.Float32)
            .alias("cos_month"),
        ]
    )

    # NYC TLC Tariff Schedules & Regulations
    df = df.with_columns(
        [
            (pl.col("dt") >= pl.datetime(2012, 9, 4))
            .cast(pl.Int8)
            .alias("is_post_sept2012"),
            (pl.col("dt") >= pl.datetime(2009, 11, 1))
            .cast(pl.Int8)
            .alias("is_post_nov2009"),
            (
                (pl.col("dayofweek") <= 5)
                & (pl.col("hour") >= 16)
                & (pl.col("hour") < 20)
            )
            .cast(pl.Int8)
            .alias("is_peak_rush"),
            ((pl.col("hour") >= 20) | (pl.col("hour") < 6))
            .cast(pl.Int8)
            .alias("is_overnight"),
            (pl.col("dayofweek") >= 6).cast(pl.Int8).alias("is_weekend"),
        ]
    )

    p_lon = df["pickup_longitude"].to_numpy().astype(np.float64)
    p_lat = df["pickup_latitude"].to_numpy().astype(np.float64)
    d_lon = df["dropoff_longitude"].to_numpy().astype(np.float64)
    d_lat = df["dropoff_latitude"].to_numpy().astype(np.float64)

    # Geodesic Distances & Bearings
    haversine_km = haversine_np(p_lon, p_lat, d_lon, d_lat).astype(np.float32)
    bearing_deg = bearing_np(p_lon, p_lat, d_lon, d_lat).astype(np.float32)
    diff_lon = (d_lon - p_lon).astype(np.float32)
    diff_lat = (d_lat - p_lat).astype(np.float32)
    abs_diff_lon = np.abs(diff_lon)
    abs_diff_lat = np.abs(diff_lat)
    euclidean_dist = np.sqrt(diff_lon**2 + diff_lat**2).astype(np.float32)

    mid_lon = ((p_lon + d_lon) / 2.0).astype(np.float32)
    mid_lat = ((p_lat + d_lat) / 2.0).astype(np.float32)

    # Isotropic Equirectangular Projection Centered at Midtown NYC
    p_x_km = (p_lon - REF_LON) * KM_PER_LON
    p_y_km = (p_lat - REF_LAT) * KM_PER_LAT
    d_x_km = (d_lon - REF_LON) * KM_PER_LON
    d_y_km = (d_lat - REF_LAT) * KM_PER_LAT

    # Rotated Metric Manhattan Grid Coordinates (29.0 degrees clockwise)
    p_street_km = (p_x_km * COS_MANHATTAN - p_y_km * SIN_MANHATTAN).astype(np.float32)
    p_avenue_km = (p_x_km * SIN_MANHATTAN + p_y_km * COS_MANHATTAN).astype(np.float32)
    d_street_km = (d_x_km * COS_MANHATTAN - d_y_km * SIN_MANHATTAN).astype(np.float32)
    d_avenue_km = (d_x_km * SIN_MANHATTAN + d_y_km * COS_MANHATTAN).astype(np.float32)

    # Decompose Travel Vectors into Signed & Unsigned Street- and Avenue-Parallel Distances
    diff_street_km = (d_street_km - p_street_km).astype(np.float32)
    diff_avenue_km = (d_avenue_km - p_avenue_km).astype(np.float32)
    abs_diff_street_km = np.abs(diff_street_km).astype(np.float32)
    abs_diff_avenue_km = np.abs(diff_avenue_km).astype(np.float32)
    manhattan_rot_km = (abs_diff_street_km + abs_diff_avenue_km).astype(np.float32)

    tortuosity = (manhattan_rot_km + 0.05) / (haversine_km + 0.05)
    tortuosity = np.clip(tortuosity, 1.0, 5.0).astype(np.float32)

    # Analytical TLC Statutory Tariff Baseline
    # Grid distance in miles (1 km ≈ 0.621371 miles)
    grid_dist_miles = (manhattan_rot_km * 0.621371).astype(np.float32)
    is_post_sept2012_arr = df["is_post_sept2012"].to_numpy().astype(np.int8)
    is_post_nov2009_arr = df["is_post_nov2009"].to_numpy().astype(np.int8)
    is_peak_rush_arr = df["is_peak_rush"].to_numpy().astype(np.int8)
    is_overnight_arr = df["is_overnight"].to_numpy().astype(np.int8)

    mileage_rate = np.where(is_post_sept2012_arr == 1, 2.50, 2.00).astype(np.float32)
    statutory_base_fare = (
        2.50
        + mileage_rate * grid_dist_miles
        + np.where(is_post_nov2009_arr == 1, 0.50, 0.0).astype(np.float32)
        + np.where(is_overnight_arr == 1, 0.50, 0.0).astype(np.float32)
        + np.where(is_peak_rush_arr == 1, 1.00, 0.0).astype(np.float32)
    ).astype(np.float32)

    feature_dict = {
        "haversine_km": haversine_km,
        "bearing_deg": bearing_deg,
        "diff_lon": diff_lon,
        "diff_lat": diff_lat,
        "abs_diff_lon": abs_diff_lon,
        "abs_diff_lat": abs_diff_lat,
        "euclidean_dist": euclidean_dist,
        "mid_lon": mid_lon,
        "mid_lat": mid_lat,
        "p_street_km": p_street_km,
        "p_avenue_km": p_avenue_km,
        "d_street_km": d_street_km,
        "d_avenue_km": d_avenue_km,
        "diff_street_km": diff_street_km,
        "diff_avenue_km": diff_avenue_km,
        "abs_diff_street_km": abs_diff_street_km,
        "abs_diff_avenue_km": abs_diff_avenue_km,
        "manhattan_rot_km": manhattan_rot_km,
        "tortuosity": tortuosity,
        "grid_dist_miles": grid_dist_miles,
        "statutory_base_fare": statutory_base_fare,
    }

    # Transit hubs & landmarks
    for code, (a_lon, a_lat) in AIRPORTS.items():
        dist_p = haversine_np(p_lon, p_lat, a_lon, a_lat).astype(np.float32)
        dist_d = haversine_np(d_lon, d_lat, a_lon, a_lat).astype(np.float32)
        feature_dict[f"{code}_pickup_dist"] = dist_p
        feature_dict[f"{code}_dropoff_dist"] = dist_d
        feature_dict[f"{code}_min_dist"] = np.minimum(dist_p, dist_d)

    for code, (l_lon, l_lat) in LANDMARKS.items():
        dist_p = haversine_np(p_lon, p_lat, l_lon, l_lat).astype(np.float32)
        dist_d = haversine_np(d_lon, d_lat, l_lon, l_lat).astype(np.float32)
        feature_dict[f"{code}_pickup_dist"] = dist_p
        feature_dict[f"{code}_dropoff_dist"] = dist_d

    # Major toll facilities
    for code, (t_lon, t_lat) in TOLL_FACILITIES.items():
        dist_p = haversine_np(p_lon, p_lat, t_lon, t_lat).astype(np.float32)
        dist_d = haversine_np(d_lon, d_lat, t_lon, t_lat).astype(np.float32)
        feature_dict[f"{code}_pickup_dist"] = dist_p
        feature_dict[f"{code}_dropoff_dist"] = dist_d
        feature_dict[f"{code}_min_dist"] = np.minimum(dist_p, dist_d)

    # Directional radial distance differences (dropoff_dist - pickup_dist for hubs)
    for hub in ["midtown", "jfk", "lga", "ewr"]:
        feature_dict[f"{hub}_dist_diff"] = (
            feature_dict[f"{hub}_dropoff_dist"] - feature_dict[f"{hub}_pickup_dist"]
        ).astype(np.float32)

    # Expanded JFK statutory boundary (3.8 km radius)
    feature_dict["is_jfk_trip"] = (
        (feature_dict["jfk_pickup_dist"] < 3.8)
        | (feature_dict["jfk_dropoff_dist"] < 3.8)
    ).astype(np.int8)
    feature_dict["is_lga_trip"] = (
        (feature_dict["lga_pickup_dist"] < 2.5)
        | (feature_dict["lga_dropoff_dist"] < 2.5)
    ).astype(np.int8)
    feature_dict["is_ewr_trip"] = (
        (feature_dict["ewr_pickup_dist"] < 2.5)
        | (feature_dict["ewr_dropoff_dist"] < 2.5)
    ).astype(np.int8)
    feature_dict["is_any_airport"] = (
        feature_dict["is_jfk_trip"]
        | feature_dict["is_lga_trip"]
        | feature_dict["is_ewr_trip"]
    ).astype(np.int8)

    # Expanded Manhattan boundaries up to latitude 40.88
    p_in_manhattan = (
        (p_lat >= 40.70) & (p_lat <= 40.88) & (p_lon >= -74.02) & (p_lon <= -73.93)
    )
    d_in_manhattan = (
        (d_lat >= 40.70) & (d_lat <= 40.88) & (d_lon >= -74.02) & (d_lon <= -73.93)
    )
    feature_dict["is_intra_manhattan"] = (p_in_manhattan & d_in_manhattan).astype(np.int8)
    feature_dict["is_manhattan_to_outer"] = (
        (p_in_manhattan & (~d_in_manhattan)) | ((~p_in_manhattan) & d_in_manhattan)
    ).astype(np.int8)
    feature_dict["is_jfk_flat_route"] = (
        (p_in_manhattan & (feature_dict["jfk_dropoff_dist"] < 3.8))
        | (d_in_manhattan & (feature_dict["jfk_pickup_dist"] < 3.8))
    ).astype(np.int8)

    # Domain-specific TLC Statutory Tariff Features
    jfk_flat_fare = np.where(
        feature_dict["is_jfk_flat_route"] == 1,
        np.where(is_post_sept2012_arr == 1, 52.0, 45.0),
        0.0,
    ).astype(np.float32)
    feature_dict["jfk_flat_fare"] = jfk_flat_fare

    ewr_surcharge = np.where(
        feature_dict["ewr_dropoff_dist"] < 2.5,
        17.50,
        0.0,
    ).astype(np.float32)
    feature_dict["ewr_surcharge"] = ewr_surcharge

    airport_tariff_baseline = np.where(
        feature_dict["is_jfk_flat_route"] == 1,
        jfk_flat_fare
        + np.where(is_post_nov2009_arr == 1, 0.50, 0.0).astype(np.float32)
        + np.where(is_peak_rush_arr == 1, 4.50, 0.0).astype(np.float32),
        np.where(
            feature_dict["ewr_dropoff_dist"] < 2.5,
            statutory_base_fare + 17.50,
            statutory_base_fare,
        ),
    ).astype(np.float32)
    feature_dict["airport_tariff_baseline"] = airport_tariff_baseline

    # Spatial River Crossing Indicators (East River & Hudson River)
    p_in_bk_qn = (p_lon > -73.95) & (p_lat <= 40.78)
    d_in_bk_qn = (d_lon > -73.95) & (d_lat <= 40.78)
    feature_dict["is_east_river_crossing"] = (
        (p_in_manhattan & d_in_bk_qn) | (d_in_manhattan & p_in_bk_qn)
    ).astype(np.int8)

    p_in_nj = p_lon < -74.03
    d_in_nj = d_lon < -74.03
    feature_dict["is_hudson_river_crossing"] = (
        ((p_lon >= -74.02) & d_in_nj) | ((d_lon >= -74.02) & p_in_nj)
    ).astype(np.int8)

    feature_dict["p_grid_lat"] = np.round(p_lat, 2).astype(np.float32)
    feature_dict["p_grid_lon"] = np.round(p_lon, 2).astype(np.float32)
    feature_dict["d_grid_lat"] = np.round(d_lat, 2).astype(np.float32)
    feature_dict["d_grid_lon"] = np.round(d_lon, 2).astype(np.float32)

    # 64-Center MiniBatchKMeans Spatial Cluster Distance Features
    global KMEANS_MODEL
    p_coords = np.column_stack([p_lon, p_lat])
    d_coords = np.column_stack([d_lon, d_lat])

    if KMEANS_MODEL is None:
        rng = np.random.RandomState(RANDOM_SEED)
        n_samples = min(len(df), 1_000_000)
        idx = (
            rng.choice(len(df), size=n_samples, replace=False)
            if len(df) > n_samples
            else np.arange(len(df))
        )
        train_sample_coords = np.vstack([p_coords[idx], d_coords[idx]])
        kmeans = MiniBatchKMeans(
            n_clusters=64,
            batch_size=10000,
            random_state=RANDOM_SEED,
            n_init=3,
            max_iter=100,
        )
        kmeans.fit(train_sample_coords)
        KMEANS_MODEL = kmeans

    centers_f32 = KMEANS_MODEL.cluster_centers_.astype(np.float32)

    def compute_cluster_dists(coords, centers):
        n = len(coords)
        chunk_size = 500_000
        dists = np.empty((n, len(centers)), dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = coords[start:end]
            diff = chunk[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists[start:end] = np.sqrt(np.sum(diff**2, axis=2), dtype=np.float32)
        return dists

    p_cluster_dists = compute_cluster_dists(p_coords.astype(np.float32), centers_f32)
    d_cluster_dists = compute_cluster_dists(d_coords.astype(np.float32), centers_f32)

    p_cluster_id = np.argmin(p_cluster_dists, axis=1).astype(np.int8)
    d_cluster_id = np.argmin(d_cluster_dists, axis=1).astype(np.int8)
    p_min_cluster_dist = np.min(p_cluster_dists, axis=1).astype(np.float32)
    d_min_cluster_dist = np.min(d_cluster_dists, axis=1).astype(np.float32)

    p_cluster_dists_copy = p_cluster_dists.copy()
    d_cluster_dists_copy = d_cluster_dists.copy()
    rows = np.arange(len(df))
    p_cluster_dists_copy[rows, p_cluster_id] = np.inf
    d_cluster_dists_copy[rows, d_cluster_id] = np.inf
    p_2nd_cluster_dist = np.min(p_cluster_dists_copy, axis=1).astype(np.float32)
    d_2nd_cluster_dist = np.min(d_cluster_dists_copy, axis=1).astype(np.float32)
    del p_cluster_dists_copy, d_cluster_dists_copy

    p_c_lon = centers_f32[p_cluster_id, 0]
    p_c_lat = centers_f32[p_cluster_id, 1]
    d_c_lon = centers_f32[d_cluster_id, 0]
    d_c_lat = centers_f32[d_cluster_id, 1]
    cluster_center_dist = haversine_np(p_c_lon, p_c_lat, d_c_lon, d_c_lat).astype(np.float32)

    feature_dict["p_min_cluster_dist"] = p_min_cluster_dist
    feature_dict["d_min_cluster_dist"] = d_min_cluster_dist
    feature_dict["p_2nd_cluster_dist"] = p_2nd_cluster_dist
    feature_dict["d_2nd_cluster_dist"] = d_2nd_cluster_dist
    feature_dict["cluster_center_dist"] = cluster_center_dist
    feature_dict["is_same_cluster"] = (p_cluster_id == d_cluster_id).astype(np.int8)

    for k in range(8):
        feature_dict[f"p_c_dist_{k}"] = p_cluster_dists[:, k]
        feature_dict[f"d_c_dist_{k}"] = d_cluster_dists[:, k]

    del p_cluster_dists, d_cluster_dists, p_coords, d_coords

    new_cols_df = pl.DataFrame(feature_dict)
    df = pl.concat([df, new_cols_df], how="horizontal")
    df = df.drop("dt")

    df = df.with_columns(
        [
            pl.col("passenger_count").cast(pl.Int8).alias("passenger_count"),
            (pl.col("passenger_count") == 1).cast(pl.Int8).alias("is_single_passenger"),
            (pl.col("haversine_km") * pl.col("passenger_count").cast(pl.Float32)).alias(
                "dist_x_passengers"
            ),
        ]
    )

    return df


# -----------------------------------------------------------------------------
# Main Pipeline Execution
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()

    # Step 1: Load Test Dataset
    test_df = pl.read_csv(TEST_PATH)
    test_keys = test_df["key"].to_list()

    # Step 2: Load and Clean Training Dataset
    train_raw = pl.read_csv(
        TRAIN_PATH,
        n_rows=N_ROWS_TO_LOAD,
        schema_overrides={
            "key": pl.Utf8,
            "fare_amount": pl.Float32,
            "pickup_datetime": pl.Utf8,
            "pickup_longitude": pl.Float64,
            "pickup_latitude": pl.Float64,
            "dropoff_longitude": pl.Float64,
            "dropoff_latitude": pl.Float64,
            "passenger_count": pl.Int16,
        },
    )

    clean_train = train_raw.filter(
        pl.col("dropoff_longitude").is_not_null()
        & pl.col("dropoff_latitude").is_not_null()
        & (pl.col("fare_amount") >= MIN_FARE)
        & (pl.col("fare_amount") <= MAX_FARE)
        & (pl.col("passenger_count") >= MIN_PASSENGERS)
        & (pl.col("passenger_count") <= MAX_PASSENGERS)
        & (pl.col("pickup_longitude") >= MIN_LON)
        & (pl.col("pickup_longitude") <= MAX_LON)
        & (pl.col("pickup_latitude") >= MIN_LAT)
        & (pl.col("pickup_latitude") <= MAX_LAT)
        & (pl.col("dropoff_longitude") >= MIN_LON)
        & (pl.col("dropoff_longitude") <= MAX_LON)
        & (pl.col("dropoff_latitude") >= MIN_LAT)
        & (pl.col("dropoff_latitude") <= MAX_LAT)
    )
    del train_raw
    gc.collect()

    # Refined Physical Cleaning: Retain valid zero-distance trips ($2.50 to $25.00),
    # eliminate distance > 2 km with fare < $3.00, and non-airport distance < 0.2 km with fare > $75.00
    p_lon_np = clean_train["pickup_longitude"].to_numpy()
    p_lat_np = clean_train["pickup_latitude"].to_numpy()
    d_lon_np = clean_train["dropoff_longitude"].to_numpy()
    d_lat_np = clean_train["dropoff_latitude"].to_numpy()
    fare_np = clean_train["fare_amount"].to_numpy()

    trip_dist_km = haversine_np(p_lon_np, p_lat_np, d_lon_np, d_lat_np).astype(np.float32)

    is_zero_dist = (trip_dist_km < 0.05) | (
        (p_lon_np == d_lon_np) & (p_lat_np == d_lat_np)
    )
    invalid_zero_dist = is_zero_dist & (fare_np > 25.00)
    contradictory_cheap = (trip_dist_km > 2.0) & (fare_np < 3.00)

    is_near_airport = (
        (((p_lon_np - -73.7781) * 84.14)**2 + ((p_lat_np - 40.6413) * 111.03)**2 < 3.8**2)
        | (((d_lon_np - -73.7781) * 84.14)**2 + ((d_lat_np - 40.6413) * 111.03)**2 < 3.8**2)
        | (((p_lon_np - -73.8740) * 84.14)**2 + ((p_lat_np - 40.7769) * 111.03)**2 < 3.8**2)
        | (((d_lon_np - -73.8740) * 84.14)**2 + ((d_lat_np - 40.7769) * 111.03)**2 < 3.8**2)
        | (((p_lon_np - -74.1745) * 84.14)**2 + ((p_lat_np - 40.6895) * 111.03)**2 < 3.8**2)
        | (((d_lon_np - -74.1745) * 84.14)**2 + ((d_lat_np - 40.6895) * 111.03)**2 < 3.8**2)
    )
    contradictory_expensive = (~is_near_airport) & (trip_dist_km < 0.2) & (fare_np > 75.00)

    keep_mask = ~(invalid_zero_dist | contradictory_cheap | contradictory_expensive)
    clean_train = clean_train.filter(keep_mask)

    del p_lon_np, p_lat_np, d_lon_np, d_lat_np, fare_np, trip_dist_km
    del is_zero_dist, invalid_zero_dist, contradictory_cheap, is_near_airport, contradictory_expensive, keep_mask
    gc.collect()

    # Step 3: Strict Isolation Train/Validation Split
    clean_train = clean_train.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
    val_raw = clean_train.slice(0, VAL_SIZE)
    train_raw = clean_train.slice(VAL_SIZE, clean_train.shape[0] - VAL_SIZE)
    del clean_train
    gc.collect()

    # Step 4: Feature Engineering (Fit K-Means strictly on train_split)
    train_split = engineer_features(train_raw)
    del train_raw
    gc.collect()

    val_split = engineer_features(val_raw)
    del val_raw
    gc.collect()

    test_feat = engineer_features(test_df)
    del test_df
    gc.collect()

    metadata_cols = ["key", "fare_amount", "pickup_datetime"]
    feature_cols = [c for c in train_split.columns if c not in metadata_cols]

    # Convert directly to float32 NumPy arrays (unscaled for tree models)
    X_train = train_split.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_split["fare_amount"].to_numpy().astype(np.float32)

    X_val = val_split.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_split["fare_amount"].to_numpy().astype(np.float32)

    X_test = test_feat.select(feature_cols).to_numpy().astype(np.float32)

    del train_split, val_split, test_feat
    gc.collect()

    # Clean non-finite entries
    np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 5: Train High-Capacity LightGBM Leaf-Wise Regressor
    lgb_train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_val_data = lgb.Dataset(
        X_val, label=y_val, reference=lgb_train_data, free_raw_data=False
    )

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "device": "gpu",
        "learning_rate": 0.08,
        "num_leaves": 512,
        "max_depth": 13,
        "min_child_samples": 100,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbose": -1,
    }

    try:
        lgb_model = lgb.train(
            lgb_params,
            lgb_train_data,
            num_boost_round=1600,
            valid_sets=[lgb_val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    except Exception:
        lgb_params["device"] = "cpu"
        lgb_params["n_jobs"] = 32
        lgb_model = lgb.train(
            lgb_params,
            lgb_train_data,
            num_boost_round=1600,
            valid_sets=[lgb_val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

    val_preds_lgb = np.clip(lgb_model.predict(X_val), 2.50, None)
    test_preds_lgb = np.clip(lgb_model.predict(X_test), 2.50, None)

    del lgb_train_data, lgb_val_data, lgb_model
    gc.collect()

    # Step 6: Train High-Capacity GPU-Accelerated XGBoost Regressor
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 12,
        "learning_rate": 0.08,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 5.0,
        "random_state": 42,
    }

    try:
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1600,
            evals=[(dval, "val")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
    except Exception:
        xgb_params_fallback = dict(xgb_params)
        xgb_params_fallback.pop("device", None)
        xgb_params_fallback["tree_method"] = "gpu_hist"
        try:
            xgb_model = xgb.train(
                xgb_params_fallback,
                dtrain,
                num_boost_round=1600,
                evals=[(dval, "val")],
                early_stopping_rounds=40,
                verbose_eval=False,
            )
        except Exception:
            xgb_params_fallback["tree_method"] = "hist"
            xgb_params_fallback["n_jobs"] = 32
            xgb_model = xgb.train(
                xgb_params_fallback,
                dtrain,
                num_boost_round=1600,
                evals=[(dval, "val")],
                early_stopping_rounds=40,
                verbose_eval=False,
            )

    val_preds_xgb = np.clip(xgb_model.predict(dval), 2.50, None).astype(np.float32)
    test_preds_xgb = np.clip(xgb_model.predict(dtest), 2.50, None).astype(np.float32)

    del dtrain, dval, dtest, xgb_model
    gc.collect()

    # Step 7: Train High-Capacity GPU-Accelerated CatBoost Regressor
    cb_params = {
        "iterations": 1600,
        "learning_rate": 0.08,
        "depth": 8,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "GPU",
        "random_seed": 42,
        "verbose": False,
    }

    try:
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=40,
            verbose=False,
        )
    except Exception:
        cb_params_cpu = dict(cb_params)
        cb_params_cpu["task_type"] = "CPU"
        cb_params_cpu["thread_count"] = 32
        cb_model = CatBoostRegressor(**cb_params_cpu)
        cb_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=40,
            verbose=False,
        )

    val_preds_cb = np.clip(cb_model.predict(X_val), 2.50, None).astype(np.float32)
    test_preds_cb = np.clip(cb_model.predict(X_test), 2.50, None).astype(np.float32)

    del X_train, cb_model
    gc.collect()

    # Step 8: Optimal 3D Simplex Convex Blending on Holdout Validation Split
    best_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    best_ensemble_mse = float("inf")
    step = 0.01
    n_steps = 101

    for i in range(n_steps):
        w1 = i * step
        for j in range(n_steps - i):
            w2 = j * step
            w3 = 1.0 - w1 - w2
            candidate_blend = np.clip(
                w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_cb, 2.50, None
            )
            diff = candidate_blend - y_val
            mse = float(np.dot(diff, diff) / len(diff))
            if mse < best_ensemble_mse:
                best_ensemble_mse = mse
                best_weights = (w1, w2, w3)

    w1_best, w2_best, w3_best = best_weights
    val_preds_final = np.clip(
        w1_best * val_preds_lgb + w2_best * val_preds_xgb + w3_best * val_preds_cb,
        2.50,
        None,
    )
    test_preds_final = np.clip(
        w1_best * test_preds_lgb + w2_best * test_preds_xgb + w3_best * test_preds_cb,
        2.50,
        None,
    )

    final_val_rmse = float(np.sqrt(np.mean((val_preds_final - y_val) ** 2)))

    # Step 8: Persist Submission and Run Integrity Assertions
    submission_df = pl.DataFrame(
        {"key": test_keys, "fare_amount": test_preds_final.astype(np.float64)}
    )
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.write_csv(submission_path)

    assert os.path.exists(submission_path), "Submission file does not exist!"
    assert (
        len(submission_df) == 9914
    ), f"Expected 9,914 rows, found {len(submission_df)}"
    assert list(submission_df.columns) == [
        "key",
        "fare_amount",
    ], f"Invalid submission headers: {submission_df.columns}"
    assert (
        not submission_df["fare_amount"].is_null().any()
    ), "Submission contains null values!"
    assert (
        not submission_df["fare_amount"].is_nan().any()
    ), "Submission contains NaN values!"
    assert (
        submission_df["fare_amount"] >= 2.50
    ).all(), "Predictions violate statutory minimum fare!"

    print(f"Final Validation Score: {final_val_rmse:.4f}")


if __name__ == "__main__":
    main()