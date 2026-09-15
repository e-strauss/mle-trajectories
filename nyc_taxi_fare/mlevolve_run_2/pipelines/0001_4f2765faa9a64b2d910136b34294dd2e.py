import gc
import json
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

N_ROWS_TO_LOAD = 10_000_000
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


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive rich domain-specific geospatial, temporal, and interaction features."""
    feats = pd.DataFrame(index=df.index)

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
    feats["is_lga_trip"] = (
        (feats["p_to_lga_km"] < 2.5) | (feats["d_to_lga_km"] < 2.5)
    ).astype(np.int8)

    # 4. Temporal Features
    dt = pd.to_datetime(
        df["pickup_datetime"].astype(str).str.slice(0, 19),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    year = dt.dt.year.values.astype(np.int16)
    month = dt.dt.month.values.astype(np.int8)
    day = dt.dt.day.values.astype(np.int8)
    dayofweek = dt.dt.dayofweek.values.astype(np.int8)
    hour = dt.dt.hour.values.astype(np.int8)
    minute = dt.dt.minute.values.astype(np.int8)
    hour_cont = (hour + minute / 60.0).astype(np.float32)

    feats["year"] = year
    feats["month"] = month
    feats["day"] = day
    feats["dayofweek"] = dayofweek
    feats["hour"] = hour
    feats["is_weekend"] = (dayofweek >= 5).astype(np.int8)

    # Cyclical encodings
    feats["hour_sin"] = np.sin(2 * np.pi * hour_cont / 24.0).astype(np.float32)
    feats["hour_cos"] = np.cos(2 * np.pi * hour_cont / 24.0).astype(np.float32)
    feats["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    feats["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0).astype(np.float32)
    feats["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    feats["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)

    # 5. NYC TLC Regulatory Rules
    is_rush_hour = ((dayofweek < 5) & (hour >= 16) & (hour < 20)).astype(np.int8)
    is_overnight = ((hour >= 20) | (hour < 6)).astype(np.int8)
    is_post_2012_hike = ((year > 2012) | ((year == 2012) & (month >= 9))).astype(
        np.int8
    )

    feats["is_rush_hour"] = is_rush_hour
    feats["is_overnight"] = is_overnight
    feats["is_post_2012_hike"] = is_post_2012_hike

    # 6. Passenger Counts & Interactions
    p_cnt = df["passenger_count"].fillna(1).clip(1, 6).values.astype(np.int8)
    feats["passenger_count"] = p_cnt
    feats["is_solo_passenger"] = (p_cnt == 1).astype(np.int8)

    # Interactions with distance
    feats["dist_x_rush"] = (feats["haversine_km"] * is_rush_hour).astype(np.float32)
    feats["dist_x_hike"] = (feats["haversine_km"] * is_post_2012_hike).astype(
        np.float32
    )
    feats["dist_x_overnight"] = (feats["haversine_km"] * is_overnight).astype(
        np.float32
    )

    return feats


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

feature_cols = list(X_train.columns)
X_test = X_test[feature_cols]

# Persist processed datasets & metadata
train_parquet_path = os.path.join(WORKING_DIR, "train_features.parquet")
val_parquet_path = os.path.join(WORKING_DIR, "val_features.parquet")
test_parquet_path = os.path.join(WORKING_DIR, "test_features.parquet")

X_train_to_save = X_train.copy()
X_train_to_save["fare_amount"] = y_train
X_train_to_save.to_parquet(train_parquet_path, index=False)
del X_train_to_save

X_val_to_save = X_val.copy()
X_val_to_save["fare_amount"] = y_val
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

del train_split, val_split
gc.collect()

# -----------------------------------------------------------------------------
# 4. Model Architecture & Training
# -----------------------------------------------------------------------------
model_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 1200,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

model = lgb.LGBMRegressor(**model_params)

training_callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=200),
]

model.fit(
    X_train[feature_cols],
    y_train,
    eval_set=[(X_val[feature_cols], y_val)],
    callbacks=training_callbacks,
)

# -----------------------------------------------------------------------------
# 5. Model Persistence & Validation Evaluation
# -----------------------------------------------------------------------------
best_model_path = os.path.join(WORKING_DIR, "best_lgbm_model.joblib")
joblib.dump(model, best_model_path)
best_model = joblib.load(best_model_path)

val_raw_preds = best_model.predict(X_val[feature_cols])
val_preds = np.clip(val_raw_preds, 2.50, 500.0)
val_rmse = float(root_mean_squared_error(y_val, val_preds))

# -----------------------------------------------------------------------------
# 6. Test Inference & Submission Generation
# -----------------------------------------------------------------------------
test_raw_preds = best_model.predict(X_test[feature_cols])
test_preds = np.clip(test_raw_preds, 2.50, 500.0)

submission_df = pd.DataFrame(
    {
        "key": test_keys,
        "fare_amount": test_preds.astype(np.float64),
    }
)
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Score: {val_rmse:.5f}")
