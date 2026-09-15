import os
import gc
import math
import random
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
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

    # Newark Airport Interstate Surcharge Indicator
    df["is_ewr_trip"] = ((df["p_dist_ewr"] < 3.5) | (df["d_dist_ewr"] < 3.5)).astype(
        np.float32
    )

    # High-density midtown zone indicator and distance interaction
    is_midtown_zone = ((df["p_dist_midtown"] < 2.5) | (df["d_dist_midtown"] < 2.5)).astype(
        np.float32
    )
    df["is_midtown_zone"] = is_midtown_zone
    df["haversine_midtown"] = (df["haversine_km"] * is_midtown_zone).astype(np.float32)

    # Official TLC Newark Airport Interstate Surcharge ($17.50)
    df["newark_surcharge"] = (df["is_ewr_trip"] * 17.50).astype(np.float32)

    # East River Crossing Indicator (detours and tolls)
    df["east_river_crossing"] = (
        ((p_lon < -73.965) & (d_lon > -73.965))
        | ((p_lon > -73.965) & (d_lon < -73.965))
    ).astype(np.float32)

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

    # Theoretical Base Fare Prior (Domain Knowledge Integration)
    miles = df["haversine_km"] * 0.621371
    meter_rate_per_mile = np.where(is_post_sept_2012, 2.50, 2.00)
    df["theoretical_fare"] = (
        2.50
        + (miles * meter_rate_per_mile)
        + df["rush_hour_surcharge"]
        + df["overnight_surcharge"]
        + df["newark_surcharge"]
        + df["jfk_rush_surcharge"]
        + 0.50
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

    n_sample_rows = 40_000_000
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
        "is_jfk_manhattan",
        "is_ewr_trip",
        "is_midtown_zone",
        "haversine_midtown",
        "newark_surcharge",
        "east_river_crossing",
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
        "theoretical_fare",
        "passenger_count",
        "is_solo_passenger",
    ]

    X_train = df_train[feature_cols]
    y_train = df_train["fare_amount"].to_numpy(dtype=np.float32)

    X_val = df_val[feature_cols]
    y_val = df_val["fare_amount"].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols]
    test_keys = test_df["key"].to_numpy()

    del df_train, df_val
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, test_keys


# -------------------------------------------------------------------------
# Step 2: Training, Evaluation & Optimal Blending
# -------------------------------------------------------------------------
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, test_keys = load_and_preprocess_data()

    print(
        f"Dataset shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    # Initialize and train high-capacity LightGBM Regressor (leaf-wise best-first growth)
    print("Training LightGBM Regressor (leaf-wise best-first growth)...")
    lgb_model = lgb.LGBMRegressor(
        num_leaves=255,
        learning_rate=0.06,
        min_child_samples=50,
        colsample_bytree=0.85,
        subsample=0.85,
        subsample_freq=1,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        objective="regression",
    )

    try:
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )
    except Exception:
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=30,
            verbose=False,
        )

    val_preds_lgb = lgb_model.predict(X_val)
    test_preds_lgb = lgb_model.predict(X_test)
    val_rmse_lgb = root_mean_squared_error(y_val, val_preds_lgb)
    print(f"LightGBM Val RMSE: {val_rmse_lgb:.4f}")

    # Initialize and train GPU-Accelerated XGBoost Regressor (depth-wise growth)
    print("Training GPU-Accelerated XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.07,
        max_depth=11,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=25,
        reg_lambda=2.0,
        reg_alpha=0.5,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds_xgb = xgb_model.predict(X_val)
    test_preds_xgb = xgb_model.predict(X_test)
    val_rmse_xgb = root_mean_squared_error(y_val, val_preds_xgb)
    print(f"XGBoost Val RMSE: {val_rmse_xgb:.4f}")

    # Initialize and train GPU-Accelerated CatBoost Regressor (oblivious decision trees)
    print("Training GPU-Accelerated CatBoost Regressor (oblivious decision trees)...")
    try:
        cat_model = CatBoostRegressor(
            depth=8,
            learning_rate=0.08,
            iterations=1000,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            random_seed=42,
            early_stopping_rounds=30,
            verbose=100,
        )
        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100,
        )
    except Exception as e:
        print(f"CatBoost GPU initialization encountered: {e}. Falling back to CPU...")
        cat_model = CatBoostRegressor(
            depth=8,
            learning_rate=0.08,
            iterations=1000,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="CPU",
            random_seed=42,
            early_stopping_rounds=30,
            verbose=100,
        )
        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100,
        )

    val_preds_cat = cat_model.predict(X_val)
    test_preds_cat = cat_model.predict(X_test)
    val_rmse_cat = root_mean_squared_error(y_val, val_preds_cat)
    print(f"CatBoost Val RMSE: {val_rmse_cat:.4f}")

    # Optimal Convex Blending: Solve analytical non-negative least-squares weights on validation set
    print("Solving optimal non-negative least-squares blend weights...")
    val_matrix = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_cat])
    weights, _ = nnls(val_matrix, y_val)
    weight_sum = np.sum(weights)
    if weight_sum > 1e-8:
        weights = weights / weight_sum
    else:
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    val_preds = (
        weights[0] * val_preds_lgb
        + weights[1] * val_preds_xgb
        + weights[2] * val_preds_cat
    )
    test_preds = (
        weights[0] * test_preds_lgb
        + weights[1] * test_preds_xgb
        + weights[2] * test_preds_cat
    )

    # Consistent regulatory post-processing: NYC TLC legal minimum fare is $2.50
    val_preds = np.clip(val_preds, 2.50, None)
    test_preds = np.clip(test_preds, 2.50, None)

    val_rmse = root_mean_squared_error(y_val, val_preds)
    print(
        f"Optimal Blend Weights (LGB: {weights[0]:.4f}, XGB: {weights[1]:.4f}, Cat: {weights[2]:.4f}) | Blended Val RMSE: {val_rmse:.4f}"
    )

    # Export Submission and Verify Integrity
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
