import os
import gc
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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

    # TLC Regulatory Surcharges:
    # Rush-hour: $1.00 weekdays 16:00 to 20:00
    is_rush_hour = (dayofweeks < 5) & (hours >= 16) & (hours < 20)
    df["rush_hour_surcharge"] = is_rush_hour.astype(np.float32)

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
        + 0.50
    ).astype(np.float32)

    # Passenger count feature
    df["passenger_count"] = df["passenger_count"].astype(np.float32)
    df["is_solo_passenger"] = (df["passenger_count"] == 1.0).astype(np.float32)

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

    n_sample_rows = 12_000_000
    train_df = pd.read_csv(train_path, nrows=n_sample_rows, usecols=use_cols)

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
        "is_jfk_manhattan",
        "is_ewr_trip",
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
        "rush_hour_surcharge",
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
# Step 2: Model Architecture Design
# -------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Pre-LayerNorm residual block with SiLU activations and dropout regularization
    for smooth representation of spatial-topological manifolds.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.norm1(self.fc1(x))))
        out = self.drop2(self.act2(self.norm2(self.fc2(out))))
        return out + residual


class PhysicsResidualSpatialNet(nn.Module):
    """
    Physics-Guided Residual Spatial Manifold Network.
    Routes deterministic regulatory tariffs through a direct linear prior stream,
    while deep residual blocks learn non-linear spatial residuals.
    """

    def __init__(
        self,
        in_features: int = 47,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.in_features = in_features

        # Direct linear pathway: learns global physics scaling and regulatory base fares
        self.linear_prior = nn.Linear(in_features, 1)

        # Deep non-linear spatial manifold stream
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = self.linear_prior(x)
        feat = self.input_proj(x)
        for block in self.res_blocks:
            feat = block(feat)
        residual = self.head(feat)
        return (prior + residual).squeeze(1)


# -------------------------------------------------------------------------
# Step 3: Training, Evaluation & Optimal Blending
# -------------------------------------------------------------------------
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val, X_test, test_keys = load_and_preprocess_data()

    # Leak-Free Feature Normalization for Neural Architecture
    scaler = StandardScaler()
    X_train_np = (
        X_train.to_numpy(dtype=np.float32)
        if hasattr(X_train, "to_numpy")
        else np.asarray(X_train, dtype=np.float32)
    )
    X_val_np = (
        X_val.to_numpy(dtype=np.float32)
        if hasattr(X_val, "to_numpy")
        else np.asarray(X_val, dtype=np.float32)
    )
    X_test_np = (
        X_test.to_numpy(dtype=np.float32)
        if hasattr(X_test, "to_numpy")
        else np.asarray(X_test, dtype=np.float32)
    )

    X_train_scaled = scaler.fit_transform(X_train_np).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_np).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_np).astype(np.float32)

    train_x_tensor = torch.from_numpy(X_train_scaled)
    train_y_tensor = torch.from_numpy(y_train)
    val_x_tensor = torch.from_numpy(X_val_scaled).to(device)
    val_y_tensor = torch.from_numpy(y_val).to(device)
    test_x_tensor = torch.from_numpy(X_test_scaled).to(device)

    # Initialize PhysicsResidualSpatialNet
    model = PhysicsResidualSpatialNet(
        in_features=47, hidden_dim=256, num_blocks=3, dropout=0.10
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)

    epochs = 5
    batch_size = 16384
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_val_rmse = float("inf")
    best_model_path = "./working/best_physics_residual_net.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = running_loss / max(1, num_batches)

        # Fast validation pass on GPU
        model.eval()
        val_preds_batches = []
        with torch.no_grad():
            for idx in range(0, len(val_x_tensor), 32768):
                batch_val_x = val_x_tensor[idx : idx + 32768]
                val_preds_batches.append(model(batch_val_x))
            epoch_val_preds = torch.cat(val_preds_batches, dim=0)
            epoch_val_rmse = torch.sqrt(
                torch.mean((epoch_val_preds - val_y_tensor) ** 2)
            ).item()

        if epoch_val_rmse < best_val_rmse:
            best_val_rmse = epoch_val_rmse
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {epoch_val_rmse:.4f}"
        )

    # Release training tensors from host memory
    del train_loader, train_dataset, train_x_tensor, train_y_tensor, X_train_scaled
    gc.collect()

    # Load best checkpoint and compute neural network predictions
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_list = []
    with torch.no_grad():
        for idx in range(0, len(val_x_tensor), 32768):
            val_preds_list.append(model(val_x_tensor[idx : idx + 32768]).cpu().numpy())
    val_preds_nn = np.concatenate(val_preds_list, axis=0)

    with torch.no_grad():
        test_preds_nn = model(test_x_tensor).cpu().numpy()

    # Release GPU tensors prior to XGBoost execution
    del val_x_tensor, val_y_tensor, test_x_tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    val_rmse_nn = root_mean_squared_error(y_val, val_preds_nn)
    print(f"PhysicsResidualSpatialNet Best Val RMSE: {val_rmse_nn:.4f}")

    # Initialize and train GPU-Accelerated XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.08,
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

    # Optimal Convex Blending: Solve analytical least-squares weight on validation set
    diff = val_preds_xgb - val_preds_nn
    denom = float(np.dot(diff, diff))
    if denom > 1e-8:
        alpha = float(np.clip(np.dot(y_val - val_preds_nn, diff) / denom, 0.0, 1.0))
    else:
        alpha = 0.50

    val_preds = alpha * val_preds_xgb + (1.0 - alpha) * val_preds_nn
    test_preds = alpha * test_preds_xgb + (1.0 - alpha) * test_preds_nn

    # Consistent regulatory post-processing: NYC TLC legal minimum fare is $2.50
    val_preds = np.clip(val_preds, 2.50, None)
    test_preds = np.clip(test_preds, 2.50, None)

    val_rmse = root_mean_squared_error(y_val, val_preds)
    print(
        f"Optimal Blend Weight (XGBoost: {alpha:.4f}, NN: {1.0 - alpha:.4f}) | Blended Val RMSE: {val_rmse:.4f}"
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
