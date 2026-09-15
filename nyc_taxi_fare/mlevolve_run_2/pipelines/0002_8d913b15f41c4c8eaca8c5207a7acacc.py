import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
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
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NYC Metropolitan coordinate boundaries and fare bounds
LON_MIN, LON_MAX = -74.45, -72.80
LAT_MIN, LAT_MAX = 40.45, 41.85
FARE_MIN, FARE_MAX = 2.50, 400.00

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
    abs_dlon = np.abs(dlon)
    abs_dlat = np.abs(dlat)
    features["abs_diff_lon"] = abs_dlon
    features["abs_diff_lat"] = abs_dlat
    features["euclidean_dist"] = np.sqrt(abs_dlon**2 + abs_dlat**2).astype(np.float32)

    # 2. Haversine Great-Circle Distance
    hav_km = haversine_km(p_lon, p_lat, d_lon, d_lat)
    features["haversine_km"] = hav_km
    features["log_haversine"] = np.log1p(hav_km)

    # 3. Rotated Manhattan Grid Distance (29 degrees grid rotation)
    rot_lon = dlon * COS_29 - dlat * SIN_29
    rot_lat = dlon * SIN_29 + dlat * COS_29
    manhattan_km = ((np.abs(rot_lon) + np.abs(rot_lat)) * 111.0).astype(np.float32)
    features["manhattan_rot_km"] = manhattan_km
    features["log_manhattan"] = np.log1p(manhattan_km)

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

    # Regulated Manhattan <-> JFK flat rate route indicator
    features["is_manhattan_jfk_flat_route"] = (
        ((features["pickup_to_midtown"] < 5.0) & (features["dropoff_to_jfk"] < 3.0))
        | ((features["dropoff_to_midtown"] < 5.0) & (features["pickup_to_jfk"] < 3.0))
    ).astype(np.float32)

    # 6. Bridge & Tunnel Funneling Detour (East/Hudson River water barrier crossing)
    choke_min_routes = np.full(len(df), np.inf, dtype=np.float32)
    for c_lon, c_lat in CHOKEPOINTS:
        p_to_c = haversine_km(p_lon, p_lat, c_lon, c_lat)
        c_to_d = haversine_km(c_lon, c_lat, d_lon, d_lat)
        route_dist = p_to_c + c_to_d
        choke_min_routes = np.minimum(choke_min_routes, route_dist)
    features["chokepoint_route_km"] = choke_min_routes
    features["chokepoint_excess_km"] = np.maximum(0.0, choke_min_routes - hav_km)

    # 7. Passenger Count & Interaction
    p_count = df["passenger_count"].values.astype(np.float32)
    p_count = np.clip(p_count, 1.0, 6.0)
    features["passenger_count"] = p_count
    features["dist_x_passenger"] = (hav_km * p_count).astype(np.float32)

    # 8. Micro-Spatial Quantization (2km grid binning)
    p_grid_lon = np.floor(p_lon * 50.0).astype(np.int32)
    p_grid_lat = np.floor(p_lat * 50.0).astype(np.int32)
    d_grid_lon = np.floor(d_lon * 50.0).astype(np.int32)
    d_grid_lat = np.floor(d_lat * 50.0).astype(np.int32)
    features["is_same_grid"] = (
        (p_grid_lon == d_grid_lon) & (p_grid_lat == d_grid_lat)
    ).astype(np.float32)

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
    features["year"] = year.astype(np.float32)
    features["month"] = month.astype(np.float32)
    features["dayofweek"] = dow.astype(np.float32)
    features["hour_fraction"] = hour_fraction
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

    # NYC TLC Mandated Surcharge Flags
    features["tlc_peak_surcharge"] = ((dow < 5) & (hour >= 16) & (hour < 20)).astype(
        np.float32
    )
    features["tlc_overnight_surcharge"] = ((hour >= 20) | (hour < 6)).astype(np.float32)
    hike_ts = pd.Timestamp("2012-09-04", tz="UTC")
    features["is_post_2012_rate_hike"] = (dt_utc >= hike_ts).values.astype(np.float32)
    mta_ts = pd.Timestamp("2009-11-01", tz="UTC")
    features["is_post_mta_tax"] = (dt_utc >= mta_ts).values.astype(np.float32)

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
# Chunked Balanced Ingestion of Training Data (~10M rows)
# -------------------------------------------------------------------------
print("Starting uniform chunked data ingestion from train.csv...")
chunk_size = 1_000_000
samples_per_chunk = 180_000
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
col_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

chunk_count = 0
for chunk in pd.read_csv(
    TRAIN_CSV, chunksize=chunk_size, usecols=use_cols, dtype=col_dtypes
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

    if len(filtered) > samples_per_chunk:
        sampled = filtered.sample(
            n=samples_per_chunk, random_state=RANDOM_SEED + chunk_count
        )
    else:
        sampled = filtered

    sampled_chunks.append(sampled)

raw_train_df = pd.concat(sampled_chunks, ignore_index=True)
del sampled_chunks
print(f"Loaded and cleaned {len(raw_train_df):,} trips across {chunk_count} chunks.")

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
# Model Architecture: Deep Tabular Residual Network (TaxiFareResNet)
# -------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    """Pre-LayerNorm Residual Block for continuous tabular representation learning."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class TaxiFareResNet(nn.Module):
    """Deep Tabular ResNet tailored for continuous NYC Taxi fare regression."""

    def __init__(
        self,
        in_features: int,
        d_model: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        self.blocks = nn.ModuleList(
            [
                ResNetBlock(d_model=d_model, expansion=2, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


# -------------------------------------------------------------------------
# Feature Preprocessing & Leak-Free Normalization
# -------------------------------------------------------------------------
X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
y_train = train_df["fare_amount"].to_numpy(dtype=np.float32)

X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val = val_df["fare_amount"].to_numpy(dtype=np.float32)

X_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)

X_train_raw = np.nan_to_num(X_train_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
X_val_raw = np.nan_to_num(X_val_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
X_test_raw = np.nan_to_num(X_test_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler(copy=False)
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

joblib.dump(scaler, os.path.join(WORKING_DIR, "feature_scaler.joblib"))

# -------------------------------------------------------------------------
# PyTorch DataLoaders
# -------------------------------------------------------------------------
batch_size = 8192
val_batch_size = 16384

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test))

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

# -------------------------------------------------------------------------
# Model Instantiation & Training Setup
# -------------------------------------------------------------------------
in_features = len(feature_cols)
model = TaxiFareResNet(
    in_features=in_features,
    d_model=256,
    n_blocks=4,
    dropout=0.1,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epochs = 6
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-5
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"Initialized TaxiFareResNet with {in_features} inputs and {total_params:,} parameters on {device}."
)

# -------------------------------------------------------------------------
# Training Loop with Early Model Checkpointing
# -------------------------------------------------------------------------
best_val_rmse = float("inf")
best_model_path = os.path.join(WORKING_DIR, "best_taxifare_resnet.pt")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for bx, by in train_loader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(by)
        total_samples += len(by)

    scheduler.step()
    epoch_train_loss = running_loss / total_samples

    # Validation pass
    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device, non_blocking=True)
            batch_preds = model(bx)
            val_preds_list.append(batch_preds.cpu().numpy())

    epoch_val_preds = np.concatenate(val_preds_list)
    epoch_val_preds = np.clip(epoch_val_preds, 2.50, 400.0)
    epoch_val_rmse = float(np.sqrt(np.mean((epoch_val_preds - y_val) ** 2)))

    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch + 1:02d}/{num_epochs:02d} - Train Loss: {epoch_train_loss:.4f} - Val RMSE: {epoch_val_rmse:.4f} - LR: {current_lr:.6f}"
    )

    if epoch_val_rmse < best_val_rmse:
        best_val_rmse = epoch_val_rmse
        torch.save(model.state_dict(), best_model_path)

# -------------------------------------------------------------------------
# Best Model Evaluation & Holdout Validation Metric
# -------------------------------------------------------------------------
model.load_state_dict(
    torch.load(best_model_path, map_location=device, weights_only=True)
)
model.eval()

final_val_preds_list = []
with torch.no_grad():
    for bx, _ in val_loader:
        bx = bx.to(device, non_blocking=True)
        batch_preds = model(bx)
        final_val_preds_list.append(batch_preds.cpu().numpy())

final_val_preds = np.concatenate(final_val_preds_list)
# Apply NYC TLC regulatory minimum fare threshold ($2.50 base flag drop)
final_val_preds = np.clip(final_val_preds, 2.50, 400.0)
final_val_rmse = float(np.sqrt(np.mean((final_val_preds - y_val) ** 2)))

# -------------------------------------------------------------------------
# Test Set Inference and Submission Export
# -------------------------------------------------------------------------
test_preds_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device, non_blocking=True)
        batch_preds = model(bx)
        test_preds_list.append(batch_preds.cpu().numpy())

test_preds = np.concatenate(test_preds_list)
test_preds = np.clip(test_preds, 2.50, 400.0)

submission_df = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": test_preds.astype(np.float64)}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
assert len(submission_df) == 9914, f"Expected 9,914 rows, got {len(submission_df)}"
assert submission_df["fare_amount"].isna().sum() == 0, "Submission contains NaN values"

# -------------------------------------------------------------------------
# Final Score Output (Mandatory Last Line)
# -------------------------------------------------------------------------
print(f"Final Validation Score: {final_val_rmse:.5f}")
