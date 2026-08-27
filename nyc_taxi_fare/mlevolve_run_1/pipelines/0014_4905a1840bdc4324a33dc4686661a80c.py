import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb

print("Starting end-to-end ML pipeline...")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load a robust sample of train.csv in chunks to prevent OOM
train_chunks = []
chunk_size = 500000
total_rows_to_load = 10000000
loaded_rows = 0

for chunk in pd.read_csv("./input/train.csv", chunksize=chunk_size):
    chunk = chunk.dropna(subset=["cost", "origin_x", "origin_y", "dest_x", "dest_y"])
    valid_mask = (
        (chunk["origin_x"] >= -180)
        & (chunk["origin_x"] <= 180)
        & (chunk["origin_y"] >= -90)
        & (chunk["origin_y"] <= 90)
        & (chunk["dest_x"] >= -180)
        & (chunk["dest_x"] <= 180)
        & (chunk["dest_y"] >= -90)
        & (chunk["dest_y"] <= 90)
        & (chunk["cost"] >= 0)
    )
    chunk = chunk[valid_mask]
    train_chunks.append(chunk)
    loaded_rows += len(chunk)
    if loaded_rows >= total_rows_to_load:
        break

train_df = pd.concat(train_chunks, ignore_index=True)
if len(train_df) > total_rows_to_load:
    train_df = train_df.iloc[:total_rows_to_load]

test_df = pd.read_csv("./input/test.csv")

print(f"Train shape after cleaning: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. Train / Validation Split (80/20) prior to feature engineering
np.random.seed(42)
shuffled_indices = np.random.permutation(len(train_df))
split_idx = int(len(train_df) * 0.8)

train_idx = shuffled_indices[:split_idx]
val_idx = shuffled_indices[split_idx:]

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)


# 3. Feature Engineering function
def engineer_features(df, origin_km=None, dest_km=None, o_x_freq=None, o_y_freq=None, d_x_freq=None, d_y_freq=None):
    df = df.copy()
    if "start_time" in df.columns:
        dt = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
        df["hour"] = dt.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["month"] = dt.dt.month.fillna(1).astype(int)
        df["year"] = dt.dt.year.fillna(2012).astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    dx = df["dest_x"] - df["origin_x"]
    dy = df["dest_y"] - df["origin_y"]
    df["euclidean_dist"] = np.sqrt(dx**2 + dy**2)
    df["abs_dx"] = np.abs(dx)
    df["abs_dy"] = np.abs(dy)
    df["manhattan_dist"] = np.abs(dx) + np.abs(dy)
    df["distance_per_unit"] = df["euclidean_dist"] / (df["unit_count"] + 1e-5)
    df["origin_x_bin"] = np.round(df["origin_x"], 2)
    df["origin_y_bin"] = np.round(df["origin_y"], 2)
    df["dest_x_bin"] = np.round(df["dest_x"], 2)
    df["dest_y_bin"] = np.round(df["dest_y"], 2)

    if origin_km is not None:
        df["origin_cluster"] = origin_km.predict(df[["origin_x", "origin_y"]])
    if dest_km is not None:
        df["dest_cluster"] = dest_km.predict(df[["dest_x", "dest_y"]])

    if o_x_freq is not None:
        df["origin_x_bin_freq"] = df["origin_x_bin"].map(o_x_freq).fillna(0)
    if o_y_freq is not None:
        df["origin_y_bin_freq"] = df["origin_y_bin"].map(o_y_freq).fillna(0)
    if d_x_freq is not None:
        df["dest_x_bin_freq"] = df["dest_x_bin"].map(d_x_freq).fillna(0)
    if d_y_freq is not None:
        df["dest_y_bin_freq"] = df["dest_y_bin"].map(d_y_freq).fillna(0)

    lat1 = np.radians(df["origin_y"])
    lon1 = np.radians(df["origin_x"])
    lat2 = np.radians(df["dest_y"])
    lon2 = np.radians(df["dest_x"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    df["haversine_dist"] = c * 6371.0

    y_bearing = np.sin(dlon) * np.cos(lat2)
    x_bearing = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y_bearing, x_bearing))

    df = df.fillna(0)
    return df


print("Fitting KMeans and frequency encodings on train split...")
origin_kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=10)
origin_kmeans.fit(train_split[["origin_x", "origin_y"]])
dest_kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=10)
dest_kmeans.fit(train_split[["dest_x", "dest_y"]])

train_orig_x_bin = np.round(train_split["origin_x"], 2)
train_orig_y_bin = np.round(train_split["origin_y"], 2)
train_dest_x_bin = np.round(train_split["dest_x"], 2)
train_dest_y_bin = np.round(train_split["dest_y"], 2)

o_x_freq = train_orig_x_bin.value_counts(normalize=True).to_dict()
o_y_freq = train_orig_y_bin.value_counts(normalize=True).to_dict()
d_x_freq = train_dest_x_bin.value_counts(normalize=True).to_dict()
d_y_freq = train_dest_y_bin.value_counts(normalize=True).to_dict()

print("Engineering features for train split...")
train_split = engineer_features(train_split, origin_kmeans, dest_kmeans, o_x_freq, o_y_freq, d_x_freq, d_y_freq)
print("Engineering features for val split...")
val_split = engineer_features(val_split, origin_kmeans, dest_kmeans, o_x_freq, o_y_freq, d_x_freq, d_y_freq)
print("Engineering features for test set...")
test_df = engineer_features(test_df, origin_kmeans, dest_kmeans, o_x_freq, o_y_freq, d_x_freq, d_y_freq)

# Save processed splits
train_split.to_parquet("./working/train_split.parquet", index=False)
val_split.to_parquet("./working/val_split.parquet", index=False)
test_df.to_parquet("./working/test_df.parquet", index=False)

# 4. Model Design & Training (Ensemble of CatBoostRegressor, LGBMRegressor, and XGBRegressor)
print("Initializing CatBoostRegressor, LGBMRegressor, and XGBRegressor...")
cb_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.04,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    task_type="CPU",
    thread_count=-1,
    verbose=False,
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.04,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.04,
    max_depth=8,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

exclude_cols = ["record_id", "start_time", "cost"]
feature_cols = [c for c in train_split.columns if c not in exclude_cols]

X_train = train_split[feature_cols]
y_train = train_split["cost"]
X_val = val_split[feature_cols]
y_val = val_split["cost"]

print("Training CatBoost regressor...")
cb_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False,
)

print("Training LGBM regressor...")
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)

print("Training XGB regressor...")
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Generating predictions on validation set...")
cb_val_preds = cb_model.predict(X_val)
lgb_val_preds = lgb_model.predict(X_val)
xgb_val_preds = xgb_model.predict(X_val)
val_preds = (cb_val_preds + lgb_val_preds + xgb_val_preds) / 3.0
score = np.sqrt(mean_squared_error(y_val, val_preds))

print("Generating predictions on test set...")
cb_test_preds = cb_model.predict(test_df[feature_cols])
lgb_test_preds = lgb_model.predict(test_df[feature_cols])
xgb_test_preds = xgb_model.predict(test_df[feature_cols])
test_preds = (cb_test_preds + lgb_test_preds + xgb_test_preds) / 3.0
test_preds = np.clip(test_preds, 0, None)

submission = pd.DataFrame({"record_id": test_df["record_id"], "cost": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved successfully to ./submission/submission.csv")

print(f"Final Validation Score: {score}")
