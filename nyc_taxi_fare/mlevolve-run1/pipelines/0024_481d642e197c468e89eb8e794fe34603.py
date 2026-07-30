import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Loading and processing test dataset...")
test_df = pd.read_csv("./input/test.csv")


def engineer_features(df):
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["hour"] = df["start_time"].dt.hour
    df["dayofweek"] = df["start_time"].dt.dayofweek
    df["month"] = df["start_time"].dt.month
    df["year"] = df["start_time"].dt.year
    df["day"] = df["start_time"].dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    for col in ["origin_x", "origin_y", "dest_x", "dest_y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(-180, 180)
            df[col] = df[col].fillna(0)

    df["origin_x_bin"] = np.round(df["origin_x"], 2)
    df["origin_y_bin"] = np.round(df["origin_y"], 2)
    df["dest_x_bin"] = np.round(df["dest_x"], 2)
    df["dest_y_bin"] = np.round(df["dest_y"], 2)

    df["dx"] = df["dest_x"] - df["origin_x"]
    df["dy"] = df["dest_y"] - df["origin_y"]
    df["euclidean_dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["manhattan_dist"] = np.abs(df["dx"]) + np.abs(df["dy"])
    df["bearing"] = np.arctan2(df["dy"], df["dx"])
    if "unit_count" in df.columns:
        df["unit_count_x_dist"] = df["unit_count"] * df["euclidean_dist"]
        df["unit_count_x_manhattan"] = df["unit_count"] * df["manhattan_dist"]
        df["unit_count_x_hour"] = df["unit_count"] * df["hour"]

    return df


test_df = engineer_features(test_df)
test_df.to_parquet("./working/test_processed.parquet")
print(f"Processed test set shape: {test_df.shape}")

print("Loading and processing training dataset in chunks...")
chunk_size = 2_000_000
train_chunks = []
total_rows_processed = 0

for chunk in pd.read_csv("./input/train.csv", chunksize=chunk_size):
    chunk = chunk[chunk["cost"] > 0].copy()
    chunk = chunk[chunk["cost"] < 50000].copy()

    for col in ["origin_x", "origin_y", "dest_x", "dest_y"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            chunk = chunk[(chunk[col] >= -180) & (chunk[col] <= 180)]

    chunk = engineer_features(chunk)

    if len(chunk) > 400_000:
        chunk = chunk.sample(n=400_000, random_state=42)

    train_chunks.append(chunk)
    total_rows_processed += len(chunk)
    print(f"Processed chunk, accumulated rows: {total_rows_processed}")

    if total_rows_processed >= 25_000_000:
        break

train_df = pd.concat(train_chunks, ignore_index=True)
print(f"Combined training data shape: {train_df.shape}")

from sklearn.cluster import KMeans

print("Fitting KMeans spatial clusters...")
kmeans_origin = KMeans(n_clusters=20, random_state=42, n_init=10)
kmeans_dest = KMeans(n_clusters=20, random_state=42, n_init=10)

train_origin_coords = train_df[["origin_x", "origin_y"]].values
train_dest_coords = train_df[["dest_x", "dest_y"]].values

kmeans_origin.fit(train_origin_coords)
kmeans_dest.fit(train_dest_coords)

train_df["origin_cluster"] = kmeans_origin.predict(train_origin_coords)
train_df["dest_cluster"] = kmeans_dest.predict(train_dest_coords)

train_origin_centroids = kmeans_origin.cluster_centers_[train_df["origin_cluster"]]
train_df["origin_centroid_dist"] = np.sqrt((train_df["origin_x"] - train_origin_centroids[:, 0])**2 + (train_df["origin_y"] - train_origin_centroids[:, 1])**2)

train_dest_centroids = kmeans_dest.cluster_centers_[train_df["dest_cluster"]]
train_df["dest_centroid_dist"] = np.sqrt((train_df["dest_x"] - train_dest_centroids[:, 0])**2 + (train_df["dest_y"] - train_dest_centroids[:, 1])**2)

train_df["centroid_dist_interaction"] = train_df["origin_centroid_dist"] * train_df["dest_centroid_dist"]

test_origin_coords = test_df[["origin_x", "origin_y"]].values
test_dest_coords = test_df[["dest_x", "dest_y"]].values

test_df["origin_cluster"] = kmeans_origin.predict(test_origin_coords)
test_df["dest_cluster"] = kmeans_dest.predict(test_dest_coords)

test_origin_centroids = kmeans_origin.cluster_centers_[test_df["origin_cluster"]]
test_df["origin_centroid_dist"] = np.sqrt((test_df["origin_x"] - test_origin_centroids[:, 0])**2 + (test_df["origin_y"] - test_origin_centroids[:, 1])**2)

test_dest_centroids = kmeans_dest.cluster_centers_[test_df["dest_cluster"]]
test_df["dest_centroid_dist"] = np.sqrt((test_df["dest_x"] - test_dest_centroids[:, 0])**2 + (test_df["dest_y"] - test_dest_centroids[:, 1])**2)

test_df["centroid_dist_interaction"] = test_df["origin_centroid_dist"] * test_df["dest_centroid_dist"]

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_idx = int(len(train_df) * 0.8)

train_split = train_df.iloc[:split_idx]
val_split = train_df.iloc[split_idx:]

train_split.to_parquet("./working/train_processed.parquet")
val_split.to_parquet("./working/val_processed.parquet")

print(f"Train split shape: {train_split.shape}")
print(f"Validation split shape: {val_split.shape}")

exclude_cols = ["record_id", "start_time", "cost"]
feature_cols = [c for c in train_split.columns if c not in exclude_cols]
input_dim = len(feature_cols)

print(f"Detected input feature dimension: {input_dim}")


import lightgbm as lgb

X_train = train_split[feature_cols].values.astype(np.float32)
y_train = train_split["cost"].values.astype(np.float32)

X_val = val_split[feature_cols].values.astype(np.float32)
y_val = val_split["cost"].values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

model = lgb.LGBMRegressor(
    n_estimators=2500,
    num_leaves=511,
    learning_rate=0.015,
    min_child_samples=50,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)],
)

final_preds = model.predict(X_val)
score = math.sqrt(mean_squared_error(y_val, final_preds))

X_test = test_df[feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
test_preds = model.predict(X_test)

sub = pd.DataFrame(
    {
        "record_id": test_df["record_id"],
        "cost": test_preds,
    }
)
sub.to_csv("./submission/submission.csv", index=False)
print("Submission saved successfully.")

print(f"Final Validation Score: {score}")
