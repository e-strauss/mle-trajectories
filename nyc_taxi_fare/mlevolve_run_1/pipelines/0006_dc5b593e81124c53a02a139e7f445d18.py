import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Ensure directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

INPUT_DIR = "./input"
WORKING_DIR = "./working"

print("Starting data processing and feature engineering...")


def process_chunk(df, is_train=True):
    if is_train:
        df = df.dropna(subset=["cost"])
        df = df[df["cost"] > 0]

    lon_min, lon_max = -75.0, -72.0
    lat_min, lat_max = 40.0, 42.0

    for col in ["origin_x", "dest_x"]:
        df[col] = df[col].clip(lon_min, lon_max)
        df[col] = df[col].fillna(df[col].median())

    for col in ["origin_y", "dest_y"]:
        df[col] = df[col].clip(lat_min, lat_max)
        df[col] = df[col].fillna(df[col].median())

    df["unit_count"] = df["unit_count"].fillna(1)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["hour"] = df["start_time"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["start_time"].dt.dayofweek.fillna(0).astype(int)
    df["month"] = df["start_time"].dt.month.fillna(1).astype(int)
    df["year"] = df["start_time"].dt.year.fillna(2012).astype(int)

    dx = df["dest_x"] - df["origin_x"]
    dy = df["dest_y"] - df["origin_y"]
    df["euclidean_dist"] = np.sqrt(dx**2 + dy**2)
    df["manhattan_dist"] = np.abs(dx) + np.abs(dy)
    df["bearing"] = np.arctan2(dy, dx)

    df["unit_distance_ratio"] = df["euclidean_dist"] / (df["unit_count"] + 1e-5)
    df["manhattan_euclidean_ratio"] = df["manhattan_dist"] / (df["euclidean_dist"] + 1e-5)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.drop(columns=["start_time"])
    return df


train_path = os.path.join(INPUT_DIR, "train.csv")
chunk_size = 2_000_000
train_chunks = []

print("Processing training data chunks...")
for chunk in pd.read_csv(train_path, chunksize=chunk_size):
    processed_chunk = process_chunk(chunk, is_train=True)
    train_chunks.append(processed_chunk)

full_train_df = pd.concat(train_chunks, ignore_index=True)
print(f"Processed training data shape: {full_train_df.shape}")

train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
print(f"Train split shape: {train_df.shape}, Validation split shape: {val_df.shape}")

test_path = os.path.join(INPUT_DIR, "test.csv")
test_df = pd.read_csv(test_path)
processed_test_df = process_chunk(test_df, is_train=False)
print(f"Processed test data shape: {processed_test_df.shape}")

train_df.to_parquet(os.path.join(WORKING_DIR, "train_processed.parquet"), index=False)
val_df.to_parquet(os.path.join(WORKING_DIR, "val_processed.parquet"), index=False)
processed_test_df.to_parquet(
    os.path.join(WORKING_DIR, "test_processed.parquet"), index=False
)

print("Data processing and feature engineering completed successfully.")

print("Initializing XGBoost Regressor model for large-scale tabular regression...")
model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=10,
    reg_alpha=1.0,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)

print("Loading preprocessed datasets for training...")
train_df = pd.read_parquet(os.path.join(WORKING_DIR, "train_processed.parquet"))
val_df = pd.read_parquet(os.path.join(WORKING_DIR, "val_processed.parquet"))
test_df = pd.read_parquet(os.path.join(WORKING_DIR, "test_processed.parquet"))

exclude_cols = ["record_id", "cost"]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df["cost"]
X_val = val_df[feature_cols]
y_val = val_df["cost"]

print(
    f"Training XGBoost model on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Evaluating model on validation set...")
val_preds = model.predict(X_val)
val_preds = np.clip(val_preds, 0, None)
score = np.sqrt(mean_squared_error(y_val, val_preds))

print("Generating predictions on test set...")
X_test = test_df[feature_cols]
test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, None)

submission = pd.DataFrame({"record_id": test_df["record_id"], "cost": test_preds})

submission_path = "./submission/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

print(f"Final Validation Score: {score}")
