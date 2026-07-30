import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

print("Loading train and test data...")
test_df = pd.read_csv("./input/test.csv")

dtypes = {
    "unit_count": "int32",
    "origin_x": "float32",
    "origin_y": "float32",
    "dest_x": "float32",
    "dest_y": "float32",
    "cost": "float32",
}

train_chunks = []
chunk_size = 5_000_000
for chunk in pd.read_csv(
    "./input/train.csv", dtype=dtypes, chunksize=chunk_size, low_memory=False
):
    train_chunks.append(chunk)
train_df = pd.concat(train_chunks, ignore_index=True)
del train_chunks

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Compute imputation values from training data to prevent data leakage
dest_x_median = train_df["dest_x"].median()
dest_y_median = train_df["dest_y"].median()
origin_x_median = train_df["origin_x"].median()
origin_y_median = train_df["origin_y"].median()


def engineer_features(df, dest_x_med, dest_y_med, origin_x_med, origin_y_med):
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["hour"] = df["start_time"].dt.hour.astype("float32")
    df["dayofweek"] = df["start_time"].dt.dayofweek.astype("float32")
    df["month"] = df["start_time"].dt.month.astype("float32")
    df["year"] = df["start_time"].dt.year.astype("float32")

    df["dest_x"] = df["dest_x"].fillna(dest_x_med)
    df["dest_y"] = df["dest_y"].fillna(dest_y_med)
    df["origin_x"] = df["origin_x"].fillna(origin_x_med)
    df["origin_y"] = df["origin_y"].fillna(origin_y_med)

    dx = df["dest_x"] - df["origin_x"]
    dy = df["dest_y"] - df["origin_y"]
    df["euclidean_dist"] = np.sqrt(dx**2 + dy**2).astype("float32")
    df["manhattan_dist"] = (np.abs(dx) + np.abs(dy)).astype("float32")

    df = df.drop(columns=["start_time"])
    return df


print("Engineering features for train set...")
train_df = engineer_features(
    train_df, dest_x_median, dest_y_median, origin_x_median, origin_y_median
)

print("Engineering features for test set...")
test_df = engineer_features(
    test_df, dest_x_median, dest_y_median, origin_x_median, origin_y_median
)

# Clean target anomalies
train_df = train_df[(train_df["cost"] >= 0) & (train_df["cost"] <= 10000)]

# Train/Validation split (90/10)
train_split, val_split = train_test_split(train_df, test_size=0.1, random_state=42)

print(f"Train split shape: {train_split.shape}, Val split shape: {val_split.shape}")

# Save to parquet for fast loading
train_split.to_parquet("./working/train_processed.parquet", index=False)
val_split.to_parquet("./working/val_processed.parquet", index=False)
test_df.to_parquet("./working/test_processed.parquet", index=False)

print("Data processing and feature engineering completed successfully.")

print("Loading processed train and validation sets for modeling...")
train_df = pd.read_parquet("./working/train_processed.parquet")
val_df = pd.read_parquet("./working/val_processed.parquet")
test_df = pd.read_parquet("./working/test_processed.parquet")

drop_cols = ["record_id", "cost"]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols]
y_train = train_df["cost"]

X_val = val_df[feature_cols]
y_val = val_df["cost"]

print(
    f"Training features shape: {X_train.shape}, Validation features shape: {X_val.shape}"
)

model_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 100,
    "random_state": 42,
    "n_jobs": -1,
}

model = lgb.LGBMRegressor(**model_params)

print("Training LightGBM model with early stopping...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

print("Generating predictions on validation set...")
val_preds = model.predict(X_val)

score = np.sqrt(mean_squared_error(y_val, val_preds))

print("Generating predictions on test set and saving submission...")
X_test = test_df[feature_cols]
test_preds = model.predict(X_test)

submission = pd.DataFrame({"record_id": test_df["record_id"], "cost": test_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
