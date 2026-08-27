import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit


INPUT_DIR = "./input"


def clip_predictions(predictions, mode):
    """Apply the original non-negative clipping only when predicting."""
    if mode == "fit":
        return predictions
    return np.clip(predictions, 0, None)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record one CSV read. The original's chunked reads,
    #    intermediate parquet files, test-set processing, and submission output
    #    are omitted because they do not contribute to the validation score.
    data = skrub.as_data_op(
        os.path.join(INPUT_DIR, "train.csv")
    ).skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original removed missing/non-positive targets before
    #    splitting, so this row filtering remains before mark_as_X/mark_as_y.
    data = data.dropna(subset=["cost"])
    data = data[data["cost"] > 0].reset_index(drop=True)

    # Mark the raw target. The original single train_test_split becomes an
    # equivalent one-split ShuffleSplit attached to mark_as_X.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["record_id", "cost"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    # Coordinate medians are now learned from each training fold through
    # SimpleImputer, fixing the original full-table preprocessing leak.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coords = coords.assign(
        origin_x=coords["origin_x"].clip(-75.0, -72.0),
        dest_x=coords["dest_x"].clip(-75.0, -72.0),
        origin_y=coords["origin_y"].clip(40.0, 42.0),
        dest_y=coords["dest_y"].clip(40.0, 42.0),
    ).skb.apply(SimpleImputer(strategy="median"))

    unit_count = X["unit_count"].fillna(1)
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce"
    )
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    dx = coords["dest_x"] - coords["origin_x"]
    dy = coords["dest_y"] - coords["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = (
        dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
    )
    bearing = dy.skb.apply_func(np.arctan2, dx)

    lat1 = coords["origin_y"].skb.apply_func(np.radians)
    lon1 = coords["origin_x"].skb.apply_func(np.radians)
    lat2 = coords["dest_y"].skb.apply_func(np.radians)
    lon2 = coords["dest_x"].skb.apply_func(np.radians)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    haversine_a = (
        (dlat / 2.0).skb.apply_func(np.sin) ** 2
        + lat1.skb.apply_func(np.cos)
        * lat2.skb.apply_func(np.cos)
        * ((dlon / 2.0).skb.apply_func(np.sin) ** 2)
    )
    haversine_dist = (
        2
        * 6371.0
        * haversine_a.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)
    )

    X_feat = X.assign(
        origin_x=coords["origin_x"],
        dest_x=coords["dest_x"],
        origin_y=coords["origin_y"],
        dest_y=coords["dest_y"],
        unit_count=unit_count,
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=bearing,
        haversine_dist=haversine_dist,
        origin_x_bin=coords["origin_x"].skb.apply_func(np.round, 2),
        origin_y_bin=coords["origin_y"].skb.apply_func(np.round, 2),
        dest_x_bin=coords["dest_x"].skb.apply_func(np.round, 2),
        dest_y_bin=coords["dest_y"].skb.apply_func(np.round, 2),
        unit_distance_ratio=euclidean_dist / (unit_count + 1e-5),
        manhattan_euclidean_ratio=(
            manhattan_dist / (euclidean_dist + 1e-5)
        ),
        hour_sin=(2 * np.pi * hour / 24).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24).skb.apply_func(np.cos),
        month_sin=(2 * np.pi * month / 12).skb.apply_func(np.sin),
        month_cos=(2 * np.pi * month / 12).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # Same XGBoost model family and fit hyperparameters. The original
    # early_stopping_rounds=50 depended on passing the outer validation set as
    # eval_set; a DataOps estimator does not receive its held-out fold during
    # fit, so early stopping and eval_set are removed while outer validation is
    # driven by the ShuffleSplit above.
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=12,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    pred_raw = X_feat.skb.apply(model, y=y)

    # Preserve the original post-prediction clipping. It is gated because a
    # prediction node evaluates to the fitted estimator in fit mode.
    pred = pred_raw.skb.apply_func(
        clip_predictions, skrub.eval_mode()
    )

    # 4. Score. No cv= here: the splitter on mark_as_X drives validation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="neg_root_mean_squared_error",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {-variant_score}")
        print(
            "Final Validation Performance: "
            f"{-search.results_['mean_test_score'].iloc[0]}"
        )
