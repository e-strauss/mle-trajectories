import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit


INPUT_DIR = "./input"


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one read of the complete training table. The original's
    # chunked reads, parquet round-trip, directory creation, test-set processing,
    # submission generation, and progress printing are omitted because they do not
    # contribute to the validation score.
    data = skrub.as_data_op(
        os.path.join(INPUT_DIR, "train.csv")
    ).skb.apply_func(pd.read_csv)

    # The original removes invalid target rows before train_test_split. Row filtering
    # changes which rows are scored, so it remains before the X/y marks.
    data = data.dropna(subset=["cost"])
    data = data[data["cost"] > 0].reset_index(drop=True)

    # 2. Prepare Data — mark the RAW target and design matrix. The original's single
    # 80/20 train_test_split becomes an equivalent one-split ShuffleSplit.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["record_id", "cost"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. Coordinate medians are
    # learned separately from each training fold with SimpleImputer, fixing the
    # original full-table preprocessing leak while preserving median-fill semantics.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]].assign(
        origin_x=X["origin_x"].clip(-75.0, -72.0),
        dest_x=X["dest_x"].clip(-75.0, -72.0),
        origin_y=X["origin_y"].clip(40.0, 42.0),
        dest_y=X["dest_y"].clip(40.0, 42.0),
    )
    coords = coords.skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce"
    )
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    unit_count = X["unit_count"].fillna(1)
    dx = coords["dest_x"] - coords["origin_x"]
    dy = coords["dest_y"] - coords["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = (
        dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
    )
    bearing = dy.skb.apply_func(np.arctan2, dx)

    X_features = X.assign(
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
        unit_distance_ratio=euclidean_dist / (unit_count + 1e-5),
        manhattan_euclidean_ratio=manhattan_dist / (euclidean_dist + 1e-5),
        hour_sin=(2 * np.pi * hour / 24).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24).skb.apply_func(np.cos),
        month_sin=(2 * np.pi * month / 12).skb.apply_func(np.sin),
        month_cos=(2 * np.pi * month / 12).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # Same XGBoost model family and hyperparameters as the original. The original
    # used the outer holdout itself as eval_set. A CV estimator does not receive its
    # held-out scoring rows during fit, so eval_set and early_stopping_rounds are
    # removed rather than leaking the validation fold into model fitting.
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
        random_state=42,
        n_jobs=-1,
    )
    raw_pred = X_features.skb.apply(model, y=y)

    # Preserve the original post-prediction clipping. Prediction arithmetic is
    # gated because a prediction node evaluates to the fitted estimator in fit mode.
    def clip_predictions(values, mode):
        if mode == "fit":
            return values
        return np.clip(values, 0, None)

    pred = raw_pred.skb.apply_func(
        clip_predictions, skrub.eval_mode()
    )

    # 4. Score. No cv= here — the ShuffleSplit attached to mark_as_X drives.
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
