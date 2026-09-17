import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set out of each outer fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's chunked reads,
    #    intermediate parquet files, test-set processing, and submission output
    #    are omitted because they do not contribute to the validation score.
    data = skrub.as_data_op(
        os.path.join("./input", "train.csv")
    ).skb.apply_func(pd.read_csv)

    # 2. Prepare Data — preserve the original training-row filters before marking,
    #    since they change which rows are scored. Mark the raw target and attach
    #    the original single 80/20 split to the design matrix.
    data = data.dropna(subset=["cost"])
    data = data[data["cost"] > 0].reset_index(drop=True)

    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. The original calculated
    #    coordinate medians before its validation split (and separately by input
    #    chunk). SimpleImputer preserves the intended median imputation while
    #    learning those medians from each fold's training rows, fixing that leak.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coords = coords.assign(
        origin_x=coords["origin_x"].clip(-75.0, -72.0),
        dest_x=coords["dest_x"].clip(-75.0, -72.0),
        origin_y=coords["origin_y"].clip(40.0, 42.0),
        dest_y=coords["dest_y"].clip(40.0, 42.0),
    )
    coords = coords.skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce"
    )
    dx = coords["dest_x"] - coords["origin_x"]
    dy = coords["dest_y"] - coords["origin_y"]

    features = X.assign(
        origin_x=coords["origin_x"],
        dest_x=coords["dest_x"],
        origin_y=coords["origin_y"],
        dest_y=coords["dest_y"],
        unit_count=X["unit_count"].fillna(1),
        hour=start_time.dt.hour.fillna(0).astype(int),
        dayofweek=start_time.dt.dayofweek.fillna(0).astype(int),
        month=start_time.dt.month.fillna(1).astype(int),
        year=start_time.dt.year.fillna(2012).astype(int),
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=(
            dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
        ),
        bearing=dy.skb.apply_func(np.arctan2, dx),
    ).drop(columns=["start_time"])

    # The original early-stopped on the same holdout it subsequently scored,
    # which is leaky and cannot be reproduced honestly by outer CV. Keep its
    # 20% eval fraction, random state, patience, and estimator count, but carve
    # the early-stopping set from each outer fold's training rows.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
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
    pred_raw = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        },
    )

    # Preserve the original post-prediction clipping. In fit mode the node holds
    # the fitted estimator rather than predictions, so arithmetic is gated.
    def clip_predictions(values, mode):
        if mode == "fit":
            return values
        return np.clip(values, 0, None)

    pred = pred_raw.skb.apply_func(
        clip_predictions, skrub.eval_mode()
    )

    # 4. Score. No cv= here — mark_as_X's ShuffleSplit drives validation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="neg_root_mean_squared_error",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
