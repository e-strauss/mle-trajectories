import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set out of the current outer fold's training rows."""

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


def clip_predictions(values, mode):
    """Reproduce the original non-negative prediction clipping."""
    if mode == "fit":
        return values
    return np.clip(values, 0, None)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the training CSV read. The original's chunked read,
    # parquet round-trip, test-set processing, and submission output are omitted
    # because they are not part of producing the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # The original removes invalid target rows before splitting. Row filtering
    # changes which rows are scored, so it remains before both marks.
    data = data.dropna(subset=["cost"])
    data = data[data["cost"] > 0].reset_index(drop=True)

    # 2. Prepare Data — mark the raw target and feature matrix. The original's
    # single train_test_split(test_size=0.2, random_state=42) is represented by
    # the equivalent one-split ShuffleSplit attached to mark_as_X.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. The coordinate medians
    # are now fitted only on each outer fold's training rows, fixing the original
    # preprocessing leakage while preserving median-imputation semantics.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coords = coords.assign(
        origin_x=coords["origin_x"].clip(-75.0, -72.0),
        dest_x=coords["dest_x"].clip(-75.0, -72.0),
        origin_y=coords["origin_y"].clip(40.0, 42.0),
        dest_y=coords["dest_y"].clip(40.0, 42.0),
    ).skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
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
    unit_count = X["unit_count"].fillna(1)

    features = X.assign(
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
    )

    features = features.assign(
        unit_distance_ratio=(
            features["euclidean_dist"] / (features["unit_count"] + 1e-5)
        ),
        manhattan_euclidean_ratio=(
            features["manhattan_dist"] / (features["euclidean_dist"] + 1e-5)
        ),
        hour_sin=(2 * np.pi * features["hour"] / 24).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * features["hour"] / 24).skb.apply_func(np.cos),
        month_sin=(2 * np.pi * features["month"] / 12).skb.apply_func(np.sin),
        month_cos=(2 * np.pi * features["month"] / 12).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # The original used the scored holdout itself for early stopping, which is
    # leaky and cannot be reproduced honestly in outer CV. Preserve its 20%
    # split fraction and patience by carving the eval set from each outer fold's
    # training rows instead.
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

    pred_raw = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        },
    )

    # Reproduce the original clipping after model.predict.
    pred = pred_raw.skb.apply_func(
        clip_predictions,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the ShuffleSplit on mark_as_X drives validation.
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
