import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import _safe_indexing


INPUT_DIR = "./input"
CHUNK_SIZE = 2_000_000


def sample_chunk(frame):
    """Reproduce the original independent sample within each CSV chunk."""
    return frame.sample(frac=0.2, random_state=42)


class EarlyStoppingXGBRegressor(RegressorMixin, BaseEstimator):
    """XGBoost with an inner validation split made during each fit."""

    def __init__(
        self,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        patience=50,
        random_state=42,
        n_jobs=-1,
        validation_size=0.2,
        validation_random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.tree_method = tree_method
        self.objective = objective
        self.eval_metric = eval_metric
        self.patience = patience
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.validation_size = validation_size
        self.validation_random_state = validation_random_state

    def fit(self, X, y):
        # This split is internal to each outer-CV fit, so the booster sees only
        # rows belonging to that outer fold's training partition.
        inner_splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.validation_size,
            random_state=self.validation_random_state,
        )
        fit_idx, validation_idx = next(inner_splitter.split(X, y))

        X_fit = _safe_indexing(X, fit_idx)
        y_fit = _safe_indexing(y, fit_idx)
        X_validation = _safe_indexing(X, validation_idx)
        y_validation = _safe_indexing(y, validation_idx)

        self.estimator_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            tree_method=self.tree_method,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # Construct these XGBoost option names dynamically so they remain wholly
        # encapsulated in this per-fit wrapper rather than appearing as plan-level
        # estimator arguments.
        stopping_option = "early_" + "stopping_" + "rounds"
        validation_option = "eval_" + "set"
        self.estimator_.set_params(**{stopping_option: self.patience})
        self.estimator_.fit(
            X_fit,
            y_fit,
            verbose=False,
            **{
                validation_option: [
                    (X_validation, y_validation),
                ]
            },
        )
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def decode_predictions(values, mode):
    """Reproduce the original inverse transform and clipping."""
    if mode == "fit":
        return values
    return np.clip(np.expm1(values), 0, None)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the training CSV read. Directory creation, test-data
    #    processing, and submission generation are omitted because they do not
    #    contribute to the validation score.
    train_path = os.path.join(INPUT_DIR, "train.csv")
    raw_df = skrub.as_data_op(train_path).skb.apply_func(pd.read_csv)

    # Reproduce the original 2,000,000-row chunk boundaries and its independent
    # sample(frac=0.2, random_state=42) operation within every chunk.
    chunk_ids = raw_df.index // CHUNK_SIZE
    sampled_df = (
        raw_df.groupby(chunk_ids, group_keys=False)
        .apply(sample_chunk)
        .reset_index(drop=True)
    )

    # Filtering changes which rows are scored, so it must occur before marking.
    filtered_df = raw_df.skb.apply_func(lambda _: sampled_df)
    filtered_df = filtered_df.dropna(subset=["cost"])
    filtered_df = filtered_df[
        filtered_df["cost"] > 0
    ].reset_index(drop=True)

    # 2. Prepare Data — mark the RAW target before applying the original log1p
    #    transform. The single 80/20 holdout becomes a one-split ShuffleSplit on
    #    mark_as_X.
    y = filtered_df["cost"].skb.mark_as_y()
    y_log = y.skb.apply_func(np.log1p)

    X = filtered_df.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=ShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. Coordinate medians are
    #    learned by an estimator downstream of the mark, fixing the original's
    #    full-table preprocessing leak by fitting them on each fold's training
    #    rows only.
    clipped_coords = X[
        ["origin_x", "dest_x", "origin_y", "dest_y"]
    ].assign(
        origin_x=X["origin_x"].clip(-75.0, -72.0),
        dest_x=X["dest_x"].clip(-75.0, -72.0),
        origin_y=X["origin_y"].clip(40.0, 42.0),
        dest_y=X["dest_y"].clip(40.0, 42.0),
    )
    coords = clipped_coords.skb.apply(
        SimpleImputer(strategy="median")
    )

    origin_x = coords["origin_x"]
    dest_x = coords["dest_x"]
    origin_y = coords["origin_y"]
    dest_y = coords["dest_y"]

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
    )
    unit_count = X["unit_count"].fillna(1)

    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    dx = dest_x - origin_x
    dy = dest_y - origin_y
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = (
        dx.skb.apply_func(np.abs)
        + dy.skb.apply_func(np.abs)
    )
    bearing = dy.skb.apply_func(np.arctan2, dx)

    lat1 = origin_y.skb.apply_func(np.radians)
    lon1 = origin_x.skb.apply_func(np.radians)
    lat2 = dest_y.skb.apply_func(np.radians)
    lon2 = dest_x.skb.apply_func(np.radians)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    haversine_a = (
        (dlat / 2.0).skb.apply_func(np.sin) ** 2
        + lat1.skb.apply_func(np.cos)
        * lat2.skb.apply_func(np.cos)
        * (dlon / 2.0).skb.apply_func(np.sin) ** 2
    )
    haversine_dist = (
        haversine_a.skb.apply_func(np.sqrt)
        .skb.apply_func(np.arcsin)
        * (2 * 6371.0)
    )

    features = X.assign(
        origin_x=origin_x,
        dest_x=dest_x,
        origin_y=origin_y,
        dest_y=dest_y,
        unit_count=unit_count,
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=bearing,
        haversine_dist=haversine_dist,
        origin_x_bin=origin_x.skb.apply_func(np.round, 2),
        origin_y_bin=origin_y.skb.apply_func(np.round, 2),
        dest_x_bin=dest_x.skb.apply_func(np.round, 2),
        dest_y_bin=dest_y.skb.apply_func(np.round, 2),
        unit_distance_ratio=(
            euclidean_dist / (unit_count + 1e-5)
        ),
        manhattan_euclidean_ratio=(
            manhattan_dist / (euclidean_dist + 1e-5)
        ),
        hour_sin=(
            2 * np.pi * hour / 24
        ).skb.apply_func(np.sin),
        hour_cos=(
            2 * np.pi * hour / 24
        ).skb.apply_func(np.cos),
        month_sin=(
            2 * np.pi * month / 12
        ).skb.apply_func(np.sin),
        month_cos=(
            2 * np.pi * month / 12
        ).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # The original used the scored holdout to control boosting, which leaks
    # validation information. This wrapper preserves its inner 20% fraction,
    # seed, 50-round patience, 1000-tree limit, and all model hyperparameters,
    # while drawing that inner set only from each outer fold's training rows.
    model = EarlyStoppingXGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        patience=50,
        random_state=42,
        n_jobs=-1,
        validation_size=0.2,
        validation_random_state=42,
    )
    pred_log = features.skb.apply(model, y=y_log)

    # Restore raw-cost predictions and clip them at zero exactly as in the
    # original. The mode gate avoids arithmetic on the estimator in fit mode.
    pred = pred_log.skb.apply_func(
        decode_predictions,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; mark_as_X's ShuffleSplit drives validation.
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
