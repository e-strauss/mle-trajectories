import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit


INPUT_DIR = "./input"
CHUNK_SIZE = 2_000_000


def sample_chunk(df):
    """Reproduce sample(frac=0.4, random_state=42) for each input chunk."""
    return df.sample(frac=0.4, random_state=42)


class EarlyStoppingXGBRegressor(RegressorMixin, BaseEstimator):
    """XGBoost regressor with a per-fit inner early-stopping split."""

    def __init__(
        self,
        n_estimators=1500,
        learning_rate=0.03,
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
        validation_size=0.2,
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
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.validation_size = validation_size

    def fit(self, X, y):
        # This inner split is made separately within every outer fold's training
        # rows. It preserves early stopping without exposing scored holdout rows.
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.validation_size,
            random_state=self.random_state,
        )
        fit_idx, eval_idx = next(splitter.split(X, y))

        if hasattr(X, "iloc"):
            X_fit = X.iloc[fit_idx]
            X_eval = X.iloc[eval_idx]
        else:
            X_array = np.asarray(X)
            X_fit = X_array[fit_idx]
            X_eval = X_array[eval_idx]

        y_array = np.asarray(y).reshape(-1)
        y_fit = y_array[fit_idx]
        y_eval = y_array[eval_idx]

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
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(
            X_fit,
            y_fit,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def inverse_and_clip_predictions(values, mode):
    """Reproduce the original expm1 followed by clipping at zero."""
    if mode == "fit":
        return values
    return np.clip(np.expm1(values), 0, None)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read. Test-data processing, directory
    #    creation, and submission generation are omitted because they do not
    #    contribute to the validation score.
    data = skrub.as_data_op(f"{INPUT_DIR}/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Reproduce the original 40% sampling independently within
    #    each conceptual 2,000,000-row read chunk. Sampling and target-dependent
    #    filtering occur before the marks because they determine which rows are
    #    scored.
    chunked = data.assign(_chunk=data.index // CHUNK_SIZE)
    sampled = (
        chunked.groupby("_chunk", group_keys=True)
        .apply(sample_chunk)
        .reset_index(drop=True)
        .drop(columns=["_chunk"], errors="ignore")
    )
    filtered = sampled.dropna(subset=["cost"])
    filtered = filtered[filtered["cost"] > 0].reset_index(drop=True)

    # Mark the RAW target, then record the original log1p training transform.
    # Predictions are transformed back to raw cost before scoring.
    y = filtered["cost"].skb.mark_as_y()
    y_log = y.skb.apply_func(np.log1p)

    # The original single 80/20 outer holdout becomes a one-split ShuffleSplit
    # on mark_as_X. The identifier and target are excluded from model features.
    X = filtered.drop(columns=["record_id", "cost"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. Coordinate medians are
    #    now learned only from each outer fold's training rows, fixing the
    #    original full-table preprocessing leakage.
    coordinates = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coordinates = coordinates.assign(
        origin_x=coordinates["origin_x"].clip(-75.0, -72.0),
        dest_x=coordinates["dest_x"].clip(-75.0, -72.0),
        origin_y=coordinates["origin_y"].clip(40.0, 42.0),
        dest_y=coordinates["dest_y"].clip(40.0, 42.0),
    ).skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(pd.to_datetime, errors="coerce")
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    origin_x = coordinates["origin_x"]
    dest_x = coordinates["dest_x"]
    origin_y = coordinates["origin_y"]
    dest_y = coordinates["dest_y"]
    unit_count = X["unit_count"].fillna(1)

    dx = dest_x - origin_x
    dy = dest_y - origin_y
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = (
        dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
    )
    bearing = dy.skb.apply_func(np.arctan2, dx)

    lat1 = origin_y.skb.apply_func(np.radians)
    lon1 = origin_x.skb.apply_func(np.radians)
    lat2 = dest_y.skb.apply_func(np.radians)
    lon2 = dest_x.skb.apply_func(np.radians)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = (dlat / 2.0).skb.apply_func(np.sin)
    sin_dlon = (dlon / 2.0).skb.apply_func(np.sin)
    haversine_a = (
        sin_dlat**2
        + lat1.skb.apply_func(np.cos)
        * lat2.skb.apply_func(np.cos)
        * sin_dlon**2
    )
    haversine_dist = (
        haversine_a.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)
        * (2 * 6371.0)
    )

    X_features = X.assign(
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
        unit_distance_ratio=euclidean_dist / (unit_count + 1e-5),
        manhattan_euclidean_ratio=manhattan_dist / (euclidean_dist + 1e-5),
        origin_dest_x_prod=origin_x * dest_x,
        origin_dest_y_prod=origin_y * dest_y,
        unit_weighted_euclidean=euclidean_dist * unit_count,
        unit_weighted_manhattan=manhattan_dist * unit_count,
        hour_sin=(2 * np.pi * hour / 24).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24).skb.apply_func(np.cos),
        month_sin=(2 * np.pi * month / 12).skb.apply_func(np.sin),
        month_cos=(2 * np.pi * month / 12).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # The wrapper preserves every original XGBoost hyperparameter and performs
    # its legitimate early-stopping split inside fit on each fold's training
    # rows. The original reused its reported holdout for early stopping; that
    # leakage cannot be reproduced in an honest outer-validation plan.
    pred_log = X_features.skb.apply(
        EarlyStoppingXGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
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
            validation_size=0.2,
        ),
        y=y_log,
    )

    # Reproduce the original prediction decoding: expm1, then clip at zero.
    pred = pred_log.skb.apply_func(
        inverse_and_clip_predictions,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here -- the ShuffleSplit on mark_as_X drives.
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
