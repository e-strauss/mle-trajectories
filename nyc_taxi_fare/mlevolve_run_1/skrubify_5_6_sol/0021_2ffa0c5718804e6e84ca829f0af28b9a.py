import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from each outer fold's training rows."""

    def __init__(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        # This is an inner, per-fit split used only for early stopping. The outer
        # validation split remains the ShuffleSplit declared on mark_as_X.
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def restore_raw_target_domain(prediction, mode):
    """Keep predictions in the raw cost domain, gated for skrub fit mode."""
    if mode == "fit":
        return prediction
    # GetXY only subsets the raw target; it does not numerically transform it.
    # Therefore the corresponding inverse mapping is the identity.
    return prediction


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one full CSV read. The original's chunked reads and
    #    concatenation produced the same table and were only a memory-management
    #    mechanism. Test loading, parquet files, and submission generation are
    #    omitted because they do not contribute to the validation score.
    dtypes = {
        "unit_count": "int32",
        "origin_x": "float32",
        "origin_y": "float32",
        "dest_x": "float32",
        "dest_y": "float32",
        "cost": "float32",
    }
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(
        pd.read_csv,
        dtype=dtypes,
        low_memory=False,
    )

    # 2. Prepare Data — target-anomaly filtering changes which rows are scored,
    #    so it must happen before marking X and y. Mark the raw cost target.
    filtered = data[(data["cost"] >= 0) & (data["cost"] <= 10000)].reset_index(
        drop=True
    )
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["record_id", "cost"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.1, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. The original calculated
    #    coordinate medians on the complete table before its holdout split. Using
    #    SimpleImputer here preserves median imputation while fitting the medians
    #    separately on each outer fold's training rows, fixing that leakage.
    coordinate_cols = ["dest_x", "dest_y", "origin_x", "origin_y"]
    coordinates = X[coordinate_cols].skb.apply(
        SimpleImputer(strategy="median")
    )

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
    )
    hour = start_time.dt.hour.astype("float32")
    dayofweek = start_time.dt.dayofweek.astype("float32")
    month = start_time.dt.month.astype("float32")
    year = start_time.dt.year.astype("float32")

    dest_x = coordinates["dest_x"]
    dest_y = coordinates["dest_y"]
    origin_x = coordinates["origin_x"]
    origin_y = coordinates["origin_y"]

    dx = dest_x - origin_x
    dy = dest_y - origin_y

    euclidean_dist = (
        (dx**2 + dy**2).skb.apply_func(np.sqrt).astype("float32")
    )
    manhattan_dist = (
        dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
    ).astype("float32")
    bearing = dy.skb.apply_func(np.arctan2, dx).astype("float32")

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
        * ((dlon / 2.0).skb.apply_func(np.sin) ** 2)
    )
    haversine_dist = (
        2
        * 6371.0
        * haversine_a.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)
    ).astype("float32")

    midpoint_x = ((origin_x + dest_x) / 2.0).astype("float32")
    midpoint_y = ((origin_y + dest_y) / 2.0).astype("float32")
    is_rush_hour = (
        (
            ((hour >= 7) & (hour <= 9))
            | ((hour >= 16) & (hour <= 19))
        )
        & (dayofweek < 5)
    ).astype("float32")
    unit_density = (
        X["unit_count"] / (euclidean_dist + 1e-5)
    ).astype("float32")

    features = X.assign(
        dest_x=dest_x,
        dest_y=dest_y,
        origin_x=origin_x,
        origin_y=origin_y,
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=bearing,
        haversine_dist=haversine_dist,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        is_rush_hour=is_rush_hour,
        unit_density=unit_density,
    ).drop(columns=["start_time"])

    # The original early-stopped on the same 10% holdout that it reported as
    # validation, which leaks validation information into model selection. Keep
    # its 10% split fraction and 50-round patience, but carve the early-stopping
    # set from each outer fold's training rows instead.
    split_training = features.skb.apply(
        GetXY(test_size=0.1, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = split_training["X"]
    y_fit = split_training.get("y", y)
    X_eval = split_training["X_val"]
    y_eval = split_training["y_val"]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=100,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    pred_inner = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_eval, y_eval)],
            "callbacks": [
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ],
        },
    )

    # y_fit is a row subset of the raw marked target rather than a numerical
    # target transform. Apply its identity inverse explicitly and gate it on
    # eval_mode so fit mode continues to pass through the fitted estimator.
    pred = pred_inner.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the ShuffleSplit declared on mark_as_X drives.
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
