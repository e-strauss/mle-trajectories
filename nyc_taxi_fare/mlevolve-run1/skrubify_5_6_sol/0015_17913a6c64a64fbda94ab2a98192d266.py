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
        # This is an inner split used only for early stopping. The outer
        # validation split is the ShuffleSplit declared on mark_as_X.
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
    """Keep predictions in the raw cost domain.

    GetXY subsets the raw marked target but does not numerically transform it,
    so the corresponding inverse is the identity. The eval-mode gate is still
    required because a prediction node evaluates to an estimator in fit mode.
    """
    if mode == "fit":
        return prediction
    return prediction


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one read of the complete training table. The
    #    original's chunked read, test-set read, parquet files, and submission
    #    generation are omitted because they do not produce the validation score.
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
    #    so it remains before both marks. Mark the raw cost target. The original
    #    90/10 holdout becomes a one-split ShuffleSplit on mark_as_X.
    filtered_data = data[
        (data["cost"] >= 0) & (data["cost"] <= 10000)
    ]

    y = filtered_data["cost"].skb.mark_as_y()
    X = filtered_data.drop(columns=["record_id", "cost"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.1, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. The original computed
    #    coordinate medians before splitting. SimpleImputer preserves per-column
    #    median imputation while learning the medians only from each outer fold's
    #    training rows, fixing that leakage.
    coordinate_columns = X[["dest_x", "dest_y", "origin_x", "origin_y"]]
    imputed_coordinates = coordinate_columns.skb.apply(
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

    dx = imputed_coordinates["dest_x"] - imputed_coordinates["origin_x"]
    dy = imputed_coordinates["dest_y"] - imputed_coordinates["origin_y"]

    euclidean_dist = (
        (dx**2 + dy**2)
        .skb.apply_func(np.sqrt)
        .astype("float32")
    )
    manhattan_dist = (
        dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
    ).astype("float32")
    bearing = dy.skb.apply_func(np.arctan2, dx).astype("float32")

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
        dest_x=imputed_coordinates["dest_x"],
        dest_y=imputed_coordinates["dest_y"],
        origin_x=imputed_coordinates["origin_x"],
        origin_y=imputed_coordinates["origin_y"],
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=bearing,
        is_rush_hour=is_rush_hour,
        unit_density=unit_density,
    ).drop(columns=["start_time"])

    # The original early-stopped on the same rows it reported as validation,
    # which leaks validation information into model selection. Preserve its 10%
    # eval fraction, random state, 1,500-tree limit, and 50-round patience, but
    # carve the eval set from each outer fold's training rows.
    split_parts = features.skb.apply(
        GetXY(test_size=0.1, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = split_parts["X"]
    y_fit = split_parts.get("y", y)
    X_eval = split_parts["X_val"]
    y_eval = split_parts["y_val"]

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

    # y_fit is a row subset of the raw marked target, not a numerical target
    # transform. Its inverse is therefore identity, gated on eval mode so fit
    # mode passes through the fitted estimator unchanged.
    pred = pred_inner.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the ShuffleSplit on mark_as_X drives.
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
