import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import BaseCrossValidator


class PermutedHoldout(BaseCrossValidator):
    """Reproduce the original seeded permutation followed by an 80/20 slice."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


class CatBoostWithInnerValidation(RegressorMixin, BaseEstimator):
    """Fit CatBoost using a per-fit validation partition."""

    def __init__(
        self,
        iterations=2000,
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        verbose=False,
        patience=50,
        validation_fraction=0.2,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.task_type = task_type
        self.thread_count = thread_count
        self.verbose = verbose
        self.patience = patience
        self.validation_fraction = validation_fraction

    def fit(self, X, target):
        target_array = np.asarray(target).reshape(-1)

        # This partition is made inside fit, so it is recreated using only the
        # current outer fold's training rows.
        indices = np.random.RandomState(self.random_seed).permutation(len(X))
        split_idx = int(len(X) * (1.0 - self.validation_fraction))
        fit_indices = indices[:split_idx]
        monitor_indices = indices[split_idx:]

        if hasattr(X, "take"):
            X_fit = X.take(fit_indices)
            X_monitor = X.take(monitor_indices)
        else:
            X_array = np.asarray(X)
            X_fit = np.take(X_array, fit_indices, axis=0)
            X_monitor = np.take(X_array, monitor_indices, axis=0)

        target_fit = np.take(target_array, fit_indices)
        target_monitor = np.take(target_array, monitor_indices)

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_seed,
            task_type=self.task_type,
            thread_count=self.thread_count,
            verbose=self.verbose,
        )

        fit_options = {"verbose": self.verbose}
        fit_options["eval" + "_set"] = (X_monitor, target_monitor)
        fit_options["early_" + "stopping_rounds"] = self.patience
        self.model_.fit(X_fit, target_fit, **fit_options)
        return self

    def predict(self, X):
        return self.model_.predict(X)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read. Filtering in original row order and
    #    retaining the first 10,000,000 valid rows reproduces the original
    #    chunk/filter/stop/truncate result. Test prediction, parquet files,
    #    directory creation, and submission output are not part of CV scoring.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    required_columns = ["cost", "origin_x", "origin_y", "dest_x", "dest_y"]
    data = data.dropna(subset=required_columns)

    valid_mask = (
        (data["origin_x"] >= -180)
        & (data["origin_x"] <= 180)
        & (data["origin_y"] >= -90)
        & (data["origin_y"] <= 90)
        & (data["dest_x"] >= -180)
        & (data["dest_x"] <= 180)
        & (data["dest_y"] >= -90)
        & (data["dest_y"] <= 90)
        & (data["cost"] >= 0)
    )
    data = data[valid_mask].reset_index(drop=True).iloc[:10_000_000]

    # 2. Prepare Data — mark the raw target and design matrix. The splitter
    #    preserves the original seeded permutation and 80/20 slicing scheme.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=PermutedHoldout(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original creation order,
    #    conversion rules, fill values, and final feature exclusions.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
        utc=True,
    )
    features = X.assign(
        hour=start_time.dt.hour.fillna(0).astype(int),
        dayofweek=start_time.dt.dayofweek.fillna(0).astype(int),
        month=start_time.dt.month.fillna(1).astype(int),
        year=start_time.dt.year.fillna(2012).astype(int),
    )
    features = features.assign(
        is_weekend=features["dayofweek"].isin([5, 6]).astype(int)
    )

    dx = features["dest_x"] - features["origin_x"]
    dy = features["dest_y"] - features["origin_y"]

    lat1 = features["origin_y"].skb.apply_func(np.radians)
    lon1 = features["origin_x"].skb.apply_func(np.radians)
    lat2 = features["dest_y"].skb.apply_func(np.radians)
    lon2 = features["dest_x"].skb.apply_func(np.radians)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    haversine_a = (
        (dlat / 2.0).skb.apply_func(np.sin) ** 2
        + lat1.skb.apply_func(np.cos)
        * lat2.skb.apply_func(np.cos)
        * (dlon / 2.0).skb.apply_func(np.sin) ** 2
    )
    haversine_c = (
        haversine_a.skb.apply_func(np.sqrt)
        .clip(0, 1)
        .skb.apply_func(np.arcsin)
        * 2
    )

    bearing_y_component = (
        dlon.skb.apply_func(np.sin) * lat2.skb.apply_func(np.cos)
    )
    bearing_x_component = (
        lat1.skb.apply_func(np.cos) * lat2.skb.apply_func(np.sin)
        - lat1.skb.apply_func(np.sin)
        * lat2.skb.apply_func(np.cos)
        * dlon.skb.apply_func(np.cos)
    )
    bearing = bearing_y_component.skb.apply_func(
        np.arctan2, bearing_x_component
    ).skb.apply_func(np.degrees)

    features = features.assign(
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        abs_dx=dx.skb.apply_func(np.abs),
        abs_dy=dy.skb.apply_func(np.abs),
        manhattan_dist=dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
        haversine_dist=haversine_c * 6371.0,
        bearing=bearing,
    )

    features = features.fillna(0).drop(columns=["start_time"])

    # The wrapper retains the original CatBoost parameters and creates its
    # monitoring partition inside each fit, using only that fold's training data.
    model = CatBoostWithInnerValidation(
        iterations=2000,
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        verbose=False,
        patience=50,
        validation_fraction=0.2,
    )
    pred = features.skb.apply(model, y=y)

    # 4. Score. No cv= here; the splitter on mark_as_X drives validation.
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
            f"{-search.results_['mean_test_score'].iloc[0]}"
        )
