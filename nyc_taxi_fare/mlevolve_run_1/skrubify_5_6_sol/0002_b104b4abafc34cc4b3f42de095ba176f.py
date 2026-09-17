import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split


class OriginalPermutationSplit(BaseCrossValidator):
    """Reproduce the original seeded permutation and first-80% training split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping eval set out of the current fold's training rows."""

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


class CatBoostRegressorCloneable(CatBoostRegressor):
    """Work around CatBoost's non-standard sklearn cloning behavior."""

    def __sklearn_clone__(self):
        return CatBoostRegressorCloneable(**self.get_params(deep=False))


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record one CSV read. The original's chunk loop retained the
    #    first 3,000,000 rows that passed its filters, so filtering the complete
    #    recorded table and taking its first 3,000,000 valid rows is equivalent.
    #    Test-set loading, parquet files, directories, and submission generation
    #    are omitted because they do not contribute to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    required = ["cost", "origin_x", "origin_y", "dest_x", "dest_y"]
    filtered = data.dropna(subset=required)
    valid_mask = (
        (filtered["origin_x"] >= -180)
        & (filtered["origin_x"] <= 180)
        & (filtered["origin_y"] >= -90)
        & (filtered["origin_y"] <= 90)
        & (filtered["dest_x"] >= -180)
        & (filtered["dest_x"] <= 180)
        & (filtered["dest_y"] >= -90)
        & (filtered["dest_y"] <= 90)
        & (filtered["cost"] >= 0)
    )
    filtered = filtered[valid_mask]
    filtered = filtered.iloc[:3_000_000].reset_index(drop=True)

    # 2. Prepare data: mark the RAW target and design matrix. The custom splitter
    #    exactly preserves the original np.random.seed(42), permutation, and
    #    first-80%/remaining-20% split rather than changing which permutation
    #    segment is used for validation.
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationSplit(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, followed by the original CatBoost model.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce", utc=True
    )
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    dx = X["dest_x"] - X["origin_x"]
    dy = X["dest_y"] - X["origin_y"]

    lat1 = X["origin_y"].skb.apply_func(np.radians)
    lon1 = X["origin_x"].skb.apply_func(np.radians)
    lat2 = X["dest_y"].skb.apply_func(np.radians)
    lon2 = X["dest_x"].skb.apply_func(np.radians)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    half_dlat_sin = (dlat / 2.0).skb.apply_func(np.sin)
    half_dlon_sin = (dlon / 2.0).skb.apply_func(np.sin)
    a = (
        half_dlat_sin**2
        + lat1.skb.apply_func(np.cos)
        * lat2.skb.apply_func(np.cos)
        * half_dlon_sin**2
    )
    sqrt_a = a.skb.apply_func(np.sqrt)
    clipped_sqrt_a = sqrt_a.skb.apply_func(np.clip, 0, 1)
    haversine_c = 2 * clipped_sqrt_a.skb.apply_func(np.arcsin)

    features = X.assign(
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        is_weekend=dayofweek.isin([5, 6]).astype(int),
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        abs_dx=dx.skb.apply_func(np.abs),
        abs_dy=dy.skb.apply_func(np.abs),
        haversine_dist=haversine_c * 6371.0,
    )

    # These columns were excluded from feature_cols in the original. Applying
    # fillna after dropping them preserves the values and order of every column
    # actually supplied to CatBoost.
    features = features.drop(columns=["record_id", "start_time"]).fillna(0)

    # The original early-stopped on the same validation rows it reported, which
    # leaks validation information and cannot be reproduced honestly by outer
    # CV. Keep iterations=1500 and patience=50, but carve a 20% eval set from
    # each outer fold's training rows through recorded GetXY operations.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = CatBoostRegressorCloneable(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        verbose=False,
    )

    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
            "verbose": False,
        },
    )

    # 4. Score. No cv= here -- the splitter on mark_as_X drives.
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
