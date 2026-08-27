import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.model_selection import BaseCrossValidator


class OriginalPermutationHoldout(BaseCrossValidator):
    """Reproduce np.random.seed(...), permutation, then an 80/20 positional split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_rows = len(X)
        indices = np.random.RandomState(self.random_state).permutation(n_rows)
        split_idx = int(n_rows * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


class CatBoostRegressorCloneable(CatBoostRegressor):
    """Work around CatBoost's non-standard sklearn cloning behavior."""

    def __sklearn_clone__(self):
        return CatBoostRegressorCloneable(**self.get_params(deep=False))


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read. The original chunked read was only an
    #    OOM precaution; test-data loading, parquet files, directories, submission
    #    generation, and progress printing are omitted because they do not produce
    #    the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Preserve the original cleaning and its selection of the first 3,000,000
    # valid rows. Row filtering and truncation happen before marking because they
    # determine which rows participate in validation.
    data = data.dropna(
        subset=["cost", "origin_x", "origin_y", "dest_x", "dest_y"]
    )
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
    data = data[valid_mask].reset_index(drop=True)
    data = data.iloc[:3_000_000].reset_index(drop=True)

    # 2. Prepare Data — mark the raw target and design matrix. The custom
    #    one-split cross-validator exactly preserves the original permutation:
    #    the first floor(80%) shuffled rows train and the remaining rows validate.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationHoldout(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original column creation
    #    order and constants.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce", utc=True
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
    features = features.assign(
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        abs_dx=dx.skb.apply_func(np.abs),
        abs_dy=dy.skb.apply_func(np.abs),
    )

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
    haversine_root = haversine_a.skb.apply_func(np.sqrt)
    haversine_clipped = haversine_root.clip(0, 1)
    haversine_c = 2 * haversine_clipped.skb.apply_func(np.arcsin)

    features = features.assign(haversine_dist=haversine_c * 6371.0)
    features = features.fillna(0)
    model_features = features.drop(columns=["record_id", "start_time"])

    # CatBoost keeps every original constructor hyperparameter. The original used
    # the outer validation rows as an eval_set for early stopping; a DataOps CV
    # estimator cannot receive its held-out fold as fit-time eval_set, so
    # early_stopping_rounds=50 is removed while the 1,500-iteration limit remains.
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
    pred = model_features.skb.apply(model, y=y)

    # 4. Score — no cv= here; the original permutation holdout on mark_as_X drives.
    #    Validation predictions were not clipped in the original, so no clipping
    #    is applied here. Its clipping affected submission predictions only.
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
