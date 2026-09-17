import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split


class OriginalPermutationSplit(BaseCrossValidator):
    """Reproduce the original seeded permutation and 80/20 index split."""

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
    """Carve an early-stopping eval set out of the outer fold's training rows."""

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
    """Work around CatBoost's sklearn cloning issue with learning_rate."""

    def __sklearn_clone__(self):
        return CatBoostRegressorCloneable(**self.get_params(deep=False))


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read. The original chunk loop retained the first
    #    10,000,000 rows surviving its filters. Reading once, applying the same
    #    filters in source order, and taking the first 10,000,000 valid rows is
    #    equivalent. Test loading, parquet files, directories, and submission
    #    generation are omitted because they do not produce the validation score.
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
    data = filtered[valid_mask].iloc[:10_000_000].reset_index(drop=True)

    # 2. Prepare data: mark the RAW target and design matrix. The custom splitter
    #    exactly reproduces np.random.seed(42), np.random.permutation, and the
    #    original floor-based 80/20 split, including which end of the permutation
    #    is used for validation.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationSplit(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. The source subsequently accesses and
    #    excludes start_time, so its schema establishes that this column is
    #    present; its conditional guard is therefore represented by these same
    #    recorded datetime operations.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce", utc=True
    )
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)
    is_weekend = dayofweek.isin([5, 6]).astype(int)

    dx = X["dest_x"] - X["origin_x"]
    dy = X["dest_y"] - X["origin_y"]

    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    abs_dx = dx.skb.apply_func(np.abs)
    abs_dy = dy.skb.apply_func(np.abs)
    manhattan_dist = abs_dx + abs_dy

    lat1 = X["origin_y"].skb.apply_func(np.radians)
    lon1 = X["origin_x"].skb.apply_func(np.radians)
    lat2 = X["dest_y"].skb.apply_func(np.radians)
    lon2 = X["dest_x"].skb.apply_func(np.radians)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_half_dlat = (dlat / 2.0).skb.apply_func(np.sin)
    sin_half_dlon = (dlon / 2.0).skb.apply_func(np.sin)
    cos_lat1 = lat1.skb.apply_func(np.cos)
    cos_lat2 = lat2.skb.apply_func(np.cos)

    haversine_a = (
        sin_half_dlat**2
        + cos_lat1 * cos_lat2 * sin_half_dlon**2
    )
    sqrt_a = haversine_a.skb.apply_func(np.sqrt)
    haversine_c = sqrt_a.clip(0, 1).skb.apply_func(np.arcsin) * 2
    haversine_dist = haversine_c * 6371.0

    sin_dlon = dlon.skb.apply_func(np.sin)
    cos_dlon = dlon.skb.apply_func(np.cos)
    sin_lat1 = lat1.skb.apply_func(np.sin)
    sin_lat2 = lat2.skb.apply_func(np.sin)

    y_bearing = sin_dlon * cos_lat2
    x_bearing = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon
    bearing_radians = y_bearing.skb.apply_func(np.arctan2, x_bearing)
    bearing = bearing_radians.skb.apply_func(np.degrees)

    features = X.assign(
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        is_weekend=is_weekend,
        euclidean_dist=euclidean_dist,
        abs_dx=abs_dx,
        abs_dy=abs_dy,
        manhattan_dist=manhattan_dist,
        haversine_dist=haversine_dist,
        bearing=bearing,
    ).fillna(0)

    # Match feature_cols: every engineered column except record_id, start_time,
    # and cost. cost was already removed when constructing the marked X.
    features = features.drop(columns=["record_id", "start_time"])

    # The original early-stopped on the same validation rows it reported, which
    # leaks validation information and cannot be reproduced honestly in outer CV.
    # Keep its 20% eval fraction, patience, and iteration count by carving the
    # eval set from each outer fold's training rows with a visible GetXY node.
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
        iterations=2000,
        learning_rate=0.04,
        depth=8,
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

    # 4. Score. No cv= here -- the splitter on mark_as_X drives. Scikit-learn's
    #    RMSE scorer is negated because grid-search scores are higher-is-better.
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
