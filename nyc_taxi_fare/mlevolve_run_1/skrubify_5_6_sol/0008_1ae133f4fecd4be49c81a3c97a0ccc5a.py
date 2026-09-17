import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split


class OriginalPermutationSplit(BaseCrossValidator):
    """Reproduce the original seeded permutation and 80/20 index slicing."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        permutation = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield permutation[:split_idx], permutation[split_idx:]


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping eval set out of this outer fold's training rows."""

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


class CloneableCatBoostRegressor(CatBoostRegressor):
    """Work around CatBoost's sklearn cloning issue with learning_rate."""

    def __sklearn_clone__(self):
        return CloneableCatBoostRegressor(**self.get_params(deep=False))


def average_predictions(catboost_predictions, lightgbm_predictions, mode):
    """Average predictions exactly as the original ensemble did."""
    if mode == "fit":
        return catboost_predictions
    return (catboost_predictions + lightgbm_predictions) / 2.0


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record one CSV read. The original chunk loop retained the
    #    first 10,000,000 rows passing these filters; filtering the complete table
    #    and taking that prefix has the same row semantics. Test prediction,
    #    parquet files, directories, and submission generation are omitted.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

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
    data = data.iloc[:10_000_000].reset_index(drop=True)

    # 2. Prepare data: mark the raw target and design matrix. The custom splitter
    #    preserves the original use of the first 80% of np.random.permutation(42)
    #    for training and the remaining 20% for validation.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationSplit(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. The original explicitly supports and uses
    #    start_time in this dataset, so its conditional body is recorded directly.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce", utc=True
    )
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    dx = X["dest_x"] - X["origin_x"]
    dy = X["dest_y"] - X["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)

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
    sin_lat1 = lat1.skb.apply_func(np.sin)
    sin_lat2 = lat2.skb.apply_func(np.sin)

    haversine_a = (
        sin_half_dlat**2
        + cos_lat1 * cos_lat2 * sin_half_dlon**2
    )
    haversine_root = haversine_a.skb.apply_func(np.sqrt)
    haversine_c = haversine_root.skb.apply_func(np.clip, 0, 1)
    haversine_c = haversine_c.skb.apply_func(np.arcsin) * 2

    y_bearing = dlon.skb.apply_func(np.sin) * cos_lat2
    x_bearing = (
        cos_lat1 * sin_lat2
        - sin_lat1 * cos_lat2 * dlon.skb.apply_func(np.cos)
    )
    bearing_radians = y_bearing.skb.apply_func(np.arctan2, x_bearing)

    features = X.assign(
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        is_weekend=dayofweek.isin([5, 6]).astype(int),
        euclidean_dist=euclidean_dist,
        abs_dx=dx.skb.apply_func(np.abs),
        abs_dy=dy.skb.apply_func(np.abs),
        manhattan_dist=(
            dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
        ),
        distance_per_unit=euclidean_dist / (X["unit_count"] + 1e-5),
        origin_x_bin=X["origin_x"].skb.apply_func(np.round, decimals=2),
        origin_y_bin=X["origin_y"].skb.apply_func(np.round, decimals=2),
        dest_x_bin=X["dest_x"].skb.apply_func(np.round, decimals=2),
        dest_y_bin=X["dest_y"].skb.apply_func(np.round, decimals=2),
        haversine_dist=haversine_c * 6371.0,
        bearing=bearing_radians.skb.apply_func(np.degrees),
    ).fillna(0)

    features = features.drop(columns=["record_id", "start_time"])

    # The original used the scored validation rows for early stopping. Reusing
    # outer test rows would leak, so both models instead receive the same 20%
    # early-stopping set carved from each outer fold's training rows.
    split_data = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = split_data["X"]
    y_fit = split_data.get("y", y)
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]

    catboost_model = CloneableCatBoostRegressor(
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
    catboost_pred = X_fit.skb.apply(
        catboost_model,
        y=y_fit,
        fit_kwargs={
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
            "verbose": False,
        },
    )

    lightgbm_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.04,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    lightgbm_pred = X_fit.skb.apply(
        lightgbm_model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "eval_metric": "rmse",
            "callbacks": [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        },
    )

    # 4. Average the two validation predictions exactly as in the original.
    #    No clipping is applied here because the original clipped only its
    #    discarded test/submission predictions, not validation predictions.
    pred = catboost_pred.skb.apply_func(
        average_predictions,
        lightgbm_pred,
        skrub.eval_mode(),
    )

    # 5. Score. No cv= here -- the splitter on mark_as_X drives.
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
