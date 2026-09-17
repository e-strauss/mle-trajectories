import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping evaluation set out of each fold's training rows."""

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
    """Apply the original non-negative clipping only in prediction modes."""
    if mode == "fit":
        return None
    return np.clip(values, 0, None)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read.
    # The original's chunked reads, parquet round-trip, test-set processing, and
    # submission generation are omitted because they do not produce the
    # validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # The original removes invalid target rows before splitting. Row filtering
    # therefore remains before mark_as_X/mark_as_y and the index is reset to
    # reproduce pd.concat(..., ignore_index=True).
    filtered = data.dropna(subset=["cost"])
    filtered = filtered[filtered["cost"] > 0].reset_index(drop=True)

    # 2. Prepare data: mark the RAW target and design matrix. The original single
    # 80/20 train_test_split becomes an equivalent one-split ShuffleSplit.
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing / feature engineering, then the model.
    # Coordinate clipping is recorded explicitly. The original calculated
    # coordinate medians before its outer split (within each input chunk), which
    # leaked validation information. SimpleImputer learns the same per-column
    # median operation from each outer fold's training rows instead.
    coordinates = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coordinates = coordinates.assign(
        origin_x=coordinates["origin_x"].clip(-75.0, -72.0),
        dest_x=coordinates["dest_x"].clip(-75.0, -72.0),
        origin_y=coordinates["origin_y"].clip(40.0, 42.0),
        dest_y=coordinates["dest_y"].clip(40.0, 42.0),
    )
    coordinates = coordinates.skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce"
    )
    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)
    unit_count = X["unit_count"].fillna(1)

    dx = coordinates["dest_x"] - coordinates["origin_x"]
    dy = coordinates["dest_y"] - coordinates["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = (
        dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
    )
    bearing = dy.skb.apply_func(np.arctan2, dx)

    lat1 = coordinates["origin_y"].skb.apply_func(np.radians)
    lon1 = coordinates["origin_x"].skb.apply_func(np.radians)
    lat2 = coordinates["dest_y"].skb.apply_func(np.radians)
    lon2 = coordinates["dest_x"].skb.apply_func(np.radians)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine_a = (
        (dlat / 2.0).skb.apply_func(np.sin) ** 2
        + lat1.skb.apply_func(np.cos)
        * lat2.skb.apply_func(np.cos)
        * (dlon / 2.0).skb.apply_func(np.sin) ** 2
    )
    haversine_dist = (
        haversine_a.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)
        * (2 * 6371.0)
    )

    features = X.assign(
        origin_x=coordinates["origin_x"],
        dest_x=coordinates["dest_x"],
        origin_y=coordinates["origin_y"],
        dest_y=coordinates["dest_y"],
        unit_count=unit_count,
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=bearing,
        haversine_dist=haversine_dist,
        origin_x_bin=coordinates["origin_x"].skb.apply_func(np.round, 2),
        origin_y_bin=coordinates["origin_y"].skb.apply_func(np.round, 2),
        dest_x_bin=coordinates["dest_x"].skb.apply_func(np.round, 2),
        dest_y_bin=coordinates["dest_y"].skb.apply_func(np.round, 2),
        unit_distance_ratio=euclidean_dist / (unit_count + 1e-5),
        manhattan_euclidean_ratio=(
            manhattan_dist / (euclidean_dist + 1e-5)
        ),
        hour_sin=(2 * np.pi * hour / 24).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24).skb.apply_func(np.cos),
        month_sin=(2 * np.pi * month / 12).skb.apply_func(np.sin),
        month_cos=(2 * np.pi * month / 12).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # The original used its scored holdout as XGBoost's early-stopping eval set,
    # which leaks information from the reported validation rows. To retain its
    # n_estimators, patience, and 20% eval fraction honestly, GetXY carves an
    # inner 20% eval set from each outer fold's training rows.
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
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=12,
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

    # Preserve the original prediction post-processing exactly.
    pred = pred_raw.skb.apply_func(
        clip_predictions,
        skrub.eval_mode(),
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
