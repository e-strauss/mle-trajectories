import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.impute import SimpleImputer


class SampledChunkHoldout(BaseCrossValidator):
    """Reproduce per-chunk 40% sampling followed by one 80/20 holdout."""

    def __init__(
        self,
        chunk_size=2_000_000,
        sample_frac=0.4,
        test_size=0.2,
        random_state=42,
    ):
        self.chunk_size = chunk_size
        self.sample_frac = sample_frac
        self.test_size = test_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        groups = pd.DataFrame(groups)
        raw_indices = groups["_raw_index"].to_numpy(dtype=np.int64)
        raw_n_rows = int(groups["_raw_n"].iloc[0])

        # The original reset random_state=42 for every call to chunk.sample().
        sampled_raw_indices = []
        for chunk_start in range(0, raw_n_rows, self.chunk_size):
            chunk_len = min(self.chunk_size, raw_n_rows - chunk_start)
            sample_size = int(round(chunk_len * self.sample_frac))
            rng = np.random.RandomState(self.random_state)
            offsets = rng.choice(chunk_len, size=sample_size, replace=False)
            sampled_raw_indices.extend((chunk_start + offsets).tolist())

        # Invalid targets were removed before marking. Intersecting here reproduces
        # sampling before process_chunk dropped those invalid rows.
        position_by_raw_index = {
            int(raw_index): position
            for position, raw_index in enumerate(raw_indices)
        }
        selected_positions = np.asarray(
            [
                position_by_raw_index[raw_index]
                for raw_index in sampled_raw_indices
                if raw_index in position_by_raw_index
            ],
            dtype=np.int64,
        )

        selected_order = np.arange(len(selected_positions))
        train_order, test_order = train_test_split(
            selected_order,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield selected_positions[train_order], selected_positions[test_order]


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping eval set out of this fold's training rows."""

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


def decode_predictions(predictions, mode):
    """Undo the log target transform and clip exactly as in the original."""
    if mode == "fit":
        return None
    return np.clip(np.expm1(predictions), 0, None)


INPUT_DIR = "./input"

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read. The test-set processing, directory creation,
    #    and submission writing are omitted because they do not produce the
    #    cross-validated score.
    data = skrub.as_data_op(
        os.path.join(INPUT_DIR, "train.csv")
    ).skb.apply_func(pd.read_csv)

    # Preserve original CSV positions and row count so the custom splitter can
    # reproduce the original 2,000,000-row chunk sampling without a chunked read.
    indexed = data.reset_index(names="_raw_index")
    indexed = indexed.assign(_raw_n=data.shape[0])

    # Row removal changes which rows can be trained/scored and therefore occurs
    # before the marks. Sampling itself is reproduced by the custom splitter:
    # it samples the original raw chunk positions first, then intersects them
    # with these valid-target rows.
    filtered = indexed.dropna(subset=["cost"])
    filtered = filtered[filtered["cost"] > 0].reset_index(drop=True)

    # 2. Prepare data: mark the RAW target. The log1p transform used for fitting
    #    occurs only after mark_as_y, and predictions are inverted before scoring.
    y = filtered["cost"].skb.mark_as_y()
    y_log = y.skb.apply_func(np.log1p)

    X = filtered.drop(
        columns=["cost", "record_id", "_raw_index", "_raw_n"]
    ).skb.mark_as_X(
        cv=SampledChunkHoldout(
            chunk_size=2_000_000,
            sample_frac=0.4,
            test_size=0.2,
            random_state=42,
        ),
        split_kwargs={
            "groups": filtered[["_raw_index", "_raw_n"]],
        },
    )

    # 3. Recorded preprocessing and feature engineering. The original calculated
    #    coordinate medians before its holdout split, which leaked validation
    #    information. SimpleImputer learns the same per-column medians from each
    #    fold's training rows only.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]].assign(
        origin_x=X["origin_x"].clip(-75.0, -72.0),
        dest_x=X["dest_x"].clip(-75.0, -72.0),
        origin_y=X["origin_y"].clip(40.0, 42.0),
        dest_y=X["dest_y"].clip(40.0, 42.0),
    )
    coords = coords.skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce"
    )

    base = X.assign(
        origin_x=coords["origin_x"],
        dest_x=coords["dest_x"],
        origin_y=coords["origin_y"],
        dest_y=coords["dest_y"],
        unit_count=X["unit_count"].fillna(1),
        hour=start_time.dt.hour.fillna(0).astype(int),
        dayofweek=start_time.dt.dayofweek.fillna(0).astype(int),
        month=start_time.dt.month.fillna(1).astype(int),
        year=start_time.dt.year.fillna(2012).astype(int),
    )

    dx = base["dest_x"] - base["origin_x"]
    dy = base["dest_y"] - base["origin_y"]

    features = base.assign(
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
        bearing=dy.skb.apply_func(np.arctan2, dx),
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
    haversine_dist = (
        2
        * 6371.0
        * haversine_a.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)
    )

    features = features.assign(
        haversine_dist=haversine_dist,
        origin_x_bin=features["origin_x"].skb.apply_func(np.round, 2),
        origin_y_bin=features["origin_y"].skb.apply_func(np.round, 2),
        dest_x_bin=features["dest_x"].skb.apply_func(np.round, 2),
        dest_y_bin=features["dest_y"].skb.apply_func(np.round, 2),
        unit_distance_ratio=(
            features["euclidean_dist"] / (features["unit_count"] + 1e-5)
        ),
        manhattan_euclidean_ratio=(
            features["manhattan_dist"] / (features["euclidean_dist"] + 1e-5)
        ),
        origin_dest_x_prod=features["origin_x"] * features["dest_x"],
        origin_dest_y_prod=features["origin_y"] * features["dest_y"],
        unit_weighted_euclidean=(
            features["euclidean_dist"] * features["unit_count"]
        ),
        unit_weighted_manhattan=(
            features["manhattan_dist"] * features["unit_count"]
        ),
        hour_sin=(
            2 * np.pi * features["hour"] / 24
        ).skb.apply_func(np.sin),
        hour_cos=(
            2 * np.pi * features["hour"] / 24
        ).skb.apply_func(np.cos),
        month_sin=(
            2 * np.pi * features["month"] / 12
        ).skb.apply_func(np.sin),
        month_cos=(
            2 * np.pi * features["month"] / 12
        ).skb.apply_func(np.cos),
    ).drop(columns=["start_time"])

    # The original early-stopped on the same holdout it reported, which leaks
    # validation information and cannot be reproduced honestly by outer CV.
    # Keep n_estimators=1500 and patience=50, but carve a 20% eval set from each
    # outer fold's own training rows using the recorded GetXY operation.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y_log,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y_log)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = xgb.XGBRegressor(
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
    )
    pred_log = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        },
    )

    pred = pred_log.skb.apply_func(
        decode_predictions,
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
