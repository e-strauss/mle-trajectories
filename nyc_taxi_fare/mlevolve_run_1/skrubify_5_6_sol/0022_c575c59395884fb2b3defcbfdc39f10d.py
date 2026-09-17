import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import BaseCrossValidator, train_test_split


class ChunkSampleHoldout(BaseCrossValidator):
    """Reproduce per-2M-row sampling followed by the original 80/20 split."""

    def __init__(
        self,
        chunk_size=2_000_000,
        sample_frac=0.2,
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
        raw_positions = np.asarray(X["_raw_position"], dtype=np.int64)
        raw_size = int(np.asarray(X["_raw_size"])[0])

        # pandas.sample is called separately with the same seed on every chunk,
        # exactly as in the original chunk-reading loop.
        sampled_raw_positions = []
        for start in range(0, raw_size, self.chunk_size):
            stop = min(start + self.chunk_size, raw_size)
            chunk_positions = pd.Series(np.arange(start, stop, dtype=np.int64))
            sampled = chunk_positions.sample(
                frac=self.sample_frac,
                random_state=self.random_state,
            ).to_numpy()
            sampled_raw_positions.append(sampled)

        sampled_raw_positions = np.concatenate(sampled_raw_positions)

        # Rows with invalid costs were filtered before marking. Intersecting with
        # the retained raw positions reproduces sampling-before-filtering.
        position_lookup = {
            int(raw_position): current_position
            for current_position, raw_position in enumerate(raw_positions)
        }
        selected = np.asarray(
            [
                position_lookup[int(raw_position)]
                for raw_position in sampled_raw_positions
                if int(raw_position) in position_lookup
            ],
            dtype=np.int64,
        )

        train_idx, validation_idx = train_test_split(
            selected,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield train_idx, validation_idx


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set out of the current outer fold's training rows."""

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


def restore_raw_predictions(values, mode):
    """Undo the target log transform and preserve the original clipping."""
    if mode == "fit":
        return None
    return np.clip(np.expm1(values), 0, None)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The test-data processing, directory
    #    creation, and submission writing are omitted because they do not
    #    contribute to the validation score.
    data = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
        .assign(
            _raw_position=lambda frame: frame.index,
            _raw_size=lambda frame: frame.shape[0],
        )
    )

    # 2. Prepare data. Cost filtering changes the scored row population and
    #    therefore occurs before the marks. The custom splitter preserves the
    #    original order of operations: sample 20% independently from each
    #    2,000,000-row chunk, then perform one 80/20 train/validation split.
    data = data.dropna(subset=["cost"])
    data = data[data["cost"] > 0].reset_index(drop=True)

    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost"]).skb.mark_as_X(
        cv=ChunkSampleHoldout(
            chunk_size=2_000_000,
            sample_frac=0.2,
            test_size=0.2,
            random_state=42,
        ),
        split_kwargs={},
    )

    # The model is trained on log1p(cost), but the raw target remains marked so
    # scoring occurs in the original cost domain.
    y_log = y.skb.apply_func(np.log1p)

    # 3. Recorded preprocessing and feature engineering. Median imputation is
    #    now fitted only on each outer fold's training rows, fixing the original
    #    full-sampled-table leakage while preserving the imputation semantics.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coords = coords.assign(
        origin_x=coords["origin_x"].clip(-75.0, -72.0),
        dest_x=coords["dest_x"].clip(-75.0, -72.0),
        origin_y=coords["origin_y"].clip(40.0, 42.0),
        dest_y=coords["dest_y"].clip(40.0, 42.0),
    ).skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
    )
    unit_count = X["unit_count"].fillna(1)

    dx = coords["dest_x"] - coords["origin_x"]
    dy = coords["dest_y"] - coords["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)

    lat1 = coords["origin_y"].skb.apply_func(np.radians)
    lon1 = coords["origin_x"].skb.apply_func(np.radians)
    lat2 = coords["dest_y"].skb.apply_func(np.radians)
    lon2 = coords["dest_x"].skb.apply_func(np.radians)
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

    hour = start_time.dt.hour.fillna(0).astype(int)
    dayofweek = start_time.dt.dayofweek.fillna(0).astype(int)
    month = start_time.dt.month.fillna(1).astype(int)
    year = start_time.dt.year.fillna(2012).astype(int)

    features = X.assign(
        origin_x=coords["origin_x"],
        dest_x=coords["dest_x"],
        origin_y=coords["origin_y"],
        dest_y=coords["dest_y"],
        unit_count=unit_count,
        hour=hour,
        dayofweek=dayofweek,
        month=month,
        year=year,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=dy.skb.apply_func(np.arctan2, dx),
        haversine_dist=haversine_dist,
        origin_x_bin=coords["origin_x"].skb.apply_func(np.round, decimals=2),
        origin_y_bin=coords["origin_y"].skb.apply_func(np.round, decimals=2),
        dest_x_bin=coords["dest_x"].skb.apply_func(np.round, decimals=2),
        dest_y_bin=coords["dest_y"].skb.apply_func(np.round, decimals=2),
        unit_distance_ratio=euclidean_dist / (unit_count + 1e-5),
        manhattan_euclidean_ratio=manhattan_dist / (euclidean_dist + 1e-5),
        hour_sin=(2 * np.pi * hour / 24).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24).skb.apply_func(np.cos),
        month_sin=(2 * np.pi * month / 12).skb.apply_func(np.sin),
        month_cos=(2 * np.pi * month / 12).skb.apply_func(np.cos),
    ).drop(
        columns=["start_time", "record_id", "_raw_position", "_raw_size"]
    )

    # The original early-stopped on the same validation rows it reported, which
    # is leaky and cannot be reproduced honestly. Keep its 20% fraction and
    # patience by carving an eval set from each outer fold's training rows.
    X_y = features.skb.apply(GetXY(test_size=0.2, random_state=42), y=y_log, how="no_wrap")
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y_log)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
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
        restore_raw_predictions,
        skrub.eval_mode(),
    )

    # 4. Score. The splitter attached to mark_as_X drives the single holdout.
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
