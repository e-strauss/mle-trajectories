import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import BaseCrossValidator
from skrub import selectors as s


CHUNK_SIZE = 2_000_000
MAX_ROWS_PER_CHUNK = 400_000
MAX_TOTAL_ROWS = 10_000_000


def cap_chunk_rows(chunk):
    """Reproduce the original per-read-chunk sampling operation."""
    if len(chunk) > MAX_ROWS_PER_CHUNK:
        return chunk.sample(n=MAX_ROWS_PER_CHUNK, random_state=42)
    return chunk


class Float32Finite(TransformerMixin, BaseEstimator):
    """Reproduce DataFrame.values.astype(float32) and np.nan_to_num."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        # Return a DataFrame, not the bare ndarray: skrub rejects a transformer
        # that takes a pandas container and hands back numpy. Same values, same
        # float32 dtype, same column names and order.
        return pd.DataFrame(values, columns=X.columns, index=X.index)


class Float32TargetRegressor(RegressorMixin, BaseEstimator):
    """Convert the training target exactly as the original did inside each fit."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_float32 = np.asarray(y, dtype=np.float32)
        y_float32 = np.nan_to_num(
            y_float32, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_float32)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class ShuffledFirst80Split(BaseCrossValidator):
    """Match sample(frac=1, random_state=42), then take the first 80% for train."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(indices) * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the original chunked CSV ingestion is replaced by one
    #    recorded read. Test-set processing, parquet dumps, directory creation,
    #    and submission generation are omitted because they do not produce the
    #    validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Preserve the original physical 2,000,000-row chunk
    #    boundaries so its filtering, per-chunk 400,000-row cap, and
    #    10,000,000-retained-row stopping rule select the same training rows.
    chunked = data.assign(_source_chunk=data.index // CHUNK_SIZE)

    filtered = chunked[chunked["cost"] > 0]
    filtered = filtered[filtered["cost"] < 50_000]

    # Coordinate conversion and bounds filtering occurred before sampling in
    # the original and therefore remain before the X/y marks.
    origin_x = filtered["origin_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    origin_y = filtered["origin_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    dest_x = filtered["dest_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    dest_y = filtered["dest_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )

    filtered = filtered.assign(
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
    )
    valid_coordinates = (
        filtered["origin_x"].between(-180, 180)
        & filtered["origin_y"].between(-180, 180)
        & filtered["dest_x"].between(-180, 180)
        & filtered["dest_y"].between(-180, 180)
    )
    filtered = filtered[valid_coordinates]

    # GroupBy.apply reproduces the original single sampling operation on each
    # physical input chunk.
    # group_keys=True + reset_index keeps the grouping column: pandas 3 excludes
    # a label-keyed grouper from the frames it passes to apply(), so with
    # group_keys=False "_source_chunk" would be missing from the result.
    sampled = filtered.groupby(
        "_source_chunk", group_keys=True, sort=True
    ).apply(cap_chunk_rows).reset_index(level=0)

    retained_per_chunk = sampled.groupby("_source_chunk").size().sort_index()
    retained_before_chunk = retained_per_chunk.cumsum().shift(fill_value=0)
    included_chunks = retained_before_chunk[
        retained_before_chunk < MAX_TOTAL_ROWS
    ].index
    prepared = sampled[
        sampled["_source_chunk"].isin(included_chunks)
    ].reset_index(drop=True)
    prepared = prepared.drop(columns=["_source_chunk"])

    # Mark the RAW target as required. The original float32/nan_to_num target
    # conversion is performed inside Float32TargetRegressor.fit, so it is
    # recomputed per fold without changing the scoring domain.
    y = prepared["cost"].skb.mark_as_y()

    # The custom one-fold splitter exactly reproduces the original seeded
    # shuffle followed by the first-80%/last-20% positional split.
    X = prepared.drop(columns=["cost"]).skb.mark_as_X(
        cv=ShuffledFirst80Split(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Explicit conversions preserve the
    # original dtype handling; clipping and fillna are retained exactly.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce", utc=True
    )
    origin_x = (
        X["origin_x"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )
    origin_y = (
        X["origin_y"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )
    dest_x = (
        X["dest_x"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )
    dest_y = (
        X["dest_y"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .clip(-180, 180)
        .fillna(0)
    )

    dx = dest_x - origin_x
    dy = dest_y - origin_y

    features = X.assign(
        start_time=start_time,
        hour=start_time.dt.hour,
        dayofweek=start_time.dt.dayofweek,
        month=start_time.dt.month,
        year=start_time.dt.year,
        day=start_time.dt.day,
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
        dx=dx,
        dy=dy,
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=(
            dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
        ),
    )

    # Match the dynamically discovered feature_cols: every column except the
    # identifier and timestamp. The target is already absent -- mark_as_X was
    # given `prepared.drop(columns=["cost"])` -- and s.cols() raises on a column
    # that is not there, so "cost" must NOT be listed here.
    model_features = features.skb.drop(
        s.cols("record_id", "start_time")
    ).skb.apply(Float32Finite())

    # Same LightGBM family and constructor hyperparameters. The original early
    # stopping used the outer validation rows as eval_set; a direct CV estimator
    # has no separate eval_set, so early stopping and logging callbacks are
    # removed while n_estimators=300 is retained.
    base_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model = Float32TargetRegressor(estimator=base_model)
    pred = model_features.skb.apply(model, y=y)

    # 4. Score. No cv= here — the splitter attached to mark_as_X drives.
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
