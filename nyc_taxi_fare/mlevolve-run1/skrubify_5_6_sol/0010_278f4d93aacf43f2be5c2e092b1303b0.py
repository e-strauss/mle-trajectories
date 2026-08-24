import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import BaseCrossValidator


CHUNK_SIZE = 2_000_000
MAX_ROWS_PER_CHUNK = 400_000
MAX_ACCUMULATED_ROWS = 15_000_000


def sample_original_chunk(group):
    """Reproduce the original per-CSV-chunk sampling rule."""
    if len(group) > MAX_ROWS_PER_CHUNK:
        return group.sample(n=MAX_ROWS_PER_CHUNK, random_state=42)
    return group


class OriginalShuffledHoldout(BaseCrossValidator):
    """Reproduce sample(frac=1, random_state=42), then an 80/20 slice."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        shuffled_indices = rng.choice(len(X), size=len(X), replace=False)
        split_idx = int(len(X) * self.train_fraction)
        yield shuffled_indices[:split_idx], shuffled_indices[split_idx:]


class OptionalUnitCountInteraction(TransformerMixin, BaseEstimator):
    """Add the conditional unit_count feature exactly when the column exists."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if "unit_count" in X_out.columns:
            X_out["unit_count_x_dist"] = (
                X_out["unit_count"] * X_out["euclidean_dist"]
            )
        return X_out


class Float32FiniteMatrix(TransformerMixin, BaseEstimator):
    """Match values.astype(float32) followed by np.nan_to_num."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        # Return a DataFrame, not the bare ndarray: skrub rejects a transformer
        # that takes a pandas container and hands back numpy. Same values, same
        # float32 dtype, same column names and order.
        return pd.DataFrame(values, columns=X.columns, index=X.index)


class Float32FiniteLGBMRegressor(RegressorMixin, BaseEstimator):
    """Apply the original target conversion inside each fold's estimator fit."""

    def __init__(
        self,
        n_estimators=600,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # The original converted the training target to float32 and replaced
        # non-finite values immediately before fitting. Keeping this conversion
        # inside the estimator lets the plan mark and score against the raw
        # target while repeating the fit-time conversion independently per fold.
        y_fit = np.asarray(y, dtype=np.float32).reshape(-1)
        y_fit = np.nan_to_num(y_fit, nan=0.0, posinf=0.0, neginf=0.0)

        self.estimator_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X, y_fit)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one full CSV read. Test-set processing, parquet
    #    intermediates, directory creation, and submission generation are dropped
    #    because they do not contribute to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # The original physically read two-million-row chunks. Retain the
    # score-affecting chunk boundaries, per-chunk filtering/sampling, and 15M-row
    # stopping rule while dropping the chunked I/O mechanism itself.
    data = data.assign(_raw_chunk=data.index // CHUNK_SIZE)

    # Original target and coordinate row filters. These change which rows are
    # scored, so they occur before mark_as_X / mark_as_y.
    data = data[(data["cost"] > 0) & (data["cost"] < 50_000)]

    origin_x = data["origin_x"].skb.apply_func(pd.to_numeric, errors="coerce")
    origin_y = data["origin_y"].skb.apply_func(pd.to_numeric, errors="coerce")
    dest_x = data["dest_x"].skb.apply_func(pd.to_numeric, errors="coerce")
    dest_y = data["dest_y"].skb.apply_func(pd.to_numeric, errors="coerce")

    data = data.assign(
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
    )
    coordinate_mask = (
        data["origin_x"].between(-180, 180)
        & data["origin_y"].between(-180, 180)
        & data["dest_x"].between(-180, 180)
        & data["dest_y"].between(-180, 180)
    )
    data = data[coordinate_mask]

    # Reproduce `if len(chunk) > 400_000: chunk.sample(...)` independently for
    # each original CSV chunk.
    # group_keys=True + reset_index(level=0) keeps the grouping column: pandas 3
    # excludes a label-keyed grouper from the frames it passes to apply().
    data = (
        data.groupby("_raw_chunk", sort=True, group_keys=True)
        .apply(sample_original_chunk)
        .reset_index(level=0)
        .reset_index(drop=True)
    )

    # The eager loop stopped after the first complete sampled chunk that brought
    # the accumulated row count to at least 15 million.
    rows_per_chunk = data["_raw_chunk"].value_counts(sort=False).sort_index()
    cumulative_rows = rows_per_chunk.cumsum()
    included_chunks = rows_per_chunk.index[
        (cumulative_rows - rows_per_chunk) < MAX_ACCUMULATED_ROWS
    ]
    data = data[data["_raw_chunk"].isin(included_chunks)].reset_index(drop=True)
    data = data.drop(columns=["_raw_chunk"])

    # 2. Prepare Data — mark the RAW target. The custom one-fold splitter
    #    represents the original seeded full-table shuffle followed by the first
    #    80%/last 20% positional split.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=OriginalShuffledHoldout(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. Coordinate conversion,
    #    clipping, and fillna are retained even though filtering occurred above:
    #    their effects depend on the actual data and cannot be assumed to be
    #    no-ops.
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

    X_feat = X.assign(
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
    )

    dx = X_feat["dest_x"] - X_feat["origin_x"]
    dy = X_feat["dest_y"] - X_feat["origin_y"]
    X_feat = X_feat.assign(
        dx=dx,
        dy=dy,
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
        bearing=dy.skb.apply_func(np.arctan2, dx),
    )
    X_feat = X_feat.skb.apply(OptionalUnitCountInteraction())
    X_feat = X_feat.drop(columns=["start_time"])
    X_matrix = X_feat.skb.apply(Float32FiniteMatrix())

    # Same LightGBM model family and constructor hyperparameters. The original
    # used the outer validation rows as an eval_set for early stopping. A direct
    # DataOps estimator has no separate eval_set, so early stopping and logging
    # callbacks are removed while retaining n_estimators=600.
    model = Float32FiniteLGBMRegressor(
        n_estimators=600,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )
    pred = X_matrix.skb.apply(model, y=y)

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
            f"{-search.results_['mean_test_score'].iloc[0]}"
        )
