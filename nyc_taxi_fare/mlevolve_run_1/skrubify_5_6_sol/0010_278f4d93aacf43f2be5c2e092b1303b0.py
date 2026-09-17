import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    BaseCrossValidator,
    train_test_split,
)


class PandasShuffledHoldout(BaseCrossValidator):
    """Reproduce pandas sample(frac=1) followed by the positional 80/20 split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = (
            pd.Series(np.arange(len(X)))
            .sample(frac=1.0, random_state=self.random_state)
            .to_numpy()
        )
        split_idx = int(len(indices) * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


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


def cap_chunk(group):
    """Reproduce the independent 400,000-row sample within each CSV chunk."""
    group = group.drop(columns=["_chunk"], errors="ignore")
    if len(group) > 400_000:
        return group.sample(n=400_000, random_state=42)
    return group


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read. Test-set processing, intermediate
    #    parquet files, submission generation, directory creation, and progress
    #    printing are omitted because they do not contribute to validation.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Reproduce the original 2,000,000-row chunk boundaries while recording a
    # single read. The helper column is removed before marking the design matrix.
    data = data.assign(_chunk=data.index // 2_000_000)

    # Row filtering changes which rows are scored, so it must remain before
    # mark_as_X / mark_as_y.
    filtered = data[(data["cost"] > 0) & (data["cost"] < 50_000)]

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
    filtered = filtered[
        (filtered["origin_x"] >= -180)
        & (filtered["origin_x"] <= 180)
        & (filtered["origin_y"] >= -180)
        & (filtered["origin_y"] <= 180)
        & (filtered["dest_x"] >= -180)
        & (filtered["dest_x"] <= 180)
        & (filtered["dest_y"] >= -180)
        & (filtered["dest_y"] <= 180)
    ]

    # The original reset random_state=42 independently for each chunk.
    sampled = (
        filtered.groupby("_chunk", group_keys=True)
        .apply(cap_chunk)
        .reset_index(level=0)
    )

    # Keep complete chunks until the accumulated retained-row count first
    # reaches 15,000,000, including the threshold-crossing chunk in full.
    chunk_sizes = sampled.groupby("_chunk").size()
    rows_before_chunk = chunk_sizes.cumsum() - chunk_sizes
    included_chunks = rows_before_chunk[rows_before_chunk < 15_000_000].index
    sampled = sampled[sampled["_chunk"].isin(included_chunks)]
    sampled = sampled.drop(columns=["_chunk"]).reset_index(drop=True)

    # 2. Prepare Data — mark the raw target and design matrix. The custom
    #    splitter reproduces sample(frac=1, random_state=42), followed by
    #    train[:int(0.8*n)] and validation[int(0.8*n):].
    y = sampled["cost"].skb.mark_as_y()
    X = sampled.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=PandasShuffledHoldout(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
        utc=True,
    )
    X_features = X.assign(
        start_time=start_time,
        hour=start_time.dt.hour,
        dayofweek=start_time.dt.dayofweek,
        month=start_time.dt.month,
        year=start_time.dt.year,
        day=start_time.dt.day,
    )

    # These coordinates are required by the original's subsequent dx/dy
    # expressions, so a successful original execution contains all four.
    X_features = X_features.assign(
        origin_x=(
            X_features["origin_x"]
            .skb.apply_func(pd.to_numeric, errors="coerce")
            .clip(-180, 180)
            .fillna(0)
        ),
        origin_y=(
            X_features["origin_y"]
            .skb.apply_func(pd.to_numeric, errors="coerce")
            .clip(-180, 180)
            .fillna(0)
        ),
        dest_x=(
            X_features["dest_x"]
            .skb.apply_func(pd.to_numeric, errors="coerce")
            .clip(-180, 180)
            .fillna(0)
        ),
        dest_y=(
            X_features["dest_y"]
            .skb.apply_func(pd.to_numeric, errors="coerce")
            .clip(-180, 180)
            .fillna(0)
        ),
    )

    dx = X_features["dest_x"] - X_features["origin_x"]
    dy = X_features["dest_y"] - X_features["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)

    X_features = X_features.assign(
        dx=dx,
        dy=dy,
        euclidean_dist=euclidean_dist,
        manhattan_dist=(
            dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
        ),
        bearing=dy.skb.apply_func(np.arctan2, dx),
    )

    # Preserve the original schema-dependent unit_count branch without
    # assuming that the optional column exists.
    unit_count_x_dist = (
        X_features.skb.select(skrub.selectors.glob("unit_count"))
        .mul(euclidean_dist, axis=0)
        .add_suffix("_x_dist")
    )
    X_features = X_features.skb.concat([unit_count_x_dist], axis=1)
    X_features = X_features.drop(columns=["start_time"])

    # Reproduce values.astype(np.float32) and np.nan_to_num(..., 0), retaining
    # the pandas container and original feature order.
    X_features = (
        X_features.astype(np.float32)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )

    # The raw marked target is passed through unchanged. The preceding
    # cost > 0 and cost < 50000 filters already exclude NaN and infinities, so
    # the original target-side nan_to_num was a no-op for all retained rows.
    # Avoiding a target transform keeps predictions and scoring in the same
    # raw-cost domain.
    #
    # The original used the scored validation rows for early stopping, which is
    # leaky and cannot be reproduced honestly in outer CV. Keep its 20% split,
    # 600 trees, and patience of 20, but carve the eval set only from each
    # outer fold's training rows.
    X_y = X_features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=600,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )
    pred = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "callbacks": [lgb.early_stopping(20, verbose=False)],
        },
    )

    # 4. Score. No cv= here — the splitter on mark_as_X drives. The original
    #    LightGBM logging callback only printed progress and is therefore
    #    omitted. sklearn's scorer is negative because grid-search scores are
    #    always higher-is-better.
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
