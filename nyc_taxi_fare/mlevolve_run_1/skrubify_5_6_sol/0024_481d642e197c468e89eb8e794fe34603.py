import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator, train_test_split


class PandasShuffle80Split(BaseCrossValidator):
    """Reproduce sample(frac=1, random_state=...) followed by an 80/20 slice."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        permutation = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield permutation[:split_idx], permutation[split_idx:]


class Float32Finite(TransformerMixin, BaseEstimator):
    """Reproduce values.astype(float32) followed by np.nan_to_num."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(values, columns=X.columns, index=X.index)


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


def restore_raw_target_predictions(values, mode):
    """Return predictions to the raw target's floating-point scoring domain."""
    if mode == "fit":
        return values
    return np.asarray(values, dtype=np.float64)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record a single CSV read. The test-set processing,
    #    intermediate parquet files, submission generation, directory creation,
    #    and progress output are omitted because they do not produce the
    #    cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Reproduce the original chunk membership without using chunked CSV reads.
    # Each raw 2,000,000-row block is filtered and capped independently below.
    data = data.assign(
        _original_order=data.index,
        _chunk=data.index // 2_000_000,
    )

    # The original filters rows before fitting or validation. These row-changing
    # operations therefore remain before mark_as_X/mark_as_y.
    filtered = data[data["cost"] > 0]
    filtered = filtered[filtered["cost"] < 50_000]

    origin_x_for_filter = filtered["origin_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(origin_x=origin_x_for_filter)
    filtered = filtered[
        (filtered["origin_x"] >= -180) & (filtered["origin_x"] <= 180)
    ]

    origin_y_for_filter = filtered["origin_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(origin_y=origin_y_for_filter)
    filtered = filtered[
        (filtered["origin_y"] >= -180) & (filtered["origin_y"] <= 180)
    ]

    dest_x_for_filter = filtered["dest_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(dest_x=dest_x_for_filter)
    filtered = filtered[
        (filtered["dest_x"] >= -180) & (filtered["dest_x"] <= 180)
    ]

    dest_y_for_filter = filtered["dest_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(dest_y=dest_y_for_filter)
    filtered = filtered[
        (filtered["dest_y"] >= -180) & (filtered["dest_y"] <= 180)
    ]

    # Cap each filtered raw chunk at 400,000 rows. Small chunks retain all rows;
    # large chunks use the original random_state=42 sampling rule.
    chunk_sizes = filtered.groupby("_chunk")["_chunk"].transform("size")
    small_chunks = filtered[chunk_sizes <= 400_000]
    large_chunks = filtered[chunk_sizes > 400_000]
    large_chunks = large_chunks.groupby(
        "_chunk", group_keys=False, sort=True
    ).sample(n=400_000, random_state=42)

    sampled = small_chunks.skb.concat([large_chunks], axis=0)
    sampled = sampled.sort_values("_chunk", kind="stable")

    # The eager loop stops after the first complete chunk that makes the
    # accumulated sampled row count reach 25,000,000.
    sampled_chunk_counts = sampled["_chunk"].value_counts(sort=False).sort_index()
    rows_before_chunk = sampled_chunk_counts.cumsum() - sampled_chunk_counts
    included_chunks = rows_before_chunk[rows_before_chunk < 25_000_000].index
    sampled = sampled[sampled["_chunk"].isin(included_chunks)]
    sampled = sampled.reset_index(drop=True).drop(
        columns=["_chunk", "_original_order"]
    )

    # 2. Prepare Data — mark the raw target and raw design matrix. The custom
    #    splitter exactly follows the original sample(frac=1, random_state=42)
    #    and positional 80/20 split, rather than ShuffleSplit's test-first
    #    permutation convention.
    y = sampled["cost"].skb.mark_as_y()
    X = sampled.drop(columns=["cost"]).skb.mark_as_X(
        cv=PandasShuffle80Split(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. The coordinate
    #    conversion is intentionally repeated here because engineer_features()
    #    did so in the original after the filtering pass.
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

    features = X.assign(
        start_time=start_time,
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
    )

    features = features.assign(
        hour=start_time.dt.hour,
        dayofweek=start_time.dt.dayofweek,
        month=start_time.dt.month,
        year=start_time.dt.year,
        day=start_time.dt.day,
    )

    features = features.assign(
        hour_sin=(2 * np.pi * features["hour"] / 24.0).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * features["hour"] / 24.0).skb.apply_func(np.cos),
        dow_sin=(
            2 * np.pi * features["dayofweek"] / 7.0
        ).skb.apply_func(np.sin),
        dow_cos=(
            2 * np.pi * features["dayofweek"] / 7.0
        ).skb.apply_func(np.cos),
    )

    features = features.assign(
        origin_x_bin=features["origin_x"].round(2),
        origin_y_bin=features["origin_y"].round(2),
        dest_x_bin=features["dest_x"].round(2),
        dest_y_bin=features["dest_y"].round(2),
    )

    dx = features["dest_x"] - features["origin_x"]
    dy = features["dest_y"] - features["origin_y"]
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)

    features = features.assign(
        dx=dx,
        dy=dy,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=dy.skb.apply_func(np.arctan2, dx),
        unit_count_x_dist=features["unit_count"] * euclidean_dist,
        unit_count_x_manhattan=features["unit_count"] * manhattan_dist,
        unit_count_x_hour=features["unit_count"] * features["hour"],
    )

    # Fit both spatial KMeans models inside each outer training fold. This
    # naturally fixes the original's leakage from fitting KMeans on the complete
    # table before its validation split.
    origin_distances = features[["origin_x", "origin_y"]].skb.apply(
        KMeans(n_clusters=20, random_state=42, n_init=10)
    )
    dest_distances = features[["dest_x", "dest_y"]].skb.apply(
        KMeans(n_clusters=20, random_state=42, n_init=10)
    )

    origin_cluster = origin_distances.skb.apply_func(np.argmin, axis=1)
    dest_cluster = dest_distances.skb.apply_func(np.argmin, axis=1)
    origin_centroid_dist = origin_distances.min(axis=1)
    dest_centroid_dist = dest_distances.min(axis=1)

    features = features.assign(
        origin_cluster=origin_cluster,
        dest_cluster=dest_cluster,
        origin_centroid_dist=origin_centroid_dist,
        dest_centroid_dist=dest_centroid_dist,
        centroid_dist_interaction=origin_centroid_dist * dest_centroid_dist,
    )

    # Match feature_cols and the original float32/np.nan_to_num conversion.
    features = features.drop(columns=["record_id", "start_time"])
    features = features.skb.apply(Float32Finite())

    # Marked y remains raw for scoring. This downstream conversion reproduces
    # the float32 and nan_to_num target passed to LightGBM; predictions are
    # explicitly mapped back to the raw target domain below.
    y_model = y.astype(np.float32).skb.apply_func(
        np.nan_to_num,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # The original early-stopped on the same validation rows it reported, which
    # is leaky and cannot be reproduced honestly by outer CV. Keep its 20-round
    # patience and 20% fraction by carving an eval set only from each fold's own
    # training rows.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y_model,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y_model)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=2500,
        num_leaves=511,
        learning_rate=0.015,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )

    pred_model_domain = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "callbacks": [
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(50),
            ],
        },
    )

    # The estimator was fitted on the float32/finite target representation, while
    # CV scores against the raw mark_as_y node. Map predictions back to the raw
    # target's floating-point domain. The fit-mode guard is required because a
    # prediction node evaluates to the fitted estimator during fitting.
    pred = pred_model_domain.skb.apply_func(
        restore_raw_target_predictions,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here—the splitter attached to mark_as_X drives.
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
