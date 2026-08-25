import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit


CHUNK_SIZE = 2_000_000
MAX_ROWS_PER_CHUNK = 400_000
MAX_TOTAL_ROWS = 25_000_000


def cap_chunk_rows(chunk):
    """Reproduce the original per-CSV-chunk sampling rule."""
    if len(chunk) > MAX_ROWS_PER_CHUNK:
        return chunk.sample(n=MAX_ROWS_PER_CHUNK, random_state=42)
    return chunk


class PandasShuffledHoldout(BaseCrossValidator):
    """Reproduce sample(frac=1, random_state=...) followed by an 80/20 slice."""

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


class SpatialClusterFeatures(TransformerMixin, BaseEstimator):
    """Fit the two original KMeans models and append their derived features."""

    def __init__(self, n_clusters=20, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        self.origin_kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        ).fit(X[["origin_x", "origin_y"]])

        self.dest_kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        ).fit(X[["dest_x", "dest_y"]])
        return self

    def transform(self, X):
        result = X.copy()

        origin_cluster = self.origin_kmeans_.predict(
            result[["origin_x", "origin_y"]]
        )
        dest_cluster = self.dest_kmeans_.predict(
            result[["dest_x", "dest_y"]]
        )

        origin_centers = self.origin_kmeans_.cluster_centers_[origin_cluster]
        dest_centers = self.dest_kmeans_.cluster_centers_[dest_cluster]

        # Assign sequentially to preserve the original column order.
        result["origin_cluster"] = origin_cluster
        result["dest_cluster"] = dest_cluster
        result["origin_centroid_dist"] = np.sqrt(
            (result["origin_x"].to_numpy() - origin_centers[:, 0]) ** 2
            + (result["origin_y"].to_numpy() - origin_centers[:, 1]) ** 2
        )
        result["dest_centroid_dist"] = np.sqrt(
            (result["dest_x"].to_numpy() - dest_centers[:, 0]) ** 2
            + (result["dest_y"].to_numpy() - dest_centers[:, 1]) ** 2
        )
        result["origin_cluster_x_unit"] = (
            result["origin_cluster"] * result["unit_count"]
        )
        result["dest_cluster_x_unit"] = (
            result["dest_cluster"] * result["unit_count"]
        )
        return result


class ClusterCostFeatures(TransformerMixin, BaseEstimator):
    """Learn the original cluster-level target statistics per training fold."""

    def fit(self, X, y):
        target = pd.Series(np.asarray(y).reshape(-1), index=X.index)

        origin_stats = target.groupby(X["origin_cluster"]).agg(["mean", "std"])
        dest_stats = target.groupby(X["dest_cluster"]).agg(["mean", "std"])

        self.origin_cost_mean_ = origin_stats["mean"].to_dict()
        self.origin_cost_std_ = origin_stats["std"].fillna(0).to_dict()
        self.dest_cost_mean_ = dest_stats["mean"].to_dict()
        self.dest_cost_std_ = dest_stats["std"].fillna(0).to_dict()

        self.global_mean_cost_ = target.mean()
        self.global_std_cost_ = target.std()
        return self

    def transform(self, X):
        result = X.copy()

        # Assign sequentially to preserve the original feature names and order.
        result["origin_cluster_cost_mean"] = (
            result["origin_cluster"]
            .map(self.origin_cost_mean_)
            .fillna(self.global_mean_cost_)
        )
        result["origin_cluster_cost_std"] = (
            result["origin_cluster"]
            .map(self.origin_cost_std_)
            .fillna(self.global_std_cost_)
        )
        result["dest_cluster_cost_mean"] = (
            result["dest_cluster"]
            .map(self.dest_cost_mean_)
            .fillna(self.global_mean_cost_)
        )
        result["dest_cluster_cost_std"] = (
            result["dest_cluster"]
            .map(self.dest_cost_std_)
            .fillna(self.global_std_cost_)
        )
        result["origin_cluster_x_hour"] = (
            result["origin_cluster"] * result["hour"]
        )
        result["dest_cluster_x_hour"] = (
            result["dest_cluster"] * result["hour"]
        )
        result["origin_cluster_x_dow"] = (
            result["origin_cluster"] * result["dayofweek"]
        )
        result["dest_cluster_x_dow"] = (
            result["dest_cluster"] * result["dayofweek"]
        )
        return result


class Float32Finite(TransformerMixin, BaseEstimator):
    """Reproduce values.astype(float32) followed by np.nan_to_num."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return pd.DataFrame(
            values,
            columns=self.columns_,
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from this outer fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        # This is an internal early-stopping split of the current outer fold's
        # training rows. It is distinct from the validation split used to score.
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        fit_idx, eval_idx = next(splitter.split(X, y))

        # Preserve the raw target domain. The original float32 conversion did
        # not mathematically transform cost, and filtering already guarantees
        # finite, positive target values.
        target = pd.Series(np.asarray(y).reshape(-1), index=X.index)

        return {
            "X": X.take(fit_idx),
            "X_val": X.take(eval_idx),
            "y": target.take(fit_idx),
            "y_val": target.take(eval_idx),
        }

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def restore_raw_target_domain(prediction, mode):
    """Keep predictions in the raw cost domain used by mark_as_y."""
    if mode == "fit":
        return prediction
    # The fit target was only partitioned for early stopping, not numerically
    # transformed, so the required inverse mapping is the identity.
    return prediction


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read. Test-set processing, parquet files,
    #    submission generation, directory creation, and progress printing are
    #    omitted because they do not contribute to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Reproduce the original chunk boundaries and row-selection
    #    effects without retaining its chunked I/O. Filtering must happen before
    #    the marks because it changes which rows are scored.
    data = data.assign(
        _chunk=data.index // CHUNK_SIZE,
        origin_x=data["origin_x"].skb.apply_func(
            pd.to_numeric, errors="coerce"
        ),
        origin_y=data["origin_y"].skb.apply_func(
            pd.to_numeric, errors="coerce"
        ),
        dest_x=data["dest_x"].skb.apply_func(
            pd.to_numeric, errors="coerce"
        ),
        dest_y=data["dest_y"].skb.apply_func(
            pd.to_numeric, errors="coerce"
        ),
    )

    data = data[(data["cost"] > 0) & (data["cost"] < 50_000)]
    data = data[
        data["origin_x"].between(-180, 180)
        & data["origin_y"].between(-180, 180)
        & data["dest_x"].between(-180, 180)
        & data["dest_y"].between(-180, 180)
    ]

    # Each original 2M-row chunk was independently capped at 400k rows.
    sampled = (
        data.groupby("_chunk", group_keys=True)
        .apply(cap_chunk_rows, include_groups=False)
        .reset_index(level=0)
    )

    # The original stopped after appending the first chunk that made the running
    # total reach 25M rows. Include chunks whose cumulative count beforehand was
    # still below that threshold.
    rows_per_chunk = sampled["_chunk"].value_counts(sort=False).sort_index()
    rows_before_chunk = rows_per_chunk.cumsum() - rows_per_chunk
    included_chunks = rows_before_chunk[
        rows_before_chunk < MAX_TOTAL_ROWS
    ].index
    filtered = (
        sampled[sampled["_chunk"].isin(included_chunks)]
        .drop(columns=["_chunk"])
        .reset_index(drop=True)
    )

    # Mark the RAW target. The custom outer splitter reproduces the original
    # pandas shuffle followed by first-80%/last-20% slicing.
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["cost"]).skb.mark_as_X(
        cv=PandasShuffledHoldout(
            train_fraction=0.8,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
        utc=True,
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

    hour = start_time.dt.hour
    dayofweek = start_time.dt.dayofweek
    dx = dest_x - origin_x
    dy = dest_y - origin_y
    euclidean_dist = (dx**2 + dy**2).skb.apply_func(np.sqrt)
    manhattan_dist = (
        dx.skb.apply_func(np.abs)
        + dy.skb.apply_func(np.abs)
    )

    features = X.assign(
        start_time=start_time,
        hour=hour,
        dayofweek=dayofweek,
        month=start_time.dt.month,
        year=start_time.dt.year,
        day=start_time.dt.day,
        hour_sin=(2 * np.pi * hour / 24.0).skb.apply_func(np.sin),
        hour_cos=(2 * np.pi * hour / 24.0).skb.apply_func(np.cos),
        dow_sin=(2 * np.pi * dayofweek / 7.0).skb.apply_func(np.sin),
        dow_cos=(2 * np.pi * dayofweek / 7.0).skb.apply_func(np.cos),
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
        origin_x_bin=origin_x.skb.apply_func(np.round, 2),
        origin_y_bin=origin_y.skb.apply_func(np.round, 2),
        dest_x_bin=dest_x.skb.apply_func(np.round, 2),
        dest_y_bin=dest_y.skb.apply_func(np.round, 2),
        dx=dx,
        dy=dy,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=dy.skb.apply_func(np.arctan2, dx),
        unit_count_x_dist=X["unit_count"] * euclidean_dist,
        unit_count_x_manhattan=X["unit_count"] * manhattan_dist,
        unit_count_x_hour=X["unit_count"] * hour,
    )

    # KMeans and target-statistic features were fitted before validation in the
    # original. Recording them as estimators fixes that leakage: every outer
    # fold learns them only from its own training rows.
    features = features.skb.apply(
        SpatialClusterFeatures(
            n_clusters=20,
            random_state=42,
            n_init=10,
        )
    )
    features = features.skb.apply(
        ClusterCostFeatures(),
        y=y,
    )

    # These are exactly the columns excluded from the original model matrix.
    features = features.drop(columns=["record_id", "start_time"])
    features = features.skb.apply(Float32Finite())

    # The original early-stopped on the same holdout it reported, which is
    # leaky and cannot be reproduced honestly by outer CV. Keep the original
    # n_estimators=2500, patience=20, and 20% eval fraction, but carve the eval
    # set only from each outer fold's training rows.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=2500,
        num_leaves=511,
        learning_rate=0.015,
        min_child_samples=100,
        min_child_weight=10.0,
        feature_fraction_bynode=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )

    pred_fit_domain = X_fit.skb.apply(
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

    # The raw target was marked before the internal split. Explicitly map the
    # prediction node back to that target domain, using the identity inverse
    # because no mathematical target transform was applied.
    pred = pred_fit_domain.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here — the splitter on mark_as_X drives.
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
