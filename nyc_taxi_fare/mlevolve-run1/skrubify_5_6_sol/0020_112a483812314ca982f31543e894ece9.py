import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import ShuffleSplit, train_test_split


CHUNK_SIZE = 2_000_000
MAX_ROWS_PER_CHUNK = 400_000
MAX_TOTAL_ROWS = 25_000_000


class OriginalChunkSampler(TransformerMixin, BaseEstimator):
    """Reproduce the original random cap and early stop across CSV chunks."""

    def __init__(
        self,
        max_rows_per_chunk=400_000,
        max_total_rows=25_000_000,
        random_state=42,
    ):
        self.max_rows_per_chunk = max_rows_per_chunk
        self.max_total_rows = max_total_rows
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        chunks = []
        total_rows = 0

        # This loop is inside a transformer because the chunk labels and number
        # of chunks are discovered from the actual data. It exactly retains the
        # original per-chunk random sampling, accumulated-row break, and output
        # row order.
        for _, chunk in X.groupby("_chunk", sort=True):
            chunk = chunk.drop(columns=["_chunk"])
            if len(chunk) > self.max_rows_per_chunk:
                chunk = chunk.sample(
                    n=self.max_rows_per_chunk,
                    random_state=self.random_state,
                )

            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows >= self.max_total_rows:
                break

        return pd.concat(chunks, ignore_index=True)


class SpatialClusterFeatures(TransformerMixin, BaseEstimator):
    """Fit the two KMeans models and append the original cluster features."""

    def __init__(self, n_clusters=20, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        self.kmeans_origin_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.kmeans_dest_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.kmeans_origin_.fit(X[["origin_x", "origin_y"]].to_numpy())
        self.kmeans_dest_.fit(X[["dest_x", "dest_y"]].to_numpy())
        return self

    def transform(self, X):
        result = X.copy()

        origin_cluster = self.kmeans_origin_.predict(
            result[["origin_x", "origin_y"]].to_numpy()
        )
        dest_cluster = self.kmeans_dest_.predict(
            result[["dest_x", "dest_y"]].to_numpy()
        )

        origin_centers = self.kmeans_origin_.cluster_centers_[origin_cluster]
        dest_centers = self.kmeans_dest_.cluster_centers_[dest_cluster]

        # Preserve the original assignment order because feature order can affect
        # randomized feature subsampling in LightGBM.
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


class FiniteFloat32(TransformerMixin, BaseEstimator):
    """Reproduce X.values.astype(float32) followed by np.nan_to_num."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
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
            columns=self.feature_names_in_,
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
        # This is the estimator's inner early-stopping split, not the outer
        # validation split. It receives only the current outer fold's training
        # rows and therefore cannot expose scored rows to early stopping.
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def inverse_float32_target(prediction, mode):
    """Return predictions to the raw target's representation for scoring."""
    if mode == "fit":
        return prediction
    return np.asarray(prediction, dtype=np.float64)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the training CSV read. Test-set processing, parquet
    #    dumps, submission generation, directory creation, and progress printing
    #    are omitted because they do not contribute to the validation score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Reconstruct the original 2,000,000-row CSV chunk boundaries from row
    # positions. The helper column is removed by OriginalChunkSampler.
    data = data.assign(_chunk=data.index // CHUNK_SIZE)

    # 2. Prepare data — all content-dependent row filters occur before the marks
    #    because they determine which rows participate in validation.
    origin_x = data["origin_x"].skb.apply_func(
        pd.to_numeric,
        errors="coerce",
    )
    origin_y = data["origin_y"].skb.apply_func(
        pd.to_numeric,
        errors="coerce",
    )
    dest_x = data["dest_x"].skb.apply_func(
        pd.to_numeric,
        errors="coerce",
    )
    dest_y = data["dest_y"].skb.apply_func(
        pd.to_numeric,
        errors="coerce",
    )

    filtered = data.assign(
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
    )
    filtered = filtered[filtered["cost"] > 0]
    filtered = filtered[filtered["cost"] < 50000]
    filtered = filtered[
        (filtered["origin_x"] >= -180)
        & (filtered["origin_x"] <= 180)
    ]
    filtered = filtered[
        (filtered["origin_y"] >= -180)
        & (filtered["origin_y"] <= 180)
    ]
    filtered = filtered[
        (filtered["dest_x"] >= -180)
        & (filtered["dest_x"] <= 180)
    ]
    filtered = filtered[
        (filtered["dest_y"] >= -180)
        & (filtered["dest_y"] <= 180)
    ]

    # The original sampled at most 400,000 rows from each filtered chunk and
    # stopped after the first chunk taking the accumulated count to at least 25M.
    selected = filtered.skb.apply(
        OriginalChunkSampler(
            max_rows_per_chunk=MAX_ROWS_PER_CHUNK,
            max_total_rows=MAX_TOTAL_ROWS,
            random_state=42,
        )
    )

    # Mark the RAW target. The original shuffled the prepared table and retained
    # its first 80% for training and last 20% for validation. A one-split
    # ShuffleSplit is the corresponding outer validation scheme.
    y = selected["cost"].skb.mark_as_y()
    X = selected.drop(columns=["cost"]).skb.mark_as_X(
        cv=ShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering.
    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
        utc=True,
    )
    hour = start_time.dt.hour
    dayofweek = start_time.dt.dayofweek

    clean_origin_x = X["origin_x"].clip(-180, 180).fillna(0)
    clean_origin_y = X["origin_y"].clip(-180, 180).fillna(0)
    clean_dest_x = X["dest_x"].clip(-180, 180).fillna(0)
    clean_dest_y = X["dest_y"].clip(-180, 180).fillna(0)

    dx = clean_dest_x - clean_origin_x
    dy = clean_dest_y - clean_origin_y
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
        origin_x=clean_origin_x,
        origin_y=clean_origin_y,
        dest_x=clean_dest_x,
        dest_y=clean_dest_y,
        origin_x_bin=clean_origin_x.skb.apply_func(np.round, 2),
        origin_y_bin=clean_origin_y.skb.apply_func(np.round, 2),
        dest_x_bin=clean_dest_x.skb.apply_func(np.round, 2),
        dest_y_bin=clean_dest_y.skb.apply_func(np.round, 2),
        dx=dx,
        dy=dy,
        euclidean_dist=euclidean_dist,
        manhattan_dist=manhattan_dist,
        bearing=dy.skb.apply_func(np.arctan2, dx),
        unit_count_x_dist=X["unit_count"] * euclidean_dist,
        unit_count_x_manhattan=X["unit_count"] * manhattan_dist,
        unit_count_x_hour=X["unit_count"] * hour,
    )

    # The eager script fit KMeans before splitting, leaking validation
    # coordinates. Recording these models as a transformer refits them on each
    # outer fold's training rows while preserving all KMeans hyperparameters and
    # derived-feature formulas.
    features = features.skb.apply(
        SpatialClusterFeatures(
            n_clusters=20,
            random_state=42,
            n_init=10,
        )
    )

    # Match the original feature_cols list. The target is already absent from X.
    features = features.drop(columns=["record_id", "start_time"])
    features = features.skb.apply(FiniteFloat32())

    # The original converted its fitting target to float32. Marking occurred on
    # the raw target above; predictions are converted back to the raw target
    # representation below before scoring.
    y_float32 = y.astype(np.float32)

    # The original early-stopped on the same holdout it reported, which is leaky
    # and cannot be reproduced honestly in outer CV. Preserve the 20% fraction,
    # seed, 2,000-tree limit, and 20-round patience by taking the eval set only
    # from each outer fold's training rows.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y_float32,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y_float32)
    X_eval = X_y["X_val"]
    y_eval = X_y["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        num_leaves=255,
        learning_rate=0.02,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )

    pred_float32_domain = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_eval, y_eval)],
            "callbacks": [
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(50),
            ],
        },
    )

    # In fit mode the prediction node contains the fitted estimator, so the
    # inverse target operation must leave it unchanged.
    pred = pred_float32_domain.skb.apply_func(
        inverse_float32_target,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here; the ShuffleSplit attached to mark_as_X drives.
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
