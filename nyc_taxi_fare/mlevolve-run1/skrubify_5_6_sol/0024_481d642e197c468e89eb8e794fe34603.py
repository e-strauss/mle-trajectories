import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit


class OriginalShuffledHoldout(BaseCrossValidator):
    """Reproduce pandas sample(frac=1), then an 80/20 positional split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        permutation = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield permutation[:split_idx], permutation[split_idx:]


class OriginalChunkCap(TransformerMixin, BaseEstimator):
    """Reproduce the original per-chunk sample cap and total-row cutoff."""

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
        kept = []
        total = 0

        for _, chunk in X.groupby("_chunk", sort=True):
            if len(chunk) > self.max_rows_per_chunk:
                chunk = chunk.sample(
                    n=self.max_rows_per_chunk,
                    random_state=self.random_state,
                )

            kept.append(chunk)
            total += len(chunk)

            # The original stops after the complete sampled chunk that reaches
            # or exceeds this threshold.
            if total >= self.max_total_rows:
                break

        if not kept:
            return X.drop(columns=["_chunk"]).reset_index(drop=True)

        return (
            pd.concat(kept, ignore_index=True)
            .drop(columns=["_chunk"])
            .reset_index(drop=True)
        )


class EngineerFeatures(TransformerMixin, BaseEstimator):
    """Reproduce the original named feature columns and their order."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["start_time"] = pd.to_datetime(
            df["start_time"], errors="coerce", utc=True
        )
        df["hour"] = df["start_time"].dt.hour
        df["dayofweek"] = df["start_time"].dt.dayofweek
        df["month"] = df["start_time"].dt.month
        df["year"] = df["start_time"].dt.year
        df["day"] = df["start_time"].dt.day

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

        # Preserve the original schema-conditional loop.
        for col in ["origin_x", "origin_y", "dest_x", "dest_y"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].clip(-180, 180)
                df[col] = df[col].fillna(0)

        df["origin_x_bin"] = np.round(df["origin_x"], 2)
        df["origin_y_bin"] = np.round(df["origin_y"], 2)
        df["dest_x_bin"] = np.round(df["dest_x"], 2)
        df["dest_y_bin"] = np.round(df["dest_y"], 2)

        df["dx"] = df["dest_x"] - df["origin_x"]
        df["dy"] = df["dest_y"] - df["origin_y"]
        df["euclidean_dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
        df["manhattan_dist"] = np.abs(df["dx"]) + np.abs(df["dy"])
        df["bearing"] = np.arctan2(df["dy"], df["dx"])

        if "unit_count" in df.columns:
            df["unit_count_x_dist"] = (
                df["unit_count"] * df["euclidean_dist"]
            )
            df["unit_count_x_manhattan"] = (
                df["unit_count"] * df["manhattan_dist"]
            )
            df["unit_count_x_hour"] = df["unit_count"] * df["hour"]

        return df


class SpatialClusterFeatures(TransformerMixin, BaseEstimator):
    """Fit the original two KMeans models and append spatial features."""

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
        df = X.copy()

        origin_coords = df[["origin_x", "origin_y"]].to_numpy()
        dest_coords = df[["dest_x", "dest_y"]].to_numpy()

        origin_cluster = self.kmeans_origin_.predict(origin_coords)
        dest_cluster = self.kmeans_dest_.predict(dest_coords)

        origin_centroids = self.kmeans_origin_.cluster_centers_[origin_cluster]
        dest_centroids = self.kmeans_dest_.cluster_centers_[dest_cluster]

        df["origin_cluster"] = origin_cluster
        df["dest_cluster"] = dest_cluster

        df["origin_centroid_dist"] = np.sqrt(
            (df["origin_x"].to_numpy() - origin_centroids[:, 0]) ** 2
            + (df["origin_y"].to_numpy() - origin_centroids[:, 1]) ** 2
        )
        df["dest_centroid_dist"] = np.sqrt(
            (df["dest_x"].to_numpy() - dest_centroids[:, 0]) ** 2
            + (df["dest_y"].to_numpy() - dest_centroids[:, 1]) ** 2
        )
        df["centroid_dist_interaction"] = (
            df["origin_centroid_dist"] * df["dest_centroid_dist"]
        )
        return df


class Float32NanToNum(TransformerMixin, BaseEstimator):
    """Reproduce values.astype(float32) and np.nan_to_num."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return pd.DataFrame(values, columns=X.columns, index=X.index)


class EarlyStoppingLGBMRegressor(RegressorMixin, BaseEstimator):
    """LightGBM with a per-fit inner validation split for early stopping."""

    def __init__(
        self,
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
        validation_fraction=0.2,
        stopping_rounds=20,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.validation_fraction = validation_fraction
        self.stopping_rounds = stopping_rounds

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float32)
        X_array = np.nan_to_num(
            X_array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # This float32 conversion is numerically in the same cost domain; it is
        # not a target-domain transform requiring prediction inversion.
        y_array = np.asarray(y, dtype=np.float32)
        y_array = np.nan_to_num(
            y_array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # The original early-stopped on the same holdout it scored, which is
        # leaky. Preserve its 20% fraction and patience using an inner split
        # drawn only from this outer fold's training rows.
        inner_split = ShuffleSplit(
            n_splits=1,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )
        fit_idx, eval_idx = next(inner_split.split(X_array, y_array))

        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        fit_options = {
            "eval_set": [(X_array[eval_idx], y_array[eval_idx])],
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=self.stopping_rounds,
                    verbose=False,
                )
            ],
        }
        self.model_.fit(
            X_array[fit_idx],
            y_array[fit_idx],
            **fit_options,
        )
        return self

    def predict(self, X):
        X_array = np.asarray(X, dtype=np.float32)
        X_array = np.nan_to_num(
            X_array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return self.model_.predict(X_array)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read of the original training path. Test-set
    #    processing, parquet dumps, directory creation, and submission output
    #    are omitted because they do not contribute to validation scoring.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. Row filtering and the original chunk-dependent sampling
    #    happen before marking because they change which rows are scored.
    chunked = data.assign(_chunk=data.index // 2_000_000)
    filtered = chunked[
        (chunked["cost"] > 0) & (chunked["cost"] < 50_000)
    ]

    origin_x = filtered["origin_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(origin_x=origin_x)
    filtered = filtered[
        (filtered["origin_x"] >= -180) & (filtered["origin_x"] <= 180)
    ]

    origin_y = filtered["origin_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(origin_y=origin_y)
    filtered = filtered[
        (filtered["origin_y"] >= -180) & (filtered["origin_y"] <= 180)
    ]

    dest_x = filtered["dest_x"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(dest_x=dest_x)
    filtered = filtered[
        (filtered["dest_x"] >= -180) & (filtered["dest_x"] <= 180)
    ]

    dest_y = filtered["dest_y"].skb.apply_func(
        pd.to_numeric, errors="coerce"
    )
    filtered = filtered.assign(dest_y=dest_y)
    filtered = filtered[
        (filtered["dest_y"] >= -180) & (filtered["dest_y"] <= 180)
    ]

    sampled = filtered.skb.apply(
        OriginalChunkCap(
            max_rows_per_chunk=400_000,
            max_total_rows=25_000_000,
            random_state=42,
        )
    )

    # Mark the RAW cost target. The custom splitter reproduces the original
    # sample(frac=1, random_state=42) followed by its positional 80/20 split.
    y = sampled["cost"].skb.mark_as_y()
    X = sampled.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalShuffledHoldout(
            train_fraction=0.8,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, clustering, numeric conversion, and model.
    features = X.skb.apply(EngineerFeatures())

    # The original fitted KMeans before splitting and therefore leaked validation
    # coordinates. As a recorded transformer, each KMeans is now fitted only on
    # the corresponding outer fold's training rows.
    features = features.skb.apply(
        SpatialClusterFeatures(
            n_clusters=20,
            random_state=42,
            n_init=10,
        )
    )
    features = features.drop(columns=["record_id", "start_time"])
    features = features.skb.apply(Float32NanToNum())

    model = EarlyStoppingLGBMRegressor(
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
        validation_fraction=0.2,
        stopping_rounds=20,
    )
    pred = features.skb.apply(model, y=y)

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
