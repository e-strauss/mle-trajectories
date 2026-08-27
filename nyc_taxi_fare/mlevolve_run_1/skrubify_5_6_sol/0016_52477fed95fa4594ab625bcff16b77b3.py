import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit


class CoordinateCleaner(TransformerMixin, BaseEstimator):
    """Reproduce the original conditional coordinate conversion and cleaning."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for col in ("origin_x", "origin_y", "dest_x", "dest_y"):
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                result[col] = result[col].clip(-180, 180)
                result[col] = result[col].fillna(0)
        return result


class EngineerFeatures(TransformerMixin, BaseEstimator):
    """Create the original named features in their original column order."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()

        result["start_time"] = pd.to_datetime(
            result["start_time"],
            errors="coerce",
            utc=True,
        )
        result["hour"] = result["start_time"].dt.hour
        result["dayofweek"] = result["start_time"].dt.dayofweek
        result["month"] = result["start_time"].dt.month
        result["year"] = result["start_time"].dt.year
        result["day"] = result["start_time"].dt.day

        result["hour_sin"] = np.sin(
            2 * np.pi * result["hour"] / 24.0
        )
        result["hour_cos"] = np.cos(
            2 * np.pi * result["hour"] / 24.0
        )
        result["dow_sin"] = np.sin(
            2 * np.pi * result["dayofweek"] / 7.0
        )
        result["dow_cos"] = np.cos(
            2 * np.pi * result["dayofweek"] / 7.0
        )

        # The original conditionally handled these columns. Keep that
        # data-dependent branch because the data is unavailable at plan build time.
        for col in ("origin_x", "origin_y", "dest_x", "dest_y"):
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                result[col] = result[col].clip(-180, 180)
                result[col] = result[col].fillna(0)

        result["origin_x_bin"] = np.round(result["origin_x"], 2)
        result["origin_y_bin"] = np.round(result["origin_y"], 2)
        result["dest_x_bin"] = np.round(result["dest_x"], 2)
        result["dest_y_bin"] = np.round(result["dest_y"], 2)

        result["dx"] = result["dest_x"] - result["origin_x"]
        result["dy"] = result["dest_y"] - result["origin_y"]
        result["euclidean_dist"] = np.sqrt(
            result["dx"] ** 2 + result["dy"] ** 2
        )
        result["manhattan_dist"] = (
            np.abs(result["dx"]) + np.abs(result["dy"])
        )
        result["bearing"] = np.arctan2(
            result["dy"],
            result["dx"],
        )

        if "unit_count" in result.columns:
            result["unit_count_x_dist"] = (
                result["unit_count"] * result["euclidean_dist"]
            )
            result["unit_count_x_manhattan"] = (
                result["unit_count"] * result["manhattan_dist"]
            )
            result["unit_count_x_hour"] = (
                result["unit_count"] * result["hour"]
            )

        return result


class AddSpatialClusters(TransformerMixin, BaseEstimator):
    """Fit and apply the original origin and destination KMeans models."""

    def __init__(self, n_clusters=20, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        self.origin_kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.dest_kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.origin_kmeans_.fit(
            X[["origin_x", "origin_y"]].to_numpy()
        )
        self.dest_kmeans_.fit(
            X[["dest_x", "dest_y"]].to_numpy()
        )
        return self

    def transform(self, X):
        result = X.copy()
        result["origin_cluster"] = self.origin_kmeans_.predict(
            result[["origin_x", "origin_y"]].to_numpy()
        )
        result["dest_cluster"] = self.dest_kmeans_.predict(
            result[["dest_x", "dest_y"]].to_numpy()
        )
        return result


class FiniteFloat32(TransformerMixin, BaseEstimator):
    """Reproduce values.astype(float32) followed by np.nan_to_num."""

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
        return pd.DataFrame(
            values,
            columns=X.columns,
            index=X.index,
        )


class ShuffledTailHoldout(BaseCrossValidator):
    """Reproduce sample(frac=1), then the original positional 80/20 split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_rows = len(X)
        random_state = np.random.RandomState(self.random_state)
        shuffled = random_state.choice(
            n_rows,
            size=n_rows,
            replace=False,
        )
        split_idx = int(n_rows * self.train_fraction)
        yield shuffled[:split_idx], shuffled[split_idx:]


class GetXY(TransformerMixin, BaseEstimator):
    """Carve LightGBM's eval set from the current outer training fold."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_idx, val_idx = next(splitter.split(X, y))
        return {
            "X": X.iloc[train_idx],
            "X_val": X.iloc[val_idx],
            "y": y.iloc[train_idx],
            "y_val": y.iloc[val_idx],
        }

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def restore_raw_target_domain(predictions, mode):
    """Undo the target's precision-only conversion.

    Conversion to float32 does not change units, so the inverse is identity.
    """
    if mode == "fit":
        return predictions
    return predictions


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read. The original's chunked I/O, test-set
    #    processing, parquet dumps, directory creation, and submission generation
    #    are omitted because they are not part of producing a validation score.
    data = skrub.as_data_op(
        "./input/train.csv"
    ).skb.apply_func(pd.read_csv)

    # 2. Prepare Data — row filtering must precede both marks because it changes
    #    which rows are scored. Coordinate conversion is performed before the
    #    range conditions, exactly as in the original training loop.
    numeric_coords = data.skb.apply(CoordinateCleaner())

    valid_cost = (
        (numeric_coords["cost"] > 0)
        & (numeric_coords["cost"] < 50_000)
    )
    valid_origin_x = numeric_coords["origin_x"].between(-180, 180)
    valid_origin_y = numeric_coords["origin_y"].between(-180, 180)
    valid_dest_x = numeric_coords["dest_x"].between(-180, 180)
    valid_dest_y = numeric_coords["dest_y"].between(-180, 180)

    prepared = numeric_coords[
        valid_cost
        & valid_origin_x
        & valid_origin_y
        & valid_dest_x
        & valid_dest_y
    ].reset_index(drop=True)

    # Mark the RAW target. The custom splitter reproduces the original full-table
    # deterministic shuffle and positional 80/20 holdout.
    y = prepared["cost"].skb.mark_as_y()
    X = prepared.drop(columns=["cost"]).skb.mark_as_X(
        cv=ShuffledTailHoldout(
            train_fraction=0.8,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Feature engineering and preprocessing. The named-column construction is
    #    kept in a transformer because the original includes data-dependent
    #    column-existence branches and appends columns in a specific order.
    features = X.skb.apply(EngineerFeatures())

    # The original fitted KMeans before splitting, leaking validation coordinates.
    # Recording it after mark_as_X fixes that leak by refitting on each fold's
    # training rows while preserving both KMeans configurations.
    features = features.skb.apply(
        AddSpatialClusters(
            n_clusters=20,
            random_state=42,
            n_init=10,
        )
    )

    features = features.drop(
        columns=["record_id", "start_time"]
    )
    features = features.skb.apply(FiniteFloat32())

    # Reproduce the original float32 target storage and finite-value replacement
    # after marking the raw target.
    y_model = y.astype(np.float32)
    y_model = y_model.fillna(0.0)
    y_model = y_model.replace([np.inf, -np.inf], 0.0)

    # The original early-stopped on the same holdout it reported, which leaks and
    # cannot be reproduced honestly. Keep its 20% eval fraction, random seed,
    # 20-round patience, and 1,500-tree cap by carving an eval set only from each
    # outer fold's training rows.
    split_data = features.skb.apply(
        GetXY(
            test_size=0.2,
            random_state=42,
        ),
        y=y_model,
        how="no_wrap",
    )
    X_fit = split_data["X"]
    X_val = split_data["X_val"]
    y_fit = split_data.get("y", y_model)
    y_val = split_data["y_val"]

    model = lgb.LGBMRegressor(
        n_estimators=1500,
        num_leaves=255,
        learning_rate=0.02,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
    )

    pred_model_domain = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=20,
                    verbose=True,
                ),
            ],
        },
    )

    # Predictions are already in cost units, so inversion of the precision-only
    # target conversion is identity. Gate it because a prediction node evaluates
    # to the fitted estimator in fit mode.
    pred = pred_model_domain.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here — the splitter on mark_as_X drives validation.
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
