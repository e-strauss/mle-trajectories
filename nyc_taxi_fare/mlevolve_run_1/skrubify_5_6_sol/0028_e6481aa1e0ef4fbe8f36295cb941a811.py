import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator


class OriginalPermutationSplit(BaseCrossValidator):
    """Reproduce the original seeded permutation and 80/20 outer split."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from each outer fold's training rows."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        train_indices = indices[:split_idx]
        validation_indices = indices[split_idx:]

        # take() preserves the exact permutation arithmetic used by the original
        # without implementing an outer fold loop; this is only the legitimate
        # inner split used for early stopping and blend-weight optimization.
        return {
            "X": X.take(train_indices).reset_index(drop=True),
            "X_val": X.take(validation_indices).reset_index(drop=True),
            "y": y.take(train_indices).reset_index(drop=True),
            "y_val": y.take(validation_indices).reset_index(drop=True),
        }

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class SpatialFeatureEngineer(TransformerMixin, BaseEstimator):
    """Fit the original spatial, frequency, and target-mean encodings."""

    def __init__(self, n_clusters=50, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    @staticmethod
    def _hour(frame):
        dt = pd.to_datetime(frame["start_time"], errors="coerce", utc=True)
        return dt.dt.hour.fillna(0).astype(int)

    def fit(self, X, y):
        X = X.copy()
        y_values = np.asarray(y).reshape(-1)

        self.origin_kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        ).fit(X[["origin_x", "origin_y"]])

        self.dest_kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        ).fit(X[["dest_x", "dest_y"]])

        origin_x_bin = np.round(X["origin_x"], 2)
        origin_y_bin = np.round(X["origin_y"], 2)
        dest_x_bin = np.round(X["dest_x"], 2)
        dest_y_bin = np.round(X["dest_y"], 2)

        self.o_x_freq_ = origin_x_bin.value_counts(normalize=True).to_dict()
        self.o_y_freq_ = origin_y_bin.value_counts(normalize=True).to_dict()
        self.d_x_freq_ = dest_x_bin.value_counts(normalize=True).to_dict()
        self.d_y_freq_ = dest_y_bin.value_counts(normalize=True).to_dict()

        cluster_hour = X.copy()
        cluster_hour["origin_cluster"] = self.origin_kmeans_.predict(
            cluster_hour[["origin_x", "origin_y"]]
        )
        cluster_hour["dest_cluster"] = self.dest_kmeans_.predict(
            cluster_hour[["dest_x", "dest_y"]]
        )
        cluster_hour["hour"] = self._hour(cluster_hour)
        cluster_hour["cost"] = y_values

        self.origin_hour_mean_ = (
            cluster_hour.groupby(["origin_cluster", "hour"])["cost"]
            .mean()
            .to_dict()
        )
        self.dest_hour_mean_ = (
            cluster_hour.groupby(["dest_cluster", "hour"])["cost"]
            .mean()
            .to_dict()
        )
        return self

    def transform(self, X):
        df = X.copy()

        if "start_time" in df.columns:
            dt = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
            df["hour"] = dt.dt.hour.fillna(0).astype(int)
            df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
            df["month"] = dt.dt.month.fillna(1).astype(int)
            df["year"] = dt.dt.year.fillna(2012).astype(int)
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        dx = df["dest_x"] - df["origin_x"]
        dy = df["dest_y"] - df["origin_y"]
        df["euclidean_dist"] = np.sqrt(dx**2 + dy**2)
        df["abs_dx"] = np.abs(dx)
        df["abs_dy"] = np.abs(dy)
        df["manhattan_dist"] = np.abs(dx) + np.abs(dy)
        df["distance_per_unit"] = df["euclidean_dist"] / (
            df["unit_count"] + 1e-5
        )

        df["origin_x_bin"] = np.round(df["origin_x"], 2)
        df["origin_y_bin"] = np.round(df["origin_y"], 2)
        df["dest_x_bin"] = np.round(df["dest_x"], 2)
        df["dest_y_bin"] = np.round(df["dest_y"], 2)

        df["origin_cluster"] = self.origin_kmeans_.predict(
            df[["origin_x", "origin_y"]]
        )
        df["dest_cluster"] = self.dest_kmeans_.predict(
            df[["dest_x", "dest_y"]]
        )

        if "hour" in df.columns:
            df["origin_cluster_hour_mean"] = [
                self.origin_hour_mean_.get((cluster, hour), 11.35)
                for cluster, hour in zip(df["origin_cluster"], df["hour"])
            ]
            df["dest_cluster_hour_mean"] = [
                self.dest_hour_mean_.get((cluster, hour), 11.35)
                for cluster, hour in zip(df["dest_cluster"], df["hour"])
            ]

        df["origin_x_bin_freq"] = (
            df["origin_x_bin"].map(self.o_x_freq_).fillna(0)
        )
        df["origin_y_bin_freq"] = (
            df["origin_y_bin"].map(self.o_y_freq_).fillna(0)
        )
        df["dest_x_bin_freq"] = (
            df["dest_x_bin"].map(self.d_x_freq_).fillna(0)
        )
        df["dest_y_bin_freq"] = (
            df["dest_y_bin"].map(self.d_y_freq_).fillna(0)
        )

        lat1 = np.radians(df["origin_y"])
        lon1 = np.radians(df["origin_x"])
        lat2 = np.radians(df["dest_y"])
        lon2 = np.radians(df["dest_x"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
        df["haversine_dist"] = c * 6371.0

        y_bearing = np.sin(dlon) * np.cos(lat2)
        x_bearing = (
            np.cos(lat1) * np.sin(lat2)
            - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )
        df["bearing"] = np.degrees(np.arctan2(y_bearing, x_bearing))

        df["unit_euclidean_interact"] = (
            df["unit_count"] * df["euclidean_dist"]
        )
        df["unit_manhattan_interact"] = (
            df["unit_count"] * df["manhattan_dist"]
        )
        df["unit_haversine_interact"] = (
            df["unit_count"] * df["haversine_dist"]
        )

        df = df.fillna(0)
        return df.drop(columns=["record_id", "start_time"])


class EarlyStoppedWeightedEnsemble(RegressorMixin, BaseEstimator):
    """Fit the original three regressors and optimize their blend weights."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y, eval_set=None):
        if not eval_set:
            raise ValueError(
                "EarlyStoppedWeightedEnsemble requires an inner eval_set."
            )

        X_validation_raw, y_validation = eval_set[0]
        y_train = pd.Series(np.asarray(y).reshape(-1)).reset_index(drop=True)
        y_validation = pd.Series(
            np.asarray(y_validation).reshape(-1)
        ).reset_index(drop=True)

        X_train_raw = X.reset_index(drop=True)
        X_validation_raw = X_validation_raw.reset_index(drop=True)

        self.feature_engineer_ = SpatialFeatureEngineer(
            n_clusters=50,
            random_state=42,
            n_init=10,
        )
        self.feature_engineer_.fit(X_train_raw, y_train)
        X_train = self.feature_engineer_.transform(X_train_raw)
        X_validation = self.feature_engineer_.transform(X_validation_raw)

        self.cb_model_ = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=7,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            task_type="CPU",
            thread_count=-1,
            verbose=False,
        )
        self.lgb_model_ = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
        )
        self.xgb_model_ = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        )

        self.cb_model_.fit(
            X_train,
            y_train,
            eval_set=(X_validation, y_validation),
            early_stopping_rounds=50,
            verbose=False,
        )
        self.lgb_model_.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        self.xgb_model_.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )

        validation_predictions = np.column_stack(
            [
                self.cb_model_.predict(X_validation),
                self.lgb_model_.predict(X_validation),
                self.xgb_model_.predict(X_validation),
            ]
        )

        def objective(weights):
            normalized_weights = weights / np.sum(weights)
            blended_prediction = validation_predictions @ normalized_weights
            return np.sqrt(
                mean_squared_error(y_validation, blended_prediction)
            )

        result = minimize(
            objective,
            [1 / 3, 1 / 3, 1 / 3],
            bounds=[(0, 1), (0, 1), (0, 1)],
            method="Nelder-Mead",
        )
        self.best_weights_ = result.x / np.sum(result.x)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        features = self.feature_engineer_.transform(X)
        predictions = np.column_stack(
            [
                self.cb_model_.predict(features),
                self.lgb_model_.predict(features),
                self.xgb_model_.predict(features),
            ]
        )
        # The original validation predictions were not clipped. Clipping was
        # performed only for the discarded test/submission predictions.
        return predictions @ self.best_weights_


def restore_raw_target_domain(prediction, mode):
    """Identity inverse: the inner target split does not change cost's domain."""
    if mode == "fit":
        return prediction
    return prediction


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original chunk loop retained the
    #    first five million rows surviving these filters. Filtering globally and
    #    retaining the same prefix is row-for-row equivalent. Test loading,
    #    parquet output, directory creation, and submission code are omitted.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    required_columns = ["cost", "origin_x", "origin_y", "dest_x", "dest_y"]
    filtered = data.dropna(subset=required_columns)
    valid = (
        filtered["origin_x"].between(-180, 180)
        & filtered["origin_y"].between(-90, 90)
        & filtered["dest_x"].between(-180, 180)
        & filtered["dest_y"].between(-90, 90)
        & (filtered["cost"] >= 0)
    )
    filtered = filtered[valid].reset_index(drop=True).iloc[:5_000_000]

    # 2. Prepare Data — mark the RAW cost target and the raw design matrix. The
    #    custom outer splitter exactly reproduces np.random.seed(42), permutation,
    #    and the original first-80%/last-20% split.
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationSplit(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Preprocessing and model. The original used the rows it reported as
    #    validation for early stopping and blend-weight optimization. That is
    #    leaky under honest outer validation, so GetXY carves the same seeded
    #    80/20 split from each outer fold's training rows. The wrapper remains
    #    necessary for the supervised feature engineer, three-model ensemble,
    #    early stopping, and scipy blend-weight optimization.
    split_data = X.skb.apply(
        GetXY(train_fraction=0.8, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = split_data["X"]
    y_fit = split_data.get("y", y)
    X_validation = split_data["X_val"]
    y_validation = split_data["y_val"]

    pred_inner = X_fit.skb.apply(
        EarlyStoppedWeightedEnsemble(random_state=42),
        y=y_fit,
        fit_kwargs={"eval_set": [(X_validation, y_validation)]},
    )

    # y_fit is only a row subset of the raw marked target, not a mathematical
    # target transform. This explicit eval-mode-gated identity inverse keeps the
    # final prediction node in the raw cost domain used by the scorer.
    pred = pred_inner.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

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
