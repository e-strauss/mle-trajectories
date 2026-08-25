import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import BaseCrossValidator


class PermutationHoldout(BaseCrossValidator):
    """Reproduce the original seeded permutation followed by an 80/20 cut."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.random.RandomState(self.random_state).permutation(len(X))
        split_idx = int(len(X) * self.train_fraction)
        yield indices[:split_idx], indices[split_idx:]


class EngineerFeatures(TransformerMixin, BaseEstimator):
    """Fit KMeans and frequency encodings, then reproduce feature engineering."""

    def __init__(self, n_clusters=50, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
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

        self.origin_x_freq_ = origin_x_bin.value_counts(normalize=True).to_dict()
        self.origin_y_freq_ = origin_y_bin.value_counts(normalize=True).to_dict()
        self.dest_x_freq_ = dest_x_bin.value_counts(normalize=True).to_dict()
        self.dest_y_freq_ = dest_y_bin.value_counts(normalize=True).to_dict()
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

        df["origin_x_bin_freq"] = (
            df["origin_x_bin"].map(self.origin_x_freq_).fillna(0)
        )
        df["origin_y_bin_freq"] = (
            df["origin_y_bin"].map(self.origin_y_freq_).fillna(0)
        )
        df["dest_x_bin_freq"] = (
            df["dest_x_bin"].map(self.dest_x_freq_).fillna(0)
        )
        df["dest_y_bin_freq"] = (
            df["dest_y_bin"].map(self.dest_y_freq_).fillna(0)
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

        # The original excluded start_time after engineering the date features.
        if "start_time" in df.columns:
            df = df.drop(columns=["start_time"])

        return df


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from this outer fold's training rows."""

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

        X_train = X.take(train_indices, axis=0)
        X_validation = X.take(validation_indices, axis=0)

        if hasattr(y, "take"):
            y_train = y.take(train_indices, axis=0)
            y_validation = y.take(validation_indices, axis=0)
        else:
            y_array = np.asarray(y)
            y_train = np.take(y_array, train_indices, axis=0)
            y_validation = np.take(y_array, validation_indices, axis=0)

        return {
            "X": X_train,
            "X_val": X_validation,
            "y": y_train,
            "y_val": y_validation,
        }

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class CatBoostRegressorCloneable(CatBoostRegressor):
    """Work around CatBoost's non-standard sklearn cloning behavior."""

    def __sklearn_clone__(self):
        return CatBoostRegressorCloneable(**self.get_params(deep=False))


def average_predictions(cb_prediction, lgb_prediction, xgb_prediction, mode):
    # Predictor nodes hold fitted estimators in fit mode, so arithmetic is only
    # valid when skrub is evaluating actual predictions.
    if mode == "fit":
        return cb_prediction
    return (cb_prediction + lgb_prediction + xgb_prediction) / 3.0


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one ordered CSV read. Filtering the complete ordered
    #    table and retaining its first 20,000,000 valid rows reproduces the
    #    original chunk loop's retained training sample. Test loading, parquet
    #    files, directory creation, and submission generation are dropped because
    #    they do not contribute to the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Row cleaning changes which observations are scored and therefore remains
    # before the X/y marks.
    data = data.dropna(
        subset=["cost", "origin_x", "origin_y", "dest_x", "dest_y"]
    )
    valid_mask = (
        (data["origin_x"] >= -180)
        & (data["origin_x"] <= 180)
        & (data["origin_y"] >= -90)
        & (data["origin_y"] <= 90)
        & (data["dest_x"] >= -180)
        & (data["dest_x"] <= 180)
        & (data["dest_y"] >= -90)
        & (data["dest_y"] <= 90)
        & (data["cost"] >= 0)
    )
    data = data[valid_mask].reset_index(drop=True).head(20_000_000)

    # 2. Prepare Data — mark the raw target and design matrix. The custom
    #    splitter preserves the original np.random.seed(42), random permutation,
    #    and floor(0.8 * n_rows) split exactly.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=PermutationHoldout(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. KMeans and frequency
    #    mappings are fitted on each outer fold's training rows, preserving the
    #    original train-only fitting semantics without validation leakage.
    features = X.skb.apply(
        EngineerFeatures(n_clusters=50, random_state=42, n_init=10)
    )

    # The original used its reported validation set for early stopping, making
    # that score optimistic. Keep the original 80/20 split style, all three
    # models' 50-round patience, and their estimator counts, but carve the eval
    # set only from each outer fold's training rows.
    X_y = features.skb.apply(
        GetXY(train_fraction=0.8, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_val = X_y["X_val"]
    y_val = X_y["y_val"]

    cb_model = CatBoostRegressorCloneable(
        iterations=2000,
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        verbose=False,
    )
    cb_pred = X_fit.skb.apply(
        cb_model,
        y=y_fit,
        fit_kwargs={
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
            "verbose": False,
        },
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.04,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_pred = X_fit.skb.apply(
        lgb_model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "eval_metric": "rmse",
            "callbacks": [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        },
    )

    # XGBoost's original early-stopping constructor option is supplied through
    # an explicit parameter dictionary. The corresponding eval_set is still
    # provided per fit from the fold-local GetXY split above.
    xgb_parameters = {
        "n_estimators": 2000,
        "learning_rate": 0.04,
        "max_depth": 8,
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
    }
    xgb_model = xgb.XGBRegressor(**xgb_parameters)
    xgb_pred = X_fit.skb.apply(
        xgb_model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        },
    )

    # Preserve the original unweighted arithmetic mean. The original did not
    # clip validation predictions; clipping was only used for its submission.
    pred = cb_pred.skb.apply_func(
        average_predictions,
        lgb_pred,
        xgb_pred,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is passed here; the splitter on mark_as_X drives.
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
