import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit


class OriginalPermutationSplit(BaseCrossValidator):
    """Reproduce np.random.permutation followed by an 80/20 positional split."""

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
    """Fit the original KMeans and frequency encoders, then add its features."""

    def __init__(self, n_clusters=20, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None):
        self.origin_kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.origin_kmeans_.fit(X[["origin_x", "origin_y"]])

        self.dest_kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.dest_kmeans_.fit(X[["dest_x", "dest_y"]])

        self.o_x_freq_ = (
            np.round(X["origin_x"], 2).value_counts(normalize=True).to_dict()
        )
        self.o_y_freq_ = (
            np.round(X["origin_y"], 2).value_counts(normalize=True).to_dict()
        )
        self.d_x_freq_ = (
            np.round(X["dest_x"], 2).value_counts(normalize=True).to_dict()
        )
        self.d_y_freq_ = (
            np.round(X["dest_y"], 2).value_counts(normalize=True).to_dict()
        )
        return self

    def transform(self, X):
        df = X.copy()

        # Preserve the original schema-dependent start_time branch.
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

        return df.fillna(0)


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping eval set out of this fold's training rows."""

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


class CatBoostRegressorCloneable(CatBoostRegressor):
    """Work around CatBoost's non-standard sklearn cloning behavior."""

    def __sklearn_clone__(self):
        return CatBoostRegressorCloneable(**self.get_params(deep=False))


def average_predictions(cb_predictions, lgb_predictions, xgb_predictions, mode):
    # Predictor nodes evaluate to fitted estimators in fit mode.
    if mode == "fit":
        return None
    return (cb_predictions + lgb_predictions + xgb_predictions) / 3.0


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read. The original chunk loop retained the first
    #    10,000,000 valid rows in file order; filtering followed by iloc expresses
    #    the same scored table without retaining the chunked-I/O machinery.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    required = ["cost", "origin_x", "origin_y", "dest_x", "dest_y"]
    filtered = data.dropna(subset=required)
    valid_mask = (
        (filtered["origin_x"] >= -180)
        & (filtered["origin_x"] <= 180)
        & (filtered["origin_y"] >= -90)
        & (filtered["origin_y"] <= 90)
        & (filtered["dest_x"] >= -180)
        & (filtered["dest_x"] <= 180)
        & (filtered["dest_y"] >= -90)
        & (filtered["dest_y"] <= 90)
        & (filtered["cost"] >= 0)
    )
    filtered = filtered[valid_mask].iloc[:10_000_000].reset_index(drop=True)

    # Test loading, parquet dumps, directory creation, submission generation,
    # and clipped test predictions are omitted because they do not produce the
    # reported validation score.

    # 2. Prepare data: mark the RAW target and design matrix. This custom
    #    one-split CV reproduces the original seeded np.random.permutation,
    #    whose first 80% were training rows and final 20% were validation rows.
    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationSplit(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. KMeans and normalized
    #    frequency mappings are fitted independently on each outer training fold,
    #    matching the original train-split-only fitting semantics.
    features = X.skb.apply(
        EngineerFeatures(n_clusters=20, random_state=42, n_init=10)
    )
    features = features.drop(columns=["record_id", "start_time"])

    # The original early-stopped on the same validation rows it reported, which
    # leaks validation outcomes into training and cannot be reproduced honestly.
    # Keep its 20% fraction, 2,000 boosting rounds, and patience of 50, but carve
    # the eval set from each outer fold's training rows.
    X_y = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
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

    # XGBoost's patience remains attached to the estimator, while its eval_set
    # is supplied from the fold-local GetXY node above. Dict expansion avoids
    # treating the setting as an unsupported plan-level fit argument.
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.04,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        **{"early_stopping_rounds": 50},
    )
    xgb_pred = X_fit.skb.apply(
        xgb_model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        },
    )

    # Preserve the original equal-weight arithmetic mean of the three raw
    # validation predictions. The original did not clip validation predictions.
    pred = cb_pred.skb.apply_func(
        average_predictions,
        lgb_pred,
        xgb_pred,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here -- the splitter on mark_as_X drives.
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
