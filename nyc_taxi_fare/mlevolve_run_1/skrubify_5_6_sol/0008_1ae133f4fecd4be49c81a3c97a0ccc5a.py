import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator


class OriginalPermutationHoldout(BaseCrossValidator):
    """Reproduce np.random.seed(42), permutation, then the first 80% as train."""

    def __init__(self, train_fraction=0.8, random_state=42):
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_rows = len(X)
        shuffled_indices = np.random.RandomState(self.random_state).permutation(n_rows)
        split_idx = int(n_rows * self.train_fraction)
        yield shuffled_indices[:split_idx], shuffled_indices[split_idx:]


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Reproduce engineer_features, including its start_time conditional."""

    def fit(self, X, y=None):
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


class CloneableCatBoostRegressor(CatBoostRegressor):
    """Work around CatBoost's sklearn cloning issue with learning_rate."""

    def __sklearn_clone__(self):
        return CloneableCatBoostRegressor(**self.get_params(deep=False))


def average_predictions(catboost_predictions, lightgbm_predictions, mode):
    # In fit mode prediction nodes contain fitted estimators rather than arrays.
    if mode == "fit":
        return None
    return (catboost_predictions + lightgbm_predictions) / 2.0


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read. The original chunked read is replaced by one
    #    recorded read; filtering followed by iloc keeps the same first 10,000,000
    #    valid rows. Test loading, parquet dumps, directories, and submission
    #    generation are omitted because they do not produce the validation score.
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
    data = filtered[valid_mask].reset_index(drop=True)
    data = data.iloc[:10_000_000].reset_index(drop=True)

    # 2. Prepare data: mark the RAW target and design matrix. The custom splitter
    #    exactly preserves the original permutation order and its first-80% train,
    #    remaining-20% validation assignment.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost"]).skb.mark_as_X(
        cv=OriginalPermutationHoldout(train_fraction=0.8, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and the original equal-weight ensemble.
    #    FeatureEngineer preserves the data-dependent start_time branch and the
    #    exact named-column insertion order from engineer_features.
    features = X.skb.apply(FeatureEngineer())
    features = features.drop(
        columns=["record_id", "start_time"], errors="ignore"
    )

    # The original used the outer validation rows as each model's early-stopping
    # eval_set. A DataOps estimator receives only its fold's training rows, so
    # eval_set, early_stopping_rounds, and LightGBM callbacks are removed rather
    # than leaking the held-out scoring rows into fitting.
    catboost_model = CloneableCatBoostRegressor(
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
    lightgbm_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.04,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    catboost_pred = features.skb.apply(catboost_model, y=y)
    lightgbm_pred = features.skb.apply(lightgbm_model, y=y)
    pred = catboost_pred.skb.apply_func(
        average_predictions,
        lightgbm_pred,
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
