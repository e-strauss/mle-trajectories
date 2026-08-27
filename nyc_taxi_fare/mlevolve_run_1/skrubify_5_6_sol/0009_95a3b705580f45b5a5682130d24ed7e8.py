import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from each outer fold's training rows."""

    def __init__(self, test_size=0.1, random_state=42):
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
        # Prediction mode has no target and needs only the feature matrix.
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def restore_raw_target_domain(prediction, mode):
    """Identity inverse: GetXY only subsets raw y and does not transform values."""
    if mode == "fit":
        return prediction
    return prediction


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read with the original dtype and parsing
    #    options. The original chunked read produced the same concatenated table,
    #    so chunk management is omitted. Test loading, parquet dumps, directory
    #    creation, and submission generation do not contribute to validation.
    dtypes = {
        "unit_count": "int32",
        "origin_x": "float32",
        "origin_y": "float32",
        "dest_x": "float32",
        "dest_y": "float32",
        "cost": "float32",
    }
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(
        pd.read_csv,
        dtype=dtypes,
        low_memory=False,
    )

    # 2. Prepare Data — preserve the original target-anomaly filter before the
    #    marks because it changes which rows are scored. Mark the RAW target.
    #    The original single 90/10 train/validation split is represented by the
    #    same ShuffleSplit on mark_as_X.
    filtered = data[
        (data["cost"] >= 0) & (data["cost"] <= 10000)
    ].reset_index(drop=True)

    y = filtered["cost"].skb.mark_as_y()
    X = filtered.drop(columns=["record_id", "cost"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.1, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. The original calculated
    #    coordinate medians once on the complete table before splitting. Using an
    #    estimator here preserves median-imputation semantics while fitting those
    #    medians only on each outer fold's training rows, fixing that leakage.
    coordinates = X[
        ["dest_x", "dest_y", "origin_x", "origin_y"]
    ].skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime,
        errors="coerce",
    )

    dx = coordinates["dest_x"] - coordinates["origin_x"]
    dy = coordinates["dest_y"] - coordinates["origin_y"]

    features = X.assign(
        hour=start_time.dt.hour.astype("float32"),
        dayofweek=start_time.dt.dayofweek.astype("float32"),
        month=start_time.dt.month.astype("float32"),
        year=start_time.dt.year.astype("float32"),
        dest_x=coordinates["dest_x"],
        dest_y=coordinates["dest_y"],
        origin_x=coordinates["origin_x"],
        origin_y=coordinates["origin_y"],
        euclidean_dist=(
            (dx**2 + dy**2).skb.apply_func(np.sqrt).astype("float32")
        ),
        manhattan_dist=(
            dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs)
        ).astype("float32"),
    ).drop(columns=["start_time"])

    # The original used the reported validation rows both for early stopping and
    # scoring, which is leaky and cannot be reproduced honestly in outer CV.
    # Keep its 10% split fraction, seed, 50-round patience, and 1500-tree limit,
    # but carve the early-stopping set from each outer fold's training rows.
    X_y = features.skb.apply(
        GetXY(test_size=0.1, random_state=42),
        y=y,
        how="no_wrap",
    )
    X_fit = X_y["X"]
    y_fit = X_y.get("y", y)
    X_early_stop = X_y["X_val"]
    y_early_stop = X_y["y_val"]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=100,
        random_state=42,
        n_jobs=-1,
    )

    pred_fit = X_fit.skb.apply(
        model,
        y=y_fit,
        fit_kwargs={
            "eval_set": [(X_early_stop, y_early_stop)],
            "callbacks": [
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ],
        },
    )

    # GetXY only subsets the already-marked raw target; it does not numerically
    # transform it. This explicitly maps predictions back to the raw scoring
    # domain with the required eval-mode guard, so fit mode leaves the fitted
    # estimator untouched.
    pred = pred_fit.skb.apply_func(
        restore_raw_target_domain,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the ShuffleSplit declared on mark_as_X drives.
    #    Scikit-learn exposes RMSE as a negative scorer so higher remains better.
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
