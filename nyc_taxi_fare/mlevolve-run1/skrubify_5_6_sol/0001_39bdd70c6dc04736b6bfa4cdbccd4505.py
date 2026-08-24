import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit


INPUT_DIR = "./input"


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record one CSV read. The original's chunked reads,
    #    intermediate parquet files, test-set processing, and submission output
    #    are omitted because they are not part of the validation score.
    data = skrub.as_data_op(
        os.path.join(INPUT_DIR, "train.csv")
    ).skb.apply_func(pd.read_csv)

    # 2. Prepare data. The training-only row filters must happen before marking
    #    because they change which rows are scored. reset_index reproduces the
    #    original pd.concat(..., ignore_index=True).
    data = data.dropna(subset=["cost"])
    data = data[data["cost"] > 0].reset_index(drop=True)

    # Mark the raw target. The original single train_test_split becomes one
    # ShuffleSplit with the same test size and random state.
    y = data["cost"].skb.mark_as_y()
    X = data.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering. Coordinate medians are
    #    learned by SimpleImputer within each training fold, fixing the original
    #    full-table preprocessing leak while preserving median-imputation
    #    semantics.
    coordinates = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coordinates = coordinates.assign(
        origin_x=coordinates["origin_x"].clip(-75.0, -72.0),
        dest_x=coordinates["dest_x"].clip(-75.0, -72.0),
        origin_y=coordinates["origin_y"].clip(40.0, 42.0),
        dest_y=coordinates["dest_y"].clip(40.0, 42.0),
    ).skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(
        pd.to_datetime, errors="coerce"
    )
    dx = coordinates["dest_x"] - coordinates["origin_x"]
    dy = coordinates["dest_y"] - coordinates["origin_y"]

    features = X.assign(
        origin_x=coordinates["origin_x"],
        dest_x=coordinates["dest_x"],
        origin_y=coordinates["origin_y"],
        dest_y=coordinates["dest_y"],
        unit_count=X["unit_count"].fillna(1),
        hour=start_time.dt.hour.fillna(0).astype(int),
        dayofweek=start_time.dt.dayofweek.fillna(0).astype(int),
        month=start_time.dt.month.fillna(1).astype(int),
        year=start_time.dt.year.fillna(2012).astype(int),
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
        bearing=dy.skb.apply_func(np.arctan2, dx),
    ).drop(columns=["start_time"])

    # Same XGBoost model and hyperparameters as the original, except
    # early_stopping_rounds. It is removed because a directly applied estimator
    # has no separate eval_set inside the CV plan; the outer ShuffleSplit owns
    # validation.
    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    raw_pred = features.skb.apply(model, y=y)

    # Preserve the original post-prediction clipping. Prediction arithmetic is
    # gated because the prediction node contains the fitted estimator in fit
    # mode rather than an array of predictions.
    def clip_predictions(values, mode):
        if mode == "fit":
            return values
        return np.clip(values, 0, None)

    pred = raw_pred.skb.apply_func(
        clip_predictions, skrub.eval_mode()
    )

    # 4. Score. No cv= is passed here; mark_as_X's ShuffleSplit drives.
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
