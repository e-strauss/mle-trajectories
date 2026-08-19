import os

import numpy as np
import pandas as pd
import skrub
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit

INPUT_DIR = "./input"

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan. The original
    #    read train.csv in 2M-row chunks and concatenated them; one recorded read
    #    expresses the same table. subsample only speeds up previews, it never
    #    changes the CV score. Everything the original did purely to ship a
    #    submission (test.csv, parquet round-trip through ./working, submission.csv)
    #    is dropped: the deliverable is a cross-validated score.
    raw_df = (
        skrub.as_data_op(os.path.join(INPUT_DIR, "train.csv"))
        .skb.apply_func(pd.read_csv)
        .skb.subsample(n=10_000)
    )

    # 2. Row filtering from the original's process_chunk(is_train=True). Row removal
    #    must happen BEFORE the marks — it changes the number of rows, so it cannot
    #    live inside the per-fold part of the plan.
    clean_df = raw_df.dropna(subset=["cost"])
    clean_df = clean_df[clean_df["cost"] > 0]

    # 3. Prepare Data — mark the RAW target; drop the identifier from the features.
    #    The original's single 80/20 train_test_split becomes the CV splitter on
    #    mark_as_X: ShuffleSplit(n_splits=1, test_size=0.2) is exactly that split.
    y = clean_df["cost"].skb.mark_as_y()
    X = clean_df.drop(columns=["cost", "record_id"]).skb.mark_as_X(
        cv=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops (never one opaque
    #    @skrub.deferred block), replacing the original's in-place `df[col] = ...`
    #    loops over column lists. The per-column median fill becomes a
    #    SimpleImputer so the median is learned on each fold's training rows only —
    #    the original computed it on the full table, which leaks.
    coords = X[["origin_x", "dest_x", "origin_y", "dest_y"]]
    coords = coords.assign(
        origin_x=coords["origin_x"].clip(-75.0, -72.0),
        dest_x=coords["dest_x"].clip(-75.0, -72.0),
        origin_y=coords["origin_y"].clip(40.0, 42.0),
        dest_y=coords["dest_y"].clip(40.0, 42.0),
    ).skb.apply(SimpleImputer(strategy="median"))

    start_time = X["start_time"].skb.apply_func(pd.to_datetime, errors="coerce")
    dx = coords["dest_x"] - coords["origin_x"]
    dy = coords["dest_y"] - coords["origin_y"]

    X_feat = X.assign(
        origin_x=coords["origin_x"],
        dest_x=coords["dest_x"],
        origin_y=coords["origin_y"],
        dest_y=coords["dest_y"],
        unit_count=X["unit_count"].fillna(1),
        hour=start_time.dt.hour.fillna(0).astype(int),
        dayofweek=start_time.dt.dayofweek.fillna(0).astype(int),
        month=start_time.dt.month.fillna(1).astype(int),
        year=start_time.dt.year.fillna(2012).astype(int),
        euclidean_dist=(dx**2 + dy**2).skb.apply_func(np.sqrt),
        manhattan_dist=dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
        bearing=dy.skb.apply_func(np.arctan2, dx),
    ).drop(columns=["start_time"])

    # 5. Model — same params as the original, minus early_stopping_rounds/eval_set:
    #    a CV plan has no separate eval set to hand the booster, the splitter owns
    #    validation.
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
    pred_raw = X_feat.skb.apply(model, y=y)

    # 6. The original clipped predictions at 0. Any arithmetic on a prediction node
    #    must be gated on the eval mode: in "fit" mode the node evaluates to the
    #    fitted estimator, not to predictions, so an ungated np.clip raises
    #    TypeError inside the CV loop.
    def clip_predictions(p, mode):
        if mode == "fit":
            return p
        return np.clip(p, 0, None)

    pred = pred_raw.skb.apply_func(clip_predictions, skrub.eval_mode())

    # 7. Score the whole plan by CV. No cv= here — the ShuffleSplit set on
    #    mark_as_X drives. The original reported RMSE, so negate skrub's
    #    higher-is-better scorer to print the same number.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="neg_root_mean_squared_error"
        )
        print(search.results_)
        print(f"Final Validation Score: {-search.results_['mean_test_score'].iloc[0]}")
