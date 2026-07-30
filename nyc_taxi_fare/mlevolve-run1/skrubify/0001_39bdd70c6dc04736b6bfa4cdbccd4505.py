import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.impute import SimpleImputer
import xgboost as xgb
import stratum as st

INPUT_DIR = "/home/elias/github/mle-claude/tasks/od_cost_regression/input"
train_path = os.path.join(INPUT_DIR, "train.csv")

with st.config_context(eager_data_ops=False):
    raw_train_df = st.as_data_op(train_path).skb.apply_func(pd.read_csv).skb.subsample(10_000)

    # cleaning
    cleaned_train_df = raw_train_df.dropna(subset=["cost"])
    cleaned_train_df2 = cleaned_train_df[cleaned_train_df["cost"] > 0]

    X = cleaned_train_df2.drop(columns=["cost"]).skb.mark_as_X()
    y = cleaned_train_df2["cost"].skb.mark_as_y()

    lon_min, lon_max = -75.0, -72.0
    lat_min, lat_max = 40.0, 42.0

    imputer = SimpleImputer(strategy='median')
    cleaned_destinations = X[["dest_x", "dest_y"]].skb.apply(imputer)

    df = X.assign(
        origin_x = X["origin_x"].clip(lon_min, lon_max),
        dest_x = cleaned_destinations["dest_x"],
        origin_y = X["origin_y"].clip(lat_min, lat_max),
        dest_y = cleaned_destinations["dest_y"]
    )

    df = df.assign(
        unit_count=df["unit_count"].fillna(1),
        start_time = df["start_time"].skb.apply_func(pd.to_datetime,format="YYYY-MM-DD HH:MM:SS", errors="coerce")
    )

    df = df.assign(
        hour = df["start_time"].dt.hour.fillna(0).astype(int),
        dayofweek = df["start_time"].dt.dayofweek.fillna(0).astype(int),
        month = df["start_time"].dt.month.fillna(1).astype(int),
        year = df["start_time"].dt.year.fillna(2012).astype(int),
    )
    dx = df["dest_x"] - df["origin_x"]
    dy = df["dest_y"] - df["origin_y"]

    X_preprossed = df.assign(
        euclidean_dist = (dx ** 2 + dy ** 2).skb.apply_func(np.sqrt),
        manhattan_dist = dx.skb.apply_func(np.abs) + dy.skb.apply_func(np.abs),
        bearing = dy.skb.apply_func(np.arctan2, dx)
    )

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
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    pred_raw = X_preprossed.skb.apply(model)

    pred = pred_raw.skb.apply_func(lambda p,m: np.clip(p, 0, None) if m != "fit" else p)

    cv = ShuffleSplit(test_size=0.2, random_state=42)
    with st.config(scheduler=True, force_polars=False):
        search = pred.skb.make_grid_search(cv=cv, scoring="root_mean_squared_error", n_jobs=1, fitted=True, refit=False, keep_subsampling=True)
        print(search.results_)