import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — recorded CSV read as the first step of the plan.
    raw_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Row filtering: classes with fewer than n_splits (3) samples cannot be
    #    stratified by StratifiedKFold and are excluded before CV splitting,
    #    faithfully translating the original's frequency-based exclusion logic.
    #    Row filtering happens BEFORE mark_as_X / mark_as_y since it changes row count.
    target_series = raw_df["Cover_Type"] - 1
    class_counts = target_series.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    clean_df = raw_df[~target_series.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data — mark the target (shifted 1-7 to 0-6 as in original) and design matrix.
    #    The CV splitter lives on mark_as_X with StratifiedKFold(n_splits=3, shuffle=True, random_state=42).
    y = (clean_df["Cover_Type"] - 1).skb.mark_as_y()
    X = clean_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops downstream of mark_as_X.
    total_h_dist = (
        X["Horizontal_Distance_To_Hydrology"]
        + X["Horizontal_Distance_To_Roadways"]
        + X["Horizontal_Distance_To_Fire_Points"]
    )

    X_feat = X.assign(
        Hillshade_composite=(X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]) / 3,
        Elevation_at_Hydrology=X["Elevation"] - X["Vertical_Distance_To_Hydrology"],
        Hillshade_Daily_Difference=(X["Hillshade_9am"] - X["Hillshade_3pm"]).skb.apply_func(np.abs),
        log1p_Total_Horizontal_Distance=total_h_dist.skb.apply_func(np.log1p),
        Elevation_x_Vertical_Hydrology=X["Elevation"] * X["Vertical_Distance_To_Hydrology"],
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    )

    # 5. Model — RandomForestClassifier matching the original hyperparameters.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pred = X_feat.skb.apply(model, y=y)

    # 6. Score the plan by 3-fold cross-validation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}")
