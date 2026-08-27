import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — recorded read from the input directory.
    raw_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Row filtering: remove classes with fewer than n_splits (3) samples so
    #    StratifiedKFold has sufficient members per class in every fold. Row removal
    #    must happen BEFORE mark_as_X / mark_as_y since it changes table size.
    raw_target = raw_df["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    clean_df = raw_df[~raw_target.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data — mark target (shifted from 1-7 to 0-6 for multiclass indexing)
    #    and design matrix. The 3-fold StratifiedKFold splitter lives on mark_as_X.
    y = (clean_df["Cover_Type"] - 1).skb.mark_as_y()
    X = clean_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops: composite hillshade,
    #    euclidean hydrology distance, elevation at hydrology, trigonometric
    #    aspect transformations, and proximity to human features. Dropping
    #    redundant and raw component columns.
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)
    dist_hyd_sq = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )

    X_feat = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3.0,
        Euclidean_Distance_To_Hydrology=dist_hyd_sq.skb.apply_func(np.sqrt),
        Elevation_at_Hydrology=X["Elevation"] - X["Vertical_Distance_To_Hydrology"],
        Aspect_sin=aspect_rad.skb.apply_func(np.sin),
        Aspect_cos=aspect_rad.skb.apply_func(np.cos),
        Proximity_To_Human_Features=(
            X["Horizontal_Distance_To_Roadways"]
            + X["Horizontal_Distance_To_Fire_Points"]
        ),
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Aspect"])

    # 5. Model — RandomForestClassifier with the exact hyperparameters from the original.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pred = X_feat.skb.apply(model, y=y)

    # 6. Score the plan by 3-fold CV. No cv= here — the splitter on mark_as_X drives.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        print(
            f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}"
        )
