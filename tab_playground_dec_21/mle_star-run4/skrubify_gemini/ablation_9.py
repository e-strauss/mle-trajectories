import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan.
    path = "./input/train.csv"
    train_df = skrub.as_data_op(path).skb.apply_func(pd.read_csv)

    # 2. Row filtering: classes with fewer than n_splits (3) samples are filtered out
    #    before cross-validation, exactly as in the original evaluate_model_performance.
    #    Row filtering must happen before marking X and y.
    raw_target = train_df["Cover_Type"] - 1
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    clean_df = train_df[~raw_target.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data — mark target (y) and features (X).
    #    StratifiedKFold(n_splits=3, shuffle=True, random_state=42) captures the
    #    3-fold CV scheme from the original script.
    y = (clean_df["Cover_Type"] - 1).skb.mark_as_y()
    X = clean_df.skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering — recorded operations corresponding to the feature
    #    computations in the original script.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    elevation_at_hydrology = X["Elevation"] - X["Vertical_Distance_To_Hydrology"]

    # Candidate ablation features
    hillshade_diff = X["Hillshade_Noon"] - X["Hillshade_3pm"]

    h_hydro = (
        X["Horizontal_Distance_To_Hydrology"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )
    h_road = (
        X["Horizontal_Distance_To_Roadways"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )
    h_fire = (
        X["Horizontal_Distance_To_Fire_Points"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )
    total_h_dist = h_hydro + h_road + h_fire

    slope_x_vert_hydro = X["Slope"] * X["Vertical_Distance_To_Hydrology"]

    # Base drop columns defined in the original:
    base_drop_cols = [
        "Id",
        "Cover_Type",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Aspect",
    ]
    base_features = X.drop(columns=base_drop_cols, errors="ignore").assign(
        Hillshade_composite=hillshade_composite,
        Elevation_at_Hydrology=elevation_at_hydrology,
    )

    # 5. Model & Ablation Variants
    #    The original evaluates 4 feature configurations: Baseline and 3 ablations.
    #    We fuse them into one search using skrub.choose_from so all variants are
    #    scored in a single cross-validation run.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    feat_baseline = base_features.assign(
        Hillshade_Noon_to_3pm_Diff=hillshade_diff,
        Total_Horizontal_Distance=total_h_dist,
        Slope_x_Vertical_Distance_To_Hydrology=slope_x_vert_hydro,
    )

    feat_no_diff = base_features.assign(
        Total_Horizontal_Distance=total_h_dist,
        Slope_x_Vertical_Distance_To_Hydrology=slope_x_vert_hydro,
    )

    feat_no_total_dist = base_features.assign(
        Hillshade_Noon_to_3pm_Diff=hillshade_diff,
        Slope_x_Vertical_Distance_To_Hydrology=slope_x_vert_hydro,
    )

    feat_no_slope_hydro = base_features.assign(
        Hillshade_Noon_to_3pm_Diff=hillshade_diff,
        Total_Horizontal_Distance=total_h_dist,
    )

    variants = {
        "Baseline": feat_baseline.skb.apply(model, y=y),
        "Hillshade_Noon_to_3pm_Diff": feat_no_diff.skb.apply(model, y=y),
        "Total_Horizontal_Distance": feat_no_total_dist.skb.apply(model, y=y),
        "Slope_x_Vertical_Distance_To_Hydrology": feat_no_slope_hydro.skb.apply(
            model, y=y
        ),
    }

    pred = skrub.choose_from(variants, name="ablation").as_data_op()

    # 6. Score the plan by 3-fold Stratified CV.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}"
        )
