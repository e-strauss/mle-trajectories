import numpy as np
import pandas as pd
import skrub
from skrub import selectors as s
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def exclude_rare_target_classes(df, target, min_count):
    """Reproduce the original pre-CV removal of classes with too few rows."""
    class_counts = df[target].value_counts()
    problematic_classes = class_counts[class_counts < min_count].index

    if len(problematic_classes) == 0:
        return df

    return df.loc[~df[target].isin(problematic_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode, a prediction node contains the fitted estimator.
        return None
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded and therefore does not occur on import.
    #    The original try/except and progress messages are omitted because they do
    #    not contribute to the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare rows before marking X and y. The original excluded every sample
    #    whose class had fewer than three observations before cross-validation.
    #    Row filtering must occur before the marks because it changes row count.
    data_cv = data.skb.apply_func(
        exclude_rare_target_classes,
        target="Cover_Type",
        min_count=3,
    )

    # 3. Mark the RAW target and the initial design matrix. The original trained
    #    on labels shifted from 1-7 to 0-6; that transform is applied only after
    #    mark_as_y, and predictions are restored to 1-7 before scoring.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    X = data_cv.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 4. Recorded feature engineering, preserving the original operation and
    #    column-append order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    )
    features = features.assign(
        Elevation_at_Hydrology=(
            features["Elevation"]
            - features["Vertical_Distance_To_Hydrology"]
        )
    )

    clipped_road_distance = features[
        "Horizontal_Distance_To_Roadways"
    ].clip(lower=0)
    logged_road_distance = clipped_road_distance.skb.apply_func(np.log1p)
    features = features.assign(
        Horizontal_Distance_To_Roadways=logged_road_distance
    )

    features = features.assign(
        Hydro_Road_Interaction=(
            features["Horizontal_Distance_To_Hydrology"]
            * features["Horizontal_Distance_To_Roadways"]
        )
    )
    features = features.assign(
        Elevation_x_Fire_Points=(
            features["Elevation"]
            * features["Horizontal_Distance_To_Fire_Points"]
        )
    )

    features = features.skb.drop(
        s.cols(
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        )
    )

    # 5. Model — same family and hyperparameters as the original.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Restore predictions to the raw target domain required by mark_as_y.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 6. Score. No cv= is passed here; mark_as_X's StratifiedKFold drives.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
