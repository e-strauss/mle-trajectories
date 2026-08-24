import numpy as np
import pandas as pd
import skrub
from skrub import selectors as s
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def drop_underrepresented_classes(df, target_column, n_splits):
    """Remove classes that cannot participate in StratifiedKFold."""
    transformed_target = df[target_column] - 1
    class_counts = transformed_target.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index

    if len(problematic_classes):
        keep = ~transformed_target.isin(problematic_classes)
        return df.loc[keep].reset_index(drop=True)
    return df


def restore_original_labels(predictions, mode):
    """Convert model labels 0-6 back to the raw target domain 1-7."""
    if mode == "fit":
        # In fit mode this node contains the fitted estimator, not predictions.
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError handling
    #    is omitted because importing this module must not read the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original removed target classes having fewer than
    #    three rows before each CV run. Row filtering changes the sample count,
    #    so it must happen before marking X and y. It is shared across variants
    #    because all four original runs perform the identical filtering.
    data = data.skb.apply_func(
        drop_underrepresented_classes,
        target_column="Cover_Type",
        n_splits=3,
    )

    # Mark the RAW target as required. The model's 0-based target transform is
    # recorded only after mark_as_y; predictions are shifted back below.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y_model = y_raw - 1

    # Keep source columns needed by feature engineering until after the mark.
    # The original manual fold loop becomes this same StratifiedKFold splitter.
    X = data.drop(columns=["Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. These are the five features from the
    #    original baseline, in the same creation order.
    horizontal_hydrology = (
        X["Horizontal_Distance_To_Hydrology"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )
    horizontal_roadways = (
        X["Horizontal_Distance_To_Roadways"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )
    horizontal_fire_points = (
        X["Horizontal_Distance_To_Fire_Points"]
        .clip(lower=0)
        .skb.apply_func(np.log1p)
    )

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Hillshade_Noon_to_3pm_Diff=(
            X["Hillshade_Noon"] - X["Hillshade_3pm"]
        ),
        Total_Horizontal_Distance=(
            horizontal_hydrology
            + horizontal_roadways
            + horizontal_fire_points
        ),
        Slope_x_Vertical_Distance_To_Hydrology=(
            X["Slope"] * X["Vertical_Distance_To_Hydrology"]
        ),
    )

    # Match base_drop_cols. Cover_Type was already separated before marking.
    features = features.skb.drop(
        s.cols(
            "Id",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        )
    )

    # The original scores exactly four ablation variants. Fuse those exact
    # configurations into one discrete choice without inventing combinations.
    variants = {
        "Baseline": features.skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_model,
        ),
        "Hillshade_Noon_to_3pm_Diff": features.skb.drop(
            s.cols("Hillshade_Noon_to_3pm_Diff")
        ).skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_model,
        ),
        "Total_Horizontal_Distance": features.skb.drop(
            s.cols("Total_Horizontal_Distance")
        ).skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_model,
        ),
        "Slope_x_Vertical_Distance_To_Hydrology": features.skb.drop(
            s.cols("Slope_x_Vertical_Distance_To_Hydrology")
        ).skb.apply(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            y=y_model,
        ),
    }

    pred_zero_based = skrub.choose_from(
        variants,
        name="variant",
    ).as_data_op()

    # Return predictions to the raw 1-7 target domain. Accuracy is unchanged by
    # this one-to-one shift, so it matches the original 0-6 accuracy calculation.
    pred = pred_zero_based.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: mark_as_X's StratifiedKFold drives validation.
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
