import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def remove_rare_target_classes(df, target_column, min_count):
    """Reproduce the original pre-CV removal of classes too small to stratify."""
    counts = df[target_column].value_counts()
    valid_classes = counts[counts >= min_count].index
    return df.loc[df[target_column].isin(valid_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain of 1-7."""
    if mode == "fit":
        # In fit mode the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message
    #    wrapper is omitted because it does not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excludes every class represented fewer than
    #    three times before cross-validation. Row filtering must happen before
    #    mark_as_X/mark_as_y because it changes the number of samples.
    data = data.skb.apply_func(
        remove_rare_target_classes,
        target_column="Cover_Type",
        min_count=3,
    )

    # Mark the RAW 1-7 target, as required for scoring, and perform the original
    # 0-6 label shift only after the mark. Predictions are shifted back to 1-7
    # below so accuracy is evaluated in the raw target domain.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the initial design matrix before feature engineering so every
    # recorded operation below is re-run independently within each CV fold.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original formulas and
    # final feature set. Plain NumPy functions use apply_func.
    total_horizontal_distance = (
        X["Horizontal_Distance_To_Hydrology"]
        + X["Horizontal_Distance_To_Roadways"]
        + X["Horizontal_Distance_To_Fire_Points"]
    )

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Hillshade_Daily_Difference=(
            X["Hillshade_9am"] - X["Hillshade_3pm"]
        ).skb.apply_func(np.abs),
        log1p_Total_Horizontal_Distance=(
            total_horizontal_distance.skb.apply_func(np.log1p)
        ),
        Elevation_x_Vertical_Hydrology=(
            X["Elevation"] * X["Vertical_Distance_To_Hydrology"]
        ),
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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Convert 0-6 model predictions back to the marked raw target's 1-7 domain.
    # This must be gated because in fit mode pred_transformed evaluates to the
    # fitted RandomForestClassifier rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the StratifiedKFold on mark_as_X drives the
    # cross-validation, and mean_test_score matches the original mean accuracy.
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
