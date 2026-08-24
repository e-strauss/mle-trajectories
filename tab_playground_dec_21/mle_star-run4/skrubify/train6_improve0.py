import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def exclude_underrepresented_classes(df, target="Cover_Type", n_splits=3):
    """Apply the original data-dependent row filtering before CV marks."""
    class_counts = df[target].value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index

    if len(problematic_classes) == 0:
        return df

    return df.loc[~df[target].isin(problematic_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predictions from 0-6 back to the raw target domain of 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read. The original error/progress printing is
    #    omitted because it does not contribute to the cross-validated score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded classes represented fewer than three
    #    times before running StratifiedKFold. Because this changes the number of
    #    rows, the same data-dependent filtering must happen before both marks.
    data_cv = data.skb.apply_func(
        exclude_underrepresented_classes,
        target="Cover_Type",
        n_splits=3,
    )

    # Mark the RAW 1-7 target first. The original trained on labels shifted to
    # 0-6, so that transformation is recorded after mark_as_y and predictions
    # are mapped back to 1-7 below before accuracy is scored.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Hillshade columns remain temporarily available for feature engineering but
    # are removed before fitting, exactly as in the original design matrix.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. assign appends the three derived
    #    columns in the same order as the original script.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Elevation_times_Slope=X["Elevation"] * X["Slope"],
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is evaluated against the raw mark_as_y node. Restore predictions
    # from 0-6 to 1-7 only during prediction; in fit mode a prediction node holds
    # the fitted estimator and therefore must pass through unchanged.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here: the StratifiedKFold on mark_as_X drives validation.
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
