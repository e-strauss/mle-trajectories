import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def drop_classes_with_fewer_than_n_samples(df, target_column, n_samples):
    """Reproduce the original pre-CV removal of classes too small to stratify."""
    class_counts = df[target_column].value_counts()
    problematic_classes = class_counts[class_counts < n_samples].index
    if len(problematic_classes) == 0:
        return df
    return (
        df.loc[~df[target_column].isin(problematic_classes)]
        .reset_index(drop=True)
    )


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0–6 back to the raw target domain 1–7."""
    if mode == "fit":
        # In fit mode, a prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError handling
    #    is unnecessary: pd.read_csv will still raise if the file is absent.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original globally excluded classes with fewer than
    #    three samples before cross-validation. This row filtering must remain
    #    before the marks because it changes the number of rows.
    data_cv = data.skb.apply_func(
        drop_classes_with_fewer_than_n_samples,
        "Cover_Type",
        3,
    )

    # Mark the RAW target. The 1–7 to 0–6 transformation is recorded afterward,
    # and predictions are converted back to 1–7 before scoring.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The manual three-fold loop becomes the same StratifiedKFold splitter on X.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Assign appends the four engineered
    #    columns in the same order as the original script.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Slope_x_Aspect_sin=(
            X["Slope"] * aspect_radians.skb.apply_func(np.sin)
        ),
        Slope_x_Aspect_cos=(
            X["Slope"] * aspect_radians.skb.apply_func(np.cos)
        ),
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is unchanged by this one-to-one label shift, but predictions must
    # be restored to the raw marked target's domain for skrub scoring.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here—the splitter on mark_as_X drives the evaluation.
    #    The original has no test-set prediction or submission step.
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
