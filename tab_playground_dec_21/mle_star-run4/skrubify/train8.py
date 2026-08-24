import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def drop_rare_target_classes(df, target, n_splits):
    """Apply the original data-dependent row filtering before CV marks."""
    class_counts = df[target].value_counts()
    keep_rows = df[target].map(class_counts).ge(n_splits)
    return df.loc[keep_rows].reset_index(drop=True)


def restore_original_labels(predictions, mode):
    """Map model predictions from 0–6 back to the raw 1–7 target domain."""
    if mode == "fit":
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message
    #    and progress printing are omitted because they do not produce the CV score;
    #    a missing file still raises when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded target classes having fewer than
    #    three samples before running StratifiedKFold. Because this data-dependent
    #    operation removes rows, it must happen before mark_as_X/mark_as_y.
    filtered_data = data.skb.apply_func(
        drop_rare_target_classes,
        target="Cover_Type",
        n_splits=3,
    )

    # Mark the RAW target. The subtraction used for model fitting is recorded
    # afterwards, and predictions are restored to the raw 1–7 domain below.
    y_raw = filtered_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    X = filtered_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original names and column
    #    order: both new columns are appended before the source columns are dropped.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is evaluated against the raw marked target, so invert the original
    # 1-to-0 label shift. Post-prediction arithmetic is gated because in fit mode
    # the prediction node contains the fitted estimator rather than predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here — the splitter on mark_as_X drives the three folds.
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
