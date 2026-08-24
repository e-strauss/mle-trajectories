import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def drop_rare_target_classes(df, target_column, min_count):
    """Remove classes that cannot participate in the requested stratified CV."""
    counts = df[target_column].value_counts()
    rare_classes = counts[counts < min_count].index
    return df.loc[~df[target_column].isin(rare_classes)].reset_index(drop=True)


def restore_original_cover_labels(predictions, mode):
    """Map model predictions from 0–6 back to the raw 1–7 target domain."""
    if mode == "fit":
        # In fit mode, the upstream prediction node contains the fitted estimator.
        return None
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module does not access
    #    the file. The original FileNotFoundError handling is omitted because the
    #    deferred read will naturally report a missing file when scoring is run.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — the original excluded classes with fewer than three samples
    #    before cross-validation. Row filtering must occur before the marks because
    #    it changes the number of rows. Filtering before the row-wise feature
    #    engineering is equivalent to filtering the engineered table afterward.
    data_cv = data.skb.apply_func(
        drop_rare_target_classes,
        target_column="Cover_Type",
        min_count=3,
    )

    # Mark the RAW target as required. The original trained on labels shifted from
    # 1–7 to 0–6, so that transformation remains downstream of mark_as_y.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual three-fold loop becomes the same StratifiedKFold splitter here.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature values,
    #    appended-column order, and dropped columns.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    horizontal_hydrology_sq = X["Horizontal_Distance_To_Hydrology"] ** 2
    vertical_hydrology_sq = X["Vertical_Distance_To_Hydrology"] ** 2

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Euclidean_Distance_To_Hydrology=(
            horizontal_hydrology_sq + vertical_hydrology_sq
        ).skb.apply_func(np.sqrt),
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
        Proximity_To_Human_Features=(
            X["Horizontal_Distance_To_Roadways"]
            + X["Horizontal_Distance_To_Fire_Points"]
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

    # Scoring is against the marked raw target, so invert the original label shift
    # on predictions. Accuracy is unchanged from comparing 0–6 labels directly.
    pred = pred_transformed.skb.apply_func(
        restore_original_cover_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives validation.
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
