import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def drop_rare_target_classes(df, target, min_count):
    """Reproduce the original pre-CV removal of classes with too few rows."""
    counts = df[target].value_counts()
    valid_classes = counts[counts >= min_count].index
    return df[df[target].isin(valid_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        # In fit mode the upstream prediction node represents the fitted estimator.
        return None
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read. The original FileNotFoundError
    #    reporting is omitted because it is not part of producing the CV score;
    #    a missing file will still fail when the recorded plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data -- the original excludes classes represented by fewer than
    #    three rows before cross-validation. Row filtering must occur before the
    #    marks because it changes the number of rows. The class-count computation
    #    is intentionally performed on the complete input, matching the original.
    data_cv = data.skb.apply_func(
        drop_rare_target_classes,
        target="Cover_Type",
        min_count=3,
    )

    # Mark the RAW target first. Its 1-7 to 0-6 transform therefore occurs
    # downstream of mark_as_y, and predictions are mapped back to 1-7 below.
    y_raw = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The manual three-fold loop becomes the identical StratifiedKFold splitter
    # attached to mark_as_X. Id is excluded exactly as in the original.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering -- reproduce every derived feature and
    #    remove the same original columns. Fine-grained assign operations replace
    #    the original in-place dataframe mutations.
    aspect_radians = X["Aspect"].skb.apply_func(np.deg2rad)
    hydrology_distance_squared = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )

    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Euclidean_Distance_To_Hydrology=hydrology_distance_squared.skb.apply_func(
            np.sqrt
        ),
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
        Aspect_sin=aspect_radians.skb.apply_func(np.sin),
        Aspect_cos=aspect_radians.skb.apply_func(np.cos),
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

    # Accuracy is invariant to this one-to-one label shift. Restoring predictions
    # to 1-7 makes them comparable with the raw target marked above while exactly
    # reproducing accuracy computed on the original script's 0-6 labels.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score -- no cv= here; mark_as_X's StratifiedKFold drives validation.
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
