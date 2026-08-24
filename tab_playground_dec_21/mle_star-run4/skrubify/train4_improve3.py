import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def exclude_underrepresented_classes(df, target_column, min_count):
    """Remove rows whose target class cannot support stratified CV."""
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= min_count].index
    return df[df[target_column].isin(valid_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predictions from 0-6 back to the raw target domain of 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except and progress
    #    messages are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded classes having fewer than three
    #    samples before cross-validation. Row filtering must happen before the
    #    marks because it changes the number of rows.
    data = data.skb.apply_func(
        exclude_underrepresented_classes,
        target_column="Cover_Type",
        min_count=3,
    )

    # Mark the RAW 1-7 target. The model's 0-6 target transformation happens
    # after mark_as_y, and predictions are mapped back to 1-7 before scoring.
    y = data["Cover_Type"].skb.mark_as_y()
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )
    y_transformed = y - 1

    # 3. Recorded feature engineering, preserving the original feature values,
    #    names, and appended-column order.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3
    features = X.assign(Hillshade_composite=hillshade_composite)

    aspect_rad = features["Aspect"].skb.apply_func(np.deg2rad)
    features = features.assign(Aspect_rad=aspect_rad)
    features = features.assign(
        Aspect_sin=features["Aspect_rad"].skb.apply_func(np.sin),
        Aspect_cos=features["Aspect_rad"].skb.apply_func(np.cos),
    )
    features = features.drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
            "Aspect_rad",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is unchanged by this one-to-one label shift, but predictions must
    # be restored to the raw marked target's 1-7 domain. In fit mode the input is
    # the fitted estimator, so it is returned unchanged.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. The splitter on mark_as_X reproduces the original manual
    #    three-fold StratifiedKFold loop; no cv= is passed here.
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
