import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def exclude_rare_classes(df, n_splits=3):
    """Remove classes that cannot participate in the original stratified CV."""
    transformed_target = df["Cover_Type"] - 1
    class_counts = transformed_target.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index

    if len(problematic_classes) == 0:
        return df

    keep_rows = ~transformed_target.isin(problematic_classes)
    return df.loc[keep_rows].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        # In fit mode the prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's FileNotFoundError
    #    printing is omitted because it does not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — preserve the original data-dependent removal of classes
    #    having fewer than three samples. Row filtering must happen before the
    #    marks because it changes the number of rows.
    cv_data = data.skb.apply_func(exclude_rare_classes, n_splits=3)

    # Mark the RAW target first, then perform the original 1-7 to 0-6 transform
    # downstream of the mark for model fitting.
    y_raw = cv_data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Mark the unengineered design matrix. The original manual three-fold loop
    # is represented by the same StratifiedKFold splitter here.
    X = cv_data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original feature values,
    #    names, and append order before dropping the source columns.
    aspect_radians = X["Aspect"].skb.apply_func(np.radians)
    aspect_sin = aspect_radians.skb.apply_func(np.sin)
    aspect_cos = aspect_radians.skb.apply_func(np.cos)
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    features = X.assign(
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Hillshade_composite=hillshade_composite,
    ).drop(
        columns=[
            "Aspect",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The model predicts transformed labels (0-6), while scoring is against the
    # raw mark_as_y target (1-7). Restore the original domain at prediction time.
    # The eval-mode guard leaves the fitted estimator untouched in fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter attached to mark_as_X drives CV.
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
