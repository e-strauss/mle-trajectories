import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


N_SPLITS = 3


def drop_rare_classes(df, target, min_count):
    """Apply the original data-dependent row filtering before CV marks."""
    class_counts = df[target].value_counts()
    problematic_classes = class_counts[class_counts < min_count].index
    return df.loc[~df[target].isin(problematic_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw target domain 1-7."""
    if mode == "fit":
        # In fit mode, a prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except and progress
    #    messages are omitted because they do not contribute to the CV score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded every class having fewer than three
    #    samples before cross-validation. Because this changes the number of rows,
    #    it must remain before mark_as_X/mark_as_y. This data-dependent filtering
    #    operation is recorded, so importing this module still does not read data.
    data_cv = data.skb.apply_func(
        drop_rare_classes,
        target="Cover_Type",
        min_count=N_SPLITS,
    )

    # Mark the RAW 1-7 target. Its 0-6 transformation is performed only after the
    # mark, and predictions are converted back below before accuracy is scored.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # The original manual three-fold loop becomes the identical splitter on X.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving the original feature names,
    #    formulas, and column order.
    hillshade_std = X[
        ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    ].std(axis=1)

    features = X.assign(
        Hillshade_Std=hillshade_std,
        Elevation_Slope_Interaction=X["Elevation"] * X["Slope"],
        Elevation_sq=X["Elevation"] ** 2,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Restore predictions to the marked raw target domain. This is gated on
    # eval_mode because the prediction node contains an estimator during fitting.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= is supplied here; mark_as_X owns the validation scheme.
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
