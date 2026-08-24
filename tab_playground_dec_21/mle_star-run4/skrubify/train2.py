import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def exclude_rare_target_classes(df, target="Cover_Type", min_count=3):
    """Remove rows whose target class cannot support stratified 3-fold CV."""
    counts = df[target].value_counts()
    valid_classes = counts[counts >= min_count].index
    return df.loc[df[target].isin(valid_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        # In fit mode, a prediction node contains the fitted estimator.
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this module does not read
    #    train.csv. The original's eager FileNotFoundError handling is omitted;
    #    any missing-file error naturally occurs when the plan is scored.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data. The original excluded classes having fewer than three
    #    samples before cross-validation. Because this changes the number of rows,
    #    it must remain before mark_as_X/mark_as_y. The helper performs exactly
    #    that data-dependent row filter without printing progress messages.
    data_cv = data.skb.apply_func(
        exclude_rare_target_classes,
        target="Cover_Type",
        min_count=3,
    )

    # Mark the RAW target, as required for scoring in its original 1-7 domain.
    # The original model was fitted on labels shifted to 0-6, so that transform
    # occurs downstream of mark_as_y and is inverted on predictions below.
    y = data_cv["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Keep the source Hillshade columns until downstream feature engineering.
    # The original manual StratifiedKFold loop is represented by the splitter
    # attached here and nowhere else.
    X = data_cv.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. Assigning the composite before
    #    dropping its source columns preserves the original feature values,
    #    names, and column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Restore predictions to the raw 1-7 target domain used by mark_as_y.
    # The operation is gated because in fit mode pred_transformed evaluates to
    # the fitted RandomForestClassifier rather than an array of predictions.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score by the original metric and three-fold validation scheme. No cv=
    #    is passed here because mark_as_X owns the splitter.
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
