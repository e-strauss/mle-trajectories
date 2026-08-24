import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def remove_rare_target_classes(df, target, min_count):
    """Reproduce the original exclusion of classes too rare for stratified CV."""
    class_counts = df[target].value_counts()
    problematic_classes = class_counts[class_counts < min_count].index

    if len(problematic_classes) == 0:
        return df

    return (
        df.loc[~df[target].isin(problematic_classes)]
        .reset_index(drop=True)
    )


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 target domain."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original FileNotFoundError message
    #    and progress printing are omitted because they do not produce the CV score.
    train_df = (
        skrub.as_data_op("./input/train.csv")
        .skb.apply_func(pd.read_csv)
    )

    # 2. Prepare Data — preserve the original removal of target classes having
    #    fewer than three samples. Row filtering must occur before the marks because
    #    it changes the number of rows available to cross-validation.
    train_df = train_df.skb.apply_func(
        remove_rare_target_classes,
        target="Cover_Type",
        min_count=3,
    )

    # Mark the RAW 1-7 target. The model's 0-6 target transformation is recorded
    # only after mark_as_y, and predictions are mapped back to 1-7 below.
    y = train_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y - 1

    # Mark the raw design matrix before feature engineering. The manual
    # StratifiedKFold loop becomes the splitter attached to mark_as_X.
    X = train_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model — create the same composite
    #    hillshade feature, then remove the three source columns while retaining
    #    the original Aspect feature.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Accuracy is invariant to this one-to-one label shift, but predictions must
    # be returned in the raw target domain used by mark_as_y. The conversion is
    # gated because the prediction node contains an estimator during fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter on mark_as_X drives the three folds.
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
