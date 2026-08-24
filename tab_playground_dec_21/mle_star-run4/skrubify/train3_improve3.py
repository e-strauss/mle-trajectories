import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold


def exclude_rare_target_classes(df, target, n_splits):
    """Remove classes that cannot be used by the original StratifiedKFold."""
    class_counts = df[target].value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    return df.loc[~df[target].isin(problematic_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map predicted labels from 0-6 back to the raw target domain, 1-7."""
    if mode == "fit":
        return predictions
    return predictions + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original try/except and progress
    #    messages are omitted so importing this module neither reads data nor
    #    performs runtime reporting.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # The original excluded target classes having fewer than three samples before
    # running CV. Because this changes the number of rows, it must happen before
    # marking X and y. Counting raw labels is equivalent to counting labels shifted
    # from 1-7 to 0-6.
    data = data.skb.apply_func(
        exclude_rare_target_classes,
        target="Cover_Type",
        n_splits=3,
    )

    # 2. Prepare Data — mark the RAW target, then perform the original 1-to-0
    #    label shift downstream of mark_as_y. The manual StratifiedKFold loop is
    #    represented by the identical splitter on mark_as_X.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering and model. The composite is appended before
    #    the three source columns are dropped, preserving the original feature
    #    values, names, and column order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    model = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        num_leaves=25,
        min_child_samples=30,
        colsample_bytree=0.8,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # The model predicts labels in the transformed 0-6 domain, while scoring uses
    # the raw target marked above. Restore predictions to 1-7 outside fit mode;
    # during fit, a prediction node contains the fitted estimator and must pass
    # through unchanged.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; the splitter declared on mark_as_X drives the
    #    three-fold validation. Accuracy is unchanged by shifting both label
    #    domains consistently.
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
