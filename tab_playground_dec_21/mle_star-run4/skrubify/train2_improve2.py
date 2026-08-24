import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def drop_rare_target_classes(df, target, min_count):
    """Reproduce the original pre-CV removal of classes too small to stratify."""
    class_counts = df[target].value_counts()
    rare_classes = class_counts[class_counts < min_count].index
    if len(rare_classes) == 0:
        return df
    return df.loc[~df[target].isin(rare_classes)].reset_index(drop=True)


def restore_original_target_labels(predictions, mode):
    """Map model predictions from 0-6 back to the raw 1-7 scoring domain."""
    if mode == "fit":
        return predictions
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read. The original's explicit FileNotFoundError
    #    message is omitted; pd.read_csv will naturally raise if the file is absent.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — reproduce the original removal of target classes having
    #    fewer than three rows. Row filtering must occur before the marks because
    #    it changes the number of samples.
    train_df = train_df.skb.apply_func(
        drop_rare_target_classes,
        target="Cover_Type",
        min_count=3,
    )

    # Mark the RAW 1-7 target. The 0-6 transformation used for fitting is recorded
    # only after mark_as_y; predictions are shifted back to 1-7 before scoring.
    y_raw = train_df["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # Hillshade columns remain in the raw marked matrix because they are needed to
    # construct Hillshade_composite. They are dropped before the model, exactly as
    # in the original feature matrix. The manual three-fold loop becomes this
    # splitter on mark_as_X.
    X = train_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering — preserve the original feature values,
    #    names, creation order, and removal of the three source Hillshade columns.
    hillshade_composite = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    hydrology_distance_squared = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    )
    euclidean_distance = hydrology_distance_squared.skb.apply_func(np.sqrt)

    features = X.assign(Hillshade_composite=hillshade_composite)
    features = features.assign(
        Euclidean_Distance_To_Hydrology=euclidean_distance
    )
    features = features.assign(
        Elevation_Hydrology_Interaction=(
            features["Elevation"]
            * features["Euclidean_Distance_To_Hydrology"]
        )
    )
    features = features.drop(
        columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    pred_transformed = features.skb.apply(model, y=y_transformed)

    # Restore the model's 0-6 predictions to the raw target's 1-7 domain. This is
    # gated because a prediction node evaluates to the fitted estimator in fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score — no cv= here; mark_as_X's StratifiedKFold drives validation.
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
