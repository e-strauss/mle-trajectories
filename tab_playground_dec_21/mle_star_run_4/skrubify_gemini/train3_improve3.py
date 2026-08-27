import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan.
    path = "./input/train.csv"
    raw_df = skrub.as_data_op(path).skb.apply_func(pd.read_csv)

    # 2. Row filtering: exclude classes with fewer samples than n_splits (3)
    #    so StratifiedKFold can split each class across all folds.
    #    Row filtering must happen BEFORE mark_as_X / mark_as_y because it
    #    modifies the total row count.
    target_raw = raw_df["Cover_Type"] - 1
    class_counts = target_raw.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    clean_df = raw_df[~target_raw.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data — mark target (shifted from 1-7 to 0-6) and features.
    #    The CV splitter is set on mark_as_X with split_kwargs={}.
    y = (clean_df["Cover_Type"] - 1).skb.mark_as_y()
    X = clean_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops: compute
    #    Hillshade_composite by averaging the three Hillshade columns,
    #    then drop the original individual Hillshade features.
    X_feat = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        ) / 3
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])

    # 5. Model — LGBMClassifier preserving all hyperparameters from the original.
    model = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        num_leaves=25,
        min_child_samples=30,
        colsample_bytree=0.8,
    )
    pred = X_feat.skb.apply(model, y=y)

    # 6. Score the plan by 3-fold CV. No cv= here — the StratifiedKFold on
    #    mark_as_X drives the evaluation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        print(
            f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}"
        )
