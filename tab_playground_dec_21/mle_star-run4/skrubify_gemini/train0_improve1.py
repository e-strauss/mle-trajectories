import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan.
    path = "./input/train.csv"
    raw_df = skrub.as_data_op(path).skb.apply_func(pd.read_csv)

    # 2. Row filtering: classes with fewer samples than n_splits (3) are dropped
    #    before marking X and y, exactly as in the original code. Row filtering
    #    must happen before mark_as_X / mark_as_y since it changes the number of rows.
    raw_target = raw_df["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    clean_df = raw_df[~raw_target.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data
    #    Target (y): 'Cover_Type', shifted from 1-7 to 0-6 as in the original.
    #    Features (X): drop 'Id' and 'Cover_Type'.
    #    The CV splitter lives on mark_as_X; the 3-fold StratifiedKFold loop from the
    #    original is expressed directly as the CV splitter here.
    y = (clean_df["Cover_Type"] - 1).skb.mark_as_y()
    X = clean_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops:
    #    - Aspect converted to radians and split into sine and cosine components.
    #    - Hillshade composite computed as average of 9am, Noon, and 3pm columns.
    #    - Original Aspect and Hillshade columns dropped.
    aspect_rad = X["Aspect"].skb.apply_func(np.radians)
    aspect_sin = aspect_rad.skb.apply_func(np.sin)
    aspect_cos = aspect_rad.skb.apply_func(np.cos)
    hillshade_comp = (
        X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
    ) / 3

    X_feat = X.assign(
        Aspect_sin=aspect_sin,
        Aspect_cos=aspect_cos,
        Hillshade_composite=hillshade_comp,
    ).drop(
        columns=["Aspect", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    )

    # 5. Model — same RandomForestClassifier hyperparameters as the original.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pred = X_feat.skb.apply(model, y=y)

    # 6. Score the whole plan by 3-fold CV. No cv= here — the StratifiedKFold set on
    #    mark_as_X drives. mean_test_score is the mean fold accuracy.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        print(
            f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}"
        )
