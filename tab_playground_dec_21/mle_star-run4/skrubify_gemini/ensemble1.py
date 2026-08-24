import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class MultiSeedStratifiedKFold(BaseCrossValidator):
    """Replicates the original script's outer loop over multiple random seeds for StratifiedKFold."""

    def __init__(self, n_splits=3, n_repeats=3, base_random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.base_random_state = base_random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups=None):
        for r in range(self.n_repeats):
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.base_random_state + r,
            )
            yield from skf.split(X, y, groups)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan.
    train_df = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Row filtering: exclude classes with fewer than n_splits (3) samples as in the original.
    #    Row dropping must happen before mark_as_X / mark_as_y since it alters row count.
    raw_target = train_df["Cover_Type"] - 1
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_df = train_df[~raw_target.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data — mark target and design matrix.
    #    The original ran 3 repetitions of 3-fold StratifiedKFold with seeds 42, 43, 44.
    #    Expressed via MultiSeedStratifiedKFold on mark_as_X.
    y = (filtered_df["Cover_Type"] - 1).skb.mark_as_y()
    X = filtered_df.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=MultiSeedStratifiedKFold(n_splits=3, n_repeats=3, base_random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops:
    #    compute Hillshade_composite and Elevation_at_Hydrology, then drop unused columns.
    X_feat = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3.0,
        Elevation_at_Hydrology=X["Elevation"] - X["Vertical_Distance_To_Hydrology"],
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Aspect"])

    # 5. Model — soft-voting ensemble of 5 RandomForestClassifiers with seeds 100..104,
    #    matching the original per-fold sub-model averaging.
    sub_models = [
        (
            f"rf_{i}",
            RandomForestClassifier(
                n_estimators=100, random_state=100 + i, n_jobs=-1
            ),
        )
        for i in range(5)
    ]
    model = VotingClassifier(estimators=sub_models, voting="soft")
    pred = X_feat.skb.apply(model, y=y)

    # 6. Score the whole plan by CV. mean_test_score computes the mean accuracy across
    #    all folds/repeats, matching the original's final validation performance.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        print(
            f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}"
        )
