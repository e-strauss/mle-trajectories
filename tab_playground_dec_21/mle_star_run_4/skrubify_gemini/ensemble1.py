import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


# Custom splitter to reproduce the original script's multi-seed outer CV loop:
# 3 repeats of 3-fold StratifiedKFold with seeds 42, 43, 44.
class MultiSeedStratifiedKFold(BaseCrossValidator):
    def __init__(self, n_splits=3, n_repeats=3, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups=None):
        for r in range(self.n_repeats):
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state + r,
            )
            yield from skf.split(X, y)


# Custom estimator wrapper to reproduce the original's fold-level ensemble:
# 5 RandomForestClassifiers trained with seeds 100..104, averaged via predict_proba,
# and decoded using positional argmax as in the original.
class SoftVoteRFEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_sub_models=5,
        random_state_base=100,
        n_estimators=100,
        n_jobs=-1,
    ):
        self.n_sub_models = n_sub_models
        self.random_state_base = random_state_base
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.models_ = []
        for i in range(self.n_sub_models):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state_base + i,
                n_jobs=self.n_jobs,
            )
            model.fit(X, y)
            self.models_.append(model)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models_]
        return np.mean(probas, axis=0)

    def predict(self, X):
        # Reproduce positional argmax decoding from the original script
        avg_proba = self.predict_proba(X)
        return np.argmax(avg_proba, axis=1)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record CSV read as the first step of the plan.
    path = "./input/train.csv"
    raw_df = skrub.as_data_op(path).skb.apply_func(pd.read_csv)

    # 2. Row filtering — exclude classes with fewer samples than n_splits (3)
    #    before marking X and y, exactly as in the original script.
    raw_target = raw_df["Cover_Type"]
    y_shifted = raw_target - 1
    class_counts = y_shifted.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered_df = raw_df[~y_shifted.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare Data — mark target (shifted 1-7 to 0-6) and features.
    #    The CV splitter is set on mark_as_X with split_kwargs={}.
    y = (filtered_df["Cover_Type"] - 1).skb.mark_as_y()
    X = filtered_df.drop(columns=["Cover_Type", "Id"]).skb.mark_as_X(
        cv=MultiSeedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42),
        split_kwargs={},
    )

    # 4. Feature engineering as fine-grained recorded ops:
    #    Create Hillshade_composite and Elevation_at_Hydrology, then drop
    #    Hillshade_9am, Hillshade_Noon, Hillshade_3pm, and Aspect.
    X_feat = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=X["Elevation"] - X["Vertical_Distance_To_Hydrology"],
    ).drop(columns=["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Aspect"])

    # 5. Apply the 5-model Random Forest soft-voting ensemble per fold.
    model = SoftVoteRFEnsemble(
        n_sub_models=5,
        random_state_base=100,
        n_estimators=100,
        n_jobs=-1,
    )
    pred = X_feat.skb.apply(model, y=y)

    # 6. Score the plan by cross-validation using the accuracy metric.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}")
