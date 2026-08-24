import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


def drop_rare_target_classes(df, target="Cover_Type", n_splits=3):
    """Reproduce the original removal of classes with fewer than three rows."""
    class_counts = df[target].value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index
    return df.loc[~df[target].isin(problematic_classes)].reset_index(drop=True)


class ThreeSeedStratifiedKFold(BaseCrossValidator):
    """Yield the original three folds for random states 42, 43, and 44."""

    def __init__(self, n_splits=3, random_states=(42, 43, 44)):
        self.n_splits = n_splits
        self.random_states = random_states

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * len(self.random_states)

    def split(self, X, y, groups=None):
        y_array = np.asarray(y).reshape(-1)
        for random_state in self.random_states:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=random_state,
            )
            yield from splitter.split(X, y_array)


class RandomForestSoftVoteClassifier(ClassifierMixin, BaseEstimator):
    """Fit five random forests and average their predicted probabilities."""

    def __init__(
        self,
        n_sub_models=5,
        n_estimators=100,
        random_state_base=100,
        n_jobs=-1,
    ):
        self.n_sub_models = n_sub_models
        self.n_estimators = n_estimators
        self.random_state_base = random_state_base
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y_array = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y_array)
        self.models_ = []

        for model_index in range(self.n_sub_models):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state_base + model_index,
                n_jobs=self.n_jobs,
            )
            model.fit(X, y_array)
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        probabilities = [model.predict_proba(X) for model in self.models_]
        return np.mean(np.array(probabilities), axis=0)

    def predict(self, X):
        # Preserve the original script's positional np.argmax semantics.
        return np.argmax(self.predict_proba(X), axis=1)


def restore_original_target_labels(predictions, mode):
    """Map predictions from transformed labels 0-6 to raw labels 1-7."""
    if mode == "fit":
        # During fitting, a prediction node evaluates to the fitted estimator.
        return None
    return np.asarray(predictions) + 1


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read. Importing this file builds the plan
    #    without reading the data. The original error/progress printing is not
    #    part of producing the cross-validated score and is omitted.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # Row filtering changes the number of samples, so it must happen before the
    # marks. This reproduces the original exclusion of target classes represented
    # by fewer than three rows.
    data = data.skb.apply_func(
        drop_rare_target_classes,
        target="Cover_Type",
        n_splits=3,
    )

    # 2. Prepare data -- mark the RAW target and design matrix. The transformed
    #    0-6 target used by the original is derived only after mark_as_y.
    y_raw = data["Cover_Type"].skb.mark_as_y()
    y_transformed = y_raw - 1

    # The original repeated three-fold StratifiedKFold for seeds 42, 43, and 44,
    # then averaged the three run means. Yielding all nine folds from one custom
    # splitter produces the same unweighted meta-average.
    X = data.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=ThreeSeedStratifiedKFold(
            n_splits=3,
            random_states=(42, 43, 44),
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Assigning both new columns before dropping
    #    the original columns preserves the original feature values and order.
    features = X.assign(
        Hillshade_composite=(
            X["Hillshade_9am"]
            + X["Hillshade_Noon"]
            + X["Hillshade_3pm"]
        )
        / 3,
        Elevation_at_Hydrology=(
            X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
        ),
    ).drop(
        columns=[
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Aspect",
        ]
    )

    # The original hand-rolled five-forest soft vote belongs in an estimator
    # wrapper so all five models are fitted independently within every CV fold.
    pred_transformed = features.skb.apply(
        RandomForestSoftVoteClassifier(
            n_sub_models=5,
            n_estimators=100,
            random_state_base=100,
            n_jobs=-1,
        ),
        y=y_transformed,
    )

    # Restore predictions to the raw target domain for scoring. This is gated on
    # eval_mode because prediction nodes hold fitted estimators in fit mode.
    pred = pred_transformed.skb.apply_func(
        restore_original_target_labels,
        skrub.eval_mode(),
    )

    # 4. Score. No cv= here -- the splitter on mark_as_X drives validation.
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
