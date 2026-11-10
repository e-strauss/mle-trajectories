import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Provide a best-model-selection eval set from each fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this module records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original feature selection.
    # StratifiedShuffleSplit reproduces the original stratified hold-out rows.
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    cat_features = ["Sex", "Embarked", "Pclass"]

    y = data["Survived"].skb.mark_as_y()
    X = data[features].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — retain column order and categorical semantics.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    X = X.assign(
        Sex=X["Sex"].astype(str),
        Embarked=X["Embarked"].astype(str),
        Pclass=X["Pclass"].astype(str),
    )

    # The original eval_set enabled CatBoost's default best-iteration selection
    # on the scored hold-out, leaking validation information. Preserve eval-set
    # selection, but carve its rows out of the outer training fold instead.
    # No early-stopping patience was specified, so none is added.
    parts = X.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_val = parts["X_val"]
    y_val = parts["y_val"]

    pred = X_fit.skb.apply(
        CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y_fit,
        fit_kwargs={
            "cat_features": cat_features,
            "eval_set": (X_val, y_val),
            "verbose": False,
        },
    )

    # 4. Score — the splitter on mark_as_X drives the single hold-out evaluation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
