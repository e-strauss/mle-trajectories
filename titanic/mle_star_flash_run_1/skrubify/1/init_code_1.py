import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from the current fold's training rows."""

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

    # 2. Prepare data — mark the raw target and raw design matrix.
    # StratifiedShuffleSplit reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — preserve the original feature order.
    features = X.assign(
        Title=X["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        ).fillna("Missing"),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        CabinDeck=X["Cabin"].str[0].fillna("Missing"),
        Embarked=X["Embarked"].fillna("Missing"),
    )
    features = features[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title",
            "FamilySize",
            "CabinDeck",
        ]
    ]

    # Learn Fare's median on each outer training fold rather than on the whole
    # table, fixing the original preprocessing leakage. Age is left unchanged.
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Fare"]
    )

    # The original early-stopped on its scored hold-out. Instead, retain its
    # patience and iteration limit while carving a stratified 20% eval set out
    # of the outer training fold; the scored rows are never used for stopping.
    parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_val = parts["X_val"]
    y_val = parts["y_val"]

    pred = X_fit.skb.apply(
        CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=y_fit,
        fit_kwargs={
            "cat_features": ["Pclass", "Sex", "Embarked", "Title", "CabinDeck"],
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 50,
        },
    )

    # 4. Score — the splitter declared on mark_as_X drives the single hold-out.
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
