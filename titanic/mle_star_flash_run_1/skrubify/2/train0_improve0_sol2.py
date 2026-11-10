import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class GetXY(TransformerMixin, BaseEstimator):
    """Provide a fold-local evaluation set for CatBoost's best-model selection."""

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


class XGBClassifier(XGBClassifier):
    """skrub calls predict() on an intermediate model.

    Return the (n_samples, n_classes) probability matrix. The blend takes
    column 1, the positive class, as it does for CatBoost.
    """

    def predict(self, X, **kwargs):
        return super().predict_proba(X)


def blend_predictions(cb_probability, xgb_probability, mode):
    # Predictor nodes hold fitted estimators, not probabilities, in fit mode.
    # Both models' predict() returns (n_samples, 2) in classes_ order [0, 1],
    # so column 1 is the positive-class probability.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    xgb_probability = np.asarray(xgb_probability, dtype=float)[:, 1]
    return ((cb_probability + xgb_probability) / 2.0 >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — the read is recorded, so importing this file reads nothing.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and retain source columns needed
    #    for feature engineering. This splitter reproduces the stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — preserve both optional-column branches.
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Capt": "Rare",
        "Countess": "Rare",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Sir": "Rare",
    }

    has_name = X.columns.isin(["Name"]).any()
    names = X.reindex(columns=["Name"])["Name"]
    extracted_titles = names.str.extract(r" ([A-Za-z]+)\.", expand=False)
    mapped_titles = extracted_titles.map(title_mapping).fillna("Rare")
    titles = has_name.skb.if_else(mapped_titles, "Missing")

    has_cabin = X.columns.isin(["Cabin"]).any()
    cabins = X.reindex(columns=["Cabin"])["Cabin"]
    extracted_decks = cabins.astype(str).str[0].replace(
        {"n": "Missing", "N": "Missing"}
    )
    decks = has_cabin.skb.if_else(extracted_decks, "Missing")

    features = X[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    features = features.assign(Title=titles, Deck=decks)
    features = features.assign(
        FamilySize=features["SibSp"] + features["Parch"] + 1
    )
    features = features.assign(
        IsAlone=(features["FamilySize"] == 1).astype(int)
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["FamilySize"]
    )
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing"),
        Deck=features["Deck"].fillna("Missing"),
        Title=features["Title"].fillna("Missing"),
    )

    cat_features = ["Sex", "Embarked", "Pclass", "Title", "Deck"]
    features = features.astype(
        {
            "Sex": "category",
            "Embarked": "category",
            "Pclass": "category",
            "Title": "category",
            "Deck": "category",
        }
    )

    # The original gave CatBoost the scored validation set. Even without explicit
    # early stopping, that enables best-model selection on those scored rows.
    # Keep eval_set, but carve it from the outer fold's training rows instead.
    # XGBoost had no eval set, so it keeps every training row of the fold.
    pieces = features.skb.apply(GetXY(), y=y, no_wrap=True)

    cb_probability = pieces["X"].skb.apply(
        CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=pieces.get("y", y),
        fit_kwargs={
            "cat_features": cat_features,
            "eval_set": (pieces["X_val"], pieces["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    xgb_probability = features.skb.apply(
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        ),
        y=y,
    )

    # Preserve the original equal probability average and >= 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        blend_predictions, xgb_probability, skrub.eval_mode()
    )

    # 4. Score — only mark_as_X controls the outer validation split.
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
