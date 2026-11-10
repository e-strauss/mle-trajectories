import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


CAT_FEATURES = [
    "Title",
    "Title_Pclass",
    "Deck",
    "FamilyType",
    "Embarked",
    "Pclass",
    "Sex",
    "Cabin_side",
]

FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Age_is_missing",
    "Cabin_is_missing",
    "Title",
    "Deck",
    "Cabin_count",
    "Cabin_side",
    "FamilyType",
    "Title_Pclass",
    "Ticket_digit_len",
]


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the CatBoost selection split within each outer training fold."""

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
    return (((cb_probability + xgb_probability) / 2.0) >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read; importing this file performs no I/O.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and raw design matrix.
    # StratifiedShuffleSplit reproduces the original stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — preserve feature names and column order.
    extracted_titles = X["Name"].str.extract(
        r" ([A-Za-z]+)\.", expand=False
    )
    title = extracted_titles.map(
        {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        }
    ).fillna("Rare")

    cabin = X["Cabin"]
    cabin_text = cabin.astype(str)
    deck = cabin_text.str[0].where(cabin.notna(), "Missing")
    cabin_count = (
        cabin_text.str.count(r"\S+").where(cabin.notna(), 0).astype(int)
    )
    first_cabin_number = cabin_text.str.extract(r"(\d+)", expand=False)
    odd_cabin = first_cabin_number.fillna("0").astype(int) % 2 != 0
    cabin_side = (
        odd_cabin.replace({True: "Starboard", False: "Port"})
        .where(first_cabin_number.notna(), "Unknown")
        .where(cabin.notna(), "Missing")
    )

    family_size = X["SibSp"] + X["Parch"] + 1
    family_type = (
        family_size.eq(1)
        .replace({True: "Single", False: "LargeFamily"})
        .where((family_size == 1) | (family_size > 4), "SmallFamily")
        .where(family_size.notna(), "LargeFamily")
    )

    features = X.assign(
        Embarked=X["Embarked"].fillna("Missing"),
        Age_is_missing=X["Age"].isna().astype(int),
        Cabin_is_missing=cabin.isna().astype(int),
        Title=title,
        Deck=deck,
        Cabin_count=cabin_count,
        Cabin_side=cabin_side,
        FamilyType=family_type,
        Title_Pclass=title.astype(str) + "_" + X["Pclass"].astype(str),
        Ticket_digit_len=(
            X["Ticket"].astype(str).str.replace(r"\D", "", regex=True).str.len()
        ),
    )[FEATURE_COLS]

    features = features.astype(
        {
            "Title": "category",
            "Title_Pclass": "category",
            "Deck": "category",
            "FamilyType": "category",
            "Embarked": "category",
            "Pclass": "category",
            "Sex": "category",
            "Cabin_side": "category",
        }
    )

    # The original CatBoost eval_set was the scored hold-out. Even without
    # explicit patience, it can select the best iteration using those labels.
    # Keep that selection behavior, but expose a separate stratified 20% eval
    # split carved from the outer training rows instead; the original leaky
    # validation score cannot be reproduced honestly.
    # Only CatBoost uses the carved split. XGBoost keeps every training row.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probability = cb_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_parts.get("y", y),
        fit_kwargs={
            "cat_features": CAT_FEATURES,
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
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
    pred = cb_probability.skb.apply_func(
        blend_predictions, xgb_probability, skrub.eval_mode()
    )

    # 4. Score — the splitter on mark_as_X supplies the single hold-out.
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
