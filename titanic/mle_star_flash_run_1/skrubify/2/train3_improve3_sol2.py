import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn the Title/Pclass age medians from the fold's training rows."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(
            ["Title", "Pclass"], observed=True
        )["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Title", "Pclass"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        # groupby.transform drops rows whose group key is null. The raw-age
        # fallback is applied afterwards, on a separate imputer.
        valid_group = X["Title"].notna() & X["Pclass"].notna()
        filled = X["Age"].fillna(medians).where(valid_group)
        return X[["Age"]].assign(Age=filled)


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies independently of the other imputations."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"TicketCount": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose CatBoost's internal evaluation split as recorded graph nodes."""

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


FEATURES = [
    "Pclass",
    "Sex",
    "Title",
    "Age",
    "SibSp",
    "Parch",
    "PerPersonFare_Log",
    "Embarked",
    "Is_Woman_or_Child",
    "Is_Alone",
    "Age_Pclass",
]
CAT_FEATURES = ["Pclass", "Sex", "Title", "Embarked"]
TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
}


def positive_probability(values):
    """Column 1 of a probability matrix, or a vector that is already that column."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return values[:, 1]
    return values


def ensemble_predictions(cb_probs, xgb_probs, mode):
    if mode == "fit":
        return None
    blended = (
        positive_probability(cb_probs) + positive_probability(xgb_probs)
    ) / 2.0
    return (blended >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Mark the raw target and raw features. This splitter reproduces the
    #    original stratified train_test_split exactly.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    title_raw = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title_raw.map(TITLE_MAP).fillna("Rare").where(
        title_raw.notna(), "Missing"
    )
    titled = X.assign(Title=title)

    # These statistics originally used the whole table. Each now learns only
    # from the outer fold's training rows, fixing that preprocessing leakage.
    group_age = titled[["Title", "Pclass", "Age"]].skb.apply(
        GroupAgeImputer()
    )
    global_age = X[["Age"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )
    age = group_age["Age"].fillna(global_age["Age"])

    ticket_counts = X[["Ticket"]].skb.apply(TicketFrequency())
    fare = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )
    per_person_fare_log = (
        fare["Fare"] / ticket_counts["TicketCount"]
    ).skb.apply_func(np.log1p)

    features = titled.assign(
        Age=age,
        PerPersonFare_Log=per_person_fare_log,
        Is_Woman_or_Child=(
            (X["Sex"] == "female") | (age < 12) | (title == "Master")
        ).astype(int),
        Is_Alone=((X["SibSp"] + X["Parch"]) == 0).astype(int),
        Age_Pclass=age * X["Pclass"].astype(float),
        Embarked=X["Embarked"].fillna("Missing"),
    )
    features = features[FEATURES].astype(
        {
            "Pclass": "category",
            "Sex": "category",
            "Title": "category",
            "Embarked": "category",
        }
    )

    # CatBoost's original eval_set enables its default best-iteration selection,
    # even though no explicit early-stopping patience was specified. The original
    # used the scored holdout for that selection, which leaks. Preserve eval_set
    # and all model parameters, but carve a stratified 20% evaluation set from
    # the fold's training rows instead. The original specifies no separate inner
    # fraction, so its 20% holdout fraction is reused here.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)

    # XGBoost trains on all outer-training rows. The blend is the original
    # equal-weight positive-class average and >= 0.5 threshold.
    cb_probs = cb_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_parts["y"],
        fit_kwargs={
            "cat_features": CAT_FEATURES,
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    xgb_probs = features.skb.apply(
        XGBModel(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method="hist",
            objective="binary:logistic",
            random_state=42,
            eval_metric="logloss",
        ),
        y=y,
    )
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, xgb_probs, skrub.eval_mode()
    )

    # 4. Score the single ensemble, not its constituent models separately.
    #    The splitter on mark_as_X drives validation; no cv= belongs here.
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
