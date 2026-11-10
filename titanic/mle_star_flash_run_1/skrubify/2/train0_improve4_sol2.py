import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


CAT_FEATURES = [
    "Sex", "Embarked", "Pclass", "Sex_Pclass",
    "TicketPrefix", "Title", "Surname", "IsMother",
]

FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "Title", "Surname", "TicketPrefix", "TicketFrequency",
    "SurnameFrequency", "LogFare", "FareRatioByPclass",
    "Sex_Pclass", "IsMother",
]


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Learn one frequency map from the training rows."""

    def __init__(self, column, output_name):
        self.column = column
        self.output_name = output_name

    def fit(self, X, y=None):
        self.counts_ = X[self.column].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {self.output_name: X[self.column].map(self.counts_)},
            index=X.index,
        )


class GroupFareMedian(TransformerMixin, BaseEstimator):
    """Learn the original Pclass/Embarked fare medians."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Embarked"])["Fare"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Embarked"]])
        return pd.DataFrame(
            {"GroupFareMedian": self.medians_.reindex(keys).to_numpy()},
            index=X.index,
        )


class CategoryVocabulary(TransformerMixin, BaseEstimator):
    """Keep XGBoost's category codes and vocabulary stable across prediction."""

    def fit(self, X, y=None):
        self.dtypes_ = X[CAT_FEATURES].dtypes.to_dict()
        return self

    def transform(self, X):
        # Independently casting each fold to category produces different codes
        # and introduces unseen category names at prediction time. Reusing the
        # fitted dtypes keeps codes stable and represents unseen values as NaN.
        return X.astype(self.dtypes_)


class GetXY(TransformerMixin, BaseEstimator):
    """Carve CatBoost's evaluation set from the outer training rows."""

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
    # 1. Load Data — record the read without accessing the file on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and raw input columns.
    # This reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    title = X["Name"].str.extract(
        r",\s*([^\.]+)\.", expand=False
    ).str.strip()
    surname = X["Name"].str.extract(
        r"^([^,]*)", expand=False
    ).str.strip()

    ticket_clean = (
        X["Ticket"].astype(str)
        .str.replace(r"[\./]", "", regex=True)
        .str.strip()
    )
    ticket_prefix = ticket_clean.str.extract(
        r"^(\S+)\s+\S+", expand=False
    ).fillna("None")

    # Learn each statistic independently on training rows, fixing the original
    # full-table leakage. Direct application preserves the named output columns
    # and lets the group-median estimator see its complete input frame.
    ticket_frequency = X[["Ticket"]].skb.apply(
        FrequencyEncoder("Ticket", "TicketFrequency"),
        no_wrap=True,
    )
    surname_frequency = (
        X[["Name"]]
        .assign(Surname=surname)[["Surname"]]
        .skb.apply(
            FrequencyEncoder("Surname", "SurnameFrequency"),
            no_wrap=True,
        )
    )
    group_median = X[["Pclass", "Embarked", "Fare"]].skb.apply(
        GroupFareMedian(), no_wrap=True
    )
    fare_imputed = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median").set_output(transform="pandas"),
        no_wrap=True,
    )

    features = X[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ]
    features = features.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.assign(Title=title.fillna("Missing"))
    features = features.assign(Surname=surname.fillna("Missing"))
    features = features.assign(TicketPrefix=ticket_prefix)
    features = features.assign(
        TicketFrequency=ticket_frequency["TicketFrequency"]
    )
    features = features.assign(
        SurnameFrequency=surname_frequency["SurnameFrequency"]
    )
    features = features.assign(
        LogFare=fare_imputed["Fare"].clip(lower=0).skb.apply_func(np.log1p)
    )
    features = features.assign(
        FareRatioByPclass=X["Fare"] / (group_median["GroupFareMedian"] + 1e-5)
    )
    features = features.assign(
        Sex_Pclass=X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
    )
    features = features.assign(
        IsMother=(
            (X["Sex"] == "female")
            & (X["Parch"] > 0)
            & (X["Age"] > 18)
            & (title != "Miss")
        ).astype(int)
    )
    features = features[FEATURE_COLS].astype(
        {
            "Sex": "category",
            "Embarked": "category",
            "Pclass": "category",
            "Sex_Pclass": "category",
            "TicketPrefix": "category",
            "Title": "category",
            "Surname": "category",
            "IsMother": "category",
        }
    )

    # Unlike CatBoost, XGBoost consumes categorical dictionaries/codes. Freeze
    # those dictionaries using training data rather than independently inferred
    # validation dictionaries; CatBoost still receives the original values.
    xgb_features = features.skb.apply(
        CategoryVocabulary(), no_wrap=True
    )

    # The original CatBoost eval_set was the scored hold-out. Its default
    # best-iteration selection leaks even without explicit patience. Retain
    # eval_set but take its rows from the outer training fold instead; the
    # original optimistic score cannot be reproduced exactly. No patience is
    # invented, and XGBoost continues to use all outer-training rows.
    cat_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probability = cat_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cat_parts.get("y", y),
        fit_kwargs={
            "cat_features": CAT_FEATURES,
            "eval_set": (cat_parts["X_val"], cat_parts["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    xgb_probability = xgb_features.skb.apply(
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

    # 4. Score — the splitter on mark_as_X drives the single hold-out.
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
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
