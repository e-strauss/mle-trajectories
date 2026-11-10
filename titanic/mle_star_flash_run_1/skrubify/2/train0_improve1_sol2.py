import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


CAT_FEATURES = ["Sex", "Embarked", "Pclass", "Title", "WomanOrChild"]


class GroupAgeImputer(TransformerMixin, BaseEstimator):
    """Learn group medians using only the current fold's training rows."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(
            ["Title", "Pclass"], observed=True
        )["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Title", "Pclass"]])
        medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(),
            index=X.index,
        )
        return pd.DataFrame(
            {"Age": X["Age"].fillna(medians)}, index=X.index
        )


class GlobalAgeMedian(TransformerMixin, BaseEstimator):
    """Keep the fallback median separate from group-based imputation."""

    def fit(self, X, y=None):
        self.median_ = X["Age"].median()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"Age": X["Age"].fillna(self.median_)}, index=X.index
        )


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"TicketGroupSize": X["Ticket"].map(self.counts_)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose CatBoost's fold-local evaluation split in the graph."""

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


class CategoricalVocabulary(TransformerMixin, BaseEstimator):
    """Reuse the training fold's category vocabulary for XGBoost.

    A per-fold cast gives the scored rows their own codes and can name a
    category XGBoost never saw. Applying the training dtypes turns those
    values into missing. CatBoost reads the values, so it does not need this.
    """

    def fit(self, X, y=None):
        self.dtypes_ = {column: X[column].dtype for column in CAT_FEATURES}
        return self

    def transform(self, X):
        return X.astype(self.dtypes_)


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
    # 1. Load Data -- importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and raw design matrix.
    # This splitter reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    features = X[["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]]
    features = features.assign(
        Embarked=features["Embarked"].fillna("Missing")
    )
    title = (
        X["Name"]
        .str.extract(r" ([A-Za-z]+)\.", expand=False)
        .fillna("Missing")
    )
    features = features.assign(Title=title, Age=X["Age"])

    # Fix the original full-table imputation leakage. The fallback is learned
    # from RAW ages, not from ages already filled with group medians.
    group_age = features[["Title", "Pclass", "Age"]].skb.apply(
        GroupAgeImputer(), no_wrap=True
    )
    global_age = X[["Age"]].skb.apply(
        GlobalAgeMedian(), no_wrap=True
    )
    features = features.assign(
        Age=group_age["Age"].fillna(global_age["Age"])
    )
    features = features.assign(
        WomanOrChild=(
            (features["Sex"] == "female") | (features["Age"] <= 12)
        ).astype(int)
    )

    # Ticket frequencies are independently learned on the training fold.
    ticket_counts = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )
    features = features.assign(
        TicketGroupSize=ticket_counts["TicketGroupSize"]
    )
    features = features.assign(
        AdjustedFare=features["Fare"] / features["TicketGroupSize"]
    )

    # Token count matching the original len(str(cabin).split()); missing cabins are 0.
    features = features.assign(
        CabinCount=X["Cabin"].fillna("").astype(str).str.count(r"\S+")
    )
    features = features.astype(
        {
            "Sex": "category",
            "Embarked": "category",
            "Pclass": "category",
            "Title": "category",
            "WomanOrChild": "category",
        }
    )

    # The original used the scored hold-out for CatBoost's best-iteration
    # selection. Preserve evaluation-based selection, but carve the eval set
    # from the training fold; the original leaky score is not reproducible.
    # XGBoost still fits every outer-training row, on the training vocabulary.
    xgb_features = features.skb.apply(CategoricalVocabulary(), no_wrap=True)
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

    # 4. Score -- validation is controlled only by mark_as_X.
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
