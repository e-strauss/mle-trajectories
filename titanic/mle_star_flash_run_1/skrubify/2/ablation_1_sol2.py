import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from skrub import selectors as s
from xgboost import XGBClassifier


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Learn one frequency map from the fold's training rows."""

    def __init__(self, output_name):
        self.output_name = output_name

    def fit(self, X, y=None):
        self.counts_ = X.iloc[:, 0].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {self.output_name: X.iloc[:, 0].map(self.counts_)},
            index=X.index,
        )


class GroupFareMedian(TransformerMixin, BaseEstimator):
    """Learn the independent (Pclass, Embarked) fare-median lookup."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Pclass", "Embarked"])["Fare"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Pclass", "Embarked"]])
        return pd.DataFrame(
            {"GroupFareMedian": self.medians_.reindex(keys).to_numpy()},
            index=X.index,
        )


class CategoryAlignedXGBClassifier(XGBClassifier):
    """Freeze this fit's category vocabulary, then return probabilities.

    skrub calls predict() on an intermediate model. The matrix is
    (n_samples, n_classes); the blend takes column 1. Unseen scored levels
    are recoded with the training categories, so they become missing instead
    of raising or referring to a different code.
    """

    def fit(self, X, y, **kwargs):
        self.category_dtypes_ = {
            column: X[column].dtype
            for column in X.select_dtypes(include="category").columns
        }
        return super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        aligned = X.copy()
        for column, dtype in self.category_dtypes_.items():
            if column in aligned.columns:
                aligned[column] = aligned[column].astype(dtype)
        return super().predict_proba(aligned, **kwargs)


class GetXY(TransformerMixin, BaseEstimator):
    """Expose CatBoost's internal evaluation split in the recorded plan."""

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


def category_feature_names(frame):
    """Category and object columns, the original Pool's cat_features list."""
    return [
        col
        for col in frame.columns
        if isinstance(frame[col].dtype, pd.CategoricalDtype)
        or frame[col].dtype == object
    ]


def blend_predictions(cb_probability, xgb_probability, mode):
    # Predictor nodes hold fitted estimators, not probabilities, in fit mode.
    # Both predict() results are (n_samples, 2); column 1 is the positive class.
    if mode == "fit":
        return None
    cb_probability = np.asarray(cb_probability, dtype=float)[:, 1]
    xgb_probability = np.asarray(xgb_probability, dtype=float)[:, 1]
    return ((cb_probability + xgb_probability) / 2.0 >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- importing records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target. This splitter reproduces the
    # original stratified train_test_split's exact outer hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    title = (
        X["Name"]
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .str.strip()
    )
    surname = X["Name"].str.extract(r"^([^,]*)", expand=False).str.strip()
    ticket_clean = (
        X["Ticket"]
        .astype(str)
        .str.replace(r"[\./]", "", regex=True)
        .str.strip()
    )
    ticket_prefix = (
        ticket_clean.str.extract(r"^(\S+)\s+\S+", expand=False)
        .fillna("None")
        .astype(str)
    )

    # These statistics originally used the entire table. Separate fitted
    # transformers now learn each statistic on the outer training fold only.
    ticket_frequency = X[["Ticket"]].skb.apply(
        FrequencyEncoder("TicketFrequency"), no_wrap=True
    )
    surname_frequency = surname.to_frame(name="Surname").skb.apply(
        FrequencyEncoder("SurnameFrequency"), no_wrap=True
    )
    group_fare_median = X[["Pclass", "Embarked", "Fare"]].skb.apply(
        GroupFareMedian(), no_wrap=True
    )
    imputed_fare = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True)
    )

    full = X[["Age", "SibSp", "Parch", "Fare"]].assign(
        Pclass=X["Pclass"].astype(str).astype("category"),
        Sex=X["Sex"].astype(str).astype("category"),
        Embarked=X["Embarked"].fillna("Missing").astype(str).astype("category"),
        Title=title.fillna("Missing").astype(str).astype("category"),
        Surname=surname.fillna("Missing").astype(str).astype("category"),
        Sex_Pclass=(
            X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
        ).astype("category"),
        IsMother=(
            (X["Sex"] == "female")
            & (X["Parch"] > 0)
            & (X["Age"] > 18)
            & (title != "Miss")
        ).astype(int).astype(str).astype("category"),
        TicketPrefix=ticket_prefix.astype("category"),
        TicketFrequency=ticket_frequency["TicketFrequency"],
        SurnameFrequency=surname_frequency["SurnameFrequency"],
        LogFare=imputed_fare["Fare"].clip(lower=0).skb.apply_func(np.log1p),
        FareRatioByPclass=(
            X["Fare"] / (group_fare_median["GroupFareMedian"] + 1e-5)
        ),
    )

    # Preserve dtype-dependent handling of any object-typed numerical inputs.
    full = full.skb.apply(skrub.ToCategorical(), cols=s.has_dtype("object"))

    # Unlike the original full-table categorical conversion, per-fold pandas
    # conversions can produce different train/validation category vocabularies.
    # CategoryAlignedXGBClassifier freezes XGBoost's training vocabulary:
    # unseen values become native missing categories, rather than raising during
    # scoring or silently referring to different category codes. CatBoost still
    # receives the original strings and can handle unseen categories itself.

    # Enumerate exactly the four original variants, preserving column order.
    features = skrub.choose_from(
        {
            "Full Feature Pipeline": full,
            "Ablation 1 (Drop Title & Demographic Interactions)": full.drop(
                columns=["Title", "Surname", "Sex_Pclass", "IsMother"]
            ),
            "Ablation 2 (Drop Ticket, Group & Relative Economic Features)": full.drop(
                columns=[
                    "TicketPrefix",
                    "TicketFrequency",
                    "SurnameFrequency",
                    "LogFare",
                    "FareRatioByPclass",
                ]
            ),
            "Ablation 3 (Drop Base Numerical Features: Age, Fare, SibSp, Parch)": full.drop(
                columns=["Age", "Fare", "SibSp", "Parch"]
            ),
        },
        name="variant",
    ).as_data_op()

    # CatBoost originally monitored the scored hold-out. Even without explicit
    # patience, eval_set enables its default best-iteration selection. Keep
    # eval_set and all 500 iterations, but obtain evaluation rows from the outer
    # training fold instead, using the original split fraction and stratification.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    # Column names follow the chosen variant. The list is fit-only metadata.
    cat_names = cb_parts["X"].skb.apply_func(category_feature_names)
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
            "cat_features": cat_names,
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    # XGBoost fits on the full outer training fold. random_seed is what the
    # original passes; XGBoost ignores it, and that is left unchanged.
    xgb_probability = features.skb.apply(
        CategoryAlignedXGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method="hist",
            random_seed=42,
            eval_metric="logloss",
        ),
        y=y,
    )
    pred = cb_probability.skb.apply_func(
        blend_predictions, xgb_probability, skrub.eval_mode()
    )

    # 4. Score every original variant in one search. The marked splitter drives.
    # Reporting-only deltas and the "most critical" conclusion are omitted.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy",
            error_score="raise",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
