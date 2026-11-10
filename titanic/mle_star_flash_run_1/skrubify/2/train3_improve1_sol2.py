import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Learn one column's group medians and its overall fallback median."""

    def __init__(self, column, group_columns):
        self.column = column
        self.group_columns = group_columns

    def fit(self, X, y=None):
        self.group_medians_ = (
            X.groupby(self.group_columns)[self.column]
            .median()
            .reset_index()
            .drop_duplicates(subset=self.group_columns, keep="last")
        )
        self.overall_median_ = X[self.column].median()
        return self

    def transform(self, X):
        # The lookup has one row per key, so this left join cannot add rows.
        lookup = X[self.group_columns].merge(
            self.group_medians_,
            on=self.group_columns,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        group_values = pd.Series(
            lookup[self.column].to_numpy(), index=X.index
        )
        filled = (
            X[self.column]
            .fillna(group_values)
            .fillna(self.overall_median_)
        )
        return X.assign(**{self.column: filled})


class TicketGroupSize(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies from this fold's training rows only."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(
            TicketGroupSize=X["Ticket"]
            .map(self.counts_)
            .fillna(1)
            .astype(int)
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose a training-only evaluation split as recorded graph outputs."""

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
    # 1. Load Data -- record the read; importing this file reads nothing.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and raw feature matrix.
    # StratifiedShuffleSplit reproduces the original stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    titles = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Officer",
        "Rev": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Capt": "Officer",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Jonkheer": "Royalty",
        "Dona": "Royalty",
    }
    features = X.assign(Title=titles.map(title_mapping).fillna("Other"))

    # The original learned medians and ticket counts before splitting.
    # Each independent statistic now has its own fold-fitted node, fixing
    # that leakage while retaining the same imputation and counting rules.
    features = features.skb.apply(
        GroupMedianImputer("Age", ["Title", "Pclass"]),
        no_wrap=True,
    )
    features = features.skb.apply(TicketGroupSize(), no_wrap=True)
    features = features.skb.apply(
        GroupMedianImputer("Fare", ["Pclass"]),
        no_wrap=True,
    )

    fare_clean = features["Fare"]
    fare_per_ticket = fare_clean / features["TicketGroupSize"]
    features = features.assign(
        Fare=fare_clean.clip(lower=0).skb.apply_func(np.log1p),
        FarePerTicketGroup=fare_per_ticket.clip(lower=0).skb.apply_func(
            np.log1p
        ),
        HasCabin=features["Cabin"].notna().astype(int),
        Embarked=features["Embarked"].fillna("Missing"),
    )
    features = features.assign(
        Sex_Pclass=(
            features["Sex"].astype(str)
            + "_"
            + features["Pclass"].astype(str)
        ),
        IsChild=(features["Age"] < 12).astype(int),
        IsMother=(
            (features["Sex"] == "female")
            & (features["Parch"] > 0)
            & (features["Age"] > 18)
            & (features["Title"] == "Mrs")
        ).astype(int),
    )

    # Explicit selection preserves the original design-matrix column order.
    features = features[
        [
            "Title",
            "Age",
            "Pclass",
            "Sex",
            "SibSp",
            "Parch",
            "TicketGroupSize",
            "Fare",
            "FarePerTicketGroup",
            "HasCabin",
            "Embarked",
            "Sex_Pclass",
            "IsChild",
            "IsMother",
        ]
    ].astype(
        {
            "Sex": "category",
            "Embarked": "category",
            "Pclass": "category",
            "Title": "category",
            "Sex_Pclass": "category",
        }
    )

    # CatBoost's original eval_set was the scored hold-out. Even without an
    # explicit patience, it can select the best iteration using that set.
    # Keep the eval_set behavior, but expose a separate stratified 20% split
    # of the fold's training rows so scored rows cannot influence fitting.
    # Consequently, the original leaky validation score is not reproducible.
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
        y=cb_parts.get("y", y),
        fit_kwargs={
            "cat_features": ["Sex", "Embarked", "Pclass", "Title", "Sex_Pclass"],
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

    # 4. Score -- the splitter on mark_as_X owns validation.
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
