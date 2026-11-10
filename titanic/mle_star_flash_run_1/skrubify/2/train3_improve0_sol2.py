import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Learn one conditional median table from the fold's training rows."""

    def __init__(self, column, groups):
        self.column = column
        self.groups = groups

    def fit(self, X, y=None):
        self.medians_ = (
            X.groupby(self.groups, observed=True)[self.column]
            .median()
            .rename("_group_median")
        )
        return self

    def transform(self, X):
        # The groupby produces exactly one lookup value per key, so this
        # left join preserves both the number and the order of input rows.
        joined = X.join(self.medians_, on=self.groups, how="left")
        return X.assign(
            **{self.column: X[self.column].fillna(joined["_group_median"])}
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose CatBoost's per-fit evaluation split as a node in the plan."""

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


CAT_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FarePerPerson",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "Deck",
]

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
    # 1. Load Data — the file is read only when the plan is evaluated.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and reproduce the original
    #    stratified 80/20 hold-out, including its exact random seed.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering.
    #    Reindex implements the original optional-column branches: absent
    #    Name/Cabin produce Missing, while an absent Fare produces zeros.
    name = X.reindex(columns=["Name"], fill_value=np.nan)["Name"].astype("str")
    title = name.str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title.replace(
        [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona",
        ],
        "Rare",
    )
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    title = title.fillna("Missing")

    family_size = X["SibSp"] + X["Parch"] + 1
    cabin = X.reindex(columns=["Cabin"], fill_value=np.nan)["Cabin"]
    deck_initial = cabin.astype(str).str[0]
    deck = deck_initial.where(
        deck_initial.isin(["A", "B", "C", "D", "E", "F", "G", "T"]),
        "Missing",
    )

    features = X.assign(
        Title=title,
        FamilySize=family_size,
        IsAlone=(family_size == 1).astype(int),
        Fare=X.reindex(columns=["Fare"], fill_value=0.0)["Fare"],
        Deck=deck,
        Embarked=X["Embarked"].fillna("Missing"),
    )

    # Each learned aggregation is its own estimator. Unlike the original's
    # whole-table medians, these are fitted only on each fold's training rows.
    features = features.skb.apply(
        GroupMedianImputer(column="Fare", groups=["Pclass"])
    )
    features = features.assign(
        FarePerPerson=features["Fare"] / features["FamilySize"]
    )
    features = features.skb.apply(
        GroupMedianImputer(column="Age", groups=["Title", "Pclass"])
    )
    features = features.skb.apply(
        GroupMedianImputer(column="Age", groups=["Pclass"])
    )
    features = features.skb.apply(
        SimpleImputer(strategy="median"), cols=["Age"]
    )

    features = features[FEATURES].astype(
        {
            "Pclass": "category",
            "Sex": "category",
            "Embarked": "category",
            "Title": "category",
            "Deck": "category",
        }
    )

    # CatBoost's eval_set can select the best iteration even without explicit
    # early-stopping patience. Keep that behavior, but do not expose the scored
    # hold-out to fitting: carve a stratified 20% eval set from training rows.
    # This fixes the original validation leakage and can change its score.
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

    # 4. Score — the splitter on mark_as_X drives this single hold-out.
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
