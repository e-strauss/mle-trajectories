import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class ThresholdStackingClassifier(StackingClassifier):
    """Preserve the source's positive-class probability >= 0.5 decoding."""

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MedianFill(TransformerMixin, BaseEstimator):
    """Preserve the original behavior even when an entire column is missing."""

    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.medians_)


class GroupAgeFill(TransformerMixin, BaseEstimator):
    """Learn only the (Title, Pclass) age medians."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby(["Title", "Pclass"])["Age"].median()
        return self

    def transform(self, X):
        keys = pd.MultiIndex.from_frame(X[["Title", "Pclass"]])
        group_medians = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        return pd.DataFrame(
            {"Age_grouped": X["Age"].fillna(group_medians)}, index=X.index
        )


class GlobalAgeFill(TransformerMixin, BaseEstimator):
    """Use the median of RAW ages, not the median of group-imputed ages."""

    def fit(self, X, y=None):
        self.median_ = X["Age"].median()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"Age_filled": X["Age_grouped"].fillna(self.median_)},
            index=X.index,
        )


TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Capt": "Officer",
    "Lady": "Royalty",
    "Sir": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
}

NUM_COLS = [
    "Pclass_num",
    "Age_filled",
    "Fare_filled",
    "Fare_log",
    "FarePerPerson",
    "FamilySize",
    "IsAlone",
    "IsSmallFamily",
    "IsLargeFamily",
    "IsChild",
    "Age_Pclass",
    "HasCabin",
    "IsFemale",
]
CAT_COLS = ["Title", "Deck", "Embarked_clean", "Sex_clean"]


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read. Test prediction, its existence guard,
    #    and submission writing are omitted. The original train/test category
    #    union leaked schema information; categories are now learned per fold.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and preserve the original five-fold
    #    stratification. The splitter lives only on mark_as_X.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering. Schema-dependent fallbacks remain
    #    recorded too; no input file is inspected while importing this module.
    has_name = X.columns.isin(["Name"]).any()
    has_age = X.columns.isin(["Age"]).any()
    has_pclass = X.columns.isin(["Pclass"]).any()
    has_cabin = X.columns.isin(["Cabin"]).any()

    # Reindex supplies nullable columns for safe evaluation of both branches.
    raw = X.reindex(
        columns=[
            "Name", "SibSp", "Parch", "Fare", "Age",
            "Pclass", "Cabin", "Embarked", "Sex",
        ]
    )
    extracted_title = raw["Name"].astype("string").str.extract(
        r" ([A-Za-z]+)\.", expand=False
    )
    title = has_name.skb.if_else(
        extracted_title.map(TITLE_MAPPING).fillna("Other"), "Unknown"
    )

    features = X.assign(
        Title=title,
        FamilySize=X.get("SibSp", 0) + X.get("Parch", 0) + 1,
    )
    features = features.assign(
        IsAlone=(features["FamilySize"] == 1).astype(float),
        IsSmallFamily=(
            (features["FamilySize"] >= 2) & (features["FamilySize"] <= 4)
        ).astype(float),
        IsLargeFamily=(features["FamilySize"] > 4).astype(float),
    )

    # Each learned statistic is a separate estimator node. The original computed
    # medians before validation; these nodes fit them on the fold's training rows.
    fare_input = raw[["Fare"]].assign(Fare=X.get("Fare", 0.0))
    fare = fare_input.skb.apply(MedianFill())["Fare"]
    features = features.assign(Fare_filled=fare)
    features = features.assign(
        Fare_log=fare.clip(lower=0).skb.apply_func(np.log1p)
    )
    features = features.assign(
        FarePerPerson=features["Fare_log"] / features["FamilySize"]
    )

    age_input = raw[["Age", "Pclass"]].assign(
        Age=X.get("Age", 28.0),
        Title=features["Title"],
    )
    grouped_age = age_input.skb.apply(GroupAgeFill())["Age_grouped"]
    age_grouped = has_pclass.skb.if_else(grouped_age, age_input["Age"])
    age_global_input = age_input[["Age"]].assign(Age_grouped=age_grouped)
    age_filled = age_global_input.skb.apply(GlobalAgeFill())["Age_filled"]
    features = features.assign(Age_filled=age_filled.fillna(28.0))
    pclass = raw["Pclass"].astype(float)
    pclass_num = has_pclass.skb.if_else(pclass, 3.0)
    features = features.assign(
        IsChild=(features["Age_filled"] <= 12).astype(float),
        Age_Pclass=has_age.skb.if_else(
            features["Age_filled"] * pclass_num, 84.0
        ),
    )

    deck = (
        raw["Cabin"].astype(str).str[0].fillna("M").replace("n", "M")
    )
    cabin_present = (
        raw["Cabin"].notna() & (raw["Cabin"] != "")
    ).astype(float)
    features = features.assign(
        Deck=has_cabin.skb.if_else(deck, "M"),
        HasCabin=has_cabin.skb.if_else(cabin_present, 0.0),
        Embarked_clean=raw["Embarked"].fillna("S").astype(str),
        Sex_clean=raw["Sex"].fillna("male").astype(str),
        Pclass_num=pclass_num,
    )
    features = features.assign(
        IsFemale=(features["Sex_clean"] == "female").astype(float)
    )

    # Match get_dummies' sorted categories and drop-first coding. Unknown
    # categories have all-zero indicators rather than changing the fitted schema.
    categoricals = features[CAT_COLS].skb.apply(
        OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        )
    )
    design = features[NUM_COLS].skb.concat([categoricals], axis=1)

    # The original prints one ensemble score, so these models are members of one stack.
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            ),
        ),
        (
            "lr",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1000, random_state=42),
            ),
        ),
    ]

    if LGBMClassifier is not None:
        estimators.append((
            "lgb",
            LGBMClassifier(
                n_estimators=120,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        ))

    if XGBClassifier is not None:
        estimators.append((
            "xgb",
            XGBClassifier(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            ),
        ))

    if CatBoostClassifier is not None:
        estimators.append((
            "cat",
            CatBoostClassifier(
                iterations=200,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=3.0,
                subsample=0.8,
                random_seed=42,
                verbose=0,
            ),
        ))

    # The original fitted and scored the meta-learner on the same OOF matrix.
    # The meta now trains on an inner five-fold of each outer training fold.
    # Each base model still contributes only its positive probability.
    # Scaling is fitted within training data instead of on the full table.
    model = ThresholdStackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.5, random_state=42),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
    )
    # The small subclass keeps the exact threshold, including ties, within
    # the ensemble's predict method so either engine can call it directly.
    pred = design.skb.apply(model, y=y)

    # 4. Score -- the marked splitter drives the outer evaluation. This reports
    #    mean fold accuracy rather than the original pooled, leaky meta accuracy.
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
