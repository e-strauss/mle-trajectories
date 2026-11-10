import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class TicketFrequency(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"Ticket_Freq": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    def __init__(self, column, groups, fallback=None):
        self.column = column
        self.groups = groups
        self.fallback = fallback

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.groups)[self.column].median()
        return self

    def transform(self, X):
        if len(self.groups) == 1:
            keys = pd.Index(X[self.groups[0]])
        else:
            keys = pd.MultiIndex.from_frame(X[self.groups])
        fill = pd.Series(
            self.medians_.reindex(keys).to_numpy(), index=X.index
        )
        result = X[self.column].fillna(fill)
        if self.fallback is not None:
            result = result.fillna(self.fallback)
        return result.to_frame(self.column)


class CrossFittedGroupSurvival(TransformerMixin, BaseEstimator):
    """Learn one lookup, cross-fitting its training-row encodings."""

    def __init__(self, key, output, filter_column=None):
        self.key = key
        self.output = output
        self.filter_column = filter_column

    def _learn(self, X, y):
        target = pd.Series(np.asarray(y), index=X.index, name="_target")
        table = X[[self.key]].assign(_target=target)
        if self.filter_column is not None:
            table = table[X[self.filter_column] == 1]
        return table.groupby(self.key)["_target"].mean()

    def fit(self, X, y):
        self.lookup_ = self._learn(X, y)
        return self

    def fit_transform(self, X, y, **fit_params):
        # This is the encoder's INNER cross-fitting, not the outer validation
        # loop. It is necessary to retain the original OOF encoding semantics.
        y_array = np.asarray(y)
        inner_cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=42
        )
        pieces = []
        for train_idx, val_idx in inner_cv.split(X, y_array):
            lookup = self._learn(X.iloc[train_idx], y_array[train_idx])
            pieces.append(
                X.iloc[val_idx][self.key].map(lookup).fillna(0.5)
            )
        result = pd.concat(pieces).reindex(X.index)
        self.fit(X, y)
        return result.to_frame(self.output)

    def transform(self, X):
        return (
            X[self.key].map(self.lookup_).fillna(0.5).to_frame(self.output)
        )


def make_models(seed):
    """The original four configurations, with their common fold seed."""
    return (
        lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            num_leaves=8, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, random_state=seed, verbose=-1,
        ),
        CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=4,
            l2_leaf_reg=3.0, random_seed=seed, verbose=0,
        ),
        XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
            reg_lambda=1.0, random_state=seed, eval_metric="logloss",
        ),
        LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs", max_iter=1000,
            random_state=seed,
        ),
    )


class FoldSeed(TransformerMixin, BaseEstimator):
    """Recover only the original CV fold's seed from its row indices."""

    def fit(self, X, y=None, *, full_y):
        self.fit_transform(X, y, full_y=full_y)
        return self

    def fit_transform(self, X, y=None, *, full_y):
        labels = np.asarray(full_y).ravel()
        rows = np.sort(np.asarray(X.index, dtype=int))
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_idx, _) in enumerate(
            splitter.split(np.zeros((len(labels), 1)), labels)
        ):
            if np.array_equal(rows, train_idx):
                self.seed_ = 42 + fold
                return self.seed_
        raise ValueError("Could not identify the original outer CV fold.")

    def transform(self, X):
        return self.seed_


TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Don": "Royalty",
    "Dr": "Officer",
    "Major": "Officer",
    "Rev": "Officer",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
}

FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Fare_Per_Person",
    "Embarked", "Title", "FamilySize", "IsAlone", "Deck", "Ticket_Freq",
    "WCG_Survival", "Ticket_Survival",
]
CAT_COLS = ["Sex", "Embarked", "Title", "Deck", "Pclass"]
NUM_COLS = [
    "Age", "SibSp", "Parch", "Fare", "Fare_Per_Person", "FamilySize",
    "IsAlone", "Ticket_Freq", "WCG_Survival", "Ticket_Survival",
]


class SurvivalDesign(TransformerMixin, BaseEstimator):
    """Fit a fresh feature design within each ensemble calibration partition.

    Nested calibration must refit these statistics, target lookups and encoders
    on its own training rows. Keeping the reusable design here avoids duplicating
    ten copies of its graph or exposing fitted-estimator routing to the engine.
    Fold-independent text extraction and arithmetic stay in the DataOps plan.
    """

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def _demographics(self, X):
        X = X.assign(Ticket_Freq=self.frequency_.transform(X)["Ticket_Freq"])
        X = X.assign(Fare=self.fare_.transform(X)["Fare"])
        X = X.assign(Fare_Per_Person=X["Fare"] / X["Ticket_Freq"])
        X = X.assign(Age=self.age_.transform(X)["Age"])
        return X.assign(
            Is_Woman_Or_Child=(
                (X["Sex"] == "female") | (X["Age"] < 16)
                | (X["Title"] == "Master")
            ).astype(int)
        )

    def fit_transform(self, X, y, **fit_params):
        self.frequency_ = TicketFrequency().fit(X)
        self.fare_ = GroupMedianImputer(column="Fare", groups=["Pclass"]).fit(X)
        self.age_ = GroupMedianImputer(
            column="Age", groups=["Title", "Pclass"], fallback=28.0
        ).fit(X)
        X = self._demographics(X)
        self.wcg_ = CrossFittedGroupSurvival(
            key="Surname", output="WCG_Survival", filter_column="Is_Woman_Or_Child"
        )
        self.ticket_ = CrossFittedGroupSurvival(
            key="Ticket", output="Ticket_Survival"
        )
        X = X.assign(
            WCG_Survival=self.wcg_.fit_transform(X, y)["WCG_Survival"],
            Ticket_Survival=self.ticket_.fit_transform(X, y)["Ticket_Survival"],
        )[FEATURE_COLS]
        self.preprocessor_ = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    NUM_COLS,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        )),
                    ]),
                    CAT_COLS,
                ),
            ]
        ).set_output(transform="pandas")
        return self.preprocessor_.fit_transform(X, y)

    def transform(self, X):
        X = self._demographics(X)
        X = X.assign(
            WCG_Survival=self.wcg_.transform(X)["WCG_Survival"],
            Ticket_Survival=self.ticket_.transform(X)["Ticket_Survival"],
        )[FEATURE_COLS]
        return self.preprocessor_.transform(X)


class CalibratedEnsemble(ClassifierMixin, BaseEstimator):
    """The original four-model average with a training-only OOF cutoff.

    Each inner fold fits its own complete design. Target encodings are themselves
    cross-fitted within that fold's training rows, never using calibration or
    outer-scoring labels. Final members then fit all outer-training rows.
    """

    @staticmethod
    def _fit_members(X, y, seed):
        members = make_models(seed)
        for member in members:
            member.fit(X, y)
        return members

    @staticmethod
    def _average_probabilities(X, members):
        p_lgb, p_cb, p_xgb, p_lr = [
            member.predict_proba(X)[:, 1] for member in members
        ]
        return (p_lgb + p_cb + p_xgb + p_lr) / 4.0

    def fit(self, X, y, *, seed):
        labels = np.asarray(y).ravel()
        self.classes_ = np.unique(labels)
        calibration_oof = np.zeros(len(X))
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, labels)):
            design = SurvivalDesign()
            inner_fit = design.fit_transform(X.iloc[train_idx], labels[train_idx])
            inner_val = design.transform(X.iloc[val_idx])
            members = self._fit_members(inner_fit, labels[train_idx], 42 + fold)
            calibration_oof[val_idx] = self._average_probabilities(inner_val, members)

        # Preserve the original grid, >= decoding, and first-best tie handling.
        self.threshold_ = 0.5
        best_accuracy = 0.0
        for cutoff in np.linspace(0.35, 0.65, 61):
            accuracy = accuracy_score(labels, (calibration_oof >= cutoff).astype(int))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.threshold_ = float(cutoff)

        self.design_ = SurvivalDesign()
        transformed = self.design_.fit_transform(X, y)
        self.members_ = self._fit_members(transformed, y, seed)
        return self

    def predict(self, X):
        transformed = self.design_.transform(X)
        probabilities = self._average_probabilities(transformed, self.members_)
        return (probabilities >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read; test loading and submission are omitted.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Preserve the original outer split and mark the raw target.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Record the fold-independent features once. Learned statistics and
    # encoders are refitted inside every nested calibration training partition.
    title = (
        X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        .map(TITLE_MAPPING).fillna("Rare")
    )
    X = X.assign(Title=title)
    X = X.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    X = X.assign(IsAlone=(X["FamilySize"] == 1).astype(int))
    X = X.assign(
        Deck=X["Cabin"].astype(str).str[0].where(X["Cabin"].notna(), "U")
    )
    X = X.assign(Embarked=X["Embarked"].fillna("S"))
    X = X.assign(Surname=(
        X["Name"].str.extract(r"^([^,]*),", expand=False)
        .str.strip().fillna("Unknown")
    ))

    # 4. Keep the original seed 42 + outer fold as fit-only metadata. The full
    # labels identify the splitter's fold, never enter model/feature fitting.
    seed = X.skb.apply(
        FoldSeed(), no_wrap=True,
        fit_transform_kwargs={"full_y": data["Survived"]},
    )
    # The real ensemble owns its nested calibration and fitted member reuse.
    # This deliberately trades fine-grained nodes for readable, portable fitting.
    pred = X.skb.apply(CalibratedEnsemble(), y=y, fit_kwargs={"seed": seed})

    # 5. Score -- no cv override. The original tuned its cutoff on scored OOF
    # labels; that optimistic accuracy cannot be reproduced without leakage.
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
