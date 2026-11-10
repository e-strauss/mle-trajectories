import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


CAT_FEATURES = [
    "Pclass", "Sex", "Embarked", "GroupSizeBin",
    "DeckTier", "TicketPrefix", "Title",
    "Title_Pclass", "Sex_GroupSizeBin",
]

FEATURE_COLS = [
    "Age", "SibSp", "Parch", "Fare", "Fare_Per_Person", "GroupSize",
    "GroupSizeBin", "CabinRoomCount", "DeckTier", "TicketPrefix", "Title",
    "Pclass", "Sex", "Embarked", "Title_Pclass", "Sex_GroupSizeBin",
]

DECK_TIER_MAP = {
    "A": "Upper_ABC", "B": "Upper_ABC", "C": "Upper_ABC",
    "D": "Mid_DE", "E": "Mid_DE",
    "F": "Lower_FG", "G": "Lower_FG", "T": "Lower_FG",
    "Unknown": "Unknown",
}

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
    "Major": "Officer", "Capt": "Officer",
    "Don": "Royalty", "Dona": "Royalty", "Sir": "Royalty",
    "Lady": "Royalty", "Countess": "Royalty", "Jonkheer": "Royalty",
}


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies from this fold's training rows."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X["Ticket"].map(self.counts_).to_frame("TicketCount")


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the training-only CatBoost evaluation split in the graph."""

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


class TrainingCategoryDtypes(TransformerMixin, BaseEstimator):
    """Apply this fold's categorical dtypes so codes keep their meaning.

    The original cast categories before splitting, so both partitions shared
    a vocabulary. Recorded casts run separately on each partition. Unseen
    values become missing, which native XGBoost supports.
    """

    def fit(self, X, y=None):
        self.category_dtypes_ = (
            X.select_dtypes(include="category").dtypes.to_dict()
        )
        return self

    def transform(self, X):
        return X.astype(self.category_dtypes_)


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
    # 1. Load Data — importing builds the graph without reading the file.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and raw features.
    # Reproduce the original single stratified 80/20 hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    # The original computed ticket frequencies before splitting; learn them
    # only on training rows instead to remove that leakage.
    ticket_counts = X[["Ticket"]].skb.apply(
        TicketFrequency(), no_wrap=True
    )["TicketCount"]
    family_size = X["SibSp"] + X["Parch"] + 1
    group_size = family_size.skb.apply_func(np.maximum, ticket_counts)

    # Use an object-backed conversion before filling missing string values:
    # pandas string/category conversions differ across pandas versions.
    group_size_bin = (
        group_size.skb.apply_func(
            pd.cut,
            bins=[0, 1, 4, np.inf],
            labels=["Alone", "SmallGroup", "LargeGroup"],
        )
        .astype(object)
        .fillna("nan")
        .astype(str)
    )

    cabin = X["Cabin"].fillna("")
    cabin_room_count = cabin.str.count(r"\S+")
    raw_deck = cabin.str.slice(0, 1).replace("", "Unknown")
    deck_tier = raw_deck.map(DECK_TIER_MAP).fillna("Unknown")

    ticket_clean = (
        X["Ticket"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.strip()
    )
    ticket_prefix = (
        ticket_clean.str.extract(r"^(\S+)", expand=False)
        .str.upper()
        .where(
            ticket_clean.str.contains(r"^\S+\s+\S+", regex=True, na=False),
            "NUMERIC",
        )
        .where(X["Ticket"].notna(), "UNKNOWN")
    )

    raw_title = (
        X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
        .str.strip()
        .fillna("Unknown")
    )
    title = raw_title.map(TITLE_MAP).fillna("Other")

    features = X[["Age", "SibSp", "Parch", "Fare"]]
    features = features.assign(Fare_Per_Person=X["Fare"] / group_size)
    features = features.assign(GroupSize=group_size)
    features = features.assign(GroupSizeBin=group_size_bin)
    features = features.assign(CabinRoomCount=cabin_room_count)
    features = features.assign(DeckTier=deck_tier)
    features = features.assign(TicketPrefix=ticket_prefix)
    features = features.assign(Title=title)
    features = features.assign(
        Pclass=X["Pclass"].astype(str).fillna("nan")
    )
    features = features.assign(Sex=X["Sex"].astype(str).fillna("nan"))
    features = features.assign(
        Embarked=X["Embarked"].fillna("Missing").astype(str)
    )
    features = features.assign(
        Title_Pclass=features["Title"] + "_" + features["Pclass"]
    )
    features = features.assign(
        Sex_GroupSizeBin=features["Sex"] + "_" + features["GroupSizeBin"]
    )

    # Object-backed string categories are accepted by both boosting libraries.
    categoricals = features[CAT_FEATURES].astype(object).astype("category")
    features = (
        features.drop(columns=CAT_FEATURES)
        .skb.concat([categoricals], axis=1)[FEATURE_COLS]
    )

    # The original supplied the scored hold-out to CatBoost as eval_set.
    # Even without explicit patience, this can select the best iteration.
    # Preserve evaluation-based selection using a visible training-only split
    # with the original fraction, seed, and stratification instead.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)

    # XGBoost trains on all outer-training rows, after the training fold's
    # categorical dtypes are applied. The blend is the original equal-weight
    # positive-class average and >= 0.5 threshold.
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
    xgb_features = features.skb.apply(TrainingCategoryDtypes())
    xgb_probs = xgb_features.skb.apply(
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

    # 4. Score — use the marked splitter. Raise execution errors rather than
    # silently printing an unusable NaN score if a library rejects an input.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
            error_score="raise",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
