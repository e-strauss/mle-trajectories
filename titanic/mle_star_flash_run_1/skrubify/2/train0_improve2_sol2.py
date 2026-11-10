import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


CAT_FEATURES = ["Sex", "Embarked", "Pclass", "Deck", "TicketPrefix", "GroupID"]
FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Deck",
    "TicketPrefix",
    "GroupID",
    "GroupSize",
    "TicketGroupSize",
    "IsMultiPerson",
    "AdjustedFare",
    "Log_AdjustedFare",
    "Age_Pclass",
]


class FrequencyCounts(TransformerMixin, BaseEstimator):
    """Learn an independent frequency map on the training rows."""

    def fit(self, X, y=None):
        self.column_ = X.columns[0]
        self.counts_ = X[self.column_].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {self.column_: X[self.column_].map(self.counts_)},
            index=X.index,
        )


class CategoricalVocabulary(TransformerMixin, BaseEstimator):
    """Learn each column's category vocabulary on the training rows."""

    def fit(self, X, y=None):
        self.categories_ = {
            col: X[col].astype("category").cat.categories for col in X.columns
        }
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                col: pd.Categorical(X[col], categories=categories)
                for col, categories in self.categories_.items()
            },
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Reserve CatBoost's eval rows within the outer training fold."""

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
    # 1. Load Data -- record the read; importing does not access the CSV.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and raw feature frame.
    # StratifiedShuffleSplit reproduces the original stratified holdout.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    deck = X["Cabin"].fillna("Unknown").astype(str).str[0]
    deck = deck.replace({"n": "Unknown"})

    clean_ticket = (
        X["Ticket"]
        .astype(str)
        .str.replace(r"[./]", "", regex=True)
        .str.strip()
        .str.upper()
    )
    ticket_prefix = clean_ticket.str.extract(
        r"^([A-Z]+)", expand=False
    ).fillna("NONE")
    ticket_num = clean_ticket.str.extract(
        r"(\d+)$", expand=False
    ).fillna("0")
    surname = X["Name"].astype(str).str.extract(
        r"^([^,]+)", expand=False
    ).fillna("Unknown")
    group_id = surname + "_" + ticket_num

    # Fit each frequency map and the median independently on training rows:
    # the original computed these using the validation rows as well.
    group_counts = group_id.to_frame(name="GroupID").skb.apply(
        FrequencyCounts(),
        no_wrap=True,
    )
    ticket_counts = X[["Ticket"]].skb.apply(
        FrequencyCounts(),
        no_wrap=True,
    )
    fare_frame = X[["Fare"]].skb.apply(
        SimpleImputer(strategy="median").set_output(transform="pandas"),
        no_wrap=True,
    )

    group_size = group_counts["GroupID"]
    ticket_size = ticket_counts["Ticket"]
    fare_imputed = fare_frame["Fare"]
    adjusted_fare = fare_imputed / group_size

    features = X.assign(
        Fare=fare_imputed,
        Embarked=X["Embarked"].fillna("Missing"),
        Deck=deck,
        TicketPrefix=ticket_prefix,
        GroupID=group_id,
        GroupSize=group_size,
        TicketGroupSize=ticket_size,
        IsMultiPerson=(group_size > 1).astype(int),
        AdjustedFare=adjusted_fare,
        Log_AdjustedFare=adjusted_fare.skb.apply_func(
            np.maximum, 0
        ).skb.apply_func(np.log1p),
        Age_Pclass=X["Age"] * X["Pclass"],
    )

    # Preserve the original design-matrix column order.
    features = features[FEATURE_COLS]

    # The original cast to category on the whole table, so the training split's
    # dtype already held every validation-only value. CatBoost reads the values,
    # so a per-fold cast is enough for it. XGBoost reads the dtype's vocabulary
    # and rejects values missing from it; learn that vocabulary on the training
    # rows instead (unseen values become missing), keeping the column order.
    cb_features = features.astype({col: "category" for col in CAT_FEATURES})
    xgb_categories = features[CAT_FEATURES].skb.apply(CategoricalVocabulary())
    xgb_features = xgb_categories.skb.concat(
        [features.drop(columns=CAT_FEATURES)], axis=1
    )[FEATURE_COLS]

    # 4. CatBoost -- the original passed the scored holdout as eval_set, which
    #    selects the best iteration on those rows. Keep eval-based selection but
    #    reserve a stratified 20% eval set inside the outer training fold.
    cb_parts = cb_features.skb.apply(GetXY(), y=y, no_wrap=True)
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

    # 5. XGBoost -- fit on the whole outer training fold, as in the original.
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

    # 6. Preserve the original equal probability average and >= 0.5 threshold.
    pred = cb_probability.skb.apply_func(
        blend_predictions, xgb_probability, skrub.eval_mode()
    )

    # 7. Score -- use the splitter attached to mark_as_X.
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
