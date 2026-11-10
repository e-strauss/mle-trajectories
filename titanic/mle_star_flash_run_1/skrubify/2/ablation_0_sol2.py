import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class GetXY(TransformerMixin, BaseEstimator):
    """Provide CatBoost's evaluation rows from the current training fold."""

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


# Both models' predict() returns (n_samples, 2) in classes_ order [0, 1];
# column 1 is the positive-class probability.
def threshold_predictions(probabilities, mode):
    if mode == "fit":
        return None
    return (np.asarray(probabilities)[:, 1] >= 0.5).astype(int)


def ensemble_predictions(cb_probabilities, xgb_probabilities, mode):
    if mode == "fit":
        return None
    cb_probabilities = np.asarray(cb_probabilities)[:, 1]
    xgb_probabilities = np.asarray(xgb_probabilities)[:, 1]
    probabilities = (cb_probabilities + xgb_probabilities) / 2.0
    return (probabilities >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the original CSV read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and original feature selection.
    # StratifiedShuffleSplit reproduces the original stratified hold-out rows.
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    cat_features = ["Sex", "Embarked", "Pclass"]
    num_features = ["Age", "SibSp", "Parch", "Fare"]

    y = data["Survived"].skb.mark_as_y()
    X = data[features].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and models.
    # The original cast categories before the split, so one vocabulary covered
    # every row. Each fold learns its own.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    X = X.assign(
        Sex=X["Sex"].astype("category"),
        Embarked=X["Embarked"].astype("category"),
        Pclass=X["Pclass"].astype("category"),
    )
    X_num = X[num_features]

    # The original used scored rows as CatBoost's eval_set. Even without
    # explicit early-stopping patience, that enables best-iteration selection.
    # Keep eval_set and all model parameters, but reserve its stratified 20%
    # from training rows instead. The outer scored rows remain untouched.
    # XGBoost still trains on the entire outer training fold, as originally.
    cb_parts = X.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_num_parts = X_num.skb.apply(GetXY(), y=y, no_wrap=True)

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
            "cat_features": cat_features,
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    xgb_probs = X.skb.apply(
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

    cb_num_probs = cb_num_parts["X"].skb.apply(
        CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        ),
        y=cb_num_parts.get("y", y),
        fit_kwargs={
            "eval_set": (cb_num_parts["X_val"], cb_num_parts["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    xgb_num_probs = X_num.skb.apply(
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        ),
        y=y,
    )

    # Preserve probability-column selection, arithmetic averaging, and >= 0.5
    # decoding exactly. Gate arithmetic because fit-mode nodes are estimators.
    variants = {
        "Full Ensemble (CatBoost + XGBoost + All Features)": (
            cb_probs.skb.apply_func(
                ensemble_predictions, xgb_probs, skrub.eval_mode()
            )
        ),
        "Ablation 1 (CatBoost Only, drop XGBoost)": (
            cb_probs.skb.apply_func(
                threshold_predictions, skrub.eval_mode()
            )
        ),
        "Ablation 2 (XGBoost Only, drop CatBoost)": (
            xgb_probs.skb.apply_func(threshold_predictions, skrub.eval_mode())
        ),
        "Ablation 3 (Drop Categorical Features)": (
            cb_num_probs.skb.apply_func(
                ensemble_predictions, xgb_num_probs, skrub.eval_mode()
            )
        ),
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all four original variants in one grid search.
    # Drop the separate reporting/conclusion logic; results retain every score.
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
