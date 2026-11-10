import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


class GetXY(TransformerMixin, BaseEstimator):
    """Produce an internal CatBoost evaluation split from training rows only."""

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
    # 1. Load Data — importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original baseline columns.
    # StratifiedShuffleSplit reproduces the original stratified hold-out rows.
    y = data["Survived"].skb.mark_as_y()
    X = data[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, exact feature subsets, and ensemble models.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    configurations = [
        (
            "Full Pipeline (Baseline)",
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
            ["Sex", "Embarked", "Pclass"],
        ),
        (
            "Ablation 1 (Drop Family Features SibSp & Parch)",
            ["Pclass", "Sex", "Age", "Fare", "Embarked"],
            ["Sex", "Embarked", "Pclass"],
        ),
        (
            "Ablation 2 (Drop Embarkation Port)",
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
            ["Sex", "Pclass"],
        ),
        (
            "Ablation 3 (Drop Continuous Signals Age & Fare)",
            ["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
            ["Sex", "Embarked", "Pclass"],
        ),
    ]

    variants = {}
    # This enumerates experiments, not columns or outer validation folds.
    for name, feature_cols, cat_cols in configurations:
        variant_features = features[feature_cols]

        # The original supplied the scored hold-out to CatBoost. Even without
        # explicit early stopping, eval_set enables its default best-iteration
        # selection. Preserve that behavior using an internal 20% evaluation
        # split, rather than leaking the scored rows. Consequently its original
        # leaky score cannot be reproduced exactly.
        cb_parts = variant_features.skb.apply(
            GetXY(test_size=0.2, random_state=42), y=y, no_wrap=True
        )

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
                "cat_features": cat_cols,
                "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
                "verbose": False,
            },
            predict_kwargs={"prediction_type": "Probability"},
        )
        # XGBoost used no evaluation set in the original. It trains on all
        # outer-fold training rows.
        xgb_probs = variant_features.skb.apply(
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
        variants[name] = cb_probs.skb.apply_func(
            ensemble_predictions, xgb_probs, skrub.eval_mode()
        )

    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score all four original experiments in one grid search.
    # Summary/conclusion printing is replaced by the per-variant score table.
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
