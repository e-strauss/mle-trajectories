import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CAT_FEATURES = ["Sex", "Embarked", "Pclass"]

CB_CONFIGS = [
    {
        "iterations": 550,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
    },
    {
        "iterations": 600,
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "random_seed": 123,
    },
    {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 2.0,
        "random_seed": 789,
    },
]

XGB_CONFIGS = [
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.025,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "random_state": 123,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "random_state": 789,
    },
]


class GetXY(TransformerMixin, BaseEstimator):
    """Expose a fold-local CatBoost fit/evaluation split in the plan."""

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


def blend_predictions(cb_a, cb_b, cb_c, xgb_a, xgb_b, xgb_c, mode):
    # Predictor nodes hold fitted estimators, not probabilities, in fit mode.
    # CatBoost predict returns (n_samples, 2); XGBModel predict is the positive
    # probability. The original averages the three CatBoost probabilities,
    # averages the three XGBoost probabilities, then applies 0.65 / 0.35 and >= 0.5.
    if mode == "fit":
        return None
    cb_mean = np.mean(
        [positive_probability(p) for p in (cb_a, cb_b, cb_c)],
        axis=0,
    )
    xgb_mean = np.mean(
        [positive_probability(p) for p in (xgb_a, xgb_b, xgb_c)],
        axis=0,
    )
    ensemble_probs = 0.65 * cb_mean + 0.35 * xgb_mean
    return (ensemble_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without reading anything during import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original seven features.
    # StratifiedShuffleSplit reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, fold-local evaluation split, and ensemble.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # CatBoost's original eval_set was the scored hold-out. Even without explicit
    # patience, it can select the best iteration using that set. Avoid that leak
    # by exposing a separate stratified 20% evaluation split inside the outer
    # training rows. Consequently, that optimistic original score is not exactly
    # reproducible. All six configurations and the voting rule remain unchanged.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    cb_probabilities = [
        cb_parts["X"].skb.apply(
            CatBoostClassifier(
                **config,
                eval_metric="Accuracy",
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
        for config in CB_CONFIGS
    ]
    # XGBoost still receives every outer-training row, as in the original.
    xgb_probabilities = [
        features.skb.apply(
            XGBModel(
                **config,
                enable_categorical=True,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
            ),
            y=y,
        )
        for config in XGB_CONFIGS
    ]
    pred = cb_probabilities[0].skb.apply_func(
        blend_predictions,
        *cb_probabilities[1:],
        *xgb_probabilities,
        skrub.eval_mode(),
    )

    # 4. Score — the splitter on mark_as_X drives the single validation fold.
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
