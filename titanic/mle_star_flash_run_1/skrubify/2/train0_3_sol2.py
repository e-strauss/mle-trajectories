import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBModel


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the CatBoost evaluation split inside each outer training fold."""

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


def ensemble_predictions(cb_probs, xgb_probs, lgb_probs, mode):
    if mode == "fit":
        return None
    blended = (
        positive_probability(cb_probs)
        + positive_probability(xgb_probs)
        + positive_probability(lgb_probs)
    ) / 3.0
    return (blended >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read without reading anything on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and retain the original feature order.
    #    StratifiedShuffleSplit reproduces the original stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, followed by the original three-model ensemble.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # CatBoost's eval_set implicitly enables best-iteration model selection even
    # without explicit early-stopping patience. The original selected that
    # iteration on its scored hold-out, leaking validation information.
    # Keep evaluation/model selection, but expose an inner stratified 20% split
    # of this fold's training rows instead. XGBoost and LightGBM still fit on all
    # outer-training rows. This necessarily differs from the original leaky score.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)

    # One scored ensemble: the original equal-weight positive-class average
    # and >= 0.5 threshold. CatBoost trains on the inner slice; XGBoost and
    # LightGBM train on all outer training rows.
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
            "cat_features": ["Sex", "Embarked", "Pclass"],
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
    lgb_probs = features.skb.apply(
        LGBMModel(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            verbose=-1,
        ),
        y=y,
    )
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, xgb_probs, lgb_probs, skrub.eval_mode()
    )

    # 4. Score -- only the splitter on mark_as_X controls outer validation.
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
