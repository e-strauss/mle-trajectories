import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class XGBClassifier(XGBClassifier):
    """skrub calls predict() on an intermediate model.

    Return the (n_samples, n_classes) probability matrix. The blend takes
    column 1, the positive class, as it does for CatBoost.
    """

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)


class GetXY(TransformerMixin, BaseEstimator):
    """Create CatBoost's eval set from the current fold's training rows."""

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
    # 1. Load Data -- record the read without accessing the file on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and the original seven features.
    # StratifiedShuffleSplit reproduces the original stratified hold-out rows.
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    y = data["Survived"].skb.mark_as_y()
    X = data[features].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and model.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    X = X.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # The original supplied the scored hold-out as CatBoost's eval_set.
    # Even without explicit early stopping, that can select the best iteration.
    # Preserve eval_set, but isolate it inside the outer training rows instead;
    # the original's validation-dependent model selection cannot be reproduced
    # without leakage. XGBoost still fits on all outer training rows.
    cb_parts = X.skb.apply(
        GetXY(test_size=0.2, random_state=42), y=y, no_wrap=True
    )
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
            "cat_features": ["Sex", "Embarked", "Pclass"],
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
            "verbose": False,
        },
        predict_kwargs={"prediction_type": "Probability"},
    )
    # XGBoost still fits on every outer training row, with no eval set.
    xgb_probability = X.skb.apply(
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
    pred = cb_probability.skb.apply_func(
        blend_predictions, xgb_probability, skrub.eval_mode()
    )

    # 4. Score -- the splitter on mark_as_X drives the single hold-out.
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
