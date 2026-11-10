import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class GetXY(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set from the current fold's training rows."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return {"X": X_fit, "X_val": X_val, "y": y_fit, "y_val": y_val}

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class OptimizedBoostingBlend(ClassifierMixin, BaseEstimator):
    """Fit the three original boosters and calibrate their weighted vote.

    The calibration rows come from GetXY, within the outer training fold.
    Keeping the genuine ensemble together lets both engines reuse its fitted
    members without requiring fitted-estimator references in the DataOps graph.
    """

    def fit(self, X, y, *, eval_set):
        X_eval, y_eval = eval_set
        self.classes_ = np.unique(y)
        self.cb_ = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=5,
            eval_metric="Accuracy",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=0,
        )
        self.cb_.fit(
            X, y, cat_features=CAT_FEATURES, eval_set=eval_set,
            early_stopping_rounds=50, verbose=False,
        )
        self.xgb_ = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        )
        self.xgb_.fit(X, y, eval_set=[eval_set], verbose=False)
        self.lgb_ = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        # The original callbacks=[] did not enable LightGBM early stopping.
        self.lgb_.fit(
            X, y, eval_set=[eval_set],
            categorical_feature=CAT_FEATURES, callbacks=[],
        )

        probabilities = self._member_probabilities(X_eval)
        labels = np.asarray(y_eval).ravel()
        best_score = 0.0
        best_weights = (1 / 3, 1 / 3, 1 / 3)
        best_threshold = 0.5
        # Preserve the exact grid order and strict first-improvement tie break.
        for w_cb in np.linspace(0, 1, 11):
            for w_xgb in np.linspace(0, 1 - w_cb, 11):
                w_lgb = 1.0 - w_cb - w_xgb
                if w_lgb < -1e-5:
                    continue
                w_lgb = max(0.0, w_lgb)
                blend_probs = (
                    w_cb * probabilities[:, 0]
                    + w_xgb * probabilities[:, 1]
                    + w_lgb * probabilities[:, 2]
                )
                for threshold in np.linspace(0.35, 0.65, 31):
                    predictions = (blend_probs >= threshold).astype(int)
                    score = accuracy_score(labels, predictions)
                    if score > best_score:
                        best_score = score
                        best_weights = (w_cb, w_xgb, w_lgb)
                        best_threshold = threshold
        self.weights_ = best_weights
        self.threshold_ = best_threshold
        return self

    def _member_probabilities(self, X):
        return np.column_stack([
            self.cb_.predict(X, prediction_type="Probability")[:, 1],
            self.xgb_.predict_proba(X)[:, 1],
            self.lgb_.predict_proba(X)[:, 1],
        ])

    def predict(self, X):
        probabilities = self._member_probabilities(X)
        blend_probs = (
            self.weights_[0] * probabilities[:, 0]
            + self.weights_[1] * probabilities[:, 1]
            + self.weights_[2] * probabilities[:, 2]
        )
        return (blend_probs >= self.threshold_).astype(int)


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CAT_FEATURES = ["Sex", "Embarked", "Pclass"]

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- the read is recorded, so importing does not access data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and original feature matrix.
    # This splitter reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and the original three models.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # The original early-stopped on the scored hold-out. Preserve patience and
    # iteration limits, but use a stratified 20% of this fold's training rows
    # for early stopping instead. The original leaky score cannot be reproduced
    # exactly under this correction.
    parts = features.skb.apply(GetXY(), y=y, no_wrap=True)
    X_fit = parts["X"]
    y_fit = parts.get("y", y)
    X_eval = parts["X_val"]
    y_eval = parts["y_val"]

    # All members fit the same inner training rows; the calibration slice is
    # used for early stopping and the original blend grid, never outer scoring.
    # The ensemble owns fitted-member reuse, while preprocessing and the split
    # remain visible in the plan.
    pred = X_fit.skb.apply(
        OptimizedBoostingBlend(), y=y_fit,
        fit_kwargs={"eval_set": (X_eval, y_eval)},
    )

    # 4. Score once. The original fit the weight grid on the rows it scores.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="accuracy",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
