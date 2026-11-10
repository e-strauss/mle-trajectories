import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier


class CatBoostClassifierCloneable(CatBoostClassifier):
    def __sklearn_clone__(self):
        return CatBoostClassifierCloneable(**self.get_params(deep=False))


class GetXY(TransformerMixin, BaseEstimator):
    """Create CatBoost's evaluation set from the fold's training rows only."""

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
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class ProbabilityAverage(ClassifierMixin, BaseEstimator):
    """Keep the original probability arithmetic and >= 0.5 decoding."""

    def __init__(self, catboost, xgboost):
        self.catboost = catboost
        self.xgboost = xgboost

    def fit(self, X, y, *, cb_X, cb_y, eval_set):
        self.classes_ = np.unique(y)
        self.catboost_ = clone(self.catboost)
        self.xgboost_ = clone(self.xgboost)
        self.catboost_.fit(
            cb_X, cb_y, eval_set=eval_set, verbose=False
        )
        # XGBoost still receives every outer-training row, as in the original.
        self.xgboost_.fit(X, y)
        return self

    def predict(self, X):
        cb_probs = self.catboost_.predict_proba(X)[:, 1]
        xgb_probs = self.xgboost_.predict_proba(X)[:, 1]
        return (((cb_probs + xgb_probs) / 2.0) >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the training read. Test prediction, its existence
    #    guard, and submission writing are omitted because they do not score.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original seven features.
    y = data["Survived"].skb.mark_as_y()
    X = data[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].skb.mark_as_X(
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and the original two-model ensemble.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # The original handed CatBoost the scored fold as eval_set. Even without
    # explicit patience, CatBoost can select its best iteration on those labels.
    # Keep eval-set-based selection, but move that evaluation to a recorded
    # 20% holdout of the outer-training rows (matching the outer fold fraction).
    # This fixes leakage, so the original optimistic score is not reproducible.
    # XGBoost needs no evaluation holdout and retains all outer-training rows.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)

    cb_model = CatBoostClassifierCloneable(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=["Sex", "Embarked", "Pclass"],
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss",
    )

    pred = features.skb.apply(
        ProbabilityAverage(catboost=cb_model, xgboost=xgb_model),
        y=y,
        fit_kwargs={
            "cb_X": cb_parts["X"],
            "cb_y": cb_parts["y"],
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
        },
    )

    # 4. Score — the marked splitter drives all five folds.
    #    Grid search reports mean fold accuracy rather than pooled OOF accuracy;
    #    unequal fold sizes can therefore produce a small aggregation difference.
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
