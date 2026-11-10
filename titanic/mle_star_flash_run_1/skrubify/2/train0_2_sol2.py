import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBModel


NUM_COLS = ["Age", "Fare", "SibSp", "Parch"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]
FEATURES = NUM_COLS + CAT_COLS


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


class HistGradientBoostingProbability(HistGradientBoostingClassifier):
    """The positive-class probability."""

    def predict(self, X):
        return super().predict_proba(X)[:, 1]


def positive_probability(values):
    """Column 1 of a probability matrix, or a vector that is already that column."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        return values[:, 1]
    return values


def ensemble_predictions(cb_probs, xgb_probs, hgb_probs, mode):
    if mode == "fit":
        return None
    blended = (
        positive_probability(cb_probs)
        + positive_probability(xgb_probs)
        + positive_probability(hgb_probs)
    ) / 3.0
    return (blended >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this module records, but does not read, the CSV.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and reproduce the stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — retain the original feature order and dtypes.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # CatBoost's original eval_set was the scored hold-out. Even without explicit
    # early-stopping patience, that enables best-iteration selection by default.
    # Keep the eval_set, but carve it from the outer fold's training rows instead;
    # the original leaky selection score cannot be reproduced honestly.
    cb_parts = features.skb.apply(
        GetXY(test_size=0.2, random_state=42),
        y=y,
        no_wrap=True,
    )

    hgb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CAT_COLS,
            ),
        ]
    )
    hgb_model = Pipeline(
        [
            ("preprocessor", hgb_preprocessor),
            (
                "model",
                HistGradientBoostingProbability(
                    max_iter=150,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )

    # One scored ensemble: the original equal-weight positive-class average
    # and >= 0.5 threshold. CatBoost trains on the inner slice; XGBoost and
    # the gradient-boosting pipeline train on all outer training rows.
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
            "cat_features": CAT_COLS,
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
    hgb_probs = features.skb.apply(hgb_model, y=y)
    pred = cb_probs.skb.apply_func(
        ensemble_predictions, xgb_probs, hgb_probs, skrub.eval_mode()
    )

    # 4. Score — the splitter on mark_as_X owns the outer validation.
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
