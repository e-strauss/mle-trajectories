import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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


class LGBMClassifier(LGBMClassifier):
    """Same intermediate predict() contract as XGBClassifier."""

    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CAT_FEATURES = ["Sex", "Embarked", "Pclass"]


class AlignCategories(TransformerMixin, BaseEstimator):
    """Retain the training categories when transforming validation rows."""

    def fit(self, X, y=None):
        self.categories_ = {
            col: X[col].cat.categories for col in CAT_FEATURES
        }
        return self

    def transform(self, X):
        return X.assign(
            Sex=pd.Categorical(X["Sex"], categories=self.categories_["Sex"]),
            Embarked=pd.Categorical(
                X["Embarked"], categories=self.categories_["Embarked"]
            ),
            Pclass=pd.Categorical(
                X["Pclass"], categories=self.categories_["Pclass"]
            ),
        )


class FrequencyFeature(TransformerMixin, BaseEstimator):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        self.frequencies_ = X[self.column].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        values = X[self.column].map(self.frequencies_).astype(float).fillna(0)
        return pd.DataFrame(
            {f"{self.column}_freq": values}, index=X.index
        )


class GroupStatistics(TransformerMixin, BaseEstimator):
    """One independently fitted categorical/numeric aggregation."""

    def __init__(self, category, numeric):
        self.category = category
        self.numeric = numeric

    def fit(self, X, y=None):
        # Preserve the original's data-dependent numeric-dtype condition.
        self.enabled_ = np.issubdtype(X[self.numeric].dtype, np.number)
        if self.enabled_:
            self.statistics_ = X.groupby(self.category)[self.numeric].agg(
                ["mean", "std"]
            )
        return self

    def transform(self, X):
        if not self.enabled_:
            # The original skips a non-numeric column, so this node adds none.
            return pd.DataFrame(index=X.index)
        # Group indexes are unique, so these lookups cannot expand the rows.
        # Only the two new columns leave this node; the base frame is joined later.
        return pd.DataFrame(
            {
                f"{self.numeric}_mean_by_{self.category}": X[self.category]
                .map(self.statistics_["mean"])
                .astype(float),
                f"{self.numeric}_std_by_{self.category}": X[self.category]
                .map(self.statistics_["std"])
                .astype(float),
            },
            index=X.index,
        )


class DiscoveredNumericInteractions(TransformerMixin, BaseEstimator):
    """Keep the source's dtype-dependent adjacent pairs and feature order."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_columns = [
            column for column in FEATURES
            if column not in CAT_FEATURES
            and np.issubdtype(X[column].dtype, np.number)
        ]
        result = pd.DataFrame(index=X.index)
        # The columns are discovered from the original feature block, before
        # frequency and aggregation columns are added, exactly as in the source.
        for i in range(min(3, len(numeric_columns) - 1)):
            left, right = numeric_columns[i : i + 2]
            result = result.assign(**{
                f"{left}_div_{right}": X[left] / (X[right].abs() + 1e-5),
                f"{left}_mult_{right}": X[left] * X[right],
            })
        return result


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the inner split without replacing or transforming the raw y."""

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


SEEDS = (42, 2023, 777)


def blend_predictions(*probabilities):
    """Mean of each family, then the mean of those three, threshold 0.5.

    The last argument is skrub's eval mode. In fit mode the inputs are the
    fitted estimators, not probabilities.
    """
    *matrices, mode = probabilities
    if mode == "fit":
        return None
    cb = [np.asarray(p, dtype=float)[:, 1] for p in matrices[:3]]
    xgb = [np.asarray(p, dtype=float)[:, 1] for p in matrices[3:6]]
    lgb = [np.asarray(p, dtype=float)[:, 1] for p in matrices[6:]]
    ensemble_probs = (np.mean(cb, axis=0) + np.mean(xgb, axis=0) + np.mean(lgb, axis=0)) / 3.0
    return (ensemble_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing records the read without executing it.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare Data — mark the raw target and the original feature matrix.
    #    This splitter reproduces the original stratified train_test_split.
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Preprocessing, independent learned statistics, and the ensemble.
    #    Category vocabularies are now learned inside the outer training fold,
    #    fixing their original full-table preprocessing leakage.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    X = X.assign(
        Sex=X["Sex"].astype("category"),
        Embarked=X["Embarked"].astype("category"),
        Pclass=X["Pclass"].astype("category"),
    )
    X = X.skb.apply(AlignCategories())

    # Each statistic reads the base frame and returns only its new columns.
    # Concatenation restores the original order: three frequencies, then the
    # twelve mean/std pairs in category-major numeric-minor order, then the
    # six interactions. The group-bys do not see the frequency columns.
    frequencies = [
        X[[column]].skb.apply(FrequencyFeature(column), no_wrap=True)
        for column in ("Sex", "Embarked", "Pclass")
    ]
    groups = [
        X[[category, numeric]].skb.apply(
            GroupStatistics(category, numeric), no_wrap=True
        )
        for category in ("Sex", "Embarked", "Pclass")
        for numeric in ("Age", "SibSp", "Parch", "Fare")
    ]
    # A discovered-column loop is necessary here: a nonnumeric source column
    # changes both which interactions exist and which numeric columns are paired.
    interactions = X.skb.apply(DiscoveredNumericInteractions(), no_wrap=True)
    X = X.skb.concat([*frequencies, *groups, interactions], axis=1)

    # CatBoost's default use_best_model can select iterations using eval_set,
    # even without explicit early stopping. The original exposed the scored
    # hold-out labels to that selection. Instead expose an inner 20% stratified
    # evaluation split from the outer training rows. XGBoost and LightGBM stay
    # on the full outer training fold: their eval sets did not select an
    # iteration. Each of the nine fits is its own node. The blend keeps the
    # per-family mean, the mean of those three, and the >= 0.5 threshold.
    parts = X.skb.apply(GetXY(), y=y, no_wrap=True)
    cb = [
        parts["X"].skb.apply(
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=3.0,
                subsample=0.8,
                eval_metric="Logloss",
                random_seed=seed,
                verbose=0,
            ),
            y=parts.get("y", y),
            fit_kwargs={
                "cat_features": CAT_FEATURES,
                "eval_set": (parts["X_val"], parts["y_val"]),
                "verbose": False,
            },
            predict_kwargs={"prediction_type": "Probability"},
        )
        for seed in SEEDS
    ]
    xgb = [
        X.skb.apply(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.5,
                enable_categorical=True,
                tree_method="hist",
                random_state=seed,
                eval_metric="logloss",
            ),
            y=y,
            fit_kwargs={"verbose": False},
        )
        for seed in SEEDS
    ]
    lgb = [
        X.skb.apply(
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=seed,
                verbose=-1,
            ),
            y=y,
            fit_kwargs={"callbacks": []},
        )
        for seed in SEEDS
    ]
    pred = cb[0].skb.apply_func(
        blend_predictions, *cb[1:], *xgb, *lgb, skrub.eval_mode()
    )

    # 4. Score — one ensemble variant, one stratified hold-out, accuracy.
    #    No target transformation or prediction-domain inverse is needed.
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
