import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class CatBoostClassifierCloneable(CatBoostClassifier):
    """Preserve CatBoost parameters while bypassing its sklearn cloning bug."""

    def __sklearn_clone__(self):
        return type(self)(**self.get_params(deep=False))


class FoldAveragedStackingClassifier(ClassifierMixin, BaseEstimator):
    """Train the original OOF stack and retain every inner-fold estimator."""

    def __init__(self, estimators, meta_learner, n_splits=5, random_state=42):
        self.estimators = estimators
        self.meta_learner = meta_learner
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        inner_cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof_meta = np.zeros((len(X), len(self.estimators)))
        self.fold_models_ = []

        # This is the original stack's INTERNAL training loop, not outer CV.
        # Its model fitting and OOF bookkeeping belong inside the estimator.
        for model_idx, (name, estimator) in enumerate(self.estimators):
            fitted_folds = []
            for train_idx, oof_idx in inner_cv.split(X, y_array):
                clf = clone(estimator)
                fit_kwargs = {"verbose": False} if name == "catboost" else {}
                clf.fit(X.iloc[train_idx], y_array[train_idx], **fit_kwargs)
                oof_meta[oof_idx, model_idx] = clf.predict_proba(
                    X.iloc[oof_idx]
                )[:, 1]
                fitted_folds.append(clf)
            self.fold_models_.append(fitted_folds)

        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(oof_meta, y_array)
        return self

    def predict_proba(self, X):
        meta = np.zeros((len(X), len(self.fold_models_)))
        for model_idx, fitted_folds in enumerate(self.fold_models_):
            fold_probs = np.zeros((len(X), self.n_splits))
            for fold_idx, clf in enumerate(fitted_folds):
                fold_probs[:, fold_idx] = clf.predict_proba(X)[:, 1]
            meta[:, model_idx] = fold_probs.mean(axis=1)
        return self.meta_learner_.predict_proba(meta)

    def predict(self, X):
        # Keep the original threshold and integer decoding, including ties.
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — importing this file records, but does not execute, the read.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original seven features.
    # The original stratified train_test_split is exactly one stratified
    # shuffle split; the five-fold split remains internal to stack training.
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    y = data["Survived"].skb.mark_as_y()
    X = data[features].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, followed by the original stacked model.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    X = X.assign(
        Pclass=X["Pclass"].astype("category"),
        Sex=X["Sex"].astype("category"),
        Embarked=X["Embarked"].astype("category"),
    )

    # Preserve the original feature-order-derived column lists. These learned
    # preprocessing steps are fitted separately inside every inner training fold.
    num_cols = ["Age", "SibSp", "Parch", "Fare"]
    cat_cols = ["Pclass", "Sex", "Embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    estimators = [
        (
            "catboost",
            CatBoostClassifierCloneable(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                cat_features=cat_cols,
                eval_metric="Accuracy",
                random_seed=42,
                verbose=0,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        ),
        (
            "xgb",
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
        ),
        (
            "log_reg",
            Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    (
                        "classifier",
                        LogisticRegression(
                            C=0.1, max_iter=1000, random_state=42
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf",
            Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=8,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    pred = X.skb.apply(
        FoldAveragedStackingClassifier(
            estimators=estimators,
            meta_learner=LogisticRegression(
                C=1.0, penalty="l2", random_state=42
            ),
            n_splits=5,
            random_state=42,
        ),
        y=y,
    )

    # 4. Score — outer validation is driven only by mark_as_X.
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
