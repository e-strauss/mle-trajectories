import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier


class OOFProbabilityEnsemble(TransformerMixin, BaseEstimator):
    """Produce OOF training probabilities and averaged inference probabilities."""

    def fit(self, X, y):
        y_array = np.asarray(y)
        self.cb_models_ = []
        self.xgb_models_ = []
        cb_oof = np.zeros(len(X))
        xgb_oof = np.zeros(len(X))

        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )

        # This is the original's internal OOF ensemble, not the outer validation
        # loop. Its models and OOF predictions must be rebuilt inside every fit.
        # Early stopping uses these inner OOF rows, all of which belong to the
        # outer TRAINING partition; the reported hold-out is never used here.
        for fold, (train_idx, val_idx) in enumerate(inner_cv.split(X, y_array)):
            X_tr = X.iloc[train_idx]
            X_va = X.iloc[val_idx]
            y_tr = y_array[train_idx]
            y_va = y_array[val_idx]

            cb_model = CatBoostClassifier(
                iterations=600,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=5.0,
                cat_features=["Sex", "Embarked", "Pclass"],
                eval_metric="Accuracy",
                random_seed=42 + fold,
                verbose=0,
            )
            cb_model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                verbose=False,
            )
            cb_oof[val_idx] = cb_model.predict_proba(X_va)[:, 1]
            self.cb_models_.append(cb_model)

            xgb_model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=3.0,
                enable_categorical=True,
                tree_method="hist",
                random_state=42 + fold,
                eval_metric="logloss",
                early_stopping_rounds=50,
            )
            xgb_model.fit(
                X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False
            )
            xgb_oof[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
            self.xgb_models_.append(xgb_model)

        self.oof_probabilities_ = pd.DataFrame(
            {
                "cb_probability": cb_oof,
                "xgb_probability": xgb_oof,
            },
            index=X.index,
        )
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.oof_probabilities_.copy()

    def transform(self, X):
        cb_probs = np.zeros(len(X))
        xgb_probs = np.zeros(len(X))

        # Preserve the original addition order and division by five.
        for cb_model, xgb_model in zip(self.cb_models_, self.xgb_models_):
            cb_probs += cb_model.predict_proba(X)[:, 1] / 5
            xgb_probs += xgb_model.predict_proba(X)[:, 1] / 5

        return pd.DataFrame(
            {
                "cb_probability": cb_probs,
                "xgb_probability": xgb_probs,
            },
            index=X.index,
        )


class IsotonicCalibration(TransformerMixin, BaseEstimator):
    """Learn calibration separately from the probability ensemble."""

    def fit(self, X, y):
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(X["ensemble_rank"].to_numpy(), y)
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                "calibrated_probability": self.calibrator_.predict(
                    X["ensemble_rank"].to_numpy()
                )
            },
            index=X.index,
        )


class OptimizedThreshold(ClassifierMixin, BaseEstimator):
    """Select the original threshold using calibrated OOF probabilities."""

    def fit(self, X, y):
        probabilities = X["calibrated_probability"].to_numpy()
        self.classes_ = np.unique(y)
        self.best_threshold_ = 0.5
        self.best_oof_score_ = 0.0

        for threshold in np.linspace(0.1, 0.9, 801):
            score = accuracy_score(
                y, (probabilities >= threshold).astype(int)
            )
            if score > self.best_oof_score_:
                self.best_oof_score_ = score
                self.best_threshold_ = threshold

        return self

    def predict(self, X):
        return (
            X["calibrated_probability"].to_numpy() >= self.best_threshold_
        ).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read without accessing the file at import time.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and the original seven features.
    # The script reports one stratified 80/20 hold-out score. Its five-fold
    # loop builds an internal OOF ensemble rather than five reported CV scores.
    y = data["Survived"].skb.mark_as_y()
    X = data[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing -- preserve missing-value and category handling.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    X = X.assign(
        Sex=X["Sex"].astype("category"),
        Embarked=X["Embarked"].astype("category"),
        Pclass=X["Pclass"].astype("category"),
    )

    # This node trains the internal ensemble and aggregates its probabilities.
    # Calibration and threshold learning are separate fitted nodes.
    probabilities = X.skb.apply(
        OOFProbabilityEnsemble(), y=y, no_wrap=True
    )

    # Preserve the original stateless, transductive prediction rule: rank each
    # model within the current OOF or inference batch, using average ties.
    # These ranks do not use validation targets. Native recorded Series ops
    # express the original rankdata(p) / len(p).
    cb_rank = probabilities["cb_probability"].rank(method="average", pct=True)
    xgb_rank = probabilities["xgb_probability"].rank(method="average", pct=True)
    ranked = probabilities.assign(
        ensemble_rank=(cb_rank + xgb_rank) / 2.0
    )[["ensemble_rank"]]

    calibrated = ranked.skb.apply(
        IsotonicCalibration(), y=y, no_wrap=True
    )
    pred = calibrated.skb.apply(OptimizedThreshold(), y=y)

    # 4. Score -- the marked splitter reproduces the original outer hold-out.
    # Fold-progress and fitted-threshold printing are omitted; only scoring
    # results are produced.
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
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
