import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]


class TrainingCategories(TransformerMixin, BaseEstimator):
    """Learn one native categorical dtype from the training rows."""

    def fit(self, X, y=None):
        self.column_ = X.columns[0]
        self.dtype_ = pd.Categorical(X[self.column_]).dtype
        return self

    def transform(self, X):
        return X.astype({self.column_: self.dtype_})


class FrequencyFeature(TransformerMixin, BaseEstimator):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        self.frequencies_ = X[self.column].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        encoded = X[self.column].map(self.frequencies_).astype(float).fillna(0)
        return X.assign(**{f"{self.column}_freq": encoded})


class GroupAggregationFeature(TransformerMixin, BaseEstimator):
    """Learn one categorical/numeric pair's mean and sample standard deviation."""

    def __init__(self, categorical_column, numeric_position):
        self.categorical_column = categorical_column
        self.numeric_position = numeric_position

    def fit(self, X, y=None):
        # Discover numeric columns from the ORIGINAL feature block, not from
        # previously appended frequency or aggregation features.
        numeric_columns = [
            c
            for c in FEATURES
            if c not in CAT_COLS and np.issubdtype(X[c].dtype, np.number)
        ]
        self.numeric_column_ = (
            numeric_columns[self.numeric_position]
            if self.numeric_position < min(4, len(numeric_columns))
            else None
        )
        if self.numeric_column_ is not None:
            c = self.numeric_column_
            self.lookup_ = (
                X.groupby(self.categorical_column, observed=False)[c]
                .agg(["mean", "std"])
                .reset_index()
                .rename(
                    columns={
                        "mean": f"{c}_mean_by_{self.categorical_column}",
                        "std": f"{c}_std_by_{self.categorical_column}",
                    }
                )
                .drop_duplicates(subset=self.categorical_column, keep="last")
            )
        return self

    def transform(self, X):
        if self.numeric_column_ is None:
            return X.copy()
        return X.merge(self.lookup_, on=self.categorical_column, how="left")


class DiscoveredNumericInteractions(TransformerMixin, BaseEstimator):
    """Preserve the original data-discovered interaction names and ordering."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # This is the discovered-column exception: the set and order depend on
        # runtime dtypes. Do not assume Age/SibSp/Parch/Fare are all numeric.
        numeric_columns = [
            c
            for c in FEATURES
            if c not in CAT_COLS and np.issubdtype(X[c].dtype, np.number)
        ]
        result = X.copy()
        for i in range(min(3, len(numeric_columns) - 1)):
            c1, c2 = numeric_columns[i], numeric_columns[i + 1]
            result = result.assign(
                **{
                    f"{c1}_div_{c2}": result[c1] / (result[c2].abs() + 1e-5),
                    f"{c1}_mult_{c2}": result[c1] * result[c2],
                }
            )
        return result


class GetXY(TransformerMixin, BaseEstimator):
    """Expose CatBoost's per-fit model-selection split in the recorded plan."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class SeedAveragedTriad(ClassifierMixin, BaseEstimator):
    """The original manual probability arithmetic, rerun independently per fit."""

    def __init__(self, seeds=(42, 2023, 777), use_lgb=True):
        self.seeds = seeds
        self.use_lgb = use_lgb

    def fit(self, X, y, cat_X=None, cat_y=None, cat_eval=None):
        self.classes_ = np.unique(y)
        self.catboost_models_ = []
        self.xgboost_models_ = []
        self.lightgbm_models_ = []

        for seed in self.seeds:
            cb_model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=3.0,
                subsample=0.8,
                cat_features=CAT_COLS,
                eval_metric="Logloss",
                random_seed=seed,
                verbose=0,
            )
            cb_model.fit(cat_X, cat_y, eval_set=cat_eval, verbose=False)
            self.catboost_models_.append(cb_model)

            xgb_model = XGBClassifier(
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
            )
            # With no early stopping, the original XGBoost eval_set only
            # monitored loss; omit that access to the outer validation rows.
            xgb_model.fit(X, y, verbose=False)
            self.xgboost_models_.append(xgb_model)

            if self.use_lgb:
                lgb_model = LGBMClassifier(
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
                )
                # Likewise, callbacks=[] did not enable early stopping.
                lgb_model.fit(X, y, callbacks=[])
                self.lightgbm_models_.append(lgb_model)

        return self

    def predict(self, X):
        cb_preds = np.zeros(len(X))
        xgb_preds = np.zeros(len(X))
        lgb_preds = np.zeros(len(X))

        # Keep the original accumulation order and division before summation.
        for i in range(len(self.seeds)):
            cb_preds += (
                self.catboost_models_[i].predict_proba(X)[:, 1] / len(self.seeds)
            )
            xgb_preds += (
                self.xgboost_models_[i].predict_proba(X)[:, 1] / len(self.seeds)
            )
            if self.use_lgb:
                lgb_preds += (
                    self.lightgbm_models_[i].predict_proba(X)[:, 1]
                    / len(self.seeds)
                )

        if self.use_lgb:
            probabilities = (cb_preds + xgb_preds + lgb_preds) / 3.0
        else:
            probabilities = (cb_preds + xgb_preds) / 2.0
        return (probabilities >= 0.5).astype(int)


def add_group_features(X):
    # This helper builds twelve separate recorded, stateful estimator nodes.
    X = X.skb.apply(GroupAggregationFeature("Sex", 0), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Sex", 1), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Sex", 2), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Sex", 3), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Embarked", 0), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Embarked", 1), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Embarked", 2), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Embarked", 3), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Pclass", 0), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Pclass", 1), no_wrap=True)
    X = X.skb.apply(GroupAggregationFeature("Pclass", 2), no_wrap=True)
    return X.skb.apply(GroupAggregationFeature("Pclass", 3), no_wrap=True)


def apply_ensemble(features, y, cat_parts, seeds=(42, 2023, 777), use_lgb=True):
    # Only graph construction happens here. The ensemble itself is per-fit;
    # its CatBoost model-selection inputs remain visible as DataOps.
    return features.skb.apply(
        SeedAveragedTriad(seeds=seeds, use_lgb=use_lgb),
        y=y,
        fit_kwargs={
            "cat_X": cat_parts["X"],
            "cat_y": cat_parts["y"],
            "cat_eval": (cat_parts["X_val"], cat_parts["y_val"]),
        },
    )


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read; importing this file reads no data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original seven features.
    # StratifiedShuffleSplit reproduces train_test_split(..., stratify=y).
    y = data["Survived"].skb.mark_as_y()
    X = data[FEATURES].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature branches.
    X = X.assign(Embarked=X["Embarked"].fillna("Missing"))

    # Learn native category vocabularies on training rows rather than leaking
    # the full-table vocabulary from the original pre-split astype("category").
    X = X.skb.apply(TrainingCategories(), cols=["Sex"])
    X = X.skb.apply(TrainingCategories(), cols=["Embarked"])
    X = X.skb.apply(TrainingCategories(), cols=["Pclass"])

    frequency_features = X.skb.apply(FrequencyFeature("Sex"), no_wrap=True)
    frequency_features = frequency_features.skb.apply(
        FrequencyFeature("Embarked"), no_wrap=True
    )
    frequency_features = frequency_features.skb.apply(
        FrequencyFeature("Pclass"), no_wrap=True
    )

    aggregation_features = add_group_features(frequency_features)
    full_features = aggregation_features.skb.apply(
        DiscoveredNumericInteractions(), no_wrap=True
    )
    no_aggregation_features = frequency_features.skb.apply(
        DiscoveredNumericInteractions(), no_wrap=True
    )
    no_interaction_features = aggregation_features

    # CatBoost's eval_set implicitly enables best-iteration selection even
    # without an explicit early-stopping patience. The original selected that
    # iteration using the very rows it scored. Preserve eval-based selection,
    # but carve its eval set out of the outer fold's training rows instead.
    # The original outer hold-out fraction, 0.2, is reused for this inner split;
    # consequently its leaky CatBoost score cannot be reproduced exactly.
    full_cat_parts = full_features.skb.apply(GetXY(), y=y, no_wrap=True)
    no_agg_cat_parts = no_aggregation_features.skb.apply(
        GetXY(), y=y, no_wrap=True
    )
    no_int_cat_parts = no_interaction_features.skb.apply(
        GetXY(), y=y, no_wrap=True
    )

    variants = {
        "Full Pipeline (Baseline)": apply_ensemble(
            full_features, y, full_cat_parts
        ),
        "Ablation 1 (Disable Multi-Seed Averaging)": apply_ensemble(
            full_features, y, full_cat_parts, seeds=(42,)
        ),
        "Ablation 2 (Drop LightGBM from Triad Ensemble)": apply_ensemble(
            full_features, y, full_cat_parts, use_lgb=False
        ),
        "Ablation 3 (Drop Group Aggregations)": apply_ensemble(
            no_aggregation_features, y, no_agg_cat_parts
        ),
        "Ablation 4 (Drop Numeric Interactions)": apply_ensemble(
            no_interaction_features, y, no_int_cat_parts
        ),
    }
    pred = skrub.choose_from(variants, name="variant").as_data_op()

    # 4. Score exactly the five original variants on the same outer hold-out.
    # No cv= here: the splitter on mark_as_X drives validation.
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
