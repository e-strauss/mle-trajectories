import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import (
    BaseCrossValidator,
    RepeatedStratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier


class FirstFold(BaseCrossValidator):
    """Use precisely the first fold of the original splitter."""

    def __init__(self, cv):
        self.cv = cv

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        yield next(iter(self.cv.split(X, y, groups)))


class MedianFill(TransformerMixin, BaseEstimator):
    """Learn medians without dropping or changing all-missing columns."""

    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.medians_)


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn the ticket-frequency lookup from the training rows."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return pd.DataFrame(
            {"Ticket_Count": X["Ticket"].map(self.counts_).fillna(1)},
            index=X.index,
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose CatBoost's internal model-selection split in the plan."""

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


class HoldoutProbabilityAverage(ClassifierMixin, BaseEstimator):
    """The final script's two-model average and explicit binary threshold."""

    def fit(self, X, y, cat_X=None, cat_y=None, eval_set=None):
        # Keep the original dtype-dependent categorical-feature check.
        valid_cat_features = [
            col
            for col in ["Pclass", "Sex_Num", "Title_Code"]
            if col in X.columns and str(X[col].dtype) == "category"
        ]
        self.cb_model_ = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            cat_features=valid_cat_features if valid_cat_features else None,
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        )
        self.xgb_model_ = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        )
        self.cb_model_.fit(
            cat_X, cat_y, eval_set=eval_set, verbose=False
        )
        self.xgb_model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        cb_probs = self.cb_model_.predict_proba(X)[:, 1]
        xgb_probs = self.xgb_model_.predict_proba(X)[:, 1]
        return ((cb_probs + xgb_probs) / 2.0 >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read; importing this file does not read data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and retain auxiliary raw columns
    #    for ticket/name features. The reported score uses only the first fold
    #    of the final 5-fold, one-repeat splitter, not the preceding 25 fits.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=FirstFold(
            RepeatedStratifiedKFold(
                n_splits=5, n_repeats=1, random_state=42
            )
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    #    The preceding repeated-OOF trio, surname encoding, rank averaging and
    #    threshold calibration never feed the final predictions or printed
    #    score, so that unused computation is omitted.
    #
    #    The original explicitly selects these seven columns first. Therefore
    #    its fallback branches looking for Real_Fare, Sex_Num or Ticket_Count
    #    in that initial frame are unreachable, unlike optional raw Name/Ticket.
    base = X[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ]
    base = base.assign(
        Embarked=base["Embarked"].fillna("Missing")
    )
    base = base.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # Reindex preserves the optional-Ticket behavior: an absent Ticket column
    # becomes all missing, whose mapped counts fall back to one.
    # The original full-table frequency and medians leaked validation rows;
    # these separate estimators now learn their statistics per training fold.
    ticket_counts = X.reindex(columns=["Ticket"]).skb.apply(
        TicketFrequency()
    )

    fare = base["Fare"].skb.apply_func(pd.to_numeric, errors="coerce")
    fare = fare.to_frame(name="Fare").skb.apply(MedianFill())
    real_fare = fare["Fare"] / ticket_counts["Ticket_Count"]
    log_real_fare = real_fare.clip(lower=0).skb.apply_func(np.log1p)

    sex_num = (
        base["Sex"]
        .astype(str)
        .map({"male": 0, "female": 1})
        .fillna(0)
        .astype(int)
    )
    pclass_num = (
        base["Pclass"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .fillna(3)
        .astype(int)
    )
    sibsp_num = (
        base["SibSp"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    parch_num = (
        base["Parch"]
        .skb.apply_func(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    family_size = sibsp_num + parch_num + 1

    age = base["Age"].skb.apply_func(pd.to_numeric, errors="coerce")
    age = age.to_frame(name="Age").skb.apply(MedianFill())

    features = base.reindex(columns=[]).assign(
        Sex_Num=sex_num,
        Pclass=pclass_num,
        Pclass_Sex=pclass_num * 2 + sex_num,
        Real_Fare=real_fare,
        Log_Real_Fare=log_real_fare,
        Family_Size=family_size,
        Is_Alone=(family_size == 1).astype(int),
        Is_Child=(age["Age"] < 12).astype(int),
        Age=age["Age"],
    )

    # Preserve the data-dependent presence of Title_Code without an eager
    # column-presence test. Surname itself is unused by the final models.
    has_name = X.columns.isin(["Name"]).any()
    name = X.reindex(columns=["Name"])["Name"].astype(str)
    titles = name.str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_code = (
        titles.map(
            {
                "Mr": 0,
                "Miss": 1,
                "Mrs": 2,
                "Master": 3,
                "Dr": 4,
                "Rev": 4,
                "Col": 4,
                "Major": 4,
                "Mlle": 1,
                "Ms": 1,
                "Mme": 2,
            }
        )
        .fillna(5)
        .astype(int)
    )
    features = has_name.skb.if_else(
        features.assign(Title_Code=title_code),
        features,
    )
    # All selected engineered columns are numeric; preserve the original
    # final median-fill pass and their candidate_features column order.
    X_clean = features.skb.apply(MedianFill())

    # The original CatBoost eval_set was the scored holdout. Even without
    # explicit early-stopping patience, CatBoost can select its best iteration
    # using that set. Preserve eval-set model selection, but carve its 20% set
    # out of the outer training rows instead; the original leaky score cannot
    # be reproduced honestly. XGBoost still fits every outer-training row.
    cat_parts = X_clean.skb.apply(GetXY(), y=y, no_wrap=True)
    pred = X_clean.skb.apply(
        HoldoutProbabilityAverage(),
        y=y,
        fit_kwargs={
            "cat_X": cat_parts["X"],
            "cat_y": cat_parts["y"],
            "eval_set": (cat_parts["X_val"], cat_parts["y_val"]),
        },
    )

    # 4. Score -- the splitter on mark_as_X determines the exact holdout.
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
