import pandas as pd
import skrub
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder


class TitleAgeImputer(TransformerMixin, BaseEstimator):
    """Learn title-specific age medians from the fold's training rows."""

    def fit(self, X, y=None):
        self.title_medians_ = X.groupby("Title")["Age"].median()
        return self

    def transform(self, X):
        return X.assign(
            Age=X["Age"].fillna(X["Title"].map(self.title_medians_))
        )


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the read without reading anything at import time.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and reproduce the original's
    #    stratified 80/20 hold-out with exactly the same random state.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, followed by fold-fitted preprocessing.
    X = X.assign(
        Title=X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False),
        FamilySize=X["SibSp"] + X["Parch"] + 1,
        Embarked=X["Embarked"].fillna("S"),
    )

    # The original learned these medians before splitting. Learn them on each
    # fold's training rows instead, retaining the title-then-global fill order.
    title_age = X[["Title", "Age"]].skb.apply(TitleAgeImputer())
    X = X.assign(Age=title_age["Age"])
    X = X.skb.apply(SimpleImputer(strategy="median"), cols=["Age", "Fare"])

    # OrdinalEncoder uses the same sorted category codes as LabelEncoder.
    # Fit on training rows only to fix the original full-table encoder leakage;
    # categories absent from training receive -1 at validation time.
    X = X.assign(
        Sex=X["Sex"].astype(str),
        Embarked=X["Embarked"].astype(str),
        Title=X["Title"].astype(str),
    )
    X = X.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ),
        cols=["Sex", "Embarked", "Title"],
    )

    features = X[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title",
            "FamilySize",
        ]
    ]

    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )
    pred = features.skb.apply(model, y=y)

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
