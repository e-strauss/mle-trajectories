import pandas as pd
import skrub
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read; importing this file does not read data.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and preserve feature order.
    # StratifiedShuffleSplit reproduces the original stratified hold-out split.
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    y = data["Survived"].skb.mark_as_y()
    X = data[features].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and model — retain native categorical handling
    # and every original model hyperparameter.
    X = X.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )
    pred = X.skb.apply(
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

    # 4. Score — the splitter on mark_as_X drives the single hold-out evaluation.
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
