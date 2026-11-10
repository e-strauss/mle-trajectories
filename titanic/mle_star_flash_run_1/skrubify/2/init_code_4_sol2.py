import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

num_cols = ["Age", "Fare", "SibSp", "Parch"]
cat_cols = ["Sex", "Embarked", "Pclass"]
features = num_cols + cat_cols

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the read without accessing the file on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data — mark the raw target and the original feature matrix.
    # StratifiedShuffleSplit reproduces the original stratified hold-out rows.
    y = data["Survived"].skb.mark_as_y()
    X = data[features].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing — fit each imputer and encoder per training fold.
    # Preserve the ColumnTransformer's numeric-then-categorical column order.
    num = X[num_cols].skb.apply(SimpleImputer(strategy="median"))
    cat = (
        X[cat_cols]
        .skb.apply(SimpleImputer(strategy="most_frequent"))
        .skb.apply(
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        )
    )
    X_prepared = num.skb.concat([cat], axis=1)

    pred = X_prepared.skb.apply(
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
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
