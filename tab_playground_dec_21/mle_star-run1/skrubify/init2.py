import skrub
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--search", action="store_true")

args = parser.parse_args()

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan.
    path = "./input/train.csv"
    train_df = skrub.as_data_op(path).skb.apply_func(pd.read_csv)

    # 2. Prepare Data
    #    Target (y): 'Cover_Type', shifted 1-7 -> 0-6 and cast to 'category'
    #    exactly as in the original (XGBoost expects 0-indexed multi-class labels).
    #    Features (X): all columns except 'Id' and 'Cover_Type'.
    #    The CV splitter lives on mark_as_X — grid search below reuses it, which is
    #    equivalent to the manual StratifiedKFold loop in init_code_2.py.
    y = (train_df["Cover_Type"] - 1).astype("category").skb.mark_as_y()
    X = train_df.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )

    # 3. Model — same params as the original.
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=7,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    pred = X.skb.apply(model, y=y)
    pred.skb.draw_graph().open()

    # 4. Score the whole plan by 3-fold CV. No cv= here — the StratifiedKFold set on
    #    mark_as_X drives. mean_test_score is the mean fold accuracy, matching the
    #    np.mean(fold_accuracies) final score of the original loop.
    if __name__ == "__main__" and args.search:
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        print(f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}")
