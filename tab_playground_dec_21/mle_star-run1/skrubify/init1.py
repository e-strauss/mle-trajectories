import skrub
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--search", action="store_true")

args = parser.parse_args()
with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record the CSV read as the first step of the plan.
    path = "./input/train.csv"
    train_df = skrub.as_data_op(path).skb.apply_func(pd.read_csv)

    # 2. Prepare Data
    #    Target (y): 'Cover_Type', shifted from 1-7 to 0-6 exactly as in the
    #    original. Features (X): all columns except 'Id' and 'Cover_Type'.
    #    The CV splitter lives on mark_as_X — grid search below reuses it, which is
    #    equivalent to the manual StratifiedKFold loop in init_code_1.py.
    y = (train_df["Cover_Type"] - 1).skb.mark_as_y()
    X = train_df.drop(["Id", "Cover_Type"], axis=1).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )

    # 3. Model — same params as the original. num_class == len(y.unique()) == 7 for
    #    this dataset (Cover_Type has 7 classes); it must be a concrete int since the
    #    estimator is built once at plan-construction time.
    model = LGBMClassifier(objective="multiclass", num_class=7, random_state=42)
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
