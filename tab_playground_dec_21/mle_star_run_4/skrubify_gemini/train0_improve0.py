import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to preserve exact column generation order and naming
# as the original script, which prevents altering the random seed behavior
# in RandomForest that would happen if column order changed. (Guide pitfall 20)
class InteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        soil_type_cols = [col for col in X.columns if 'Soil_Type' in col]
        wilderness_area_cols = [col for col in X.columns if 'Wilderness_Area' in col]

        if 'Elevation' in X.columns:
            for soil_col in soil_type_cols:
                X[f"{soil_col}_x_Elevation"] = X[soil_col] * X['Elevation']

        for soil_col in soil_type_cols:
            for wa_col in wilderness_area_cols:
                X[f"{soil_col}_x_{wa_col}"] = X[soil_col] * X[wa_col]

        return X


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Row filtering: handle classes with fewer samples than n_splits=3.
    #    This must happen before mark_as_X / mark_as_y because it drops rows.
    #    We use n_splits=3 (hard-coded literal as it was fixed in the script).
    raw_target = data["Cover_Type"]
    class_counts = raw_target.value_counts()
    problematic_classes = class_counts[class_counts < 3].index
    filtered = data[~raw_target.isin(problematic_classes)].reset_index(drop=True)

    # 3. Prepare data: mark the RAW target and the design matrix; the CV splitter
    #    lives on mark_as_X and nowhere else.
    y = filtered["Cover_Type"].skb.mark_as_y()
    X = filtered.drop(columns=["Id", "Cover_Type"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={}
    )

    # 4. Apply feature engineering using the custom transformer.
    X_feat = X.skb.apply(InteractionFeatures())

    # 5. Transform the target (shift by 1 to be 0-indexed) exactly as the original.
    #    Because we must mark the RAW target (1-indexed) at mark_as_y, we apply
    #    the transformation inside the plan.
    y_transformed = y - 1

    # 6. Model application
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pred_raw = X_feat.skb.apply(model, y=y_transformed)

    # 7. Invert the target transformation on the predictions, gated by eval_mode,
    #    so predictions map back to the original 1-7 scale for valid scoring against
    #    the RAW marked target.
    def invert_predictions(p, mode):
        if mode == "fit":
            return p
        return p + 1

    pred = pred_raw.skb.apply_func(invert_predictions, skrub.eval_mode())

    # 8. Score. No cv= here -- the splitter on mark_as_X drives.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}")
