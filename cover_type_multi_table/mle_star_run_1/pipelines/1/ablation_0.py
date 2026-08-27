
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# 1. Data Preparation
# ---------------------------------------------------------
input_dir = './input'
base_path = os.path.join(input_dir, 'table_0_0.csv')
base_df = pd.read_csv(base_path)

# Create merged dataset with all relational auxiliary tables
merged_df = base_df.copy()
all_files = glob.glob(os.path.join(input_dir, '*.csv'))
remaining_files = [f for f in all_files if os.path.abspath(f) != os.path.abspath(base_path)]

while remaining_files:
    merged_in_this_pass = False
    next_remaining = []
    for filepath in remaining_files:
        df_temp = pd.read_csv(filepath)
        common_keys = [c for c in df_temp.columns if c.startswith('Key_') and c in merged_df.columns]
        if common_keys:
            dup_cols = [c for c in df_temp.columns if c in merged_df.columns and c not in common_keys]
            if dup_cols:
                df_temp = df_temp.drop(columns=dup_cols)
            merged_df = pd.merge(merged_df, df_temp, on=common_keys, how='left')
            merged_in_this_pass = True
        else:
            next_remaining.append(filepath)
    remaining_files = next_remaining
    if not merged_in_this_pass:
        break


# ---------------------------------------------------------
# 2. Evaluation Helper Function
# ---------------------------------------------------------
def evaluate_pipeline(df, model_params, seed=42):
    y = (df['class'] == 2).astype(int)
    feature_cols = [col for col in df.columns if not col.startswith('Key_') and col != 'class']
    X = df[feature_cols]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof_preds)


# ---------------------------------------------------------
# 3. Model Parameters Definition
# ---------------------------------------------------------
tuned_params = {
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

default_params = {
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'random_state': 42,
    'n_jobs': -1
}

# ---------------------------------------------------------
# 4. Ablation Study Execution
# ---------------------------------------------------------
print("=== Starting Ablation Study ===")

# Baseline evaluation: Merged tables + Tuned hyperparameters
baseline_score = evaluate_pipeline(merged_df, tuned_params)
print(f"Baseline Score (Full Relational Data + Tuned Hyperparameters): ROC AUC = {baseline_score:.6f}")

# Ablation 1: Base table only (Disabling auxiliary relational table merging)
ablation1_score = evaluate_pipeline(base_df, tuned_params)
drop_ablation1 = baseline_score - ablation1_score
print(f"Ablation 1 Score (Base Table Only - No Relational Merging): ROC AUC = {ablation1_score:.6f}")
print(f"  -> Performance drop when removing Relational Merging: {drop_ablation1:+.6f}")

# Ablation 2: Default tree hyper-parameters (Disabling specific depth/leaf/subsample tuning)
ablation2_score = evaluate_pipeline(merged_df, default_params)
drop_ablation2 = baseline_score - ablation2_score
print(f"Ablation 2 Score (Default Tree Parameters - No Custom Hyperparameter Tuning): ROC AUC = {ablation2_score:.6f}")
print(f"  -> Performance drop when removing Custom Hyperparameters: {drop_ablation2:+.6f}")

# ---------------------------------------------------------
# 5. Conclusion & Contribution Ranking
# ---------------------------------------------------------
print("\n=== Ablation Results Summary ===")
print(f"1. Relational Feature Merging Impact: Performance Drop of {drop_ablation1:.6f} ROC AUC")
print(f"2. Hyperparameter Tuning Impact:     Performance Drop of {drop_ablation2:.6f} ROC AUC")

if drop_ablation1 > drop_ablation2:
    most_important = "Relational Table Merging"
    impact_diff = drop_ablation1 - drop_ablation2
else:
    most_important = "Tree Hyperparameter Tuning"
    impact_diff = drop_ablation2 - drop_ablation1

print(f"\nConclusion: {most_important} contributes the most to the overall performance (larger performance drop by {impact_diff:.6f} ROC AUC).")
