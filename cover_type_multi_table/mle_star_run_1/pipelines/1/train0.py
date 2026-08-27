
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load base table
input_dir = './input'
base_path = os.path.join(input_dir, 'table_0_0.csv')
merged_df = pd.read_csv(base_path)

# Find all remaining CSV files and merge them iteratively based on Key_* columns
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

# Map class (1, 2) to binary target (0, 1)
y = (merged_df['class'] == 2).astype(int)

# Drop Key_* columns and target class from features
feature_cols = [col for col in merged_df.columns if not col.startswith('Key_') and col != 'class']
X = merged_df[feature_cols]

# Stratified 3-fold cross validation using LightGBM
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

final_validation_score = roc_auc_score(y, oof_preds)
print(f'Final Validation Performance: {final_validation_score}')
