
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

# Create feature dataset with group-by aggregations
X_raw = merged_df.copy()

key_cols = [col for col in merged_df.columns if col.startswith('Key_')]
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in key_cols and c != 'class']

# Compute group-by aggregation features based on Key_* entity identifiers
for key in key_cols:
    if X_raw[key].nunique() > 1 and X_raw[key].nunique() < len(X_raw):
        aggs = X_raw.groupby(key)[num_cols].agg(['mean', 'std', 'min', 'max'])
        aggs.columns = [f'{key}_{c}_{stat}' for c, stat in aggs.columns]
        X_raw = X_raw.merge(aggs, on=key, how='left')

# Drop Key_* columns and target class from features
feature_cols = [col for col in X_raw.columns if not col.startswith('Key_') and col != 'class']
X = X_raw[feature_cols].copy()

# Remove zero-variance (constant) features
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X.drop(columns=constant_cols, inplace=True)

# Remove highly collinear redundant features (correlation > 0.98)
corr_matrix = X.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
if to_drop:
    X.drop(columns=to_drop, inplace=True)

# Stratified 5-fold cross validation using LightGBM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.015,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=30,
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

