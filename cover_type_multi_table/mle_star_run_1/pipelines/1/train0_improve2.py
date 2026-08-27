
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


from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Map class (1, 2) to binary target (0, 1)
y = (merged_df['class'] == 2).astype(int)

# Drop Key_* columns and target class from features
feature_cols = [col for col in merged_df.columns if not col.startswith('Key_') and col != 'class']
X = merged_df[feature_cols].copy()

# Identify categorical and string features, converting them to pandas category dtype
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype('category')

# Stratified 5-fold cross validation using LightGBM, XGBoost, and CatBoost
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_cb = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method='hist',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        eval_metric='auc'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        subsample=0.8,
        random_seed=42,
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        early_stopping_rounds=50,
        verbose=False
    )
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    oof_cb[val_idx] = cb_model.predict_proba(X_val)[:, 1]

# Rank averaging blend of OOF prediction probabilities
lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
cb_rank = rankdata(oof_cb) / len(oof_cb)

oof_preds = (lgb_rank + xgb_rank + cb_rank) / 3.0

final_validation_score = roc_auc_score(y, oof_preds)
print(f'Final Validation Performance: {final_validation_score}')

