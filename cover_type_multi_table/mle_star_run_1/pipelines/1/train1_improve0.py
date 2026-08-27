
import glob
import os
import sys
import subprocess

# Ensure necessary packages are installed
for pkg in ['lightgbm', 'xgboost', 'catboost', 'scikit-learn', 'pandas', 'numpy']:
    try:
        __import__(pkg if pkg != 'scikit-learn' else 'sklearn')
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def load_and_merge_data(input_dir='./input'):
    base_path = os.path.join(input_dir, 'table_0_0.csv')
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base table not found at {base_path}")
        
    base_df = pd.read_csv(base_path)
    
    all_files = glob.glob(os.path.join(input_dir, '*.csv'))
    other_files = [f for f in all_files if os.path.basename(f) != 'table_0_0.csv']
    
    tables = {}
    for f in other_files:
        fname = os.path.basename(f)
        tables[fname] = pd.read_csv(f)
        
    merged_any = True
    while merged_any and len(tables) > 0:
        merged_any = False
        keys_in_base = [c for c in base_df.columns if c.lower().startswith('key_') or c.lower() == 'key']
        
        for fname in list(tables.keys()):
            df = tables[fname]
            keys_in_df = [c for c in df.columns if c.lower().startswith('key_') or c.lower() == 'key']
            
            common_keys = list(set(keys_in_base).intersection(set(keys_in_df)))
            rename_map = {}
            
            if not common_keys:
                for k_base in keys_in_base:
                    base_vals = set(base_df[k_base].dropna().unique())
                    for k_df in keys_in_df:
                        df_vals = set(df[k_df].dropna().unique())
                        if len(df_vals) > 0 and len(base_vals.intersection(df_vals)) > 0.3 * min(len(base_vals), len(df_vals)):
                            rename_map[k_df] = k_base
                            common_keys.append(k_base)
                            break
                    if common_keys:
                        break
            
            if common_keys:
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                # Check for column collisions non-key
                cols_to_rename = {c: f"{os.path.splitext(fname)[0]}_{c}" for c in df.columns if c not in common_keys and not c.lower().startswith('key_')}
                df = df.rename(columns=cols_to_rename)
                
                if df.duplicated(subset=common_keys).any():
                    num_cols = df.select_dtypes(include=[np.number]).columns.difference(common_keys)
                    cat_cols = df.select_dtypes(exclude=[np.number]).columns.difference(common_keys)
                    
                    aggs = {}
                    for c in num_cols:
                        aggs[c] = ['mean', 'std', 'min', 'max']
                    for c in cat_cols:
                        aggs[c] = ['first', 'nunique']
                    
                    df_agg = df.groupby(common_keys).agg(aggs)
                    df_agg.columns = [f'{col}_{stat}' for col, stat in df_agg.columns]
                    df_agg = df_agg.reset_index()
                    
                    base_df = base_df.merge(df_agg, on=common_keys, how='left')
                else:
                    base_df = base_df.merge(df, on=common_keys, how='left')
                
                del tables[fname]
                merged_any = True
                break
                
    for fname, df in list(tables.items()):
        if len(df) == len(base_df):
            df_feats = df.drop(columns=[c for c in df.columns if c.lower().startswith('key_') or c.lower() == 'key'], errors='ignore')
            cols_to_rename = {c: f"{os.path.splitext(fname)[0]}_{c}" for c in df_feats.columns if c in base_df.columns}
            df_feats = df_feats.rename(columns=cols_to_rename)
            base_df = pd.concat([base_df.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)
            
    return base_df

def feature_engineering(df):
    X = df.copy()
    
    # 1. Missing count feature
    X['num_missing'] = X.isnull().sum(axis=1)
    
    # 2. Specific spatial/topographic feature interactions if columns match covertype dataset pattern
    h_dist_col = None
    v_dist_col = None
    for c in X.columns:
        cl = c.lower()
        if 'horiz' in cl and 'hydrolog' in cl:
            h_dist_col = c
        elif 'vert' in cl and 'hydrolog' in cl:
            v_dist_col = c
            
    if h_dist_col and v_dist_col:
        X['Euclidean_Distance_To_Hydrology'] = np.sqrt(X[h_dist_col]**2 + X[v_dist_col]**2)
        
    aspect_cols = [c for c in X.columns if 'aspect' in c.lower()]
    if aspect_cols:
        ac = aspect_cols[0]
        rad = np.radians(X[ac].fillna(0))
        X['Aspect_sin'] = np.sin(rad)
        X['Aspect_cos'] = np.cos(rad)
        
    hillshade_cols = [c for c in X.columns if 'hillshade' in c.lower()]
    if len(hillshade_cols) >= 2:
        X['Hillshade_mean'] = X[hillshade_cols].mean(axis=1)
        X['Hillshade_std'] = X[hillshade_cols].std(axis=1)
        X['Hillshade_max'] = X[hillshade_cols].max(axis=1)
        X['Hillshade_min'] = X[hillshade_cols].min(axis=1)
        
    soil_cols = [c for c in X.columns if 'soil_type' in c.lower()]
    if len(soil_cols) > 1:
        X['Soil_Type_count'] = X[soil_cols].sum(axis=1)
        
    wilderness_cols = [c for c in X.columns if 'wilderness' in c.lower()]
    if len(wilderness_cols) > 1:
        X['Wilderness_Area_count'] = X[wilderness_cols].sum(axis=1)
        
    # 3. Summary stats across numerical features
    num_cols = X.select_dtypes(include=[np.number]).columns
    # Filter out target and key columns from summary stats
    num_cols = [c for c in num_cols if not (c.lower().startswith('key_') or c.lower() == 'key' or c.lower() == 'class')]
    if len(num_cols) > 0:
        X['row_mean'] = X[num_cols].mean(axis=1)
        X['row_std'] = X[num_cols].std(axis=1)
        X['row_min'] = X[num_cols].min(axis=1)
        X['row_max'] = X[num_cols].max(axis=1)
        X['row_sum'] = X[num_cols].sum(axis=1)
        
    return X

def main():
    merged_df = load_and_merge_data('./input')
    
    if 'class' not in merged_df.columns:
        raise KeyError("'class' column not found in base table.")
        
    # Process target variable: 1 -> 0, 2 -> 1
    target = merged_df['class'].values
    if set(np.unique(target)) == {1, 2}:
        y = (target == 2).astype(int)
    else:
        y = (target == np.max(target)).astype(int)
        
    processed_df = feature_engineering(merged_df)
    
    # Drop all key columns and target
    drop_cols = [c for c in processed_df.columns if c.lower().startswith('key_') or c.lower() == 'key' or c.lower() == 'class']
    X = processed_df.drop(columns=drop_cols, errors='ignore')
    
    # Clean column names for models
    X.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(':', '_').replace(' ', '_') for c in X.columns]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # 1. LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=127,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        p_lgb = lgb_model.predict_proba(X_val)[:, 1]
        
        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            early_stopping_rounds=50
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        p_xgb = xgb_model.predict_proba(X_val)[:, 1]
        
        # 3. CatBoost
        cat_model = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            eval_metric='AUC',
            random_seed=42,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50
        )
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        p_cat = cat_model.predict_proba(X_val)[:, 1]
        
        # Ensemble prediction for validation fold
        oof_preds[val_idx] = 0.40 * p_lgb + 0.35 * p_xgb + 0.25 * p_cat

    final_validation_score = roc_auc_score(y, oof_preds)
    print(f"Final Validation Performance: {final_validation_score}")

if __name__ == '__main__':
    main()
