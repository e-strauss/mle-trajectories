
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def train_and_evaluate(
    train_path,
    xgb_native_categoricals=False,
    impute_coerce_with_median=False
):
    """
    This function encapsulates a single training and evaluation run.
    It includes data loading, preprocessing, model training, and evaluation.
    """
    # --- 1. Data Loading and Preprocessing ---
    def clean_col_names(df):
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if 'violation_description' in df.columns:
            df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
        return df

    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError:
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            
            # This is the section being ablated
            fill_value = median_val if impute_coerce_with_median else 0
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(fill_value)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    full_df['log_violation_count'] = np.log1p(full_df['violation_count'])
    features = [col for col in full_df.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = full_df[features]
    y_base = full_df['violation_count']
    y_log = full_df['log_violation_count']
    groups = full_df['street_name']
    
    n_splits = 5
    y_strat = pd.qcut(y_log, q=n_splits, labels=False, duplicates='drop')

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_idx, val_idx = next(sgkf.split(X, y_strat, groups=groups))
    except ValueError:
        gkf = GroupKFold(n_splits=n_splits)
        train_idx, val_idx = next(gkf.split(X, y_log, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_base, y_val_base = y_base.iloc[train_idx], y_base.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # --- 3. Model Training ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'eval_metric': 'RMSE',
        'random_seed': 42, 'verbose': 0, 'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # This is the section being ablated
    if xgb_native_categoricals:
        X_train_xgb, X_val_xgb = X_train.copy(), X_val.copy()
        xgb_enable_categorical = True
    else:
        X_train_xgb, X_val_xgb = X_train.copy(), X_val.copy()
        for col in cat_features:
            X_train_xgb[col] = X_train_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
        xgb_enable_categorical = False
        
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5,
        early_stopping_rounds=50, enable_categorical=xgb_enable_categorical, tree_method='hist',
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)

    # --- 4. Validation ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    return np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))

if __name__ == '__main__':
    # Create dummy data files if they don't exist for self-contained execution
    if not os.path.exists('./input'):
        os.makedirs('./input')
    
    # Dummy main data
    train_data = {
        'Street Name': [f'Street {i}' for i in range(5)] * 2,
        'Violation Description': [f'Type {i}' for i in range(2)] * 5,
        'Violation Count': np.random.randint(10, 100, 10)
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)
    
    # Dummy augmentation data
    borough_data = {'Street Name': [f'Street {i}' for i in range(5)], 'Borough': ['A', 'B', 'A', 'C', 'B']}
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)
    
    physical_data = {
        'Street Name': [f'Street {i}' for i in range(5)],
        'width': [10, 12, '10', 15, 11],
        'length': [100, 150, 120, 'N/A', 130]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

    results = {}
    train_file = './input/violations_per_street_2022.csv'

    print("Running Ablation Study...")

    # Baseline
    print("1. Running Baseline experiment...")
    results['Baseline'] = train_and_evaluate(
        train_path=train_file,
        xgb_native_categoricals=False,
        impute_coerce_with_median=False
    )

    # Ablation 1: Use native XGBoost categorical support
    print("2. Running Ablation: Native XGBoost Categorical Support...")
    results['Native XGBoost Categoricals'] = train_and_evaluate(
        train_path=train_file,
        xgb_native_categoricals=True,
        impute_coerce_with_median=False
    )
    
    # Ablation 2: Impute coerced values with median instead of zero
    print("3. Running Ablation: Median Imputation for Coerced NaN...")
    results['Median Imputation for Coerced NaN'] = train_and_evaluate(
        train_path=train_file,
        xgb_native_categoricals=False,
        impute_coerce_with_median=True
    )

    # --- Analysis ---
    baseline_rmse = results['Baseline']
    impact = {}
    
    print("\n--- Ablation Study Results ---")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # Calculate impact of Native XGBoost Categoricals
    ablation_1_rmse = results['Native XGBoost Categoricals']
    impact['Native XGBoost Categoricals'] = ablation_1_rmse - baseline_rmse
    print(f"No .cat.codes (Native XGBoost Support) RMSE: {ablation_1_rmse:.4f} (Impact: {impact['Native XGBoost Categoricals']:+.4f})")
    
    # Calculate impact of Median Imputation for Coerced NaN
    ablation_2_rmse = results['Median Imputation for Coerced NaN']
    impact['Median Imputation for Coerced NaN'] = ablation_2_rmse - baseline_rmse
    print(f"Median Imputation for Coerced NaN RMSE: {ablation_2_rmse:.4f} (Impact: {impact['Median Imputation for Coerced NaN']:+.4f})")

    # Determine the most impactful component
    most_impactful_component = max(impact, key=lambda k: abs(impact[k]))
    
    print("\n--- Conclusion ---")
    print(f"The most impactful component is: {most_impactful_component}")

