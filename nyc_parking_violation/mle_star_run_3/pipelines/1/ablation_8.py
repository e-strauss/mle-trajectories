
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import io

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, boroughs_path, physical_path, use_augmentation=True):
    """Loads, preprocesses, and prepares data for all models."""
    # Reset StringIO buffer to the beginning in case it has been read before
    if isinstance(train_path, io.StringIO):
        train_path.seek(0)
    if isinstance(boroughs_path, io.StringIO):
        boroughs_path.seek(0)
    if isinstance(physical_path, io.StringIO):
        physical_path.seek(0)
        
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    if use_augmentation:
        boroughs_df = clean_col_names(pd.read_csv(boroughs_path))
        physical_df = clean_col_names(pd.read_csv(physical_path))
        
        # Feature Engineering & Merging
        full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
        full_df = pd.merge(full_df, physical_df, on='street_name', how='left')
        
        # Impute missing values
        full_df['borough'].fillna('Unknown', inplace=True)
        numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
        for col in numerical_cols:
            if col in full_df.columns:
                median_val = full_df[col].median()
                full_df[col].fillna(median_val, inplace=True)
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
    else:
        # If no augmentation, just use the training data
        full_df = train_df
        # Add an empty borough column for consistency in categorical features
        full_df['borough'] = 'Unknown'


    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(train_path, boroughs_path, physical_path, use_augmentation, split_random_state):
    """Runs a single training and evaluation experiment with specified configurations."""
    
    # 1. Data Preparation
    train_data, cat_features = load_and_prepare_data(
        train_path, boroughs_path, physical_path, use_augmentation=use_augmentation
    )
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    # Ensure all categorical features are present in the feature list
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # 2. Validation Split
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=split_random_state
    )

    # 3. Model Training
    # Shared CatBoost parameters
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': [c for c in cat_features if c in X_train.columns], 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    
    # Model 1 (CatBoost Base)
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    # Model 2 (CatBoost Log)
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Convert categorical columns to pandas categorical type for XGBoost
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for col in cat_params['cat_features']:
        X_train_xgb[col] = X_train_xgb[col].astype('category')
        X_val_xgb[col] = X_val_xgb[col].astype('category')

    # Model 3 (XGBoost Log)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)

    # 4. Validation Performance & Ensembling
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    return np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))

def main():
    """Performs the ablation study and prints the results."""
    # Create dummy data files in memory to make the script self-contained
    train_data_csv = """street_name,violation_description,violation_count
street_a,vtype_1,100
street_a,vtype_2,50
street_b,vtype_1,200
street_b,vtype_2,10
street_c,vtype_1,5
street_c,vtype_3,15
street_d,vtype_1,80
"""
    boroughs_data_csv = """street_name,borough
street_a,Brooklyn
street_b,Manhattan
street_c,Queens
"""
    physical_data_csv = """street_name,length,width
street_a,150.5,30.2
street_b,200.0,45.1
street_d,100.1,25.0
"""
    
    train_path = io.StringIO(train_data_csv)
    boroughs_path = io.StringIO(boroughs_data_csv)
    physical_path = io.StringIO(physical_data_csv)

    print("Running Ablation Study...\n")
    results = {}

    # Baseline: Full model with all features
    baseline_rmse = run_experiment(
        train_path, boroughs_path, physical_path, use_augmentation=True, split_random_state=42
    )
    results['Baseline (Augmentation=True, Split State=42)'] = baseline_rmse
    print(f"Final Validation Performance: {baseline_rmse:.4f}")

    # Ablation 1: No Data Augmentation
    results['No Data Augmentation'] = run_experiment(
        train_path, boroughs_path, physical_path, use_augmentation=False, split_random_state=42
    )
    
    # Ablation 2: Different Validation Split
    results['Different Split (Augmentation=True, Split State=101)'] = run_experiment(
        train_path, boroughs_path, physical_path, use_augmentation=True, split_random_state=101
    )

    # Print results summary
    print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
    
    no_aug_rmse = results['No Data Augmentation']
    print(f"Ablation 'No Data Augmentation' RMSE: {no_aug_rmse:.4f} (Impact: {no_aug_rmse - baseline_rmse:+.4f})")
    
    diff_split_rmse = results['Different Split (Augmentation=True, Split State=101)']
    print(f"Ablation 'Different Split' RMSE: {diff_split_rmse:.4f} (Impact: {diff_split_rmse - baseline_rmse:+.4f})\n")

    # Determine the most impactful component
    aug_impact = no_aug_rmse - baseline_rmse
    # Use absolute difference for split impact, as it can be positive or negative
    split_impact = abs(diff_split_rmse - baseline_rmse)

    # A small threshold to consider impacts as negligible
    if aug_impact < 0.01 and split_impact < 0.01:
        print("Conclusion: Neither data augmentation nor the validation split has a significant impact on performance for this dataset.")
    elif aug_impact > split_impact:
        print("Conclusion: Data Augmentation is the most impactful component. Adding external features for borough and physical properties provides a significant performance boost.")
    else:
        print("Conclusion: The model performance is highly sensitive to the random validation split, indicating potential instability. Data augmentation has a lesser effect.")

if __name__ == '__main__':
    main()
