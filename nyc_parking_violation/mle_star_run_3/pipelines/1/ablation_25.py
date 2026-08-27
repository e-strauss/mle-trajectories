
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import io

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Dummy Data Creation ---
def create_dummy_data():
    """Creates dummy CSV files for a self-contained script."""
    os.makedirs('./input', exist_ok=True)
    
    train_data = """street_name,violation_type,violation_count
Main St,Double Parking,150
Main St,Bus Lane,50
Oak Ave,Fire Hydrant,20
Oak Ave,Double Parking,75
Maple St,No Standing,300
Elm St,Double Parking,120
Elm St,Bus Lane,40
"""
    with open('./input/violations_per_street_2022.csv', 'w') as f:
        f.write(train_data)
        
    boroughs_data = """street_name,borough
Main St,Queens
Oak Ave,Brooklyn
Maple St,Manhattan
Elm St,Queens
"""
    with open('./input/street_names_and_boroughs.csv', 'w') as f:
        f.write(boroughs_data)
        
    physical_data = """street_name,street_width,pavement_quality
Main St,30,Good
Oak Ave,25,Fair
Maple St,40,Excellent
Elm St,30,Good
Pine Ave,28,Bad
"""
    with open('./input/physical_features_per_street.csv', 'w') as f:
        f.write(physical_data)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_imputation_safeguard=True):
    """Loads and preprocesses data."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    # Impute categorical features from augmentation tables
    full_df['borough'].fillna('Unknown', inplace=True)
    if 'pavement_quality' in full_df.columns:
        full_df['pavement_quality'].fillna('Unknown', inplace=True)

    # Define and impute numerical features from augmentation tables
    numerical_cols = ['street_width']  # Explicitly define numerical columns
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            if use_imputation_safeguard:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
            else:
                 # Pass NaNs directly to models
                 full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Define all categorical features
    cat_features = ['street_name', 'violation_type', 'borough']
    if 'pavement_quality' in full_df.columns:
        cat_features.append('pavement_quality')
    
    for col in cat_features:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(ablation_name, use_median_ensemble, use_imputation_safeguard):
    """Runs a single training and validation experiment with specific configurations."""
    
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        use_imputation_safeguard=use_imputation_safeguard
    )

    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.3, random_state=42
    )

    # --- 3. Model Training ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Prepare data for XGBoost (it requires categorical columns to be of 'category' dtype)
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for col in cat_features:
        X_train_xgb[col] = X_train_xgb[col].astype('category')
        X_val_xgb[col] = X_val_xgb[col].astype('category')

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=1
    )
    # Redirect XGBoost output to prevent verbose printing
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))
    
    predictions_stack = np.stack([
        val_preds_cat_base,
        val_preds_cat_log_transformed,
        val_preds_xgb_log_transformed
    ], axis=1)

    if use_median_ensemble:
        ensemble_predictions = np.median(predictions_stack, axis=1)
    else: # Use mean
        ensemble_predictions = np.mean(predictions_stack, axis=1)

    ensemble_predictions = np.maximum(0, ensemble_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    
    print(f"{ablation_name} RMSE: {val_rmse:.4f}")
    return val_rmse

if __name__ == '__main__':
    create_dummy_data()
    
    results = {}

    # --- Baseline Experiment ---
    # Ensemble with Median, with imputation safeguard
    results['Baseline'] = run_experiment(
        "Baseline (Median Ensemble, Imputation Safeguard)",
        use_median_ensemble=True,
        use_imputation_safeguard=True
    )
    
    # --- Ablation 1: Change Ensemble Method ---
    # Ensemble with Mean
    results['No Median Ensemble'] = run_experiment(
        "Ablation: No Median Ensemble (use Mean)",
        use_median_ensemble=False,
        use_imputation_safeguard=True
    )

    # --- Ablation 2: Change Imputation Strategy ---
    # Remove safeguard, let models handle potential NaNs
    results['No Imputation Safeguard'] = run_experiment(
        "Ablation: No Imputation Safeguard",
        use_median_ensemble=True,
        use_imputation_safeguard=False
    )

    # --- Analysis ---
    baseline_rmse = results['Baseline']
    impacts = {
        'Median Ensemble Strategy': abs(results['No Median Ensemble'] - baseline_rmse),
        'Numerical Imputation Safeguard': abs(results['No Imputation Safeguard'] - baseline_rmse)
    }

    most_impactful_component = max(impacts, key=impacts.get)
    
    print(f"\n--- Ablation Study Summary ---")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Impact of 'Median Ensemble Strategy': {impacts['Median Ensemble Strategy']:.4f}")
    print(f"Impact of 'Numerical Imputation Safeguard': {impacts['Numerical Imputation Safeguard']:.4f}")
    print(f"\nMost impactful component: {most_impactful_component}")
    print(f"Final Validation Performance: {baseline_rmse:.4f}")
