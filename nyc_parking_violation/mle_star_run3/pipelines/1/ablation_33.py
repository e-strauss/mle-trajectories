
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import collections

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def setup_dummy_data():
    """Creates dummy CSV files for a self-contained execution."""
    os.makedirs('./input', exist_ok=True)
    
    train_data = {
        'Street Name': [f'Street {i}' for i in range(1, 6)] * 2,
        'Violation Description': ['No Parking'] * 5 + ['Double Parking'] * 5,
        'Violation Count': [10, 20, 15, 5, 30, 8, 18, 12, 3, 25]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    boroughs_data = {
        'street_name': [f'Street {i}' for i in range(1, 6)],
        'borough': ['Brooklyn', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx']
    }
    pd.DataFrame(boroughs_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data = {
        'street_name': [f'Street {i}' for i in range(1, 6)],
        'width': [30, 50, 35, 40, 45],
        'length': [100, 200, 150, 120, 180]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_target_encoding=True, smoothing_factor=20):
    """Loads, preprocesses, and prepares data, with toggles for ablation."""
    # --- 1. Load Data ---
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    # --- 2. Feature Engineering & Merging ---
    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    # --- Smoothed Target Encoding (Ablation Target) ---
    all_cat_features = ['street_name', 'violation_type', 'borough']
    
    if use_target_encoding:
        global_mean = full_df['violation_count'].mean()
        encoding_maps = {}

        for col in all_cat_features:
            agg = full_df.groupby(col)['violation_count'].agg(['mean', 'count'])
            # Use smoothing_factor from function argument
            agg['smoothed'] = (agg['count'] * agg['mean'] + smoothing_factor * global_mean) / (agg['count'] + smoothing_factor)
            encoding_map = agg['smoothed'].to_dict()
            encoding_maps[col] = encoding_map
            full_df[f'{col}_target_enc'] = full_df[col].map(encoding_map)

    # Cast original categorical features to 'category' dtype
    for col in all_cat_features:
        full_df[col] = full_df[col].astype('category')
        
    return full_df, all_cat_features

def run_experiment(description, use_target_encoding=True, smoothing_factor=20):
    """Runs a single training and validation experiment with a given configuration."""
    print(f"\n--- Running Experiment: {description} ---")

    # --- 1. Data Preparation ---
    train_data, cat_features_list = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        use_target_encoding=use_target_encoding,
        smoothing_factor=smoothing_factor
    )
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    # Define features based on whether target encoding was used
    if use_target_encoding:
        features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    else: # If no TE, don't include TE features and rely on original categoricals
        features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count'] and '_target_enc' not in col]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- 3. Model Training ---
    # Model 1 (CatBoost Base)
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features_list, 'early_stopping_rounds': 50,
        'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    # Model 2 (CatBoost Log)
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Model 3 (XGBoost Log)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Validation RMSE: {val_rmse:.4f}")
    return val_rmse

def ablation_study():
    """Performs an ablation study on target encoding and smoothing."""
    setup_dummy_data()
    results = collections.OrderedDict()

    # Baseline: Full model with smoothed target encoding
    results['Baseline (Smoothed Target Encoding)'] = run_experiment(
        "Baseline (Smoothed Target Encoding)",
        use_target_encoding=True,
        smoothing_factor=20
    )

    # Ablation 1: Disable target encoding entirely
    results['No Target Encoding'] = run_experiment(
        "No Target Encoding",
        use_target_encoding=False
    )
    
    # Ablation 2: Use target encoding but without smoothing
    results['No Smoothing in Target Encoding'] = run_experiment(
        "No Smoothing in Target Encoding",
        use_target_encoding=True,
        smoothing_factor=0
    )

    # --- Analysis ---
    baseline_rmse = results['Baseline (Smoothed Target Encoding)']
    impact_no_te = abs(results['No Target Encoding'] - baseline_rmse)
    impact_no_smoothing = abs(results['No Smoothing in Target Encoding'] - baseline_rmse)
    
    print("\n\n--- Ablation Study Summary ---")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Impact of removing Target Encoding: {results['No Target Encoding'] - baseline_rmse:+.4f} (New RMSE: {results['No Target Encoding']:.4f})")
    print(f"Impact of removing Smoothing: {results['No Smoothing in Target Encoding'] - baseline_rmse:+.4f} (New RMSE: {results['No Smoothing in Target Encoding']:.4f})")

    impacts = {
        'Target Encoding': impact_no_te,
        'Smoothing in Target Encoding': impact_no_smoothing
    }
    
    most_impactful_component = max(impacts, key=impacts.get)
    
    print(f"\nConclusion: The most impactful component is '{most_impactful_component}'.")

if __name__ == '__main__':
    ablation_study()
