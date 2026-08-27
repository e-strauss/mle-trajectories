
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_dummy_data():
    """Creates dummy data files for the script to run."""
    if os.path.exists('./input'):
        shutil.rmtree('./input')
    os.makedirs('./input', exist_ok=True)
    
    # Dummy main training data
    train_data = {
        'street_name': ['STREET_A', 'STREET_B', 'STREET_A', 'STREET_C', 'STREET_B', 'STREET_D', 'STREET_E'] * 10,
        'violation_description': ['PARKING', 'SPEEDING', 'SPEEDING', 'PARKING', 'RED_LIGHT', 'PARKING', 'SPEEDING'] * 10,
        'violation_count': [100, 50, 25, 200, 10, 150, 45, 90, 60, 30, 210, 5, 140, 40] * 5
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)
    
    # Dummy augmentation data
    boroughs_data = {
        'street_name': ['STREET_A', 'STREET_B', 'STREET_C', 'STREET_D'],
        'borough': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens']
    }
    pd.DataFrame(boroughs_data).to_csv('./input/street_names_and_boroughs.csv', index=False)
    
    physical_data = {
        'street_name': ['STREET_A', 'STREET_B', 'STREET_C', 'STREET_E'],
        'num_lanes': [4, 2, 4, 3],
        'speed_limit': [25, 30, 25, 35]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path):
    """Loads and preprocesses data for validation."""
    train_df = clean_col_names(pd.read_csv(train_path))
    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col != 'street_name']
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')
        
    return full_df, cat_features

def run_training_pipeline(ablation_config):
    """Runs the training and validation pipeline with a given configuration."""
    
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')

    # Apply log transformation based on config
    if ablation_config.get('use_log_transform') == 'log':
        # Add a small epsilon to avoid log(0) issues
        train_data['log_violation_count'] = np.log(train_data['violation_count'] + 1e-9)
    else: # Default to log1p
        train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- 3. Model Training ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_params = {
        'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 5, 'early_stopping_rounds': 50, 'enable_categorical': True,
        'tree_method': 'hist', 'random_state': 42,
        'n_jobs': ablation_config.get('xgb_n_jobs', -1) # Apply n_jobs from config
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    
    # Apply negative clipping based on config
    if not ablation_config.get('no_negative_clipping', False):
        ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

def main():
    """Main function to run the ablation study."""
    setup_dummy_data()
    
    ablation_results = {}
    
    # --- Baseline Experiment ---
    print("Running: Baseline")
    baseline_config = {}
    baseline_rmse = run_training_pipeline(baseline_config)
    ablation_results['Baseline'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}\n")
    
    # --- Ablation Experiments ---
    experiments = {
        "No Negative Clipping": {'no_negative_clipping': True},
        "Using np.log instead of np.log1p": {'use_log_transform': 'log'},
        "Single-Threaded XGBoost (n_jobs=1)": {'xgb_n_jobs': 1}
    }
    
    impacts = {}
    for name, config in experiments.items():
        print(f"Running: {name}")
        rmse = run_training_pipeline(config)
        ablation_results[name] = rmse
        impact = rmse - baseline_rmse
        impacts[name] = abs(impact)
        print(f"Resulting RMSE: {rmse:.4f}")
        print(f"Impact on RMSE: {impact:+.4f}\n")
        
    # --- Conclusion ---
    if impacts:
        most_impactful_component = max(impacts, key=impacts.get)
        print(f"The most impactful component is '{most_impactful_component}', as modifying it caused the largest change in RMSE.")
    else:
        print("No ablation studies were run to compare.")

    # Clean up dummy files
    shutil.rmtree('./input')

if __name__ == '__main__':
    main()
