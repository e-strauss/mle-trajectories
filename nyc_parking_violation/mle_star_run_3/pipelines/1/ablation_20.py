
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

def create_dummy_data():
    """Creates dummy CSV files for the script to run."""
    print("Creating dummy data files for ablation study...")
    os.makedirs('./input', exist_ok=True)
    
    # Main training data
    train_data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'BROADWAY', 'MAIN ST', 'OAK ST', 'MAPLE AVE', 'BROADWAY', 'PARK AVE'],
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'DOUBLE PARKING', 'DOUBLE PARKING'],
        'Violation Count': [120, 45, 95, 30, 88, 15, 25, 22]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)
    
    # Augmentation data
    borough_data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'BROADWAY', 'OAK ST', 'MAPLE AVE'],
        'Borough': ['Manhattan', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx']
    }
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'BROADWAY', 'OAK ST', 'MAPLE AVE'],
        'street_width': [30, 40, 35, 25, 28],
        'has_bike_lane': [1, 0, 1, 0, 1]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)
    print("Dummy data created.")


def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def run_experiment(ensemble_method='geometric', use_model_seed=True):
    """
    Runs a single training and evaluation experiment with specified configurations.
    
    Args:
        ensemble_method (str): 'geometric' for averaging in log-space (geometric mean) or
                               'arithmetic' for averaging in original space.
        use_model_seed (bool): If True, sets a fixed random_state for the models.
    """
    # --- 1. Data Preparation ---
    train_df = pd.read_csv('./input/violations_per_street_2022.csv')
    train_df = clean_col_names(train_df)
    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            median_val = full_df[col].median()
            if pd.isna(median_val): median_val = 0
            full_df[col].fillna(median_val, inplace=True)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')
    
    full_df['log_violation_count'] = np.log1p(full_df['violation_count'])
    features = [col for col in full_df.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = full_df[features]
    y_base = full_df['violation_count']
    y_log = full_df['log_violation_count']
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- 3. Model Training ---
    cat_seed = 42 if use_model_seed else None
    xgb_seed = 42 if use_model_seed else None

    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': cat_seed, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50, 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=xgb_seed, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    ensemble_predictions = None
    if ensemble_method == 'geometric':
        # Geometric Mean: Average predictions in log-space then transform back
        val_preds_cat_base = model_cat_base.predict(X_val)
        val_preds_cat_base_log = np.log1p(np.maximum(0, val_preds_cat_base))
        val_preds_cat_log = model_cat_log.predict(X_val)
        val_preds_xgb_log = xgb_model.predict(X_val)
        log_ensemble_preds = (val_preds_cat_base_log + val_preds_cat_log + val_preds_xgb_log) / 3.0
        ensemble_predictions = np.expm1(log_ensemble_preds)
    elif ensemble_method == 'arithmetic':
        # Arithmetic Mean: Transform all predictions to original space then average
        val_preds_cat_base = model_cat_base.predict(X_val)
        val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
        val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
        ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    
    ensemble_predictions = np.maximum(0, ensemble_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse


if __name__ == '__main__':
    create_dummy_data()
    
    print("\n--- Starting Ablation Study ---")
    
    # --- Baseline Experiment ---
    baseline_rmse = run_experiment(ensemble_method='geometric', use_model_seed=True)
    print(f"Baseline (Geometric Mean Ensemble, Fixed Model Seed) RMSE: {baseline_rmse:.4f}")

    # --- Ablation Experiments ---
    results = {}
    
    # Ablation 1: No Geometric Mean Ensemble (use Arithmetic Mean instead)
    ablation_arithmetic_rmse = run_experiment(ensemble_method='arithmetic', use_model_seed=True)
    results['No Geometric Mean Ensemble'] = ablation_arithmetic_rmse - baseline_rmse
    print(f"Ablation 'No Geometric Mean Ensemble' RMSE: {ablation_arithmetic_rmse:.4f} (Impact: {results['No Geometric Mean Ensemble']:+.4f})")
    
    # Ablation 2: No Fixed Model Seed
    ablation_noseed_rmse = run_experiment(ensemble_method='geometric', use_model_seed=False)
    results['No Fixed Model Seed'] = ablation_noseed_rmse - baseline_rmse
    print(f"Ablation 'No Fixed Model Seed' RMSE: {ablation_noseed_rmse:.4f} (Impact: {results['No Fixed Model Seed']:+.4f})")
    
    print("--- Ablation Study Conclusion ---")
    
    # Determine the most impactful component
    if not results:
        print("No ablation studies were run.")
    else:
        most_impactful_component = max(results, key=lambda k: abs(results[k]))
        impact_value = results[most_impactful_component]
        print(f"The most impactful component is '{most_impactful_component}', which changed the RMSE by {impact_value:+.4f}.")
    
    # Clean up dummy data
    shutil.rmtree('./input')
    print("\nCleaned up dummy data files.")

