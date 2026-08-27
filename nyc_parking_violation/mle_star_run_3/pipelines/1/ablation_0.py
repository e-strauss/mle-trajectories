
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Data Preparation Functions (Modified for Ablation) ---

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_augmentation=True):
    """Loads and preprocesses data. Ablation flag controls external data usage."""
    # --- 1. Load Base Data ---
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)
    
    full_df = train_df
    cat_features = ['street_name', 'violation_type']

    # --- 2. Feature Engineering & Merging (Ablation Point) ---
    if use_augmentation:
        try:
            boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
            physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
        except FileNotFoundError:
            print("Warning: Augmentation data not found, skipping augmentation.")
            return load_and_prepare_data(train_path, use_augmentation=False) # Rerun without augmentation

        full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
        full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

        # Impute missing values from augmented data
        full_df['borough'].fillna('Unknown', inplace=True)
        numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
        for col in numerical_cols:
            if col in full_df.columns:
                median_val = full_df[col].median()
                full_df[col].fillna(median_val, inplace=True)
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
        
        cat_features.append('borough')

    # Cast categorical features
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

# --- Main Ablation Study Execution ---

def run_ablation_study():
    """Performs an ablation study on the model pipeline."""
    
    train_path = './input/violations_per_street_2022.csv'
    # Check for required data file
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        print("Please ensure 'violations_per_street_2022.csv' is in the 'input' directory.")
        # Create dummy files for demonstration if they don't exist
        print("Creating dummy data files for demonstration purposes...")
        os.makedirs('./input', exist_ok=True)
        pd.DataFrame({
            'Street Name': ['A', 'B', 'C', 'A'], 
            'Violation Description': ['V1', 'V1', 'V2', 'V2'], 
            'violation_count': [10, 20, 5, 15]
        }).to_csv(train_path, index=False)
        pd.DataFrame({
            'Street Name': ['A', 'B'], 'Borough': ['MANHATTAN', 'BRONX']
        }).to_csv('./input/street_names_and_boroughs.csv', index=False)
        pd.DataFrame({
            'Street Name': ['A', 'C'], 'street_width': [30.5, 25.0]
        }).to_csv('./input/physical_features_per_street.csv', index=False)
        print("Dummy files created. Rerunning study...")

    results = {}

    # --- Experiment 1: Baseline (Full Model) ---
    print("--- Running Baseline: Full Ensemble with Data Augmentation ---")
    train_data, cat_features = load_and_prepare_data(train_path, use_augmentation=True)
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # Model Parameters
    cat_params = {'iterations': 500, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0, 'cat_features': cat_features, 'early_stopping_rounds': 50}
    xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'early_stopping_rounds': 50, 'enable_categorical': True, 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1}

    # Train Models
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # Evaluate Individual and Ensemble Models
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log = np.expm1(xgb_model.predict(X_val))
    
    ensemble_preds = (val_preds_cat_base + val_preds_cat_log + val_preds_xgb_log) / 3.0
    ensemble_preds = np.maximum(0, ensemble_preds)

    results['Baseline (Full Ensemble)'] = np.sqrt(mean_squared_error(y_val_base, ensemble_preds))
    results['Single Model (CatBoost on Base Target)'] = np.sqrt(mean_squared_error(y_val_base, val_preds_cat_base))
    results['Single Model (CatBoost on Log Target)'] = np.sqrt(mean_squared_error(y_val_base, val_preds_cat_log))
    results['Single Model (XGBoost on Log Target)'] = np.sqrt(mean_squared_error(y_val_base, val_preds_xgb_log))
    
    print(f"Baseline Validation RMSE: {results['Baseline (Full Ensemble)']:.4f}\n")


    # --- Experiment 2: No Data Augmentation ---
    print("--- Running Ablation: No Data Augmentation ---")
    train_data_no_aug, cat_features_no_aug = load_and_prepare_data(train_path, use_augmentation=False)
    
    train_data_no_aug['log_violation_count'] = np.log1p(train_data_no_aug['violation_count'])
    features_no_aug = [col for col in train_data_no_aug.columns if col not in ['violation_count', 'log_violation_count']]
    
    X_no_aug = train_data_no_aug[features_no_aug]
    y_base_no_aug = train_data_no_aug['violation_count']
    y_log_no_aug = train_data_no_aug['log_violation_count']
    
    X_train_na, X_val_na, y_train_base_na, y_val_base_na, y_train_log_na, y_val_log_na = train_test_split(
        X_no_aug, y_base_no_aug, y_log_no_aug, test_size=0.2, random_state=42
    )

    # Update cat features for models
    cat_params['cat_features'] = cat_features_no_aug

    # Train Models
    model_cat_base_na = CatBoostRegressor(**cat_params)
    model_cat_base_na.fit(X_train_na, y_train_base_na, eval_set=(X_val_na, y_val_base_na), use_best_model=True)

    model_cat_log_na = CatBoostRegressor(**cat_params)
    model_cat_log_na.fit(X_train_na, y_train_log_na, eval_set=(X_val_na, y_val_log_na), use_best_model=True)

    xgb_model_na = xgb.XGBRegressor(**xgb_params)
    xgb_model_na.fit(X_train_na, y_train_log_na, eval_set=[(X_val_na, y_val_log_na)], verbose=False)

    # Evaluate Ensemble without augmentation
    preds_cat_base_na = model_cat_base_na.predict(X_val_na)
    preds_cat_log_na = np.expm1(model_cat_log_na.predict(X_val_na))
    preds_xgb_log_na = np.expm1(xgb_model_na.predict(X_val_na))
    
    ensemble_preds_na = (preds_cat_base_na + preds_cat_log_na + preds_xgb_log_na) / 3.0
    ensemble_preds_na = np.maximum(0, ensemble_preds_na)
    
    results['Ensemble (No Data Augmentation)'] = np.sqrt(mean_squared_error(y_val_base_na, ensemble_preds_na))
    print(f"Validation RMSE without Augmentation: {results['Ensemble (No Data Augmentation)']:.4f}\n")
    
    # --- Final Results and Conclusion ---
    print("--- Ablation Study Summary ---")
    for name, score in sorted(results.items(), key=lambda item: item[1]):
        print(f"Validation RMSE for {name}: {score:.4f}")

    # Determine most impactful component
    impact_of_augmentation = results['Ensemble (No Data Augmentation)'] - results['Baseline (Full Ensemble)']
    best_single_model_rmse = min(
        results['Single Model (CatBoost on Base Target)'], 
        results['Single Model (CatBoost on Log Target)'], 
        results['Single Model (XGBoost on Log Target)']
    )
    impact_of_ensemble = best_single_model_rmse - results['Baseline (Full Ensemble)']

    print("\n--- Conclusion ---")
    if impact_of_augmentation > impact_of_ensemble:
        print(f"Data augmentation is the most impactful component, improving the RMSE by {impact_of_augmentation:.4f}.")
    else:
        print(f"Ensembling is the most impactful component, improving the RMSE by {impact_of_ensemble:.4f} over the best single model.")

if __name__ == '__main__':
    run_ablation_study()
