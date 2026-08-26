
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import copy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path):
    """Loads, preprocesses, and prepares data for all models."""
    # --- 1. Load Data ---
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    # Assume augmentation data is present and correct this time for a stable baseline
    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError:
        # In case the files are still missing, create dummy ones to avoid crashing
        print("Warning: Augmentation data not found. Creating dummy files to proceed.")
        if not os.path.exists('./input'):
            os.makedirs('./input')
        pd.DataFrame(columns=['street_name', 'borough']).to_csv('./input/street_names_and_boroughs.csv', index=False)
        pd.DataFrame(columns=['street_name']).to_csv('./input/physical_features_per_street.csv', index=False)
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    # --- 2. Feature Engineering & Merging ---
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

    # Cast categorical features to 'category' dtype
    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_ablation_study():
    """Performs an ablation study on model components and hyperparameters."""
    
    # --- 1. Data Preparation ---
    print("--- Ablation Study ---")
    print("Loading and preparing data...")
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')

    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    results = {}

    # --- 2. Baseline Model (Full Ensemble) ---
    print("\n1. Training Baseline (Full 3-Model Ensemble)...")
    base_cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**base_cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    model_cat_log = CatBoostRegressor(**base_cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log = np.expm1(xgb_model.predict(X_val))
    
    ensemble_preds = (val_preds_cat_base + val_preds_cat_log + val_preds_xgb_log) / 3.0
    baseline_rmse = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds)))
    results['Baseline (Full Ensemble)'] = baseline_rmse
    print(f"   > Baseline RMSE: {baseline_rmse:.4f}")

    # --- 3. Ablation 1: Remove XGBoost Model ---
    # Test the contribution of model diversity. Is XGBoost helping or hurting?
    print("\n2. Ablation: Remove XGBoost (Ensemble of CatBoost models only)...")
    ensemble_preds_no_xgb = (val_preds_cat_base + val_preds_cat_log) / 2.0
    ablation_1_rmse = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds_no_xgb)))
    results['Ablation (No XGBoost)'] = ablation_1_rmse
    print(f"   > RMSE without XGBoost: {ablation_1_rmse:.4f}")

    # --- 4. Ablation 2: Remove Base Target Model ---
    # Test the contribution of the model trained on the original scale.
    print("\n3. Ablation: Remove Base-Target Model (Ensemble of log-target models only)...")
    ensemble_preds_log_only = (val_preds_cat_log + val_preds_xgb_log) / 2.0
    ablation_2_rmse = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds_log_only)))
    results['Ablation (Log-Target Models Only)'] = ablation_2_rmse
    print(f"   > RMSE with log-target models only: {ablation_2_rmse:.4f}")

    # --- 5. Ablation 3: Reduce CatBoost Tree Depth ---
    # Test if the deep trees (depth=10) are overfitting.
    print("\n4. Ablation: Reduce CatBoost Tree Depth (10 -> 6)...")
    shallow_cat_params = copy.deepcopy(base_cat_params)
    shallow_cat_params['depth'] = 6
    
    model_cat_base_shallow = CatBoostRegressor(**shallow_cat_params)
    model_cat_base_shallow.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    model_cat_log_shallow = CatBoostRegressor(**shallow_cat_params)
    model_cat_log_shallow.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    val_preds_cat_base_shallow = model_cat_base_shallow.predict(X_val)
    val_preds_cat_log_shallow = np.expm1(model_cat_log_shallow.predict(X_val))
    
    ensemble_preds_shallow = (val_preds_cat_base_shallow + val_preds_cat_log_shallow + val_preds_xgb_log) / 3.0
    ablation_3_rmse = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds_shallow)))
    results['Ablation (Shallow CatBoost depth=6)'] = ablation_3_rmse
    print(f"   > RMSE with shallow CatBoost models: {ablation_3_rmse:.4f}")

    # --- 6. Conclusion ---
    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name:<35}: {score:.4f}")
    
    baseline_score = results['Baseline (Full Ensemble)']
    impacts = {
        'XGBoost Model': abs(results['Ablation (No XGBoost)'] - baseline_score),
        'Base-Target CatBoost Model': abs(results['Ablation (Log-Target Models Only)'] - baseline_score),
        'CatBoost Tree Depth (10 vs 6)': abs(results['Ablation (Shallow CatBoost depth=6)'] - baseline_score)
    }
    
    most_impactful_component = max(impacts, key=impacts.get)
    
    print("\n--- Final Conclusion ---")
    print(f"The component that contributes the most to the overall performance is the '{most_impactful_component}'.")
    print("Removing it or changing it resulted in the largest change to the validation RMSE, indicating its significant role in the model's predictive power.")


if __name__ == '__main__':
    run_ablation_study()
