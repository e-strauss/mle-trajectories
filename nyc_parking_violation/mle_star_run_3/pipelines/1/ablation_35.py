
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path):
    """Loads and preprocesses training data."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError:
        print("Warning: Augmentation data not found, proceeding without it.")
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
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(experiment_name, cat_loss, xgb_objective, early_stopping_rounds):
    """Runs a single training and validation experiment with a given configuration."""
    print(f"\n--- Running Experiment: {experiment_name} ---")
    start_time = time.time()
    
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
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
    # Model 1 (CatBoost Base)
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': cat_loss,
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'task_type': 'CPU', 'depth': 10
    }
    if early_stopping_rounds is not None:
        cat_params['early_stopping_rounds'] = early_stopping_rounds

    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=(early_stopping_rounds is not None))
    
    # Model 2 (CatBoost Log)
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=(early_stopping_rounds is not None))

    # Model 3 (XGBoost Log)
    xgb_params = {
        'objective': xgb_objective, 'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 5, 'enable_categorical': True, 'tree_method': 'hist',
        'random_state': 42, 'n_jobs': -1
    }
    if early_stopping_rounds is not None:
        xgb_params['early_stopping_rounds'] = early_stopping_rounds
        
    xgb_model = xgb.XGBRegressor(**xgb_params)
    eval_set = [(X_val, y_val_log)] if early_stopping_rounds is not None else []
    xgb_model.fit(X_train, y_train_log, eval_set=eval_set, verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))

    X_meta_val = np.column_stack((
        val_preds_cat_base, val_preds_cat_log_transformed, val_preds_xgb_log_transformed
    ))
    
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 100))
    meta_model.fit(X_meta_val, y_val_base)

    ensemble_predictions = meta_model.predict(X_meta_val)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    
    end_time = time.time()
    print(f"Validation RMSE: {val_rmse:.4f} (took {end_time - start_time:.2f}s)")
    return val_rmse

if __name__ == '__main__':
    results = {}

    # Baseline experiment
    baseline_rmse = run_experiment(
        experiment_name="Baseline",
        cat_loss='RMSE',
        xgb_objective='reg:squarederror',
        early_stopping_rounds=50
    )
    results["Baseline"] = baseline_rmse

    # Ablation 1: Change CatBoost Loss Function
    # We change the loss to Quantile to predict the median, which can be more robust to outliers.
    ablation1_rmse = run_experiment(
        experiment_name="Ablation: CatBoost with Quantile Loss",
        cat_loss='Quantile:alpha=0.5',
        xgb_objective='reg:squarederror',
        early_stopping_rounds=50
    )
    results["CatBoost Quantile Loss"] = ablation1_rmse

    # Ablation 2: Disable Early Stopping
    # We remove early stopping to see if models overfit and how much it affects performance.
    ablation2_rmse = run_experiment(
        experiment_name="Ablation: No Early Stopping",
        cat_loss='RMSE',
        xgb_objective='reg:squarederror',
        early_stopping_rounds=None
    )
    results["No Early Stopping"] = ablation2_rmse
    
    # --- Summary ---
    print("\n--- Ablation Study Summary ---")
    impacts = {}
    print(f"Baseline RMSE: {results['Baseline']:.4f}")

    # Calculate impact of Ablation 1
    impact1 = results["CatBoost Quantile Loss"] - results["Baseline"]
    impacts["CatBoost Quantile Loss"] = abs(impact1)
    print(f"Impact of CatBoost Quantile Loss: {results['CatBoost Quantile Loss']:.4f} (Change: {impact1:+.4f})")

    # Calculate impact of Ablation 2
    impact2 = results["No Early Stopping"] - results["Baseline"]
    impacts["No Early Stopping"] = abs(impact2)
    print(f"Impact of No Early Stopping: {results['No Early Stopping']:.4f} (Change: {impact2:+.4f})")

    # Determine the most impactful component
    most_impactful_component = max(impacts, key=impacts.get)
    print(f"\nConclusion: The most impactful component is '{most_impactful_component}'.")
