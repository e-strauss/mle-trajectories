
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def create_dummy_files():
    """Creates dummy CSV files for the script to run."""
    os.makedirs('./input', exist_ok=True)
    
    # Main training data
    train_data = {
        'Street Name': ['Main St', 'Main St', 'Oak Ave', 'Oak Ave', 'Pine St', 'Pine St', 'Maple Dr', 'Maple Dr', 'Elm St', 'Elm St'],
        'Violation Description': ['Parking', 'Speeding', 'Parking', 'Speeding', 'Parking', 'Speeding', 'Parking', 'Speeding', 'Parking', 'Speeding'],
        'Violation Count': [150, 50, 200, 75, 120, 40, 300, 110, 80, 25]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    # Augmentation data 1
    borough_data = {
        'Street Name': ['Main St', 'Oak Ave', 'Pine St', 'Maple Dr', 'Elm St'],
        'Borough': ['Uptown', 'Downtown', 'Uptown', 'Midtown', np.nan]
    }
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)
    
    # Augmentation data 2 - with mixed dtypes and missing values
    physical_data = {
        'Street Name': ['Main St', 'Oak Ave', 'Pine St', 'Maple Dr', 'Cedar Rd'],
        'Street Width': [12.5, 15.0, 12.5, np.nan, 10.0],
        'Pavement Quality': ['Good', 'Fair', 'Good', 'Excellent', 'Poor'],
        'Has Bike Lane': [1, 0, np.nan, 1, 0]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

def run_experiment(use_optimized_weights=True, use_differentiated_imputation=True):
    """
    Runs a single experiment with a specific configuration.
    Args:
        use_optimized_weights (bool): If True, use scipy.optimize to find ensemble weights. If False, use a simple average.
        use_differentiated_imputation (bool): If True, use dtype to decide imputation method for physical features. If False, treat all as numeric.
    Returns:
        float: The validation RMSE for the experiment.
    """

    # --- Data Loading and Preparation ---
    def clean_col_names(df):
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if 'violation_description' in df.columns:
            df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
        return df

    def load_and_prepare_data(train_path):
        train_df = clean_col_names(pd.read_csv(train_path))
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

        full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
        full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

        cat_features = ['street_name', 'violation_type', 'borough']
        full_df['borough'].fillna('Unknown', inplace=True)

        physical_cols_to_process = [col for col in physical_df.columns if col != 'street_name']
        
        # ABLATION POINT 1: Differentiated Imputation
        if use_differentiated_imputation:
            # Baseline: Differentiate imputation based on column dtype
            for col in physical_cols_to_process:
                if col in full_df.columns:
                    if full_df[col].dtype == 'object':
                        full_df[col].fillna('Unknown', inplace=True)
                        if col not in cat_features: cat_features.append(col)
                    else: # Assume numeric
                        median_val = full_df[col].median()
                        full_df[col].fillna(median_val, inplace=True)
                        full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
        else:
            # Ablation: Treat all physical features as numeric (simpler old logic)
            for col in physical_cols_to_process:
                if col in full_df.columns:
                    # FIX: Convert to numeric BEFORE calculating median to avoid TypeError
                    numeric_col = pd.to_numeric(full_df[col], errors='coerce')
                    median_val = numeric_col.median()
                    
                    # If median is NaN (e.g., column was all strings), default to 0
                    fill_value = median_val if pd.notna(median_val) else 0
                    
                    full_df[col] = numeric_col.fillna(fill_value)

        for col in cat_features:
            if col in full_df.columns:
                full_df[col] = full_df[col].astype('category')

        return full_df, cat_features

    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- Model Training (remains the same) ---
    cat_params = {
        'iterations': 100, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'random_seed': 42,
        'verbose': 0, 'cat_features': cat_features, 'early_stopping_rounds': 10, 'depth': 6
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Convert dataframe for XGBoost
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for col in cat_features:
        if col in X_train_xgb.columns:
            X_train_xgb[col] = X_train_xgb[col].astype("category")
            X_val_xgb[col] = X_val_xgb[col].astype("category")

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5,
        early_stopping_rounds=10, enable_categorical=True, tree_method='hist', random_state=42, n_jobs=1
    )
    xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)

    # --- Ensembling and Evaluation ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))

    val_predictions_stacked = np.column_stack([val_preds_cat_base, val_preds_cat_log_transformed, val_preds_xgb_log_transformed])

    # ABLATION POINT 2: Optimized vs. Simple Averaging
    if use_optimized_weights:
        # Baseline: Find optimal weights
        def rmse_objective(weights):
            weighted_preds = np.average(val_predictions_stacked, axis=1, weights=weights)
            return np.sqrt(mean_squared_error(y_val_base, weighted_preds))

        result = minimize(rmse_objective, np.array([1/3]*3), method='SLSQP', bounds=[(0,1)]*3, constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}))
        optimal_weights = result.x
    else:
        # Ablation: Use simple average weights
        optimal_weights = np.array([1/3, 1/3, 1/3])

    ensemble_predictions = np.average(val_predictions_stacked, axis=1, weights=optimal_weights)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

if __name__ == '__main__':
    create_dummy_files()

    print("--- Running Ablation Study ---")
    
    # 1. Baseline Experiment
    baseline_rmse = run_experiment(use_optimized_weights=True, use_differentiated_imputation=True)
    print(f"Baseline RMSE (Optimized Weights, Differentiated Imputation): {baseline_rmse:.4f}")
    print(f"Final Validation Performance: {baseline_rmse}")

    # 2. Ablation: No Optimized Weights
    no_opt_rmse = run_experiment(use_optimized_weights=False, use_differentiated_imputation=True)
    print(f"Ablation 'No Optimized Weights' RMSE: {no_opt_rmse:.4f}")

    # 3. Ablation: No Differentiated Imputation
    no_diff_imp_rmse = run_experiment(use_optimized_weights=True, use_differentiated_imputation=False)
    print(f"Ablation 'No Differentiated Imputation' RMSE: {no_diff_imp_rmse:.4f}")

    print("\n--- Ablation Analysis ---")
    impact_optimization = abs(no_opt_rmse - baseline_rmse)
    impact_imputation = abs(no_diff_imp_rmse - baseline_rmse)

    print(f"Impact of 'Optimized Weights': {impact_optimization:.4f}")
    print(f"Impact of 'Differentiated Imputation': {impact_imputation:.4f}")

    if impact_optimization > impact_imputation:
        most_impactful = "Optimized Weights"
    elif impact_imputation > impact_optimization:
        most_impactful = "Differentiated Imputation"
    else:
        most_impactful = "Both components had an equal impact"

    print(f"\nConclusion: The most impactful component is '{most_impactful}'.")

    # Clean up dummy files
    shutil.rmtree('./input')
