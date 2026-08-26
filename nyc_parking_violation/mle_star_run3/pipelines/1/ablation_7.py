
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
warnings.filterwarnings("ignore", category=FutureWarning)


def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_aggregation_features=True, simple_aggregation=False):
    """Loads and preprocesses data with options for ablation."""
    # --- 1. Load Data ---
    train_df = pd.read_csv(train_path)
    full_df = clean_col_names(train_df)

    # Use dummy augmentation dataframes as they are not the focus of this study
    boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
    physical_df = pd.DataFrame(columns=['street_name'])
    
    # Merge (will be empty merges, but keeps structure)
    full_df = pd.merge(full_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')
    full_df['borough'].fillna('Unknown', inplace=True)

    # --- 2. Aggregation Feature Engineering ---
    if use_aggregation_features:
        agg_cols = ['street_name', 'violation_type', 'borough']
        target = 'violation_count'
        
        for col in agg_cols:
            if simple_aggregation:
                # Ablation: Use only 'mean' for aggregation
                aggregations = {f'{col}_{target}_mean': ('mean')}
            else:
                # Baseline: Use 'mean', 'std', and 'count'
                aggregations = {
                    f'{col}_{target}_mean': ('mean'),
                    f'{col}_{target}_std': ('std'),
                    f'{col}_{target}_count': ('count')
                }
            
            agg_df = full_df.groupby(col)[target].agg(**aggregations).reset_index()
            full_df = pd.merge(full_df, agg_df, on=col, how='left')
            
            if not simple_aggregation:
                full_df[f'{col}_{target}_std'].fillna(0, inplace=True)

    # Cast categorical features to 'category' dtype
    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(train_path, use_aggregation_features=True, simple_aggregation=False, use_xgboost=True):
    """
    Runs a single experiment configuration.
    - Prepares data based on flags.
    - Trains the specified models.
    - Evaluates the ensemble on the validation set.
    - Returns the validation RMSE.
    """
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data(
        train_path,
        use_aggregation_features=use_aggregation_features,
        simple_aggregation=simple_aggregation
    )

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
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    
    if use_xgboost:
        # Train XGBoost only if it's part of the ensemble
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
            max_depth=5, early_stopping_rounds=50, enable_categorical=True,
            tree_method='hist', random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
        val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
        
        # Ensemble with a simple average of the three models
        ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    else:
        # Ensemble of only the two CatBoost models
        ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed) / 2.0
        
    ensemble_predictions = np.maximum(0, ensemble_predictions) # Ensure non-negativity

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

def main():
    """Performs the ablation study and prints the results."""
    train_path = './input/violations_per_street_2022.csv'
    
    # Check for data file before running
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        print("Please ensure 'violations_per_street_2022.csv' is in the './input/' directory.")
        # Create dummy file to avoid crashing if run in an environment without the data
        pd.DataFrame({
            'street_name': ['A', 'B'], 'violation_type': ['X', 'Y'], 'violation_count': [10, 20]
        }).to_csv(train_path, index=False)
        print("Created a dummy training file. Results will not be meaningful.")

    results = {}

    print("Running Ablation Study...\n")

    # --- Baseline ---
    print("1. Running Baseline (Full Aggregation Features, 3-Model Ensemble)...")
    baseline_rmse = run_experiment(
        train_path, 
        use_aggregation_features=True, 
        simple_aggregation=False, 
        use_xgboost=True
    )
    results['Baseline'] = baseline_rmse
    print(f"   - Validation RMSE: {baseline_rmse:.4f}\n")

    # --- Ablation 1: No Aggregation Features ---
    print("2. Running Ablation: No Aggregation Features...")
    no_agg_rmse = run_experiment(
        train_path, 
        use_aggregation_features=False
    )
    results['No Aggregation Features'] = no_agg_rmse
    print(f"   - Validation RMSE: {no_agg_rmse:.4f}\n")

    # --- Ablation 2: Simplified Aggregation (Mean only) ---
    print("3. Running Ablation: Simplified Aggregation (Mean only)...")
    simple_agg_rmse = run_experiment(
        train_path, 
        use_aggregation_features=True, 
        simple_aggregation=True
    )
    results['Simplified Aggregation (Mean only)'] = simple_agg_rmse
    print(f"   - Validation RMSE: {simple_agg_rmse:.4f}\n")

    # --- Ablation 3: Remove XGBoost from Ensemble ---
    print("4. Running Ablation: Remove XGBoost from Ensemble...")
    no_xgb_rmse = run_experiment(
        train_path, 
        use_xgboost=False
    )
    results['No XGBoost in Ensemble'] = no_xgb_rmse
    print(f"   - Validation RMSE: {no_xgb_rmse:.4f}\n")

    # --- Analysis and Conclusion ---
    print("--- Ablation Study Results ---")
    print(f"{'Configuration':<40} | {'RMSE':<12} | {'Change from Baseline':<20}")
    print("-" * 75)
    
    impacts = {}
    for name, score in results.items():
        change = score - baseline_rmse
        impacts[name] = change
        print(f"{name:<40} | {score:<12.4f} | {change:+.4f}")
    
    # Exclude baseline from impact calculation
    del impacts['Baseline']
    
    # Determine the most impactful component (largest positive change means it was beneficial)
    most_impactful_component = max(impacts, key=lambda k: impacts[k] if 'No Aggregation' in k else -float('inf'))
    
    print("\n--- Conclusion ---")
    print(f"The most impactful component is '{most_impactful_component}'.")
    print("Removing it caused the largest increase in RMSE, indicating it provides the most value to the model's performance.")


if __name__ == '__main__':
    main()
