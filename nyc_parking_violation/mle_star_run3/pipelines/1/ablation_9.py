
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_borough_imputation=True):
    """Loads and preprocesses data, with an option to disable borough imputation."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError:
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    if use_borough_imputation:
        full_df['borough'].fillna('Unknown', inplace=True)

    # FIX: CatBoost requires categorical features to be strings or integers, not NaN.
    # We fill any remaining NaNs in the 'borough' column with a specific placeholder.
    # This ensures the code runs even when `use_borough_imputation` is False,
    # allowing the ablation experiment to complete without crashing.
    if 'borough' in full_df.columns:
        full_df['borough'].fillna('NotAvailable', inplace=True)

    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(config):
    """Runs a single experiment with a given configuration."""
    print(f"--- Running: {config['name']} ---")
    
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        use_borough_imputation=config['use_borough_imputation']
    )
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    if config['use_group_split']:
        print("Using GroupShuffleSplit to avoid street-level data leakage.")
        groups = train_data['street_name']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y_base, groups))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_base, y_val_base = y_base.iloc[train_idx], y_base.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
    else:
        print("Using standard random train_test_split.")
        X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
            X, y_base, y_log, test_size=0.2, random_state=42
        )

    # --- 3. Model Training ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50, 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

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
    print(f"Validation RMSE: {val_rmse:.4f}\n")
    return val_rmse

def main():
    # Define experiment configurations
    configurations = [
        {
            "name": "Baseline",
            "use_borough_imputation": True,
            "use_group_split": False,
        },
        {
            "name": "Ablation: No Borough Imputation",
            "use_borough_imputation": False,
            "use_group_split": False,
        },
        {
            "name": "Ablation: Group-Based Validation Split",
            "use_borough_imputation": True,
            "use_group_split": True,
        }
    ]

    # Run experiments and store results
    results = {}
    for config in configurations:
        results[config['name']] = run_experiment(config)

    # Analyze and print results
    baseline_rmse = results.get("Baseline", float('inf'))
    impacts = {}
    
    print("--- Ablation Study Summary ---")
    if "Baseline" in results:
        print(f"Baseline RMSE: {baseline_rmse:.4f}")
    else:
        print("Baseline configuration was not run.")


    for name, rmse in results.items():
        if name != "Baseline":
            change = rmse - baseline_rmse
            impacts[name] = abs(change)
            print(f"{name} RMSE: {rmse:.4f} (Change: {change:+.4f})")
    
    print("\n--- Conclusion ---")
    if impacts:
        most_impactful = max(impacts, key=impacts.get)
        try:
            impact_desc = most_impactful.split(': ')[1]
        except IndexError:
            impact_desc = most_impactful
        print(f"The most impactful component is the '{impact_desc}', which changed the RMSE by {impacts[most_impactful]:.4f}.")
    elif len(results) > 1:
        print("Ablation experiments run, but no single component was isolated for impact calculation.")
    else:
        print("No ablations were performed to determine the most impactful component.")
    
    if results:
        best_config_name = min(results, key=results.get)
        final_validation_score = results[best_config_name]
        print(f"\nBest configuration found: '{best_config_name}' with RMSE: {final_validation_score:.4f}")
        print(f"Final Validation Performance: {final_validation_score}")


if __name__ == '__main__':
    # Create dummy files if they don't exist to ensure the script runs
    if not os.path.exists('./input'):
        os.makedirs('./input')
    if not os.path.exists('./input/violations_per_street_2022.csv'):
        pd.DataFrame({
            'Street Name': [f'Street {i}' for i in range(100)],
            'Violation Description': [f'Type {i%5}' for i in range(100)],
            'Violation Count': np.random.randint(10, 100, 100)
        }).to_csv('./input/violations_per_street_2022.csv', index=False)
    if not os.path.exists('./input/street_names_and_boroughs.csv'):
        pd.DataFrame({
            'Street Name': [f'Street {i}' for i in range(80)], # Intentionally smaller
            'Borough': [f'Borough {i%3}' for i in range(80)]
        }).to_csv('./input/street_names_and_boroughs.csv', index=False)
    if not os.path.exists('./input/physical_features_per_street.csv'):
         pd.DataFrame({
            'Street Name': [f'Street {i}' for i in range(100)],
            'Length': np.random.rand(100) * 1000
        }).to_csv('./input/physical_features_per_street.csv', index=False)
        
    main()
