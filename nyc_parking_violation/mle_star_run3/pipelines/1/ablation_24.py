
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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

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

def run_experiment(X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
                   ensemble_method='stacking', cat_log_depth=10, xgb_lr=0.05):
    """
    Trains and evaluates a single configuration of the model pipeline.
    """
    # --- Model 1 (CatBoost Base) ---
    cat_params_base = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params_base)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    # --- Model 2 (CatBoost Log) ---
    cat_params_log = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': cat_log_depth  # Ablation parameter
    }
    model_cat_log = CatBoostRegressor(**cat_params_log)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # --- Model 3 (XGBoost Log) ---
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=xgb_lr, # Ablation parameter
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))

    if ensemble_method == 'stacking':
        stacked_val_preds = np.c_[val_preds_cat_base, val_preds_cat_log_transformed, val_preds_xgb_log_transformed]
        alphas = np.logspace(-3, 3, 20)
        meta_model = RidgeCV(alphas=alphas)
        meta_model.fit(stacked_val_preds, y_val_base)
        ensemble_predictions = meta_model.predict(stacked_val_preds)
    elif ensemble_method == 'weighted_average':
        # Use weights that previously showed promise
        ensemble_predictions = (val_preds_cat_base * 0.5) + \
                               (val_preds_cat_log_transformed * 0.25) + \
                               (val_preds_xgb_log_transformed * 0.25)

    ensemble_predictions = np.maximum(0, ensemble_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

def main():
    """
    Main function to run the ablation study.
    Ablation targets:
    1. Ensemble Method: Stacking with RidgeCV vs. a simpler weighted average.
    2. CatBoost Architecture: Uniform hyperparameters vs. different depths for each task.
    3. XGBoost Learning Rate: The effect of a faster learning rate for the XGBoost component.
    """
    train_path = './input/violations_per_street_2022.csv'
    
    # --- 1. Data Preparation ---
    print("Loading and preparing data...")
    train_data, cat_features = load_and_prepare_data(train_path)
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )
    print("Data prepared. Starting experiments...\n")

    results = {}

    # --- Experiment 1: Baseline ---
    print("Running Baseline experiment (Stacking Ensemble, Uniform CatBoost Depth, XGBoost LR=0.05)...")
    baseline_rmse = run_experiment(X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
                                   ensemble_method='stacking', cat_log_depth=10, xgb_lr=0.05)
    results['Baseline'] = baseline_rmse
    print(f"  Validation RMSE: {baseline_rmse:.4f}\n")

    # --- Experiment 2: No Stacking (Weighted Average Ensemble) ---
    print("Running Ablation: No Stacking (Weighted Average Ensemble)...")
    ablation1_rmse = run_experiment(X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
                                    ensemble_method='weighted_average', cat_log_depth=10, xgb_lr=0.05)
    results['No Stacking (Weighted Avg)'] = ablation1_rmse
    print(f"  Validation RMSE: {ablation1_rmse:.4f}")
    print(f"  Impact: {ablation1_rmse - baseline_rmse:+.4f}\n")

    # --- Experiment 3: Differentiated CatBoost Depth ---
    print("Running Ablation: Differentiated CatBoost Depth (Log model depth=7)...")
    ablation2_rmse = run_experiment(X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
                                    ensemble_method='stacking', cat_log_depth=7, xgb_lr=0.05)
    results['Differentiated CatBoost Depth'] = ablation2_rmse
    print(f"  Validation RMSE: {ablation2_rmse:.4f}")
    print(f"  Impact: {ablation2_rmse - baseline_rmse:+.4f}\n")
    
    # --- Experiment 4: Higher XGBoost Learning Rate ---
    print("Running Ablation: Higher XGBoost Learning Rate (LR=0.1)...")
    ablation3_rmse = run_experiment(X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
                                    ensemble_method='stacking', cat_log_depth=10, xgb_lr=0.1)
    results['Higher XGBoost LR'] = ablation3_rmse
    print(f"  Validation RMSE: {ablation3_rmse:.4f}")
    print(f"  Impact: {ablation3_rmse - baseline_rmse:+.4f}\n")

    # --- Conclusion ---
    impacts = {
        'Stacking Ensemble': abs(results['No Stacking (Weighted Avg)'] - baseline_rmse),
        'Uniform CatBoost Depth': abs(results['Differentiated CatBoost Depth'] - baseline_rmse),
        'XGBoost Learning Rate': abs(results['Higher XGBoost LR'] - baseline_rmse)
    }

    if not impacts:
        print("Could not run ablation studies.")
        return

    most_impactful_component = max(impacts, key=impacts.get)
    
    print("--- Ablation Study Conclusion ---")
    print(f"The most impactful component was '{most_impactful_component}'.")
    print(f"Removing/altering it changed the validation RMSE by {impacts[most_impactful_component]:.4f}.")

if __name__ == '__main__':
    # Create dummy data if it doesn't exist for script execution
    if not os.path.exists('./input/violations_per_street_2022.csv'):
        print("Creating dummy input files...")
        os.makedirs('./input', exist_ok=True)
        dummy_train_data = {
            'Street Name': [f'Street {i}' for i in range(50)] * 2,
            'Violation Description': ['Fire Hydrant'] * 50 + ['No Parking'] * 50,
            'violation_count': np.random.randint(1, 500, 100)
        }
        pd.DataFrame(dummy_train_data).to_csv('./input/violations_per_street_2022.csv', index=False)
        
        dummy_borough_data = {
            'Street Name': [f'Street {i}' for i in range(50)],
            'Borough': np.random.choice(['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'], 50)
        }
        pd.DataFrame(dummy_borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)
        
        dummy_physical_data = {
            'Street Name': [f'Street {i}' for i in range(50)],
            'street_width': np.random.uniform(20, 50, 50),
            'pavement_quality': np.random.uniform(1, 10, 50)
        }
        pd.DataFrame(dummy_physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

    main()
