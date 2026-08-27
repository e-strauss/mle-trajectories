
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

def load_and_prepare_data(train_path, use_median_imputation=True):
    """Loads, preprocesses, and prepares data."""
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
    
    if use_median_imputation:
        for col in numerical_cols:
            if col in full_df.columns:
                median_val = full_df[col].median()
                full_df[col].fillna(median_val, inplace=True)
    
    # Fallback for any remaining NaNs
    for col in numerical_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(description, X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_params_base, cat_params_log, xgb_params, cat_features):
    """Trains the ensemble and evaluates it on the validation set."""
    print(f"\n--- Running Experiment: {description} ---")
    
    # Model 1 (CatBoost Base)
    model_cat_base = CatBoostRegressor(**cat_params_base)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True, verbose=0)
    
    # Model 2 (CatBoost Log)
    model_cat_log = CatBoostRegressor(**cat_params_log)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True, verbose=0)
    
    # Model 3 (XGBoost Log)
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    
    # Ensembling
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)
    
    rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

def main():
    # --- 1. Data Preparation ---
    print("Preparing data for ablation study...")
    
    # Shared data preparation
    train_path = './input/violations_per_street_2022.csv'
    
    # Data for Baseline and "No L2" Ablation (uses median imputation)
    train_data_median, cat_features_median = load_and_prepare_data(train_path, use_median_imputation=True)
    train_data_median['log_violation_count'] = np.log1p(train_data_median['violation_count'])
    features_median = [col for col in train_data_median.columns if col not in ['violation_count', 'log_violation_count']]
    X_median = train_data_median[features_median]
    y_base_median = train_data_median['violation_count']
    y_log_median = train_data_median['log_violation_count']
    X_train_median, X_val_median, y_train_base_median, y_val_base_median, y_train_log_median, y_val_log_median = train_test_split(
        X_median, y_base_median, y_log_median, test_size=0.2, random_state=42
    )

    # Data for "No Median Imputation" Ablation
    train_data_no_median, cat_features_no_median = load_and_prepare_data(train_path, use_median_imputation=False)
    train_data_no_median['log_violation_count'] = np.log1p(train_data_no_median['violation_count'])
    features_no_median = [col for col in train_data_no_median.columns if col not in ['violation_count', 'log_violation_count']]
    X_no_median = train_data_no_median[features_no_median]
    y_base_no_median = train_data_no_median['violation_count']
    y_log_no_median = train_data_no_median['log_violation_count']
    X_train_no_median, X_val_no_median, y_train_base_no_median, y_val_base_no_median, y_train_log_no_median, y_val_log_no_median = train_test_split(
        X_no_median, y_base_no_median, y_log_no_median, test_size=0.2, random_state=42
    )
    
    # --- 2. Define Model Configurations ---
    base_cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'cat_features': cat_features_median,
        'early_stopping_rounds': 50, 'task_type': 'CPU', 'depth': 10
    }
    
    # Baseline configs with L2 regularization
    baseline_cat_params_base = {**base_cat_params, 'l2_leaf_reg': 5}
    baseline_cat_params_log = {**base_cat_params, 'l2_leaf_reg': 2}
    
    xgb_params = {
        'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 5, 'early_stopping_rounds': 50, 'enable_categorical': True,
        'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1
    }

    results = {}

    # --- 3. Run Experiments ---
    
    # Baseline Experiment
    baseline_rmse = run_experiment(
        "Baseline (Full Model with L2 Reg and Median Imputation)",
        X_train_median, X_val_median, y_train_base_median, y_val_base_median, y_train_log_median, y_val_log_median,
        baseline_cat_params_base, baseline_cat_params_log, xgb_params, cat_features_median
    )
    results['Baseline'] = baseline_rmse
    
    # Ablation 1: No L2 Regularization
    no_l2_cat_params = copy.deepcopy(base_cat_params) # Params without l2_leaf_reg
    ablation1_rmse = run_experiment(
        "Ablation 1: No L2 Regularization",
        X_train_median, X_val_median, y_train_base_median, y_val_base_median, y_train_log_median, y_val_log_median,
        no_l2_cat_params, no_l2_cat_params, xgb_params, cat_features_median
    )
    results['No_L2_Regularization'] = ablation1_rmse
    
    # Ablation 2: No Median Imputation
    # Use the data prepared without median imputation
    ablation2_rmse = run_experiment(
        "Ablation 2: No Median Imputation (uses 0-fill instead)",
        X_train_no_median, X_val_no_median, y_train_base_no_median, y_val_base_no_median, y_train_log_no_median, y_val_log_no_median,
        {**baseline_cat_params_base, 'cat_features': cat_features_no_median}, 
        {**baseline_cat_params_log, 'cat_features': cat_features_no_median},
        xgb_params,
        cat_features_no_median
    )
    results['No_Median_Imputation'] = ablation2_rmse
    
    # --- 4. Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    
    impact_l2 = results['No_L2_Regularization'] - results['Baseline']
    impact_imputation = results['No_Median_Imputation'] - results['Baseline']
    
    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    print(f"Impact of removing L2 Regularization: RMSE changed by {impact_l2:+.4f}")
    print(f"Impact of removing Median Imputation: RMSE changed by {impact_imputation:+.4f}")
    
    if abs(impact_l2) > abs(impact_imputation):
        print("\nConclusion: L2 Regularization is the most impactful component studied. Removing it caused the largest change in performance.")
    elif abs(impact_imputation) > abs(impact_l2):
        print("\nConclusion: The median imputation strategy is the most impactful component studied. Replacing it with 0-filling caused the largest change in performance.")
    else:
        print("\nConclusion: Both L2 regularization and the median imputation strategy had a similar impact on performance.")

if __name__ == '__main__':
    main()
