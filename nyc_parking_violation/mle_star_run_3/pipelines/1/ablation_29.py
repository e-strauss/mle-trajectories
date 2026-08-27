
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

def load_and_prepare_data(train_path):
    """Loads, preprocesses, and prepares data for all models."""
    input_dir = os.path.dirname(train_path)
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    # Load augmentation data
    try:
        boroughs_df = clean_col_names(pd.read_csv(os.path.join(input_dir,'street_names_and_boroughs.csv')))
        physical_df = clean_col_names(pd.read_csv(os.path.join(input_dir,'physical_features_per_street.csv')))
    except FileNotFoundError:
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])

    # Feature Engineering & Merging
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

def run_experiment(train_data, cat_features, all_models_on_log=False, xgb_enable_categorical=True, random_state_split=42):
    """
    Runs a single training and validation experiment with a specific configuration.
    """
    # Create log-transformed target
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]

    # Validation Split
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=random_state_split
    )

    # --- Model Training ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }

    # Model 1 (CatBoost): Target depends on the experiment config
    model_cat_1 = CatBoostRegressor(**cat_params)
    if all_models_on_log:
        model_cat_1.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)
    else:
        model_cat_1.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    # Model 2 (CatBoost Log)
    model_cat_2 = CatBoostRegressor(**cat_params)
    model_cat_2.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Model 3 (XGBoost Log)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50,
        enable_categorical=xgb_enable_categorical, # Ablation parameter
        tree_method='hist', random_state=42, n_jobs=-1
    )

    if not xgb_enable_categorical:
        # Manually encode categoricals if native support is disabled
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        for col in cat_features:
            if col in X_train_xgb.columns:
                # Use .cat.codes for integer encoding
                X_train_xgb[col] = X_train_xgb[col].cat.codes
                X_val_xgb[col] = X_val_xgb[col].cat.codes
        xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)
        val_preds_xgb_log = xgb_model.predict(X_val_xgb)
    else:
        # The .astype('category') already prepares data for native support
        xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
        val_preds_xgb_log = xgb_model.predict(X_val)


    # --- Validation Performance & Ensembling ---
    val_preds_cat_2_log = model_cat_2.predict(X_val)
    val_preds_cat_2_transformed = np.expm1(val_preds_cat_2_log)
    val_preds_xgb_transformed = np.expm1(val_preds_xgb_log)

    if all_models_on_log:
        val_preds_cat_1_log = model_cat_1.predict(X_val)
        val_preds_cat_1_transformed = np.expm1(val_preds_cat_1_log)
        ensemble_predictions = (val_preds_cat_1_transformed + val_preds_cat_2_transformed + val_preds_xgb_transformed) / 3.0
    else:
        val_preds_cat_1_base = model_cat_1.predict(X_val)
        ensemble_predictions = (val_preds_cat_1_base + val_preds_cat_2_transformed + val_preds_xgb_transformed) / 3.0

    ensemble_predictions = np.maximum(0, ensemble_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

if __name__ == '__main__':
    # Create dummy data for a self-contained script
    if not os.path.exists('./input'):
        os.makedirs('./input')
    
    pd.DataFrame({
        'Street Name': ['BROADWAY', 'WALL STREET', '5TH AVE', 'PARK AVE', 'MADISON AVE', 'BROADWAY', 'WALL STREET'],
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'DOUBLE PARKING', 'FIRE HYDRANT'],
        'Violation Count': [1500, 300, 1200, 800, 950, 400, 250]
    }).to_csv('./input/violations_per_street_2022.csv', index=False)

    pd.DataFrame({
        'Street Name': ['BROADWAY', 'WALL STREET', '5TH AVE', 'PARK AVE', 'MADISON AVE'],
        'Borough': ['Manhattan', 'Manhattan', 'Manhattan', 'Manhattan', 'Manhattan']
    }).to_csv('./input/street_names_and_boroughs.csv', index=False)

    pd.DataFrame({
        'Street Name': ['BROADWAY', 'WALL STREET', '5TH AVE', 'PARK AVE'],
        'pavement_quality': [8, 9, 8, 7],
        'street_width': [35.5, 22.0, 40.0, 38.0]
    }).to_csv('./input/physical_features_per_street.csv', index=False)
    
    # --- Ablation Study ---
    print("Starting ablation study...")
    
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')

    # Baseline Experiment
    baseline_rmse = run_experiment(train_data.copy(), cat_features)
    print(f"Final Validation Performance: {baseline_rmse}")


    # Ablation 1: Different Random State for Data Splitting
    ablation1_rmse = run_experiment(train_data.copy(), cat_features, random_state_split=101)

    # Ablation 2: All models trained on log-transformed target
    ablation2_rmse = run_experiment(train_data.copy(), cat_features, all_models_on_log=True)

    # Ablation 3: Disable native categorical feature support in XGBoost
    ablation3_rmse = run_experiment(train_data.copy(), cat_features, xgb_enable_categorical=False)
    
    results = {
        "Baseline": baseline_rmse,
        "No Fixed Split (random_state=101)": ablation1_rmse,
        "Ensemble: All Models on Log Target": ablation2_rmse,
        "No XGBoost Native Categoricals": ablation3_rmse
    }

    impact = {
        "Random State for Split": ablation1_rmse - baseline_rmse,
        "Ensemble Target Strategy": ablation2_rmse - baseline_rmse,
        "XGBoost Native Categoricals": ablation3_rmse - baseline_rmse,
    }

    print("\n--- Ablation Study Results ---")
    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    print(f"Ablation 'Different Random State': {results['No Fixed Split (random_state=101)']:.4f} (Impact: {impact['Random State for Split']:+.4f})")
    print(f"Ablation 'All Models on Log Target': {results['Ensemble: All Models on Log Target']:.4f} (Impact: {impact['Ensemble Target Strategy']:+.4f})")
    print(f"Ablation 'No XGBoost Native Categoricals': {results['No XGBoost Native Categoricals']:.4f} (Impact: {impact['XGBoost Native Categoricals']:+.4f})")
    
    # Determine the most impactful component
    most_impactful_component = max(impact, key=lambda k: abs(impact[k]))
    
    print("\n--- Conclusion ---")
    print(f"The most impactful component is: '{most_impactful_component}'")
