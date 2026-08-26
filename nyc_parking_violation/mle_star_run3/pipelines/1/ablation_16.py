
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import io

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_dummy_data():
    """Creates dummy CSV files required for the script to run."""
    os.makedirs('./input', exist_ok=True)

    violations_data = """street_name,violation_type,violation_count
    Main Street,Double Parking,150
    Main Street,Fire Hydrant,50
    Oak Avenue,No Standing,200
    Oak Avenue,Bus Lane,75
    Pine Lane,Double Parking,30
    Elm Street,Fire Hydrant,90
    Elm Street,No Standing,120
    Maple Drive,Bus Lane,40
    Maple Drive,Fire Hydrant,60
    First Street,Double Parking,210
    Second Street,No Standing,180
    Third Avenue,Bus Lane,95
    """

    boroughs_data = """street_name,borough
    Main Street,Queens
    Oak Avenue,Brooklyn
    Pine Lane,Manhattan
    Elm Street,Bronx
    Maple Drive,Staten Island
    First Street,Manhattan
    Second Street,Brooklyn
    Third Avenue,Queens
    """

    physical_data = """street_name,length_in_meters,width_in_meters
    Main Street,500,12
    Oak Avenue,300,10
    Pine Lane,150,8
    Elm Street,400,11
    Maple Drive,250,9
    First Street,600,15
    Second Street,550,13
    Third Avenue,450,12
    """

    with open('./input/violations_per_street_2022.csv', 'w') as f:
        f.write(violations_data)
    with open('./input/street_names_and_boroughs.csv', 'w') as f:
        f.write(boroughs_data)
    with open('./input/physical_features_per_street.csv', 'w') as f:
        f.write(physical_data)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_freq_encoding=True):
    """Loads, preprocesses, and prepares data for all models."""
    # --- 1. Load Data ---
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    # --- 2. Feature Engineering & Merging ---
    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)

    # --- Frequency Encoding Ablation ---
    if use_freq_encoding:
        street_name_freq = full_df['street_name'].value_counts().to_dict()
        borough_freq = full_df['borough'].value_counts().to_dict()
        full_df['street_name_freq'] = full_df['street_name'].map(street_name_freq)
        full_df['borough_freq'] = full_df['borough'].map(borough_freq)

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

def run_experiment(description, use_freq_encoding=True, use_rounding=True):
    """Runs a single experiment with specified configurations."""
    print(f"--- Running: {description} ---")

    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        use_freq_encoding=use_freq_encoding
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
        'cat_features': cat_features, 'early_stopping_rounds': 50
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

    # --- Rounding Ablation ---
    if use_rounding:
        ensemble_predictions = ensemble_predictions.round()

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Validation RMSE: {val_rmse:.4f}\n")
    return val_rmse

if __name__ == '__main__':
    setup_dummy_data()
    results = {}

    results['Baseline'] = run_experiment(
        "Baseline (with Frequency Encoding and Rounding)",
        use_freq_encoding=True,
        use_rounding=True
    )

    results['No Frequency Encoding'] = run_experiment(
        "Ablation: No Frequency Encoding",
        use_freq_encoding=False,
        use_rounding=True
    )

    results['No Rounding'] = run_experiment(
        "Ablation: No Prediction Rounding",
        use_freq_encoding=True,
        use_rounding=False
    )
    
    # --- 5. Final Analysis ---
    print("--- Ablation Study Summary ---")
    baseline_rmse = results['Baseline']
    
    impact_freq_encoding = results['No Frequency Encoding'] - baseline_rmse
    impact_rounding = results['No Rounding'] - baseline_rmse

    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Impact of removing Frequency Encoding: {impact_freq_encoding:+.4f} RMSE")
    print(f"Impact of removing Prediction Rounding: {impact_rounding:+.4f} RMSE")

    impacts = {
        "Frequency Encoding": abs(impact_freq_encoding),
        "Prediction Rounding": abs(impact_rounding)
    }
    
    most_impactful_component = max(impacts, key=impacts.get)
    
    print("\n--- Conclusion ---")
    print(f"The most impactful component is: **{most_impactful_component}**")
    if impact_freq_encoding > 0:
        print("-> Frequency Encoding is beneficial as removing it increased the error.")
    if impact_rounding < 0:
        print("-> Prediction Rounding is harmful as removing it decreased the error.")
    elif impact_rounding > 0:
        print("-> Prediction Rounding is beneficial as removing it increased the error.")

