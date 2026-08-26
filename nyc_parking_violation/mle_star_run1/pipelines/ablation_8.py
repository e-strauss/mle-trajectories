
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import sys

# --- Helper function to run an experiment ---
def run_experiment(name, features_to_use, df, target_col):
    """
    Trains a model with a specific feature set and evaluates its performance.
    """
    print(f"--- Running Experiment: {name} ---")
    
    # Define features and target for this run
    X = df[features_to_use]
    y = df[target_col]
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val) # Original scale for RMSE calculation

    # Initialize and train the LightGBM model
    # verbosity=-1 suppresses fitting output for cleaner logs
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)

    # Predict on validation set and transform back to original scale
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0 # Ensure predictions are non-negative
    
    # Calculate and return RMSE
    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f"Validation RMSE: {rmse:.4f}\n")
    return rmse

def perform_ablation_study():
    """
    Performs an ablation study on different feature engineering strategies.
    """
    # --- 1. Data Loading and Preprocessing (Common for all experiments) ---
    train_path = './input/violations_per_street_2022.csv'
    
    try:
        df_train = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}", file=sys.stderr)
        print("Aborting study. Please ensure the dataset is available.", file=sys.stderr)
        return

    # Standardize column names for consistency
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering (Create all features upfront) ---
    
    # Component 1: Aggregate (Mean Encoded) Features
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
    df = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    # Component 2: Simple Label Encoded Features
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])
    
    # Log-transform the target variable (kept constant across experiments)
    target_col = 'violation_count'
    log_target_col = 'log_target'
    df[log_target_col] = np.log1p(df[target_col])

    # --- 3. Define Feature Sets for Ablation ---
    features_all = [
        'street_name_encoded', 
        'violation_description_encoded', 
        'description_mean_count', 
        'street_mean_count'
    ]
    features_label_only = [
        'street_name_encoded', 
        'violation_description_encoded'
    ]
    features_mean_only = [
        'description_mean_count', 
        'street_mean_count'
    ]

    # --- 4. Run Experiments and Collect Results ---
    results = {}

    # Experiment 1: Baseline (using both feature types)
    results['Baseline (All Features)'] = run_experiment(
        "Baseline (Mean + Label Features)",
        features_all,
        df,
        log_target_col
    )

    # Experiment 2: Ablation of Mean Encoded Features
    results['Ablation (Label Features Only)'] = run_experiment(
        "Ablation 1 (Label Features Only)",
        features_label_only,
        df,
        log_target_col
    )

    # Experiment 3: Ablation of Label Encoded Features
    results['Ablation (Mean Features Only)'] = run_experiment(
        "Ablation 2 (Mean Features Only)",
        features_mean_only,
        df,
        log_target_col
    )

    # --- 5. Analyze Results and Determine Most Impactful Component ---
    print("--- Ablation Study Summary ---")
    baseline_rmse = results['Baseline (All Features)']
    
    # Calculate performance degradation by removing each component
    # A larger positive number means removing it hurt the model more
    degradation_from_removing_mean = results['Ablation (Label Features Only)'] - baseline_rmse
    degradation_from_removing_label = results['Ablation (Mean Features Only)'] - baseline_rmse

    print(f"Baseline RMSE (Mean + Label Features): {baseline_rmse:.4f}")
    print(f"Removing Mean Encoded Features led to a performance change of: {degradation_from_removing_mean:+.4f} RMSE.")
    print(f"Removing Label Encoded Features led to a performance change of: {degradation_from_removing_label:+.4f} RMSE.")

    print("\n--- Conclusion ---")
    if degradation_from_removing_label > degradation_from_removing_mean and degradation_from_removing_label > 0:
        print("The 'Label Encoded Features' contribute the most to the overall performance.")
    elif degradation_from_removing_mean > degradation_from_removing_label and degradation_from_removing_mean > 0:
        print("The 'Mean Encoded Features' contribute the most to the overall performance.")
    else:
        print("Neither feature set showed a dominant contribution, or their removal improved the model, suggesting redundancy or negative interaction.")

if __name__ == '__main__':
    perform_ablation_study()
