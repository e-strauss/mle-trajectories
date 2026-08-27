
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore')

def run_ablation_study():
    # --- Data Loading and Preparation ---
    # Use a sample dataframe if the file doesn't exist
    data = {
        'street_name': ['BROADWAY', 'MAIN ST', 'PARK AVE', 'BROADWAY', 'MAIN ST'] * 20,
        'violation_description': ['PHTO SCHOOL ZN SPEED VIOLATION', 'FAILURE TO STOP AT RED LIGHT', 'PHTO SCHOOL ZN SPEED VIOLATION', 'NO PARKING-STREET CLEANING', 'FAILURE TO STOP AT RED LIGHT'] * 20,
        'violation_count': [120, 55, 80, 45, 60, 130, 50, 85, 40, 65, 115, 58, 78, 48, 62, 125, 52, 82, 42, 68] * 5
    }
    df = pd.DataFrame(data)
    
    # --- Feature Engineering (Common for all experiments) ---
    description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])
    
    features = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    target = 'violation_count'
    df['log_target'] = np.log1p(df[target])

    X = df[features]
    y = df['log_target']
    
    results = {}

    # --- Experiment 1: Baseline ---
    # This represents the original script's performance with a fixed data split and default model complexity.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42) # n_estimators default is 100
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    baseline_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Baseline'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # --- Ablation 1: Removing Fixed Data Split Random State ---
    # This tests the model's sensitivity to the specific data points in the train/validation sets.
    # A volatile score would indicate the model is not robust.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None) # Changed
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    ablation1_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation 1 (No Fixed Split)'] = ablation1_rmse
    print(f"Ablation RMSE (No Fixed Split): {ablation1_rmse:.4f}")

    # --- Ablation 2: Reducing Model Complexity (n_estimators) ---
    # This tests the importance of having a sufficiently complex model. We reduce the number of boosting
    # rounds from the default of 100 to 10.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=10) # Changed
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    ablation2_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation 2 (Reduced Estimators)'] = ablation2_rmse
    print(f"Ablation RMSE (Reduced Estimators from 100 to 10): {ablation2_rmse:.4f}")

    # --- Conclusion ---
    degradation1 = results['Ablation 1 (No Fixed Split)'] - results['Baseline']
    degradation2 = results['Ablation 2 (Reduced Estimators)'] - results['Baseline']

    print("\n--- Ablation Study Summary ---")
    print(f"Baseline Performance (RMSE): {results['Baseline']:.4f}")
    print(f"Performance Degradation from removing fixed data split: {degradation1:.4f}")
    print(f"Performance Degradation from reducing model estimators: {degradation2:.4f}")

    if degradation2 > degradation1:
        print("\nConclusion: Model Complexity (number of estimators) contributes more to performance than the specific data split.")
    else:
        print("\nConclusion: The specific Train/Validation Data Split contributes more to performance than the number of model estimators.")

if __name__ == '__main__':
    run_ablation_study()
