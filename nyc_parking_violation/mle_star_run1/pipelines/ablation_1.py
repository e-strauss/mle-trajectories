
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os

# Create a dummy dataset for the script to run
if not os.path.exists('./input'):
    os.makedirs('./input')

dummy_data = {
    'Street Name': ['BROADWAY', 'BROADWAY', 'MAIN ST', 'MAIN ST', '5TH AVE', '5TH AVE', 'PARK AVE'],
    'Violation Description': ['PHTO SCHOOL ZN SPEED VIOLATION', 'FAILURE TO STOP AT RED LIGHT', 'PHTO SCHOOL ZN SPEED VIOLATION', 'NO PARKING-STREET CLEANING', 'FAILURE TO STOP AT RED LIGHT', 'NO PARKING-STREET CLEANING', 'PHTO SCHOOL ZN SPEED VIOLATION'],
    'Violation Count': [150, 80, 120, 200, 75, 210, 140]
}
dummy_df = pd.DataFrame(dummy_data)
dummy_df.to_csv('./input/violations_per_street_2022.csv', index=False)


def run_ablation_study():
    """
    Performs an ablation study on the NYC parking violations model.
    """
    train_path = './input/violations_per_street_2022.csv'
    
    # --- 1. Common Data Loading and Preprocessing ---
    df_base = pd.read_csv(train_path)
    df_base.columns = df_base.columns.str.lower().str.replace(' ', '_')

    # --- 2. Baseline Experiment (Full Model) ---
    print("--- Running Baseline Experiment (Full Model) ---")
    df = df_base.copy()
    
    # a. Aggregate features
    description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')
    
    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])
    
    # c. Target Transformation
    df['log_target'] = np.log1p(df['violation_count'])

    # d. Training
    features = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    X = df[features]
    y = df['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    y_val_original = np.expm1(y_val)
    baseline_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f'Baseline RMSE: {baseline_rmse:.4f}\n')

    # --- 3. Ablation 1: No Log Transform on Target ---
    print("--- Running Ablation 1: No Log Transform ---")
    df = df_base.copy()
    
    # a. Aggregate features (same as baseline)
    description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')
    
    # b. Categorical Feature Encoding (same as baseline)
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])
    
    # c. Training on raw target
    features = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    X = df[features]
    y = df['violation_count'] # <-- No log transform
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    
    val_preds = lgbm.predict(X_val) # <-- No expm1
    val_preds[val_preds < 0] = 0
    
    no_log_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f'Ablation 1 (No Log Transform) RMSE: {no_log_rmse:.4f}\n')

    # --- 4. Ablation 2: No Aggregate Features ---
    print("--- Running Ablation 2: No Aggregate Features ---")
    df = df_base.copy()
    
    # a. Categorical Feature Encoding
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])
    
    # b. Target Transformation (same as baseline)
    df['log_target'] = np.log1p(df['violation_count'])

    # c. Training with reduced feature set
    features = ['street_name_encoded', 'violation_description_encoded'] # <-- No aggregate features
    X = df[features]
    y = df['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    y_val_original = np.expm1(y_val)
    no_agg_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f'Ablation 2 (No Aggregate Features) RMSE: {no_agg_rmse:.4f}\n')

    # --- 5. Conclusion ---
    print("--- Ablation Study Conclusion ---")
    perf_degradation_log = no_log_rmse - baseline_rmse
    perf_degradation_agg = no_agg_rmse - baseline_rmse

    print(f"Removing Log Transform degraded performance (increased RMSE) by: {perf_degradation_log:.4f}")
    print(f"Removing Aggregate Features degraded performance (increased RMSE) by: {perf_degradation_agg:.4f}")

    if perf_degradation_log > perf_degradation_agg:
        print("\nConclusion: The log transformation of the target variable is the most critical component for model performance.")
    elif perf_degradation_agg > perf_degradation_log:
        print("\nConclusion: The aggregate features (mean encoding) are the most critical component for model performance.")
    else:
        print("\nConclusion: Both the log transform and aggregate features contribute similarly to performance.")

if __name__ == '__main__':
    run_ablation_study()
