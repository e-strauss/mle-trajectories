
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
from io import StringIO
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_ablation_study():
    """
    Performs an ablation study on the modeling script to determine the contribution
    of different feature groups and model training techniques.
    """
    # --- 1. Common Data Loading and Preprocessing ---
    # Using a dummy dataset for demonstration, mimicking the original structure.
    # In a real scenario, this would load from a file.
    data = """street_name,violation_description,violation_count
BWAY,NO PARKING,150
BWAY,DOUBLE PARKING,50
LEX AVE,NO PARKING,200
LEX AVE,FIRE HYDRANT,30
5TH AVE,NO STANDING,180
5TH AVE,DOUBLE PARKING,70
BWAY,NO PARKING,160
LEX AVE,NO PARKING,210
5TH AVE,NO STANDING,190
BWAY,DOUBLE PARKING,65
"""
    # Fallback for systems without a file, use an in-memory string
    try:
        df_train = pd.read_csv('./input/violations_per_street_2022.csv')
    except FileNotFoundError:
        print("Info: Could not find input file, using a sample in-memory dataset.", file=sys.stderr)
        df_train = pd.read_csv(StringIO(data))


    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # Feature Engineering: Aggregates
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    # Feature Engineering: Label Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])
    
    # Log-transform the target
    df_train['log_target'] = np.log1p(df_train['violation_count'])

    results = {}

    # --- Experiment 0: Baseline Model ---
    # Uses all features and default fitting.
    features_base = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    X = df_train[features_base]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    baseline_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Baseline (All Features)'] = baseline_rmse
    print(f"Baseline RMSE (All Features): {baseline_rmse:.4f}")

    # --- Ablation 1: Remove Street-Related Features ---
    # This tests the predictive power of violation descriptions alone.
    features_ab1 = ['violation_description_encoded', 'description_mean_count']
    X = df_train[features_ab1]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    ablation1_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation 1 (Description Only)'] = ablation1_rmse
    print(f"Ablation 1 RMSE (Description Features Only): {ablation1_rmse:.4f}")

    # --- Ablation 2: Remove Violation-Description-Related Features ---
    # This tests the predictive power of street names alone.
    features_ab2 = ['street_name_encoded', 'street_mean_count']
    X = df_train[features_ab2]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    ablation2_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation 2 (Street Only)'] = ablation2_rmse
    print(f"Ablation 2 RMSE (Street Features Only): {ablation2_rmse:.4f}")

    # --- Ablation 3: Inform LGBM about Categorical Features ---
    # Tests the importance of correctly treating integer-encoded features as categoricals.
    features_base = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    categorical_features_names = ['street_name_encoded', 'violation_description_encoded']
    X = df_train[features_base]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train, categorical_feature=categorical_features_names)
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    ablation3_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation 3 (Add Categorical Hint)'] = ablation3_rmse
    print(f"Ablation 3 RMSE (With Categorical Feature Hint): {ablation3_rmse:.4f}")

    # --- Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    
    impact = {
        'Street Features': results['Ablation 1 (Description Only)'] - results['Baseline (All Features)'],
        'Description Features': results['Ablation 2 (Street Only)'] - results['Baseline (All Features)'],
        'Categorical Hint': results['Baseline (All Features)'] - results['Ablation 3 (Add Categorical Hint)']
    }

    # Find the feature removal that caused the biggest increase in error (most important)
    most_impactful_feature_removal = max(impact, key=lambda k: impact[k] if 'Hint' not in k else -1)
    
    print(f"Removing Street Features increased RMSE by: {impact['Street Features']:.4f}")
    print(f"Removing Description Features increased RMSE by: {impact['Description Features']:.4f}")
    print(f"Adding Categorical Hint improved RMSE by: {impact['Categorical Hint']:.4f}")

    print(f"\nThe most impactful component is: {most_impactful_feature_removal}. Removing it caused the largest performance degradation.")

if __name__ == '__main__':
    run_ablation_study()
