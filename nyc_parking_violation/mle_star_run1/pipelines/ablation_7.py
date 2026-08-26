
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore')

def run_ablation_study():
    """
    Performs an ablation study on the model training process.
    """
    # --- 1. Data Loading and Feature Engineering (Common for all experiments) ---
    try:
        df = pd.read_csv('./input/violations_per_street_2022.csv')
    except FileNotFoundError:
        print("Error: Training file not found at ./input/violations_per_street_2022.csv")
        print("Please ensure the dataset is available in the correct directory.")
        return

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Aggregate features
    description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    # Label Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])

    features = [
        'street_name_encoded',
        'violation_description_encoded',
        'description_mean_count',
        'street_mean_count'
    ]
    target = 'violation_count'

    # Log-transform target and split data
    df['log_target'] = np.log1p(df[target])
    X = df[features]
    y = df['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    results = {}

    # --- Experiment 1: Baseline (Tuned Hyperparameters + Early Stopping) ---
    lgbm_baseline = lgb.LGBMRegressor(
        random_state=42,
        learning_rate=0.05,
        num_leaves=41,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=2000
    )
    lgbm_baseline.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    val_preds_log = lgbm_baseline.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    baseline_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Baseline (Tuned Params + Early Stopping)'] = baseline_rmse

    # --- Experiment 2: Ablation of Hyperparameter Tuning ---
    # Revert to default parameters but keep early stopping
    lgbm_no_tuning = lgb.LGBMRegressor(random_state=42, n_estimators=2000)
    lgbm_no_tuning.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    val_preds_log = lgbm_no_tuning.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    ablation_no_tuning_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation (No Hyperparameter Tuning)'] = ablation_no_tuning_rmse

    # --- Experiment 3: Ablation of Early Stopping ---
    # Keep tuned parameters but train for the full n_estimators, risking overfitting
    lgbm_no_early_stop = lgb.LGBMRegressor(
        random_state=42,
        learning_rate=0.05,
        num_leaves=41,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=2000
    )
    lgbm_no_early_stop.fit(X_train, y_train)
    val_preds_log = lgbm_no_early_stop.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    ablation_no_early_stop_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation (No Early Stopping)'] = ablation_no_early_stop_rmse

    # --- 4. Print Results and Conclusion ---
    print("--- Ablation Study Results (Lower RMSE is better) ---")
    for name, score in results.items():
        print(f"{name}: RMSE = {score:.4f}")
    
    print("\n--- Performance Impact Analysis ---")
    degradation_tuning = results['Ablation (No Hyperparameter Tuning)'] - results['Baseline (Tuned Params + Early Stopping)']
    degradation_early_stop = results['Ablation (No Early Stopping)'] - results['Baseline (Tuned Params + Early Stopping)']
    
    print(f"Removing Hyperparameter Tuning caused a performance degradation (RMSE increase) of: {degradation_tuning:.4f}")
    print(f"Removing Early Stopping caused a performance degradation (RMSE increase) of: {degradation_early_stop:.4f}")

    print("\n--- Conclusion ---")
    if degradation_tuning > degradation_early_stop:
        print("Hyperparameter Tuning contributes the most to the model's performance.")
    elif degradation_early_stop > degradation_tuning:
        print("Early Stopping contributes the most to the model's performance.")
    else:
        print("Both components have a similar impact on performance.")

if __name__ == '__main__':
    run_ablation_study()
