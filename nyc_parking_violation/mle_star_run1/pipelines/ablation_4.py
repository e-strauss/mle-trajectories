
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

def run_ablation_study():
    """
    Performs an ablation study on the training pipeline to evaluate the impact of:
    1. Robust K-Fold Target Encoding vs. Simple (Leaky) Target Encoding.
    2. Clamping negative predictions to zero.
    """
    # --- Data Loading ---
    # Create a dummy CSV file if it doesn't exist
    file_path = './input/violations_per_street_2022.csv'
    if not os.path.exists(file_path):
        print(f"'{file_path}' not found. Creating a dummy file for demonstration.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Generate more representative dummy data
        num_rows = 10000
        streets = [f'STREET_{i}' for i in range(500)]
        descs = [f'DESC_{i}' for i in range(50)]
        dummy_data = {
            'street_name': np.random.choice(streets, num_rows),
            'violation_description': np.random.choice(descs, num_rows),
            'violation_count': np.random.randint(1, 100, num_rows)
        }
        pd.DataFrame(dummy_data).to_csv(file_path, index=False)
        print("Dummy file created.")

    try:
        df_base = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {file_path}")
        return

    df_base.columns = df_base.columns.str.lower().str.replace(' ', '_')

    # --- Common Preprocessing ---
    # Log-transform target
    df_base['log_target'] = np.log1p(df_base['violation_count'])

    # Label encode categorical features
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_base['street_name_encoded'] = le_street.fit_transform(df_base['street_name'])
    df_base['violation_description_encoded'] = le_desc.fit_transform(df_base['violation_description'])
    
    # --- Store Results ---
    results = {}

    # --- Experiment 1: Baseline (Robust K-Fold Encoding + Clamping) ---
    df = df_base.copy()
    
    # a. Aggregate features using a robust K-Fold target encoding scheme
    df['description_mean_count'] = np.nan
    df['street_mean_count'] = np.nan
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_index, val_index in kf.split(df):
        df_train_fold, df_val_fold = df.iloc[train_index], df.iloc[val_index]
        
        description_agg_fold = df_train_fold.groupby('violation_description')['violation_count'].mean()
        street_agg_fold = df_train_fold.groupby('street_name')['violation_count'].mean()
        
        df.loc[df.index[val_index], 'description_mean_count'] = df_val_fold['violation_description'].map(description_agg_fold)
        df.loc[df.index[val_index], 'street_mean_count'] = df_val_fold['street_name'].map(street_agg_fold)
        
    global_desc_mean = df['violation_count'].mean()
    df['description_mean_count'].fillna(global_desc_mean, inplace=True)
    df['street_mean_count'].fillna(global_desc_mean, inplace=True)

    # b. Train model
    features = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    target = 'log_target'
    X_train, X_val, y_train, y_val = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # c. Validate and clamp negatives
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    baseline_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Baseline (K-Fold Encoding + Clamping)'] = baseline_score
    print(f"Baseline (K-Fold Encoding + Clamping) Validation RMSE: {baseline_score:.4f}")

    # --- Experiment 2: Ablation 1 (Simple/Leaky Target Encoding) ---
    df = df_base.copy()

    # a. Aggregate features using simple, leaky target encoding
    description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    # b. Train model
    X_train, X_val, y_train, y_val = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # c. Validate and clamp negatives
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    ablation1_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation (Leaky Encoding)'] = ablation1_score
    print(f"Ablation 1 (Leaky Encoding) Validation RMSE: {ablation1_score:.4f}")

    # --- Experiment 3: Ablation 2 (No Negative Clamping) ---
    df = df_base.copy()
    
    # a. Use the same robust K-Fold encoding as baseline
    df['description_mean_count'] = np.nan
    df['street_mean_count'] = np.nan
    
    for train_index, val_index in kf.split(df):
        df_train_fold, df_val_fold = df.iloc[train_index], df.iloc[val_index]
        description_agg_fold = df_train_fold.groupby('violation_description')['violation_count'].mean()
        street_agg_fold = df_train_fold.groupby('street_name')['violation_count'].mean()
        df.loc[df.index[val_index], 'description_mean_count'] = df_val_fold['violation_description'].map(description_agg_fold)
        df.loc[df.index[val_index], 'street_mean_count'] = df_val_fold['street_name'].map(street_agg_fold)
        
    df['description_mean_count'].fillna(global_desc_mean, inplace=True)
    df['street_mean_count'].fillna(global_desc_mean, inplace=True)
    
    # b. Train model
    X_train, X_val, y_train, y_val = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # c. Validate WITHOUT clamping negatives
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log) # No clamping: val_preds[val_preds < 0] = 0 is removed
    
    ablation2_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    results['Ablation (No Negative Clamping)'] = ablation2_score
    print(f"Ablation 2 (No Negative Clamping) Validation RMSE: {ablation2_score:.4f}")

    # --- Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    
    # Lower RMSE is better. The baseline is expected to be the best.
    best_score = min(results.values())
    
    degradation_leaky = results['Ablation (Leaky Encoding)'] - best_score
    degradation_no_clamp = results['Ablation (No Negative Clamping)'] - best_score
    
    impact = {
        'Robust K-Fold Target Encoding': degradation_leaky,
        'Clamping Negative Predictions': degradation_no_clamp
    }

    if not impact:
        print("Could not determine the most impactful component.")
        return

    most_impactful_component = max(impact, key=impact.get)
    
    print(f"Performance degradation by removing/changing K-Fold Encoding: {degradation_leaky:.4f}")
    print(f"Performance degradation by removing Negative Clamping: {degradation_no_clamp:.4f}")
    
    if max(impact.values()) > 0:
        print(f"\nThe component that contributes the most to performance is: '{most_impactful_component}'.")
    else:
        print("\nNo component showed a significant contribution in this study.")


if __name__ == '__main__':
    run_ablation_study()
