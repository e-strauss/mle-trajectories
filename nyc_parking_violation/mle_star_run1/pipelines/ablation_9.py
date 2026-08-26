
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import sys
import subprocess
import warnings

warnings.filterwarnings('ignore')

# --- Helper function to run a single experiment ---
def run_experiment(df_original, use_street_agg=True, use_desc_agg=True, clamp_negatives=True):
    """
    Runs a single training and validation experiment with specified components enabled or disabled.
    
    Args:
        df_original (pd.DataFrame): The initial, clean dataframe.
        use_street_agg (bool): If True, include the street_mean_count feature.
        use_desc_agg (bool): If True, include the description_mean_count feature.
        clamp_negatives (bool): If True, clamp negative predictions to 0.

    Returns:
        float: The Root Mean Squared Error on the validation set.
    """
    df = df_original.copy()

    features = [
        'street_name_encoded',
        'violation_description_encoded'
    ]

    # --- Feature Engineering ---
    if use_desc_agg:
        description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
        df = pd.merge(df, description_agg, on='violation_description', how='left')
        features.append('description_mean_count')

    if use_street_agg:
        street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
        df = pd.merge(df, street_agg, on='street_name', how='left')
        features.append('street_mean_count')

    # --- Model Training ---
    target = 'violation_count'
    df['log_target'] = np.log1p(df[target])

    X = df[features]
    y = df['log_target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- Validation ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)

    if clamp_negatives:
        val_preds[val_preds < 0] = 0

    return np.sqrt(mean_squared_error(y_val_original, val_preds))


def main():
    """
    Main function to orchestrate the ablation study.
    """
    # Create a dummy dataframe for the script to run, as per the problem description
    # This ensures the script is self-contained.
    data_path = './input/violations_per_street_2022.csv'
    if not os.path.exists(data_path):
        print("Input file not found. Creating a dummy file for demonstration.")
        os.makedirs('./input', exist_ok=True)
        dummy_data = {
            'Street Name': ['BROADWAY', 'BROADWAY', 'MAIN ST', 'MAIN ST', '5TH AVE', '5TH AVE', 'PARK AVE', 'PARK AVE', 'WALL ST', 'WALL ST'],
            'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'FIRE HYDRANT'],
            'Violation Count': [150, 50, 120, 40, 200, 60, 180, 55, 250, 70] * 10 # Larger data for stability
        }
        pd.DataFrame(dummy_data).to_csv(data_path, index=False)

    # --- 1. Data Loading and Basic Preparation ---
    try:
        df_train = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {data_path}")
        return

    # Standardize column names
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')
    
    # Pre-encode categorical features
    df_train['street_name_encoded'] = LabelEncoder().fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = LabelEncoder().fit_transform(df_train['violation_description'])


    # --- 2. Run Ablation Experiments ---
    results = {}

    # Experiment 1: Baseline (all components enabled)
    results['Baseline (All Features & Clamping)'] = run_experiment(
        df_train,
        use_street_agg=True,
        use_desc_agg=True,
        clamp_negatives=True
    )

    # Experiment 2: Ablation of Street Aggregate Feature
    results['Ablation (No Street Aggregate)'] = run_experiment(
        df_train,
        use_street_agg=False,
        use_desc_agg=True,
        clamp_negatives=True
    )

    # Experiment 3: Ablation of Description Aggregate Feature
    results['Ablation (No Description Aggregate)'] = run_experiment(
        df_train,
        use_street_agg=True,
        use_desc_agg=False,
        clamp_negatives=True
    )

    # Experiment 4: Ablation of Negative Prediction Clamping
    results['Ablation (No Negative Clamping)'] = run_experiment(
        df_train,
        use_street_agg=True,
        use_desc_agg=True,
        clamp_negatives=False
    )
    
    # --- 3. Report Results ---
    print("--- Ablation Study Results (Lower RMSE is better) ---")
    baseline_score = results['Baseline (All Features & Clamping)']
    degradations = {}

    for name, score in results.items():
        degradation = score - baseline_score
        degradations[name] = degradation
        print(f"{name:<40}: RMSE = {score:.4f} (Performance Change: {degradation:+.4f})")
    
    # Remove baseline for finding the worst degradation
    del degradations['Baseline (All Features & Clamping)']
    
    # Find the component whose removal caused the biggest drop in performance
    most_impactful_component = max(degradations, key=degradations.get)
    
    # Translate experiment name to component name
    if 'No Street Aggregate' in most_impactful_component:
        conclusion = "The 'Street Aggregate' feature"
    elif 'No Description Aggregate' in most_impactful_component:
        conclusion = "The 'Description Aggregate' feature"
    else: # No Negative Clamping
        conclusion = "Clamping negative predictions to zero"

    print("\n--- Conclusion ---")
    print(f"{conclusion} contributes the most to the overall performance.")

if __name__ == '__main__':
    main()
