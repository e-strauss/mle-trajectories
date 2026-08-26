
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import warnings

# Suppress verbose LightGBM and other warnings for cleaner output
warnings.filterwarnings('ignore')

def run_experiment(df_processed, split_random_state, model_random_state):
    """
    Runs a single training and evaluation experiment with specific configurations.

    Args:
        df_processed (pd.DataFrame): The pre-processed dataframe with all features.
        split_random_state (int or None): The random state for train_test_split.
        model_random_state (int or None): The random state for the LGBMRegressor.

    Returns:
        float: The Root Mean Squared Error on the validation set.
    """
    features = [
        'street_name_encoded',
        'violation_description_encoded',
        'description_mean_count',
        'street_mean_count'
    ]
    target = 'log_target'

    X = df_processed[features]
    y = df_processed[target]

    # Split data using the specified random state for the split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=split_random_state)
    y_val_original = np.expm1(y_val)

    # Initialize and train the model with the specified random state
    lgbm = lgb.LGBMRegressor(random_state=model_random_state, verbosity=-1)
    lgbm.fit(X_train, y_train)

    # Evaluate performance
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return rmse

def main():
    """
    Performs an ablation study on the effect of fixed random states
    for the data split and the model algorithm.
    """
    parser = argparse.ArgumentParser(description="Ablation study for NYC parking violations.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data file.')
    args = parser.parse_args()

    try:
        df_train = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Warning: Training file not found at {args.train_path}. Using a sample dataset for demonstration.")
        # Create a small, reproducible dummy dataframe if the file is not found
        sample_data = {
            'Street Name': [f'STREET_{i%50}' for i in range(2000)],
            'Violation Description': [f'DESC_{i%20}' for i in range(2000)],
            'Violation Count': np.random.RandomState(42).randint(1, 150, 2000)
        }
        df_train = pd.DataFrame(sample_data)

    # --- 1. Data Cleaning and Feature Engineering ---
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')

    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])

    df_train['log_target'] = np.log1p(df_train['violation_count'])

    # --- 2. Run Ablation Study ---
    results = {}

    # Baseline: Both data split and model algorithm are deterministic
    results['Baseline (Fixed Split, Fixed Model)'] = run_experiment(df_train, split_random_state=42, model_random_state=42)

    # Ablation 1: Remove determinism from the data split
    results['Ablation 1 (Stochastic Split, Fixed Model)'] = run_experiment(df_train, split_random_state=None, model_random_state=42)

    # Ablation 2: Remove determinism from the model's training algorithm
    results['Ablation 2 (Fixed Split, Stochastic Model)'] = run_experiment(df_train, split_random_state=42, model_random_state=None)

    # --- 3. Print Results and Conclusion ---
    print("--- Ablation Study Results ---")
    baseline_score = results['Baseline (Fixed Split, Fixed Model)']
    print(f"Baseline RMSE (Fixed Split, Fixed Model): {baseline_score:.4f}")

    ablation1_score = results['Ablation 1 (Stochastic Split, Fixed Model)']
    print(f"Ablation 1 RMSE (Stochastic Split): {ablation1_score:.4f}")

    ablation2_score = results['Ablation 2 (Fixed Split, Stochastic Model)']
    print(f"Ablation 2 RMSE (Stochastic Model): {ablation2_score:.4f}")

    # Calculate performance change (a positive value means performance got worse)
    impact_split = ablation1_score - baseline_score
    impact_model = ablation2_score - baseline_score

    print("\n--- Impact Analysis ---")
    print(f"Impact of removing fixed data split state: {impact_split:+.4f} change in RMSE")
    print(f"Impact of removing fixed model algorithm state: {impact_model:+.4f} change in RMSE")

    print("\n--- Conclusion ---")
    if abs(impact_split) > abs(impact_model):
        print("The 'Data Split Random State' contributes most to performance variance.")
        print("This means the model's performance is highly sensitive to the specific data it is trained and validated on. Fixing the data split ('random_state' in train_test_split) is crucial for achieving reproducible results.")
    elif abs(impact_model) > abs(impact_split):
        print("The 'Model Algorithm Random State' contributes most to performance variance.")
        print("This means the LightGBM algorithm has inherent stochasticity that affects the outcome. Fixing the model's 'random_state' is crucial for achieving reproducible results.")
    else:
        print("Both the data split and model algorithm randomness have a similar, minimal impact on performance, indicating a stable pipeline.")

if __name__ == '__main__':
    main()
