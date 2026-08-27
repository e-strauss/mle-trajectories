
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import sys
import subprocess

# --- Helper function to run a single experiment ---

def run_experiment(use_statistical_features=True, fix_model_random_state=True, train_path='./input/violations_per_street_2022.csv'):
    """
    Runs a single training and validation experiment with specified configurations.

    Args:
        use_statistical_features (bool): If True, include frequency and diversity features.
        fix_model_random_state (bool): If True, use a fixed random_state for the LGBMRegressor.
        train_path (str): Path to the training data.

    Returns:
        float: The RMSE score on the validation set.
    """
    try:
        df = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}", file=sys.stderr)
        # Create a small dummy dataframe to prevent crashing, but results will be meaningless.
        print("Warning: Using a dummy dataset. Results will not be meaningful.", file=sys.stderr)
        data = {
            'street_name': [f'street_{i}' for i in range(100)],
            'violation_description': [f'desc_{i % 5}' for i in range(100)],
            'violation_count': np.random.randint(1, 100, 100)
        }
        df = pd.DataFrame(data)

    # --- 1. Basic Cleaning ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---

    # a. Always include target-encoded mean counts
    description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    features = ['description_mean_count', 'street_mean_count']

    # b. Conditionally include statistical features (Ablation Component 1)
    if use_statistical_features:
        street_name_freq_map = df['street_name'].value_counts()
        df['street_name_freq'] = df['street_name'].map(street_name_freq_map)

        street_name_diversity_map = df.groupby('street_name')['violation_description'].nunique()
        df['street_name_desc_diversity'] = df['street_name'].map(street_name_diversity_map)

        violation_desc_freq_map = df['violation_description'].value_counts()
        df['violation_description_freq'] = df['violation_description'].map(violation_desc_freq_map)

        violation_desc_diversity_map = df.groupby('violation_description')['street_name'].nunique()
        df['violation_description_street_diversity'] = df['violation_description'].map(violation_desc_diversity_map)
        
        features.extend([
            'street_name_freq', 'street_name_desc_diversity',
            'violation_description_freq', 'violation_description_street_diversity'
        ])

    # --- 3. Model Training ---
    target = 'violation_count'
    df['log_target'] = np.log1p(df[target])

    # Use a fixed random state for the split to ensure consistency across experiments
    X = df[features]
    y = df['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    # Initialize model with or without a fixed random state (Ablation Component 2)
    if fix_model_random_state:
        lgbm = lgb.LGBMRegressor(random_state=42)
    else:
        lgbm = lgb.LGBMRegressor()

    lgbm.fit(X_train, y_train)

    # --- 4. Validation ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return rmse

# --- Main Ablation Study Execution ---

if __name__ == '__main__':
    # Ensure lightgbm is installed
    try:
        import lightgbm as lgb
    except ImportError:
        print("Installing lightgbm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])

    # Define experiments
    experiments = {
        "Baseline": {
            "description": "All features and fixed model random state.",
            "params": {"use_statistical_features": True, "fix_model_random_state": True}
        },
        "Ablation 1 (No Statistical Features)": {
            "description": "Removes frequency and diversity features.",
            "params": {"use_statistical_features": False, "fix_model_random_state": True}
        },
        "Ablation 2 (No Fixed Model Random State)": {
            "description": "Uses default (random) LGBM initialization.",
            "params": {"use_statistical_features": True, "fix_model_random_state": False}
        }
    }

    results = {}
    print("--- Starting Ablation Study ---")

    # Run all experiments
    for name, config in experiments.items():
        print(f"Running: {name}...")
        results[name] = run_experiment(**config['params'])

    # Print results
    print("\n--- Ablation Study Results (RMSE) ---")
    baseline_score = results.get("Baseline", float('inf'))
    
    for name, score in results.items():
        degradation = score - baseline_score if name != "Baseline" else 0.0
        print(f"{name:<35}: {score:.4f} (Performance Degradation: {degradation:+.4f})")
    
    # Determine the most impactful component
    print("\n--- Conclusion ---")
    
    degradations = {
        "Statistical Features": results.get("Ablation 1 (No Statistical Features)", baseline_score) - baseline_score,
        "Fixed Model Random State": results.get("Ablation 2 (No Fixed Model Random State)", baseline_score) - baseline_score,
    }

    if not degradations or max(degradations.values()) <= 0:
        print("No single component removal led to a clear performance degradation.")
    else:
        most_impactful_component = max(degradations, key=degradations.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'.")
        print("Removing it caused the largest increase in RMSE, indicating its high importance.")

