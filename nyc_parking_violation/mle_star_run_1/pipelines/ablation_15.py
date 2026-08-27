
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import sys
import subprocess

# Attempt to install lightgbm if not already installed
try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
        import lightgbm as lgb
    except Exception as e:
        print(f"Failed to install and import lightgbm: {e}")
        sys.exit(1)


def run_experiment(lgbm_params, experiment_name, df_train_full):
    """
    Runs a single training and validation experiment with a specific configuration.

    Args:
        lgbm_params (dict): Parameters for the LGBMRegressor.
        experiment_name (str): Name of the experiment for logging.
        df_train_full (pd.DataFrame): The full, clean training dataframe.

    Returns:
        float: The validation RMSE score for the experiment.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")
    
    # Use a copy to ensure each experiment is independent
    df_train = df_train_full.copy()

    # --- 1. Feature Engineering (Replicated from original script) ---
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    le_street = LabelEncoder()
    le_desc = LabelEncoder()

    all_streets = df_train['street_name'].unique()
    all_descs = df_train['violation_description'].unique()
    
    le_street.fit(np.append(all_streets, '<unknown>'))
    le_desc.fit(np.append(all_descs, '<unknown>'))

    df_train['street_name_encoded'] = le_street.transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.transform(df_train['violation_description'])
    
    # --- 2. Model Training ---
    features = [
        'street_name_encoded', 
        'violation_description_encoded', 
        'description_mean_count', 
        'street_mean_count'
    ]
    target = 'violation_count'
    
    df_train['log_target'] = np.log1p(df_train[target])

    X = df_train[features]
    y = df_train['log_target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    # Initialize and train the model with the specified parameters
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train)

    # --- 3. Validation Performance ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    validation_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f'Validation RMSE: {validation_score:.4f}')
    
    return validation_score


def main():
    """
    Main function to orchestrate the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    if not os.path.exists(train_path):
        print("Input file not found. Creating a dummy dataset for demonstration.")
        os.makedirs('./input', exist_ok=True)
        data = {
            'street_name': np.repeat([f'STREET_{i}' for i in range(50)], 20),
            'violation_description': np.tile([f'DESC_{i}' for i in range(10)], 100),
            'violation_count': np.random.randint(1, 150, 1000)
        }
        pd.DataFrame(data).to_csv(train_path, index=False)

    try:
        df_train_full = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}")
        return
        
    df_train_full.columns = df_train_full.columns.str.lower().str.replace(' ', '_')

    # --- Define and Run Experiments ---
    experiments = {}

    # Baseline: Default LGBM parameters from the original script
    baseline_params = {'random_state': 42}
    baseline_score = run_experiment(baseline_params, "Baseline (Default Model)", df_train_full)
    experiments["Baseline"] = baseline_score

    # Ablation 1: Reduce Model Complexity (num_leaves)
    # The default is 31. A lower value creates a simpler model, which can prevent overfitting.
    complexity_params = {'random_state': 42, 'num_leaves': 10}
    experiments["Model Complexity (num_leaves)"] = run_experiment(complexity_params, "Ablation: Reduced Complexity (num_leaves=10)", df_train_full)

    # Ablation 2: Add L2 Regularization
    # The default is 0. Adding regularization can also help prevent overfitting.
    regularization_params = {'random_state': 42, 'reg_lambda': 2.0}
    experiments["L2 Regularization"] = run_experiment(regularization_params, "Ablation: Added L2 Regularization (reg_lambda=2.0)", df_train_full)

    # --- Analyze and Conclude ---
    print("\n" + "="*30)
    print("Ablation Study Summary")
    print("="*30)
    
    impacts = {}
    for name, score in experiments.items():
        if name != "Baseline":
            degradation = score - baseline_score
            print(f"Modifying '{name}' resulted in a performance change of: {degradation:+.4f} RMSE")
            impacts[name] = abs(degradation)

    if not impacts:
        print("\nConclusion: No ablations were run to compare against the baseline.")
        return

    # Determine the most impactful component based on the largest change in score
    most_impactful_component = max(impacts, key=impacts.get)
    
    print("\n--- Conclusion ---")
    print(f"The component that contributes most to the overall performance is: '{most_impactful_component}'.")
    print(f"Altering it from the baseline setting caused the largest change in model performance, indicating the model is highly sensitive to this parameter.")

if __name__ == '__main__':
    main()
