
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_experiment(
    use_length_features=True,
    use_pavement_data=True,
    use_early_stopping=True
):
    """
    Runs a single training and validation experiment with specified configurations.
    This function isolates the training and validation logic and does not use test data.
    """
    # Define file paths
    base_dir = "./input" if os.path.exists("./input") else "."
    violations_2022_path = os.path.join(base_dir, "violations_per_street_2022.csv")
    street_details_path = os.path.join(base_dir, "street_details.csv")
    pavement_maintenance_path = os.path.join(base_dir, "pavement_maintenance_history.csv")

    # Load core training data
    try:
        train_df = pd.read_csv(violations_2022_path)
    except FileNotFoundError:
        print(f"Error: Training data not found at {violations_2022_path}.")
        print("Please ensure 'violations_per_street_2022.csv' is in the './input/' directory.")
        return float('inf')
        
    train_df.rename(columns={
        'Street Name': 'street_name',
        'Violation Description': 'violation_type',
        'violation_count': 'target'
    }, inplace=True)
    full_df = train_df

    # Load and merge augmentation data
    if os.path.exists(street_details_path):
        street_details = pd.read_csv(street_details_path)
        full_df = pd.merge(full_df, street_details, on='street_name', how='left')

    # Ablation Point 1: Pavement Maintenance Data
    if use_pavement_data and os.path.exists(pavement_maintenance_path):
        pavement_maintenance = pd.read_csv(pavement_maintenance_path)
        pavement_maintenance['last_maintenance_year'] = pd.to_datetime(pavement_maintenance['last_maintenance_date'], errors='coerce').dt.year
        pavement_agg = pavement_maintenance.groupby('street_name').agg({
            'last_maintenance_year': 'max',
            'pavement_condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        full_df = pd.merge(full_df, pavement_agg, on='street_name', how='left')
        
        current_year = 2023
        full_df['years_since_maintenance'] = current_year - full_df['last_maintenance_year']
        full_df.drop('last_maintenance_year', axis=1, inplace=True)

    # Ablation Point 2: Simple Length-Based Features
    if use_length_features:
        full_df['street_name_len'] = full_df['street_name'].astype(str).apply(len)
        full_df['violation_type_len'] = full_df['violation_type'].astype(str).apply(len)

    # Impute missing values
    for col in full_df.columns:
        if full_df[col].isnull().any():
            if full_df[col].dtype == 'object':
                fill_val = full_df[col].mode()[0] if not full_df[col].mode().empty else "missing"
                full_df[col] = full_df[col].fillna(fill_val)
            elif pd.api.types.is_numeric_dtype(full_df[col]):
                fill_val = full_df[col].median()
                full_df[col] = full_df[col].fillna(fill_val)

    # Encode Categorical Features
    categorical_features_base = ['street_name', 'violation_type', 'borough']
    if 'pavement_condition' in full_df.columns:
        categorical_features_base.append('pavement_condition')
    
    categorical_features_exist = [c for c in categorical_features_base if c in full_df.columns]

    for col in categorical_features_exist:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))

    # Split data for validation
    X = full_df.drop(columns=['target'])
    y = full_df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features_indices = [X_train.columns.get_loc(c) for c in categorical_features_exist if c in X_train]

    # Ablation Point 3: Early Stopping
    model_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 10,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 0, # Silence CatBoost output for the study
        'task_type': "CPU"
    }
    if use_early_stopping:
        model_params['early_stopping_rounds'] = 50

    model = CatBoostRegressor(**model_params)
    
    # Only use eval_set if early stopping is enabled
    eval_set = (X_val, y_val) if use_early_stopping else None
    
    model.fit(X_train, y_train, cat_features=cat_features_indices, eval_set=eval_set, use_best_model=use_early_stopping)

    # Evaluate on the validation set
    val_preds = model.predict(X_val)
    val_preds = np.maximum(0, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    return rmse


if __name__ == '__main__':
    results = {}
    print("Running ablation study...")

    # --- Baseline Experiment ---
    baseline_rmse = run_experiment(
        use_length_features=True,
        use_pavement_data=True,
        use_early_stopping=True
    )
    if baseline_rmse == float('inf'):
        exit()
        
    results['Baseline'] = baseline_rmse
    print(f"\nBaseline (all components enabled) RMSE: {baseline_rmse:.4f}")

    # --- Ablation Experiments ---
    
    # Ablation 1: No Pavement Maintenance Data
    no_pave_rmse = run_experiment(
        use_length_features=True,
        use_pavement_data=False,
        use_early_stopping=True
    )
    results['No Pavement Maintenance Data'] = no_pave_rmse
    print(f"Ablation 'No Pavement Data' RMSE: {no_pave_rmse:.4f} (Impact: {no_pave_rmse - baseline_rmse:+.4f})")

    # Ablation 2: No Simple Length Features
    no_len_rmse = run_experiment(
        use_length_features=False,
        use_pavement_data=True,
        use_early_stopping=True
    )
    results['No Simple Length Features'] = no_len_rmse
    print(f"Ablation 'No Length Features' RMSE: {no_len_rmse:.4f} (Impact: {no_len_rmse - baseline_rmse:+.4f})")
    
    # Ablation 3: No Early Stopping
    no_es_rmse = run_experiment(
        use_length_features=True,
        use_pavement_data=True,
        use_early_stopping=False
    )
    results['No Early Stopping'] = no_es_rmse
    print(f"Ablation 'No Early Stopping' RMSE: {no_es_rmse:.4f} (Impact: {no_es_rmse - baseline_rmse:+.4f})")
    
    # --- Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    
    impacts = {
        'Pavement Maintenance Data': abs(no_pave_rmse - baseline_rmse),
        'Simple Length Features': abs(no_len_rmse - baseline_rmse),
        'Early Stopping': abs(no_es_rmse - baseline_rmse)
    }

    if not impacts:
        print("No ablation results to compare.")
    else:
        most_impactful_component = max(impacts, key=impacts.get)
        print(f"The most impactful component is '{most_impactful_component}' with an absolute impact of {impacts[most_impactful_component]:.4f} on RMSE.")
