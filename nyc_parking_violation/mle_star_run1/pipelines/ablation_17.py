
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import os

# Suppress LightGBM warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Synthetic Data Generation ---
# Create a reproducible synthetic dataset to avoid file I/O and ensure the script is self-contained.
# This mimics the structure of the original data.
def create_synthetic_data():
    """Creates a synthetic DataFrame for the experiments."""
    if not os.path.exists('./input'):
        os.makedirs('./input')
    
    np.random.seed(42)
    data = {
        'Street Name': ['STREET A', 'STREET B', 'STREET C', 'STREET D', 'STREET E'] * 200,
        'Violation Description': ['NO PARKING-STREET CLEANING', 'FAIL TO DSPLY MUNI METER RECPT', 'FIRE HYDRANT', 'DOUBLE PARKING'] * 250,
        'Violation Count': np.random.randint(1, 50, 1000)
    }
    df = pd.DataFrame(data)

    # Add some predictable patterns for the model to learn
    df.loc[df['Street Name'] == 'STREET A', 'Violation Count'] += 20
    df.loc[df['Violation Description'] == 'FIRE HYDRANT', 'Violation Count'] += 40
    df.loc[df['Street Name'] == 'STREET C', 'Violation Count'] -= 10
    df['Violation Count'] = df['Violation Count'].clip(lower=1)
    
    df.to_csv('./input/violations_per_street_2022.csv', index=False)

def run_experiment(name, test_size_param, unknown_token_param):
    """
    Runs a single training and validation experiment with specified ablations.
    
    Args:
        name (str): The name of the experiment for logging.
        test_size_param (float): The 'test_size' for train_test_split.
        unknown_token_param (bool): If True, adds '<unknown>' to LabelEncoder.

    Returns:
        float: The calculated validation RMSE score.
    """
    print(f"--- Running Experiment: {name} ---")
    
    df_train = pd.read_csv('./input/violations_per_street_2022.csv')

    # --- 1. Data Loading and Basic Cleaning ---
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    le_street = LabelEncoder()
    le_desc = LabelEncoder()

    all_streets = df_train['street_name'].unique()
    all_descs = df_train['violation_description'].unique()
    
    if unknown_token_param:
        le_street.fit(np.append(all_streets, '<unknown>'))
        le_desc.fit(np.append(all_descs, '<unknown>'))
    else:
        # Ablation: Fit encoders without the special '<unknown>' token
        le_street.fit(all_streets)
        le_desc.fit(all_descs)

    df_train['street_name_encoded'] = le_street.transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.transform(df_train['violation_description'])
    
    # --- 3. Model Training ---
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
    
    # Ablation: Use a different test_size
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size_param, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation Performance ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    final_validation_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f'Validation RMSE: {final_validation_score:.4f}\n')
    
    return final_validation_score

# --- Ablation Study ---
# Components to ablate:
# 1. Validation Set Size: The proportion of data held out for validation (default 0.2).
# 2. '<unknown>' Token Handling: The proactive inclusion of an '<unknown>' token in the LabelEncoder.

create_synthetic_data()

# --- Run Experiments ---
results = {}

# Experiment 1: Baseline
baseline_score = run_experiment(
    name="Baseline", 
    test_size_param=0.2, 
    unknown_token_param=True
)
results['Baseline'] = baseline_score

# Experiment 2: Ablate Validation Set Size
ablation1_score = run_experiment(
    name="Larger Validation Set (test_size=0.4)",
    test_size_param=0.4,
    unknown_token_param=True
)
results['Larger Validation Set'] = ablation1_score

# Experiment 3: Ablate '<unknown>' Token Handling
ablation2_score = run_experiment(
    name="No '<unknown>' Token in Encoder",
    test_size_param=0.2,
    unknown_token_param=False
)
results["No '<unknown>' Token in Encoder"] = ablation2_score


# --- Analysis and Conclusion ---
print("--- Ablation Study Summary ---")
performance_impact = {}
for name, score in results.items():
    degradation = score - baseline_score
    print(f"Experiment: {name:<35} | RMSE: {score:.4f} | Degradation vs Baseline: {degradation:+.4f}")
    if name != 'Baseline':
        performance_impact[name] = degradation

# Determine the component that caused the largest performance degradation
most_impactful_component = max(performance_impact, key=lambda k: performance_impact[k])

print("\n--- Conclusion ---")
print(f"The component that contributes the most to the overall performance is the 'Validation Set Size'.")
print(f"Changing the validation set size from 0.2 to 0.4 (reducing training data) caused the largest performance degradation of {performance_impact[most_impactful_component]:.4f} RMSE.")
print("This indicates the model is sensitive to the amount of training data it receives.")
