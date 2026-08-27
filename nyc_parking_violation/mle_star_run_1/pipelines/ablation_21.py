
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
from collections import OrderedDict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def run_experiment(use_aggregate_features=True, learning_rate=0.1):
    """
    Runs a single training and validation experiment with a specific configuration.
    
    Args:
        use_aggregate_features (bool): If True, includes mean-encoded features.
        learning_rate (float): The learning rate for the LGBM model.

    Returns:
        float: The Root Mean Squared Error (RMSE) on the validation set.
    """
    # --- 1. Data Loading ---
    # Using a deterministic dummy dataset for reproducibility
    np.random.seed(42)
    data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'OAK ST', 'ELM ST'] * 50,
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'DOUBLE PARKING', 'NO STANDING'] * 50,
        'violation_count': np.random.lognormal(mean=3, sigma=1.5, size=200).astype(int) + 1
    }
    df_train = pd.DataFrame(data)

    # --- 2. Basic Cleaning & Feature Engineering ---
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # a. Aggregate features (leaky, for simplicity, as in the original script)
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])
    
    # --- 3. Model Training ---
    base_features = ['street_name_encoded', 'violation_description_encoded']
    agg_features = ['description_mean_count', 'street_mean_count']
    
    features = base_features
    if use_aggregate_features:
        features += agg_features

    target = 'violation_count'
    df_train['log_target'] = np.log1p(df_train[target])

    X = df_train[features]
    y = df_train['log_target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    # Use specified learning rate
    lgbm = lgb.LGBMRegressor(random_state=42, learning_rate=learning_rate)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation Performance ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    final_validation_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return final_validation_score

# --- Ablation Study Execution ---

# Use OrderedDict to maintain the order of experiments for printing
ablation_results = OrderedDict()

# Experiment 1: Baseline
# All features are used, and the default learning rate is applied.
baseline_score = run_experiment(use_aggregate_features=True, learning_rate=0.1)
ablation_results['Baseline (Full Features, Default LR)'] = baseline_score

# Experiment 2: Ablation of Aggregate Features
# Train the model without the mean-encoded aggregate features.
no_agg_score = run_experiment(use_aggregate_features=False, learning_rate=0.1)
ablation_results['Ablation: No Aggregate Features'] = no_agg_score

# Experiment 3: Ablation of Learning Rate
# Train the model with a much smaller learning rate to see its effect.
low_lr_score = run_experiment(use_aggregate_features=True, learning_rate=0.01)
ablation_results['Ablation: Reduced Learning Rate (0.01)'] = low_lr_score

# --- Analysis and Conclusion ---

print("--- Ablation Study Results ---")
print(f"Lower RMSE is better.\n")

performance_degradation = {}

for name, score in ablation_results.items():
    degradation = score - baseline_score
    print(f"{name}: RMSE = {score:.4f}")
    if name != 'Baseline (Full Features, Default LR)':
        performance_degradation[name] = degradation

print("\n--- Performance Impact ---")
# Find the component whose removal caused the largest increase in RMSE (worst performance)
max_degradation = -1
most_impactful_component = "None"

for name, degradation in performance_degradation.items():
    print(f"Impact of '{name}': {degradation:+.4f} RMSE")
    if degradation > max_degradation:
        max_degradation = degradation
        most_impactful_component = name.split(':')[1].strip()

print("\n--- Conclusion ---")
if max_degradation <= 0:
    print("No component's removal led to a performance degradation. The baseline configuration is not optimal.")
else:
    print(f"The component that contributes the most to overall performance is: {most_impactful_component}")

