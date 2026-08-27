
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def run_experiment(train_path, use_std_features=True, nan_fill_strategy='median'):
    """
    Runs a single training and validation experiment with specified configurations.

    Args:
        train_path (str): Path to the training data file.
        use_std_features (bool): If True, includes standard deviation in aggregate features.
        nan_fill_strategy (str): Strategy for filling NaNs in features, either 'median' or 'zero'.

    Returns:
        float: The Root Mean Squared Error (RMSE) for the experiment.
    """
    try:
        df = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Warning: Training file not found at {train_path}. Using a small dummy dataset.")
        data = {
            'Street Name': [f'Street {i}' for i in range(100)] + ['Single Appearance St'],
            'Violation Description': [f'Desc {i % 10}' for i in range(101)],
            'Violation Count': np.random.randint(1, 500, 101)
        }
        df = pd.DataFrame(data)

    # --- 1. Basic Cleaning ---
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---
    # a. Create aggregate features (mean and std)
    description_agg = df.groupby('violation_description')['violation_count'].agg(['mean', 'std']).reset_index()
    description_agg.columns = ['violation_description', 'description_mean_count', 'description_std_count']
    
    street_agg = df.groupby('street_name')['violation_count'].agg(['mean', 'std']).reset_index()
    street_agg.columns = ['street_name', 'street_mean_count', 'street_std_count']
    
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    # b. Categorical Feature Encoding
    df['street_name_encoded'] = LabelEncoder().fit_transform(df['street_name'])
    df['violation_description_encoded'] = LabelEncoder().fit_transform(df['violation_description'])
        
    # --- 3. Model Training ---
    # Define feature set based on experiment configuration
    features = [
        'street_name_encoded', 
        'violation_description_encoded', 
        'description_mean_count', 
        'street_mean_count'
    ]
    if use_std_features:
        features.extend(['description_std_count', 'street_std_count'])

    # Handle NaNs created by aggregation (e.g., std of a single-member group)
    for col in features:
        if df[col].isnull().any():
            if nan_fill_strategy == 'median':
                fill_value = df[col].median()
            else: # 'zero' strategy
                fill_value = 0
            df[col].fillna(fill_value, inplace=True)

    # Define target and apply log transformation
    target = 'violation_count'
    df['log_target'] = np.log1p(df[target])

    # Split data for validation
    X = df[features]
    y = df['log_target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    # Train the model
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    return np.sqrt(mean_squared_error(y_val_original, val_preds))


# --- Ablation Study Main Execution ---

# Ensure a dummy input file exists if the real one is missing
train_file_path = './input/violations_per_street_2022.csv'
if not os.path.exists(train_file_path):
    os.makedirs('./input', exist_ok=True)
    dummy_df = pd.DataFrame({
        'Street Name': [f'Main St' for _ in range(50)] + [f'Elm St' for _ in range(50)] + ['Orchard Ave'],
        'Violation Description': [f'No Parking {i%2}' for i in range(101)],
        'Violation Count': np.random.randint(10, 200, 101)
    })
    dummy_df.to_csv(train_file_path, index=False)
    print("Created dummy 'violations_per_street_2022.csv' for execution.")

# --- Run Experiments ---
print("Running ablation study...")

# Experiment 1: Baseline (Using STD features and Median NaN filling)
baseline_rmse = run_experiment(
    train_file_path,
    use_std_features=True, 
    nan_fill_strategy='median'
)
print(f"Baseline RMSE (with STD features, median-fill NaNs): {baseline_rmse:.4f}")

# Experiment 2: Ablation of Standard Deviation features
ablation1_rmse = run_experiment(
    train_file_path,
    use_std_features=False, 
    nan_fill_strategy='median'
)
print(f"Ablation 1 RMSE (No STD features): {ablation1_rmse:.4f}")

# Experiment 3: Ablation of NaN filling strategy (use zero-fill instead of median)
ablation2_rmse = run_experiment(
    train_file_path,
    use_std_features=True, 
    nan_fill_strategy='zero'
)
print(f"Ablation 2 RMSE (STD features, zero-fill NaNs): {ablation2_rmse:.4f}")

print("\n--- Ablation Analysis ---")

# Calculate performance degradation (higher value is worse)
degradation_no_std = ablation1_rmse - baseline_rmse
degradation_zero_fill = ablation2_rmse - baseline_rmse

print(f"Performance degradation from removing STD features: {degradation_no_std:+.4f} RMSE")
print(f"Performance degradation from using zero-fill for NaNs: {degradation_zero_fill:+.4f} RMSE")

# Determine the most impactful component based on the largest increase in RMSE
impacts = {
    "STD Aggregate Features": degradation_no_std,
    "Median NaN Filling Strategy": degradation_zero_fill
}

# Filter out improvements (negative degradation) before finding the max
positive_impacts = {k: v for k, v in impacts.items() if v > 0}

if not positive_impacts:
    most_impactful_component = "None of the tested components worsened performance"
else:
    most_impactful_component = max(positive_impacts, key=positive_impacts.get)

print(f"\nConclusion: The component that contributes most to performance is '{most_impactful_component}'.")
