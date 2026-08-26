
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# --- 0. Setup: Create a dummy dataset for the ablation study ---
if not os.path.exists('./input'):
    os.makedirs('./input')

data = {
    'Street Name': ['BROADWAY', 'MAIN ST', 'PARK AVE', 'BROADWAY', 'MAIN ST', 'BROADWAY', 'PARK AVE', 'ELM ST', 'OAK AVE', 'MAIN ST'] * 10,
    'Violation Description': ['NO PARKING-STREET CLEANING', 'FAIL TO DISP/PAY MUNI METER', 'NO PARKING-STREET CLEANING', 'FIRE HYDRANT', 'FAIL TO DISP/PAY MUNI METER', 'NO PARKING-STREET CLEANING', 'FAIL TO DISP/PAY MUNI METER', 'FIRE HYDRANT', 'NO PARKING-STREET CLEANING', 'FAIL TO DISP/PAY MUNI METER'] * 10,
    'Violation Count': [150, 80, 120, 30, 95, 160, 110, 25, 130, 85] * 10
}
df_dummy = pd.DataFrame(data)
# Introduce some noise/variation
np.random.seed(42)
df_dummy['Violation Count'] += np.random.randint(-10, 10, df_dummy.shape[0])
df_dummy.loc[df_dummy['Violation Count'] < 0, 'Violation Count'] = 0
df_dummy.to_csv('./input/violations_per_street_2022.csv', index=False)


# --- 1. Define Ablation Functions ---

def get_baseline_performance():
    """Runs the original script logic and returns the validation RMSE."""
    df_train = pd.read_csv('./input/violations_per_street_2022.csv')
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # a. Aggregate features
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    # b. Categorical Feature Encoding (with '<unknown>' handling)
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    all_streets = df_train['street_name'].unique()
    all_descs = df_train['violation_description'].unique()
    le_street.fit(np.append(all_streets, '<unknown>'))
    le_desc.fit(np.append(all_descs, '<unknown>'))
    df_train['street_name_encoded'] = le_street.transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.transform(df_train['violation_description'])
    
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
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    y_val_original = np.expm1(y_val)
    
    return np.sqrt(mean_squared_error(y_val_original, val_preds))

def ablate_target_encoding():
    """Ablation 1: Remove target-encoded aggregate features."""
    df_train = pd.read_csv('./input/violations_per_street_2022.csv')
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')
    
    # Target encoding is NOT performed in this ablation
    
    # Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    le_street.fit(df_train['street_name'])
    le_desc.fit(df_train['violation_description'])
    df_train['street_name_encoded'] = le_street.transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.transform(df_train['violation_description'])
    
    # Features list without target-encoded features
    features = [
        'street_name_encoded', 
        'violation_description_encoded'
    ]
    target = 'violation_count'
    
    df_train['log_target'] = np.log1p(df_train[target])
    
    X = df_train[features]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    y_val_original = np.expm1(y_val)
    
    return np.sqrt(mean_squared_error(y_val_original, val_preds))

def ablate_label_encoding():
    """Ablation 2: Remove standard label-encoded features."""
    df_train = pd.read_csv('./input/violations_per_street_2022.csv')
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # a. Aggregate features
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')
    
    # Label encoding is NOT added to features
    
    # Features list without standard label-encoded features
    features = [
        'description_mean_count', 
        'street_mean_count'
    ]
    target = 'violation_count'
    
    df_train['log_target'] = np.log1p(df_train[target])
    
    X = df_train[features]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    y_val_original = np.expm1(y_val)
    
    return np.sqrt(mean_squared_error(y_val_original, val_preds))
    
def ablate_column_standardization():
    """Ablation 3: Remove the column name standardization step."""
    # Note: This requires using original column names throughout the function.
    df_train = pd.read_csv('./input/violations_per_street_2022.csv')
    # Column standardization is NOT performed
    
    # Use original column names
    desc_col = 'Violation Description'
    street_col = 'Street Name'
    target_col = 'Violation Count'

    # a. Aggregate features
    description_agg = df_train.groupby(desc_col)[target_col].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby(street_col)[target_col].mean().to_frame('street_mean_count')
    df_train = pd.merge(df_train, description_agg, on=desc_col, how='left')
    df_train = pd.merge(df_train, street_agg, on=street_col, how='left')

    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    le_street.fit(df_train[street_col])
    le_desc.fit(df_train[desc_col])
    df_train['street_name_encoded'] = le_street.transform(df_train[street_col])
    df_train['violation_description_encoded'] = le_desc.transform(df_train[desc_col])
    
    features = [
        'street_name_encoded', 
        'violation_description_encoded', 
        'description_mean_count', 
        'street_mean_count'
    ]
    
    df_train['log_target'] = np.log1p(df_train[target_col])
    
    X = df_train[features]
    y = df_train['log_target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    y_val_original = np.expm1(y_val)
    
    return np.sqrt(mean_squared_error(y_val_original, val_preds))


# --- 2. Run Ablation Study and Report Results ---

results = {}
results['Baseline (All Features)'] = get_baseline_performance()
results['Ablation 1 (No Target Encoding)'] = ablate_target_encoding()
results['Ablation 2 (No Label Encoding)'] = ablate_label_encoding()
results['Ablation 3 (No Column Standardization)'] = ablate_column_standardization()

print("--- Ablation Study Results (Validation RMSE) ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# --- 3. Analyze and Conclude ---
baseline_score = results['Baseline (All Features)']
performance_degradation = {
    "Target Encoded Features": results['Ablation 1 (No Target Encoding)'] - baseline_score,
    "Standard Label Encoded Features": results['Ablation 2 (No Label Encoding)'] - baseline_score,
    "Column Name Standardization": results['Ablation 3 (No Column Standardization)'] - baseline_score
}

# Find the component whose removal caused the largest increase in RMSE
most_impactful_component = max(performance_degradation, key=performance_degradation.get)
max_degradation = performance_degradation[most_impactful_component]

print("\n--- Conclusion ---")
if max_degradation > 0:
    print(f"The component that contributes most to the performance is: {most_impactful_component}.")
    print(f"Removing it degraded performance (increased RMSE) by: {max_degradation:.4f}.")
else:
    print("No single component removal resulted in a clear performance degradation.")
