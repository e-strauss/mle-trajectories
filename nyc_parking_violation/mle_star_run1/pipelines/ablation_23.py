
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_data(path='./input/violations_per_street_2022.csv'):
    """Loads data from the specified path or creates a dummy dataframe."""
    if os.path.exists(path):
        print(f"Loading data from {path}")
        df = pd.read_csv(path)
    else:
        print("Data file not found. Creating a dummy dataset for demonstration.")
        data = {
            'street_name': ['STREET_A'] * 5 + ['STREET_B'] * 3 + ['STREET_C'] * 2,
            'violation_description': ['DESC_1', 'DESC_1', 'DESC_2', 'DESC_2', 'DESC_3'] + ['DESC_1', 'DESC_2', 'DESC_2'] + ['DESC_3', 'DESC_3'],
            'violation_count': [10, 12, 50, 55, 20, 15, 60, 65, 22, 25]
        }
        df = pd.DataFrame(data)
    return df

def run_experiment(df_train, smoothing_factor):
    """Runs a single training and evaluation experiment."""
    
    # --- 1. Data Cleaning ---
    df = df_train.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---
    # a. Aggregate features using smoothed target encoding
    if smoothing_factor is not None:
        # Use smoothed target encoding
        global_mean = df['violation_count'].mean()
        
        # Aggregate stats for 'violation_description'
        desc_agg = df.groupby('violation_description')['violation_count'].agg(['mean', 'size'])
        desc_agg['description_mean_count'] = (desc_agg['size'] * desc_agg['mean'] + smoothing_factor * global_mean) / (desc_agg['size'] + smoothing_factor)
        
        # Aggregate stats for 'street_name'
        street_agg = df.groupby('street_name')['violation_count'].agg(['mean', 'size'])
        street_agg['street_mean_count'] = (street_agg['size'] * street_agg['mean'] + smoothing_factor * global_mean) / (street_agg['size'] + smoothing_factor)

        # Merge smoothed features
        df = pd.merge(df, desc_agg[['description_mean_count']], on='violation_description', how='left')
        df = pd.merge(df, street_agg[['street_mean_count']], on='street_name', how='left')
    else:
        # Use simple, non-smoothed mean encoding
        description_mean = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
        street_mean = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
        df = pd.merge(df, description_mean, on='violation_description', how='left')
        df = pd.merge(df, street_mean, on='street_name', how='left')


    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df['street_name_encoded'] = le_street.fit_transform(df['street_name'])
    df['violation_description_encoded'] = le_desc.fit_transform(df['violation_description'])
    
    # --- 3. Model Training ---
    features = [
        'street_name_encoded', 
        'violation_description_encoded', 
        'description_mean_count', 
        'street_mean_count'
    ]
    target = 'violation_count'
    
    df['log_target'] = np.log1p(df[target])

    X = df[features]
    y = df['log_target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return score

def main():
    """Main function to run the ablation study."""
    
    df_train = load_data()

    experiments = {
        "Baseline (Smoothing Factor=20)": {
            "smoothing_factor": 20
        },
        "Ablation 1 (Reduced Smoothing Factor=1)": {
            "smoothing_factor": 1
        },
        "Ablation 2 (No Smoothing - Simple Mean)": {
            "smoothing_factor": None
        }
    }

    results = {}
    print("\n--- Starting Ablation Study ---")
    
    for name, params in experiments.items():
        score = run_experiment(df_train.copy(), **params)
        results[name] = score
        print(f"Performance for '{name}': RMSE = {score:.4f}")

    # --- 5. Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    baseline_score = results["Baseline (Smoothing Factor=20)"]
    
    impacts = {}
    for name, score in results.items():
        if "Ablation" in name:
            degradation = score - baseline_score
            # Extract component name from experiment name
            component_name = name.split("(")[1].split(")")[0]
            impacts[component_name] = degradation
    
    if not impacts:
        print("No ablation experiments were run to compare.")
        return

    most_impactful_component = max(impacts, key=impacts.get)
    max_degradation = impacts[most_impactful_component]

    print(f"Baseline RMSE: {baseline_score:.4f}\n")
    for component, degradation in impacts.items():
        print(f"Impact of '{component}': {degradation:+.4f} RMSE")

    print("\n--- Final Conclusion ---")
    if max_degradation > 0:
        print(f"The component that contributes most to performance is '{most_impactful_component}'.")
        print(f"Modifying it caused a performance degradation of {max_degradation:.4f} RMSE.")
    else:
        print("No single component removal led to a significant performance degradation.")
        print("This may indicate that the tested components have little effect or that their effects are positive.")


if __name__ == '__main__':
    main()
