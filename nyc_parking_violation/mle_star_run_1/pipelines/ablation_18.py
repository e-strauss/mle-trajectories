
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

def create_dummy_data():
    """Creates a dummy CSV file for the script to run if it doesn't exist."""
    input_dir = './input'
    file_path = os.path.join(input_dir, 'violations_per_street_2022.csv')
    if not os.path.exists(file_path):
        print("Dummy data not found. Creating dummy 'violations_per_street_2022.csv'...")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        data = {
            'Street Name': [
                'BROADWAY', 'BROADWAY', '5TH AVE', '5TH AVE', 'MAIN ST', 'MAIN ST',
                'PARK AVE', 'PARK AVE', 'ELM ST', 'ELM ST', 'WALL ST', 'WALL ST',
                'OAK ST', 'OAK ST', 'PINE ST', 'PINE ST'
            ],
            'Violation Description': [
                'NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT',
                'NO PARKING', 'DOUBLE PARKING', 'FIRE HYDRANT', 'NO PARKING',
                'DOUBLE PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING',
                'NO PARKING', 'FIRE HYDRANT', 'DOUBLE PARKING', 'FIRE HYDRANT'
            ],
            'Violation Count': [
                150, 80, 120, 40, 200, 90, 30, 110, 75, 25, 180, 85, 160, 45, 95, 35
            ]
        }
        # Repeat the data to make it slightly larger and more variable
        df = pd.DataFrame(data)
        df_extended = pd.concat([df] * 10, ignore_index=True)
        # Add some noise
        np.random.seed(42)
        df_extended['Violation Count'] += np.random.randint(-10, 10, size=len(df_extended))
        df_extended['Violation Count'] = df_extended['Violation Count'].clip(lower=1)
        df_extended.to_csv(file_path, index=False)
        print("Dummy data created.")

def run_experiment(name, log_transform_agg_features, remove_negative_clipping):
    """
    Runs a single experiment with a specific configuration.
    
    Args:
        name (str): The name of the experiment.
        log_transform_agg_features (bool): If True, applies log transform to aggregate features.
        remove_negative_clipping (bool): If True, disables clipping of negative predictions.
        
    Returns:
        float: The Root Mean Squared Error (RMSE) for the experiment.
    """
    # --- 1. Data Loading and Basic Cleaning ---
    df_train = pd.read_csv('./input/violations_per_street_2022.csv')
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---
    # a. Aggregate features (leaky, as in the original script)
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    # --- ABLATION POINT 1: Log-transform aggregate features ---
    if log_transform_agg_features:
        df_train['description_mean_count'] = np.log1p(df_train['description_mean_count'])
        df_train['street_mean_count'] = np.log1p(df_train['street_mean_count'])

    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])
    
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
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation Performance ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    
    # --- ABLATION POINT 2: Clipping negative predictions ---
    if not remove_negative_clipping:
        val_preds[val_preds < 0] = 0
    
    final_validation_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f"Performance for '{name}': {final_validation_score:.4f}")
    
    return final_validation_score

def main():
    """Main function to run the ablation study."""
    create_dummy_data()
    
    results = {}

    print("--- Running Ablation Study ---")

    # Baseline experiment (matches the original script logic)
    baseline_score = run_experiment(
        name="Baseline", 
        log_transform_agg_features=False, 
        remove_negative_clipping=False
    )
    results["Baseline"] = baseline_score

    # Ablation 1: Add log transformation to aggregate features
    score1 = run_experiment(
        name="Ablation 1: Add Log-Transform on Aggregate Features",
        log_transform_agg_features=True,
        remove_negative_clipping=False
    )
    results["Log-Transform Aggregate Features"] = score1

    # Ablation 2: Remove the clipping of negative predictions
    score2 = run_experiment(
        name="Ablation 2: Remove Negative Prediction Clipping",
        log_transform_agg_features=False,
        remove_negative_clipping=True
    )
    results["Remove Negative Prediction Clipping"] = score2

    print("\n--- Ablation Study Conclusion ---")
    
    # Calculate performance degradation (higher is worse)
    # A positive degradation means the RMSE increased, so performance worsened.
    degradation1 = results["Log-Transform Aggregate Features"] - baseline_score
    degradation2 = results["Remove Negative Prediction Clipping"] - baseline_score
    
    impacts = {
        "Log-Transform Aggregate Features": abs(degradation1),
        "Remove Negative Prediction Clipping": abs(degradation2)
    }

    # Determine the most impactful component
    if not impacts:
        most_impactful_component = "No components were tested."
    else:
        most_impactful_component = max(impacts, key=impacts.get)
    
    print(f"Baseline RMSE: {baseline_score:.4f}")
    print(f"Ablation 'Log-Transform Aggregate Features' resulted in an RMSE of {results['Log-Transform Aggregate Features']:.4f}, a change of {degradation1:+.4f}.")
    print(f"Ablation 'Remove Negative Prediction Clipping' resulted in an RMSE of {results['Remove Negative Prediction Clipping']:.4f}, a change of {degradation2:+.4f}.")
    print("\nConclusion: Based on the absolute change in RMSE, the component that contributes the most to the overall performance is:")
    print(f"'{most_impactful_component}' with an impact of {impacts[most_impactful_component]:.4f}.")

if __name__ == '__main__':
    main()
