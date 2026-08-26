
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def load_data():
    """Creates a synthetic dataset for the experiment if the original file is not found."""
    print("Warning: Training file not found. Falling back to a synthetic dataset.")
    data = {
        'street_name': ['BROADWAY'] * 40 + ['MAIN ST'] * 30 + ['PARK AVE'] * 15 + ['OAK ST'] * 10 + ['ELM ST'] * 5,
        'violation_description': ['NO PARKING-STREET CLEANING'] * 25 + ['FAIL TO DISP/PAY METER'] * 25 + ['FIRE HYDRANT'] * 25 + ['NO STANDING-DAY/TIME LIMITS'] * 25,
        'violation_count': list(np.random.poisson(20, 25)) + list(np.random.poisson(50, 25)) + list(np.random.poisson(5, 25)) + list(np.random.poisson(30, 25))
    }
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def run_experiment(df, use_smoothing=True, use_label_encoding=True, description="Baseline"):
    """
    Runs a single training and validation experiment with specified ablations.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        use_smoothing (bool): If True, applies smoothing to target encoding. If False, uses simple mean.
        use_label_encoding (bool): If True, includes label encoded features.
        description (str): A description for the current experiment run.

    Returns:
        float: The validation RMSE of the experiment.
    """
    df_train = df.copy()

    # --- Feature Engineering ---
    # Ablation point 1: The smoothing factor for target encoding
    smoothing_factor = 20 if use_smoothing else 0
    global_mean = df_train['violation_count'].mean()

    # a. Aggregate features (Smoothed or Simple Target Encoding)
    description_agg = df_train.groupby('violation_description')['violation_count'].agg(['mean', 'size'])
    description_agg['description_smoothed_mean'] = (description_agg['mean'] * description_agg['size'] + global_mean * smoothing_factor) / (description_agg['size'] + smoothing_factor)
    
    street_agg = df_train.groupby('street_name')['violation_count'].agg(['mean', 'size'])
    street_agg['street_smoothed_mean'] = (street_agg['mean'] * street_agg['size'] + global_mean * smoothing_factor) / (street_agg['size'] + smoothing_factor)

    df_train = pd.merge(df_train, description_agg[['description_smoothed_mean']], on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg[['street_smoothed_mean']], on='street_name', how='left')

    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])

    # --- Model Training ---
    # Ablation point 2: Inclusion of label encoded features
    features = [
        'description_smoothed_mean', 
        'street_smoothed_mean'
    ]
    if use_label_encoding:
        features.extend(['street_name_encoded', 'violation_description_encoded'])
        
    target = 'violation_count'
    df_train['log_target'] = np.log1p(df_train[target])

    X = df_train[features]
    y = df_train['log_target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)

    # --- Validation ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return rmse

def main():
    """Main function to run the ablation study."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv')
    args, _ = parser.parse_known_args()

    try:
        df = pd.read_csv(args.train_path)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
    except FileNotFoundError:
        df = load_data()

    # --- Run Experiments ---
    baseline_rmse = run_experiment(df, use_smoothing=True, use_label_encoding=True, description="Baseline")
    
    # Ablation 1: Remove Smoothing (set factor to 0, which is equivalent to simple mean encoding)
    ablation_no_smoothing_rmse = run_experiment(df, use_smoothing=False, use_label_encoding=True, description="No Smoothing")
    
    # Ablation 2: Remove Label Encoded Features
    ablation_no_label_encoding_rmse = run_experiment(df, use_smoothing=True, use_label_encoding=False, description="No Label Encoding")

    # --- Report Results ---
    print("\n--- Ablation Study Results ---")
    print(f"Baseline (Smoothed Target Encoding + Label Encoding): RMSE = {baseline_rmse:.4f}")
    
    degradation_no_smoothing = ablation_no_smoothing_rmse - baseline_rmse
    print(f"Ablation 1 (No Smoothing): RMSE = {ablation_no_smoothing_rmse:.4f} (Performance Degradation: {degradation_no_smoothing:+.4f})")
    
    degradation_no_label_encoding = ablation_no_label_encoding_rmse - baseline_rmse
    print(f"Ablation 2 (No Label Encoding): RMSE = {ablation_no_label_encoding_rmse:.4f} (Performance Degradation: {degradation_no_label_encoding:+.4f})")
    
    # --- Conclusion ---
    print("\n--- Conclusion ---")
    impacts = {
        "Target Encoding Smoothing": degradation_no_smoothing,
        "Label Encoded Features": degradation_no_label_encoding,
    }

    if not impacts or max(impacts.values()) <= 0:
        print("No component removal led to a performance degradation. The baseline is optimal or all components are neutral/beneficial.")
    else:
        most_impactful_component = max(impacts, key=impacts.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

if __name__ == '__main__':
    main()
