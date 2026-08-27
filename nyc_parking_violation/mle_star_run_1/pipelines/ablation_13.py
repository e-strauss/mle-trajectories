
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os
import warnings
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

warnings.filterwarnings('ignore', category=UserWarning)

def safe_label_transform(le, series):
    """Safely transforms a series using a fitted LabelEncoder, handling unseen values."""
    known_labels = set(le.classes_)
    return series.apply(lambda s: s if s in known_labels else '<unknown>')

def run_experiment(config):
    """
    Runs a single training and validation experiment based on a configuration dictionary.
    
    Args:
        config (dict): A dictionary specifying the experiment setup.
            'agg_method' (str): 'mean' or 'median' for creating aggregate features.
            'proper_encoding' (bool): If True, fits encoders only on the training set. 
                                      If False, fits on the whole dataset before splitting (leaky).
    """
    
    # --- 1. Data Loading ---
    # Use a dummy dataframe if the file doesn't exist to ensure the script can run.
    if not os.path.exists('./input/violations_per_street_2022.csv'):
        print("Warning: Input file not found. Using a small dummy dataset for demonstration.")
        data = {
            'Street Name': [f'STREET {i}' for i in range(50)] * 10,
            'Violation Description': [f'DESC {i}' for i in range(10)] * 50,
            'Violation Count': np.random.randint(10, 500, 500)
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv('./input/violations_per_street_2022.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---
    agg_method = config.get('agg_method', 'mean')
    if agg_method == 'median':
        description_agg = df.groupby('violation_description')['violation_count'].median().to_frame('description_agg_val')
        street_agg = df.groupby('street_name')['violation_count'].median().to_frame('street_agg_val')
    else:  # Default to 'mean'
        description_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_agg_val')
        street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_agg_val')
        
    df = pd.merge(df, description_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')

    df['log_target'] = np.log1p(df['violation_count'])
    
    # --- 3. Splitting and Encoding Strategy ---
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df = val_df.copy()
    train_df = train_df.copy()

    y_val_original = np.expm1(val_df['log_target'])
    
    le_street = LabelEncoder()
    le_desc = LabelEncoder()

    if config.get('proper_encoding', False):
        # Proper method: Fit encoders ONLY on the training data.
        train_streets = np.append(train_df['street_name'].unique(), '<unknown>')
        train_descs = np.append(train_df['violation_description'].unique(), '<unknown>')
        
        le_street.fit(train_streets)
        le_desc.fit(train_descs)
        
        train_df['street_name_encoded'] = le_street.transform(safe_label_transform(le_street, train_df['street_name']))
        train_df['violation_description_encoded'] = le_desc.transform(safe_label_transform(le_desc, train_df['violation_description']))
        
        val_df['street_name_encoded'] = le_street.transform(safe_label_transform(le_street, val_df['street_name']))
        val_df['violation_description_encoded'] = le_desc.transform(safe_label_transform(le_desc, val_df['violation_description']))
    
    else:
        # Leaky method (original script): Fit encoders on the whole dataset before splitting.
        all_streets = np.append(df['street_name'].unique(), '<unknown>')
        all_descs = np.append(df['violation_description'].unique(), '<unknown>')
        
        le_street.fit(all_streets)
        le_desc.fit(all_descs)
        
        train_df['street_name_encoded'] = le_street.transform(train_df['street_name'])
        train_df['violation_description_encoded'] = le_desc.transform(train_df['violation_description'])
        
        val_df['street_name_encoded'] = le_street.transform(val_df['street_name'])
        val_df['violation_description_encoded'] = le_desc.transform(val_df['violation_description'])

    # --- 4. Model Training ---
    features = ['street_name_encoded', 'violation_description_encoded', 'description_agg_val', 'street_agg_val']
    
    X_train = train_df[features]
    y_train = train_df['log_target']
    X_val = val_df[features]

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- 5. Validation ---
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return rmse


def main():
    """Main function to run the ablation study."""
    
    if not os.path.exists('./input'):
        os.makedirs('./input')

    results = {}
    
    # --- Experiment 1: Baseline ---
    # Aggregates with 'mean', LabelEncoders fit before split (leaky).
    baseline_config = {'agg_method': 'mean', 'proper_encoding': False}
    results['Baseline'] = run_experiment(baseline_config)
    print(f"Baseline RMSE (Mean Aggregates, Leaky Encoding): {results['Baseline']:.4f}")
    
    # --- Experiment 2: Ablation of Mean Aggregates ---
    # Change aggregation method to median, keep leaky encoding.
    median_agg_config = {'agg_method': 'median', 'proper_encoding': False}
    results['Ablation: Use Median Aggregates'] = run_experiment(median_agg_config)
    print(f"Ablation RMSE (Median Aggregates, Leaky Encoding): {results['Ablation: Use Median Aggregates']:.4f}")
    
    # --- Experiment 3: Ablation of Leaky Encoding ---
    # Use proper encoding (fit on train only), keep mean aggregates.
    proper_encoding_config = {'agg_method': 'mean', 'proper_encoding': True}
    results['Ablation: Use Proper Encoding'] = run_experiment(proper_encoding_config)
    print(f"Ablation RMSE (Mean Aggregates, Proper Encoding): {results['Ablation: Use Proper Encoding']:.4f}")

    print("\n--- Ablation Study Analysis ---")
    
    degradations = {
        'Mean Aggregation Statistic': results['Ablation: Use Median Aggregates'] - results['Baseline'],
        'Leaky Label Encoding Strategy': results['Ablation: Use Proper Encoding'] - results['Baseline']
    }
    
    print(f"Impact of changing 'Mean Aggregation' to Median: {degradations['Mean Aggregation Statistic']:.4f} RMSE")
    print(f"Impact of removing 'Leaky Label Encoding': {degradations['Leaky Label Encoding Strategy']:.4f} RMSE")
    
    # A positive degradation means the original component was helpful.
    # We are looking for the component whose removal/change leads to the highest positive RMSE change.
    
    # Filter for components that worsened performance
    positive_degradations = {k: v for k, v in degradations.items() if v > 0}
    
    if not positive_degradations:
        print("\nConclusion: No component modification led to a degradation in performance.")
        # Check if any change was a significant improvement
        best_improvement = min(degradations, key=degradations.get)
        if degradations[best_improvement] < 0:
            print(f"The change that most improved performance was related to '{best_improvement}'.")
    else:
        most_impactful_component = max(positive_degradations, key=positive_degradations.get)
        print(f"\nConclusion: The '{most_impactful_component}' contributes the most to the overall performance.")
        print(f"Modifying or removing it resulted in the largest performance degradation (RMSE increase of {positive_degradations[most_impactful_component]:.4f}).")

if __name__ == '__main__':
    main()
