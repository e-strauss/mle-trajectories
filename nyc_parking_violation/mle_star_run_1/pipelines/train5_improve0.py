
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import subprocess
import sys
import os

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
        # Exit if the core dependency cannot be installed.
        # Using a return here as we are in a script context, not a function.
        # In a real script, sys.exit(1) would be appropriate.
        # For this environment, we'll let it fail on the next line.

# Define a function to handle unseen labels during transform
def safe_label_transform(le, series):
    """Safely transforms a series using a fitted LabelEncoder, handling unseen values."""
    known_labels = set(le.classes_)
    # The default value ('<unknown>') must be in the LabelEncoder's classes
    # from the initial fit.
    return series.map(lambda s: s if s in known_labels else '<unknown>')

def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data file.')
    parser.add_argument('--test-path', type=str,
                        help='Optional path to the test data file.')
    args = parser.parse_args()

    # --- 1. Data Loading and Basic Cleaning ---
    try:
        df_train = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}")
        return

    # Standardize column names
    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering ---
    
    # a. Aggregate features from the training data itself
    # Create richer set of aggregates for 'violation_description'
    description_agg = df_train.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
    description_agg.columns = ['description_mean_count', 'description_sum_count', 'description_std_count', 'description_agg_count']

    # Create richer set of aggregates for 'street_name'
    street_agg = df_train.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
    street_agg.columns = ['street_mean_count', 'street_sum_count', 'street_std_count', 'street_agg_count']

    # Merge the new aggregated features back into the training dataframe
    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    # b. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()

    # Add '<unknown>' to handle unseen values in test data later
    all_streets = pd.concat([df_train['street_name']]).unique()
    all_descs = pd.concat([df_train['violation_description']]).unique()
    
    # Fit the encoders including the '<unknown>' token
    le_street.fit(np.append(all_streets, '<unknown>'))
    le_desc.fit(np.append(all_descs, '<unknown>'))

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
    
    # Log-transform the target variable to handle skewness
    df_train['log_target'] = np.log1p(df_train[target])

    # Split 2022 data for validation
    X = df_train[features]
    y = df_train['log_target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val)

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation Performance ---
    
    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    final_validation_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f'Final Validation Performance: {final_validation_score}')

    # --- 5. Test Set Prediction ---
    
    if args.test_path:
        try:
            df_test = pd.read_csv(args.test_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}")
            return
            
        df_test.columns = df_test.columns.str.lower().str.replace(' ', '_')
        
        submission_df = df_test[['street_name', 'violation_description']].copy()
        
        y_test_original = None
        if 'violation_count' in df_test.columns:
            y_test_original = df_test['violation_count']

        # --- a. Apply feature engineering to the test set ---
        
        unseen_streets = set(df_test['street_name']) - set(le_street.classes_)
        unseen_descriptions = set(df_test['violation_description']) - set(le_desc.classes_)
        print(f"Test set has {len(unseen_streets)} unseen street names.")
        print(f"Test set has {len(unseen_descriptions)} unseen violation descriptions.")
        
        df_test = pd.merge(df_test, description_agg, on='violation_description', how='left')
        df_test = pd.merge(df_test, street_agg, on='street_name', how='left')

        # Fill NaNs for unseen keys with global averages from training
        global_desc_mean = description_agg['description_mean_count'].mean()
        global_street_mean = street_agg['street_mean_count'].mean()
        df_test['description_mean_count'].fillna(global_desc_mean, inplace=True)
        df_test['street_mean_count'].fillna(global_street_mean, inplace=True)
        
        # Apply label encoders, handling unseen values safely
        df_test['street_name_encoded'] = le_street.transform(safe_label_transform(le_street, df_test['street_name']))
        df_test['violation_description_encoded'] = le_desc.transform(safe_label_transform(le_desc, df_test['violation_description']))
        
        # --- b. Generate Predictions ---
        
        X_test = df_test[features]
        test_preds_log = lgbm.predict(X_test)
        
        test_preds = np.expm1(test_preds_log)
        test_preds[test_preds < 0] = 0
        
        submission_df['predicted_count'] = test_preds
        
        # --- c. Save Submission File ---
        submission_df.to_csv('submission.csv', index=False)
        print("submission.csv generated successfully.")
        
        # --- d. Score Test Set if ground truth is available ---
        if y_test_original is not None:
            test_rmse = np.sqrt(mean_squared_error(y_test_original, test_preds))
            print(f"Test set RMSE: {test_rmse}")

if __name__ == '__main__':
    main()
