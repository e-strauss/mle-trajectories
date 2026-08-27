
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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
        # In a real script, sys.exit(1) would be appropriate.
        # For this environment, we'll let it fail if the import doesn't work.

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

    # --- 2. Feature Engineering on Training Data ---
    
    # a. Aggregate features using K-Fold target encoding to prevent data leakage
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize new feature columns for training data
    df_train['description_mean_count'] = np.nan
    df_train['street_mean_count'] = np.nan

    # Create out-of-fold encodings for the training set to prevent leakage
    for train_index, val_index in kf.split(df_train):
        df_train_fold, df_val_fold = df_train.iloc[train_index], df_train.iloc[val_index]
        
        # Calculate aggregates on the training folds
        description_agg_fold = df_train_fold.groupby('violation_description')['violation_count'].mean()
        street_agg_fold = df_train_fold.groupby('street_name')['violation_count'].mean()

        # Apply aggregates to the validation fold
        df_train.loc[df_train.index[val_index], 'description_mean_count'] = df_val_fold['violation_description'].map(description_agg_fold)
        df_train.loc[df_train.index[val_index], 'street_mean_count'] = df_val_fold['street_name'].map(street_agg_fold)

    # b. Create aggregates on the FULL training data for test set use and for filling NaNs
    description_agg_full = df_train.groupby('violation_description')['violation_count'].mean()
    street_agg_full = df_train.groupby('street_name')['violation_count'].mean()
    global_mean = df_train['violation_count'].mean()

    # Fill any NaNs in the training set that occurred during K-Fold encoding
    df_train['description_mean_count'].fillna(df_train['violation_description'].map(description_agg_full), inplace=True)
    df_train['street_mean_count'].fillna(df_train['street_name'].map(street_agg_full), inplace=True)
    # Use global mean as a final fallback for any remaining NaNs
    df_train['description_mean_count'].fillna(global_mean, inplace=True)
    df_train['street_mean_count'].fillna(global_mean, inplace=True)

    # c. Categorical Feature Encoding
    le_street = LabelEncoder()
    le_desc = LabelEncoder()

    # Add '<unknown>' to handle unseen values in test data later
    all_streets = df_train['street_name'].unique()
    all_descs = df_train['violation_description'].unique()
    
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
        
        # Apply aggregate features from the full training data
        df_test['description_mean_count'] = df_test['violation_description'].map(description_agg_full)
        df_test['street_mean_count'] = df_test['street_name'].map(street_agg_full)

        # Fill NaNs for unseen keys with the global average from training
        df_test['description_mean_count'].fillna(global_mean, inplace=True)
        df_test['street_mean_count'].fillna(global_mean, inplace=True)
        
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
