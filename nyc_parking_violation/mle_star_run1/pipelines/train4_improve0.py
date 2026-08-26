
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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
        # In a script, sys.exit(1) would be appropriate.
        # Here we let it fail on the next line if import still fails.
        import lightgbm as lgb

def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data file.')
    parser.add_argument('--test-path', type=str,
                        help='Optional path to the test data file.')
    # In a real script, we would parse args from sys.argv.
    # For this environment, we can simulate or use defaults.
    # Using a direct call to parse_args() which works in most interactive/script runners
    # If running from a context where sys.argv is not what's expected, this might need adjustment
    # For this specific task, we'll assume standard script execution.
    args = parser.parse_args()


    # --- 1. Data Loading and Basic Cleaning ---
    try:
        df_train = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}")
        return

    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # --- 2. Feature Engineering using K-Fold Target Encoding ---
    
    # Define target and categorical features
    target_col = 'violation_count'
    categorical_cols = ['street_name', 'violation_description']

    # For robust stratification in regression, bin the target variable
    num_bins = 5
    # Ensure there are enough unique bin edges, otherwise pd.qcut will fail.
    # The number of bins should be less than the number of data points.
    if len(df_train) > num_bins:
        try:
            df_train['target_binned'] = pd.qcut(df_train[target_col], q=num_bins, labels=False, duplicates='drop')
        except ValueError: # If qcut fails due to non-unique edges even with 'drop'
            df_train['target_binned'] = pd.cut(df_train[target_col], bins=num_bins, labels=False)
    else:
        # Fallback for very small datasets
        df_train['target_binned'] = 0

    
    # Create global mean maps for test set transformation and as a fallback for training
    street_mean_map = df_train.groupby('street_name')[target_col].mean()
    desc_mean_map = df_train.groupby('violation_description')[target_col].mean()
    global_mean = df_train[target_col].mean()

    # Initialize new feature columns for target encoding
    df_train['street_name_encoded'] = np.nan
    df_train['violation_description_encoded'] = np.nan

    # Use Stratified K-Fold to create out-of-fold target encodings
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # The stratification is performed on the binned target column
    for train_idx, val_idx in skf.split(df_train, df_train['target_binned']):
        df_train_fold, df_val_fold = df_train.iloc[train_idx], df_train.iloc[val_idx]

        for col in categorical_cols:
            encoded_col_name = f'{col}_encoded'
            
            # Calculate mean target for each category on the training part of the fold
            mapping = df_train_fold.groupby(col)[target_col].mean()
            
            # Apply the mapping to the validation part of the fold
            df_train.loc[df_train.index[val_idx], encoded_col_name] = df_val_fold[col].map(mapping)

    # Fill any remaining NaNs in encoded columns.
    # First, try to fill with the category's global mean.
    df_train['street_name_encoded'].fillna(df_train['street_name'].map(street_mean_map), inplace=True)
    df_train['violation_description_encoded'].fillna(df_train['violation_description'].map(desc_mean_map), inplace=True)
    
    # If any NaNs still exist, fill with the overall global mean.
    df_train['street_name_encoded'].fillna(global_mean, inplace=True)
    df_train['violation_description_encoded'].fillna(global_mean, inplace=True)

    # Clean up the temporary binned column
    df_train = df_train.drop(columns=['target_binned'])

    # --- 3. Model Training ---
    
    # Define features and target. We use the robustly encoded features.
    features = ['street_name_encoded', 'violation_description_encoded']
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
        
        unseen_streets = set(df_test['street_name']) - set(street_mean_map.index)
        unseen_descriptions = set(df_test['violation_description']) - set(desc_mean_map.index)
        print(f"Test set has {len(unseen_streets)} unseen street names.")
        print(f"Test set has {len(unseen_descriptions)} unseen violation descriptions.")
        
        # Apply the global mean maps (calculated on the full training data) to the test set
        df_test['street_name_encoded'] = df_test['street_name'].map(street_mean_map)
        df_test['violation_description_encoded'] = df_test['violation_description'].map(desc_mean_map)

        # Fill NaNs for unseen keys with the global average from training
        df_test['street_name_encoded'].fillna(global_mean, inplace=True)
        df_test['violation_description_encoded'].fillna(global_mean, inplace=True)
        
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
