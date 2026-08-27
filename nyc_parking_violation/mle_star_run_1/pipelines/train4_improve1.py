
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
        pass

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
    # These will also be used to enrich the test set
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')

    df_train = pd.merge(df_train, description_agg, on='violation_description', how='left')
    df_train = pd.merge(df_train, street_agg, on='street_name', how='left')

    # b. Create statistical features for high-cardinality categorical variables.
    # These mappings will be saved to apply to the test set
    street_name_freq_map = df_train['street_name'].value_counts()
    df_train['street_name_freq'] = df_train['street_name'].map(street_name_freq_map)

    street_name_diversity_map = df_train.groupby('street_name')['violation_description'].nunique()
    df_train['street_name_desc_diversity'] = df_train['street_name'].map(street_name_diversity_map)

    violation_desc_freq_map = df_train['violation_description'].value_counts()
    df_train['violation_description_freq'] = df_train['violation_description'].map(violation_desc_freq_map)

    violation_desc_diversity_map = df_train.groupby('violation_description')['street_name'].nunique()
    df_train['violation_description_street_diversity'] = df_train['violation_description'].map(violation_desc_diversity_map)

    # --- 3. Model Training ---

    # Define the features to be used in the model
    features = [
        'description_mean_count',
        'street_mean_count',
        'street_name_freq',
        'street_name_desc_diversity',
        'violation_description_freq',
        'violation_description_street_diversity'
    ]
    target = 'violation_count'

    # Log-transform the target variable to handle skewness
    df_train['log_target'] = np.log1p(df_train[target])

    # Split 2022 data for validation
    X = df_train[features]
    y = df_train['log_target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_val_original = np.expm1(y_val) # Get original scale for RMSE calculation

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # --- 4. Validation Performance ---

    val_preds_log = lgbm.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0 # Ensure predictions are non-negative

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
        unseen_streets = set(df_test['street_name']) - set(df_train['street_name'])
        unseen_descriptions = set(df_test['violation_description']) - set(df_train['violation_description'])
        print(f"Test set has {len(unseen_streets)} unseen street names.")
        print(f"Test set has {len(unseen_descriptions)} unseen violation descriptions.")

        # Merge aggregate features from training data
        df_test = pd.merge(df_test, description_agg, on='violation_description', how='left')
        df_test = pd.merge(df_test, street_agg, on='street_name', how='left')

        # Map statistical features from training data
        df_test['street_name_freq'] = df_test['street_name'].map(street_name_freq_map)
        df_test['street_name_desc_diversity'] = df_test['street_name'].map(street_name_diversity_map)
        df_test['violation_description_freq'] = df_test['violation_description'].map(violation_desc_freq_map)
        df_test['violation_description_street_diversity'] = df_test['violation_description'].map(violation_desc_diversity_map)

        # Fill NaNs for unseen keys
        # For mean counts, use global mean from training
        global_desc_mean = description_agg['description_mean_count'].mean()
        global_street_mean = street_agg['street_mean_count'].mean()
        df_test['description_mean_count'].fillna(global_desc_mean, inplace=True)
        df_test['street_mean_count'].fillna(global_street_mean, inplace=True)
        
        # For frequencies and diversities, 0 is a reasonable default for unseen items
        df_test['street_name_freq'].fillna(0, inplace=True)
        df_test['street_name_desc_diversity'].fillna(0, inplace=True)
        df_test['violation_description_freq'].fillna(0, inplace=True)
        df_test['violation_description_street_diversity'].fillna(0, inplace=True)


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
