
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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
        # For this environment, we'll stop execution by returning from main.
        # This will cause a failure if main() is called without this check.

# Define a function to handle unseen labels during transform
def safe_label_transform(le, series):
    """Safely transforms a series using a fitted LabelEncoder, handling unseen values."""
    known_labels = set(le.classes_)
    # The default value ('<unknown>') must be in the LabelEncoder's classes
    # from the initial fit.
    return series.map(lambda s: s if s in known_labels else '<unknown>')

def main():
    # Check if lightgbm was imported successfully before proceeding
    if 'lgb' not in sys.modules:
        print("LightGBM is required but could not be imported. Exiting.")
        return

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

    # a. Global aggregate features from the training data for test set imputation
    description_agg = df_train.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df_train.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    
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
    
    # --- 3. Model Training with K-Fold Cross-Validation ---
    
    base_features = [
        'street_name_encoded', 
        'violation_description_encoded'
    ]
    features_to_use = base_features + ['description_mean_count', 'street_mean_count']
    target = 'violation_count'
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(df_train))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"--- Starting Fold {fold+1}/{n_splits} ---")
        
        train_fold = df_train.iloc[train_idx].copy()
        val_fold = df_train.iloc[val_idx].copy()

        # --- Target Encoding within the training fold to prevent leakage ---
        street_mean_map = train_fold.groupby('street_name')[target].mean()
        desc_mean_map = train_fold.groupby('violation_description')[target].mean()
        
        train_fold['street_mean_count'] = train_fold['street_name'].map(street_mean_map)
        train_fold['description_mean_count'] = train_fold['violation_description'].map(desc_mean_map)
        
        val_fold['street_mean_count'] = val_fold['street_name'].map(street_mean_map)
        val_fold['description_mean_count'] = val_fold['violation_description'].map(desc_mean_map)
        
        global_target_mean = train_fold[target].mean()
        val_fold['street_mean_count'].fillna(global_target_mean, inplace=True)
        val_fold['description_mean_count'].fillna(global_target_mean, inplace=True)

        # Log-transform the target variable
        train_fold['log_target'] = np.log1p(train_fold[target])
        
        X_train = train_fold[features_to_use]
        y_train = train_fold['log_target']
        X_val = val_fold[features_to_use]

        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train, y_train)

        models.append(lgbm)
        val_preds_log = lgbm.predict(X_val)
        oof_preds[val_idx] = val_preds_log

    # --- 4. Validation Performance on Out-of-Fold Predictions ---
    
    # Inverse transform the out-of-fold predictions
    oof_preds_orig_scale = np.expm1(oof_preds)
    oof_preds_orig_scale[oof_preds_orig_scale < 0] = 0

    # Calculate RMSE on the full OOF predictions against the original target
    final_validation_score = np.sqrt(mean_squared_error(df_train[target], oof_preds_orig_scale))
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
        
        # Merge with global aggregates calculated from the full training data
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
        
        # --- b. Generate Predictions by Averaging Models ---
        
        X_test = df_test[features_to_use]
        
        all_fold_preds = []
        for model in models:
            fold_preds_log = model.predict(X_test)
            all_fold_preds.append(fold_preds_log)
        
        # Average predictions from all models on the log scale
        test_preds_log = np.mean(all_fold_preds, axis=0)
        
        # Inverse transform and clip predictions
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
