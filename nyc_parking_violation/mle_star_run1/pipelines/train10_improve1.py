
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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
        # For this environment, we'll let the script fail on the next line if import fails.
        pass

def create_smoothed_target_encoding(train_series, target_series, test_series, smoothing=10):
    """
    Creates smoothed target encoding for a categorical feature.

    Args:
        train_series (pd.Series): The categorical feature from the training set.
        target_series (pd.Series): The target variable from the training set.
        test_series (pd.Series): The categorical feature from the test/validation set.
        smoothing (int): The smoothing factor (m in the formula).

    Returns:
        (pd.Series, pd.Series, float): The encoded training series, encoded test series, and global mean.
    """
    global_mean = np.mean(target_series)
    
    # Group by category and calculate mean and count
    stats = target_series.groupby(train_series).agg(['mean', 'count'])
    
    # Calculate smoothed mean
    smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
    # Map to train and test series
    encoded_train = train_series.map(smoothed_mean)
    encoded_test = test_series.map(smoothed_mean)
    
    # Fill NaNs in test set (for categories not in train set) with the global mean
    encoded_test.fillna(global_mean, inplace=True)
    
    return encoded_train, encoded_test, global_mean

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

    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')
    df_train['log_violation_count'] = np.log1p(df_train['violation_count'])

    # --- 2. Cross-Validation Setup ---
    # We use K-Fold to create features and validate the model to prevent data leakage.
    # Features like 'mean violations for a street' must be calculated on the training part of a fold
    # and then applied to the validation part.

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_train))
    models = []
    
    features = ['street_name_encoded', 'violation_description_encoded']

    print("Starting cross-validation training...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"--- Fold {fold+1}/5 ---")
        
        # --- a. Create fold data ---
        train_fold = df_train.iloc[train_idx]
        val_fold = df_train.iloc[val_idx]
        
        y_train_fold = train_fold['log_violation_count']
        y_val_fold = val_fold['log_violation_count']

        # --- b. Leak-proof Feature Engineering (within the fold) ---
        X_train_fold = pd.DataFrame(index=train_fold.index)
        X_val_fold = pd.DataFrame(index=val_fold.index)

        # Smoothed Target Encoding for 'violation_description'
        X_train_fold['violation_description_encoded'], X_val_fold['violation_description_encoded'], _ = \
            create_smoothed_target_encoding(train_fold['violation_description'], y_train_fold, val_fold['violation_description'])

        # Smoothed Target Encoding for 'street_name'
        X_train_fold['street_name_encoded'], X_val_fold['street_name_encoded'], _ = \
            create_smoothed_target_encoding(train_fold['street_name'], y_train_fold, val_fold['street_name'])
        
        # --- c. Train Model ---
        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train_fold[features], y_train_fold)
        
        # --- d. Predict on Validation Set ---
        val_preds_log = lgbm.predict(X_val_fold[features])
        oof_preds[val_idx] = val_preds_log
        
        models.append(lgbm)

    # --- 3. Final Validation Performance ---
    oof_preds_unlogged = np.expm1(oof_preds)
    # Clip predictions to be non-negative
    oof_preds_unlogged[oof_preds_unlogged < 0] = 0
    final_validation_score = np.sqrt(mean_squared_error(df_train['violation_count'], oof_preds_unlogged))
    print(f"Final Validation Performance: {final_validation_score}")
    
    # --- 4. Final Model Training on Full Data ---
    # Now we train a final model on the full dataset to be used for the test set predictions.
    # We create the features on the full dataset.
    print("\nTraining final model on all data...")
    X_full_train = pd.DataFrame(index=df_train.index)
    y_full_train = df_train['log_violation_count']

    # Target encoding for 'violation_description' on full data
    desc_encoder_map = y_full_train.groupby(df_train['violation_description']).mean()
    X_full_train['violation_description_encoded'] = df_train['violation_description'].map(desc_encoder_map)
    desc_global_mean = y_full_train.mean()

    # Target encoding for 'street_name' on full data
    street_encoder_map = y_full_train.groupby(df_train['street_name']).mean()
    X_full_train['street_name_encoded'] = df_train['street_name'].map(street_encoder_map)
    street_global_mean = y_full_train.mean()
    
    final_model = lgb.LGBMRegressor(random_state=42)
    final_model.fit(X_full_train[features], y_full_train)
    print("Final model training complete.")

    # --- 5. Test Set Prediction ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
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
        X_test = pd.DataFrame(index=df_test.index)
        
        # Map learned encodings to the test set
        X_test['violation_description_encoded'] = df_test['violation_description'].map(desc_encoder_map)
        X_test['street_name_encoded'] = df_test['street_name'].map(street_encoder_map)
        
        # Report unseen keys
        unseen_descriptions = X_test['violation_description_encoded'].isna().sum()
        unseen_streets = X_test['street_name_encoded'].isna().sum()
        print(f"Test set has {unseen_streets} rows with unseen street names.")
        print(f"Test set has {unseen_descriptions} rows with unseen violation descriptions.")

        # Fill NaNs for unseen keys with global averages from training
        X_test['violation_description_encoded'].fillna(desc_global_mean, inplace=True)
        X_test['street_name_encoded'].fillna(street_global_mean, inplace=True)
        
        # --- b. Generate Predictions ---
        test_preds_log = final_model.predict(X_test[features])
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
