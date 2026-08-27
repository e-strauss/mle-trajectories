
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
        # For this environment, we will let it fail later if import is needed.
        pass

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

    # a. Categorical Feature Encoding
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
    
    # --- 3. Model Validation (with K-Fold) ---
    
    # Log-transform the target variable to handle skewness
    df_train['log_target'] = np.log1p(df_train['violation_count'])

    base_features = [
        'street_name_encoded', 
        'violation_description_encoded'
    ]
    target = 'log_target'

    # Initialize K-Fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    val_scores = []
    
    # This loop is for robust validation metric calculation only
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"===== Fold {fold+1}/{n_splits} =====")
        
        train_fold = df_train.iloc[train_idx].copy()
        val_fold = df_train.iloc[val_idx].copy()

        # Target Encoding within the fold to prevent data leakage
        desc_map = train_fold.groupby('violation_description_encoded')[target].mean()
        street_map = train_fold.groupby('street_name_encoded')[target].mean()

        train_fold['description_mean_log_target'] = train_fold['violation_description_encoded'].map(desc_map)
        val_fold['description_mean_log_target'] = val_fold['violation_description_encoded'].map(desc_map)
        
        train_fold['street_mean_log_target'] = train_fold['street_name_encoded'].map(street_map)
        val_fold['street_mean_log_target'] = val_fold['street_name_encoded'].map(street_map)

        global_mean_log_target = train_fold[target].mean()
        val_fold['description_mean_log_target'].fillna(global_mean_log_target, inplace=True)
        val_fold['street_mean_log_target'].fillna(global_mean_log_target, inplace=True)

        features = base_features + ['description_mean_log_target', 'street_mean_log_target']

        X_train, y_train = train_fold[features], train_fold[target]
        X_val, y_val = val_fold[features], val_fold[target]
        y_val_original = np.expm1(y_val)

        lgbm = lgb.LGBMRegressor(
            n_estimators=2000,
            random_state=42,
            n_jobs=-1
        )

        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        val_preds_log = lgbm.predict(X_val)
        val_preds = np.expm1(val_preds_log)
        val_preds[val_preds < 0] = 0

        fold_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
        val_scores.append(fold_score)
        print(f'Fold {fold+1} Validation RMSE: {fold_score}')

    final_validation_score = np.mean(val_scores)
    print(f'\nFinal Validation Performance: {final_validation_score}')
    print(f'Std Dev of Validation Scores: {np.std(val_scores)}')

    # --- 4. Final Model Training on Full Data ---
    print("\n--- Training Final Model on Full Training Data ---")
    
    # a. Create target-encoded features on the full training data
    full_desc_map = df_train.groupby('violation_description_encoded')[target].mean()
    full_street_map = df_train.groupby('street_name_encoded')[target].mean()
    global_mean_log_target_full = df_train[target].mean()

    df_train['description_mean_log_target'] = df_train['violation_description_encoded'].map(full_desc_map)
    df_train['street_mean_log_target'] = df_train['street_name_encoded'].map(full_street_map)
    
    final_features = base_features + ['description_mean_log_target', 'street_mean_log_target']
    
    X_train_full = df_train[final_features]
    y_train_full = df_train[target]

    # b. Initialize and train the final LightGBM model
    final_lgbm = lgb.LGBMRegressor(
        n_estimators=1000, # Using a fixed number of estimators
        random_state=42,
        n_jobs=-1
    )
    
    final_lgbm.fit(X_train_full, y_train_full)
    print("Final model trained.")

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
        print(f"\nTest set has {len(unseen_streets)} unseen street names.")
        print(f"Test set has {len(unseen_descriptions)} unseen violation descriptions.")
        
        # Apply label encoders, handling unseen values safely
        df_test['street_name_encoded'] = le_street.transform(safe_label_transform(le_street, df_test['street_name']))
        df_test['violation_description_encoded'] = le_desc.transform(safe_label_transform(le_desc, df_test['violation_description']))
        
        # Apply the target encoding mappings from the full training set
        df_test['description_mean_log_target'] = df_test['violation_description_encoded'].map(full_desc_map)
        df_test['street_mean_log_target'] = df_test['street_name_encoded'].map(full_street_map)
        
        # Fill NaNs for unseen keys with the global training mean log target
        df_test['description_mean_log_target'].fillna(global_mean_log_target_full, inplace=True)
        df_test['street_mean_log_target'].fillna(global_mean_log_target_full, inplace=True)

        # --- b. Generate Predictions ---
        X_test = df_test[final_features]
        test_preds_log = final_lgbm.predict(X_test)
        
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
