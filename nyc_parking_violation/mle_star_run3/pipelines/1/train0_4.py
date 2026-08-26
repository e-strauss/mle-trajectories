
import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import argparse
import os
import gc

def load_data(file_path):
    """Loads data from a CSV file, raising an error if it doesn't exist."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path and try again.")
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def prepare_features(df, boroughs_df):
    """
    Engineers features for the model.
    - Merges borough information.
    - Creates interaction features.
    - Label encodes categorical columns.
    """
    df_copy = df.copy()

    # Merge with boroughs data if available
    if boroughs_df is not None:
        # Ensure column names are consistent for merging
        if 'Street Name' not in boroughs_df.columns and 'street_name' in boroughs_df.columns:
            boroughs_df.rename(columns={'street_name': 'Street Name'}, inplace=True)
        if 'Borough' not in boroughs_df.columns and 'borough' in boroughs_df.columns:
             boroughs_df.rename(columns={'borough': 'Borough'}, inplace=True)

        df_copy = pd.merge(df_copy, boroughs_df[['Street Name', 'Borough']], on='Street Name', how='left')
        # Fill missing boroughs with a placeholder 'Unknown'
        df_copy['Borough'].fillna('Unknown', inplace=True)
    else:
        # If no borough data, create a placeholder column
        df_copy['Borough'] = 'Unknown'

    # Create a unique identifier for each street-violation pair
    df_copy['street_violation_interaction'] = df_copy['Street Name'].astype(str) + "_" + df_copy['Violation Description'].astype(str)

    # Use a dictionary to store label encoders to handle unseen values in the test set
    encoders = {}
    for col in ['Street Name', 'Violation Description', 'Borough', 'street_violation_interaction']:
        if col in df_copy.columns:
            le = LabelEncoder()
            # Fit on all possible values to handle unseen ones gracefully
            all_values = df_copy[col].astype(str).unique()
            le.fit(list(all_values))
            df_copy[col] = le.transform(df_copy[col].astype(str))
            encoders[col] = le
            
    return df_copy, encoders

def apply_encoders(df, encoders):
    """Applies fitted label encoders to a new dataset."""
    df_copy = df.copy()
    
    if 'Borough' not in df_copy.columns:
        df_copy['Borough'] = 'Unknown'
    df_copy['street_violation_interaction'] = df_copy['Street Name'].astype(str) + "_" + df_copy['Violation Description'].astype(str)

    for col, le in encoders.items():
        if col in df_copy.columns:
            # Handle unseen labels by assigning them to a new 'unknown' category
            new_labels = df_copy[col].astype(str)
            known_labels = set(le.classes_)
            df_copy[col] = [label if label in known_labels else 'Unknown' for label in new_labels]
            
            # Add 'Unknown' to classes if not present
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')

            df_copy[col] = le.transform(df_copy[col])
    return df_copy


def handle_unseen_keys(train_df, test_df):
    """Identifies and reports the number of unseen key pairs in the test set."""
    train_keys = set(zip(train_df['Street Name'], train_df['Violation Description']))
    test_keys = set(zip(test_df['Street Name'], test_df['Violation Description']))
    
    unseen_keys_count = len(test_keys - train_keys)
    
    if unseen_keys_count > 0:
        print(f"Test set contains {unseen_keys_count} (Street Name, Violation Description) pairs not present in the training data.")
    else:
        print("All key pairs in the test set were seen during training.")
        
    return unseen_keys_count


def main(args):
    """Main function to run the training and prediction pipeline."""

    # --- 1. Load Data ---
    try:
        train_df_orig = load_data(args.train_path)
        # Standardize column names
        train_df_orig.columns = ['Street Name', 'Violation Description', 'violation_count']
        
        boroughs_df = None
        if args.boroughs_path:
            boroughs_df = load_data(args.boroughs_path)

    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        return

    # --- 2. Feature Engineering ---
    print("Engineering features for the training data...")
    train_features_df, encoders = prepare_features(train_df_orig, boroughs_df)
    
    features = [col for col in train_features_df.columns if col not in ['violation_count']]
    target = 'violation_count'
    
    # Ensure target is numeric, coercing errors and filling NaNs.
    train_features_df[target] = pd.to_numeric(train_features_df[target], errors='coerce').fillna(0)

    # --- 3. Validation Strategy (Grouped Holdout) ---
    print("Setting up validation strategy using GroupKFold...")
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    
    # Use the 'Street Name' integer-encoded column for grouping
    groups = train_features_df['Street Name']
    
    # Use the last fold as the validation set
    train_indices, val_indices = list(gkf.split(train_features_df, groups=groups))[-1]

    X_train, y_train = train_features_df.loc[train_indices, features], train_features_df.loc[train_indices, target]
    X_val, y_val = train_features_df.loc[val_indices, features], train_features_df.loc[val_indices, target]
    
    categorical_features = ['Street Name', 'Violation Description', 'Borough', 'street_violation_interaction']
    categorical_features_indices = [X_train.columns.get_loc(c) for c in categorical_features]

    # --- 4. Model Training ---
    print("Training models...")
    
    # CatBoost Model
    cb_model = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=10,
        loss_function='RMSE',
        verbose=200,
        random_seed=42,
        allow_writing_files=False,
        task_type='CPU', # To prevent OOM issues on systems without a configured GPU [6, 7]
    )
    cb_model.fit(X_train, y_train, 
                 cat_features=categorical_features_indices, 
                 eval_set=(X_val, y_val), 
                 early_stopping_rounds=50, 
                 use_best_model=True)
    
    # LightGBM Model
    lgb_params = {
        'objective': 'rmse', 'metric': 'rmse', 'n_estimators': 1500,
        'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
        'num_leaves': 40, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  eval_metric='rmse', 
                  callbacks=[lgb.early_stopping(50, verbose=False)]) # Using LightGBM's callback for early stopping [1, 2, 3, 4]

    # --- 5. Validation and Ensembling ---
    print("Validating models and ensembling...")
    val_preds_cb = cb_model.predict(X_val)
    val_preds_lgb = lgb_model.predict(X_val)

    # Simple average ensemble
    val_preds_ensemble = (val_preds_cb + val_preds_lgb) / 2
    val_preds_ensemble[val_preds_ensemble < 0] = 0  # Ensure non-negativity

    final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))
    print(f"Validation RMSE (CatBoost): {np.sqrt(mean_squared_error(y_val, val_preds_cb))}")
    print(f"Validation RMSE (LightGBM): {np.sqrt(mean_squared_error(y_val, val_preds_lgb))}")
    print(f"Final Validation Performance: {final_validation_score}")

    # --- 6. Test Set Prediction ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_orig = load_data(args.test_path)
            # Handle case where test file has no ground truth column
            if 'violation_count' not in test_df_orig.columns:
                test_df_orig['violation_count'] = -1 # Placeholder
            test_df_orig.columns = ['Street Name', 'Violation Description', 'violation_count']
        except FileNotFoundError as e:
            print(f"Fatal Error: {e}")
            return

        has_ground_truth = 'violation_count' in test_df_orig.columns and (test_df_orig['violation_count'] != -1).any()
        
        handle_unseen_keys(train_df_orig, test_df_orig)

        print("Retraining models on full 2022 dataset for final prediction...")
        full_X_train = train_features_df[features]
        full_y_train = train_features_df[target]

        cb_model_full = cb.CatBoostRegressor(**cb_model.get_params())
        cb_model_full.fit(full_X_train, full_y_train, cat_features=categorical_features_indices, verbose=False)

        lgb_params['n_estimators'] = lgb_model.best_iteration_ if lgb_model.best_iteration_ else lgb_params['n_estimators']
        lgb_model_full = lgb.LGBMRegressor(**lgb_params)
        lgb_model_full.fit(full_X_train, full_y_train, eval_metric='rmse')
        
        print("Preparing test features and making predictions...")
        # Apply the same feature engineering and encoding
        test_features_df = apply_encoders(test_df_orig, encoders)
        test_features_df = test_features_df[features] # Align columns

        preds_cb = cb_model_full.predict(test_features_df)
        preds_lgb = lgb_model_full.predict(test_features_df)
        
        predictions = (preds_cb + preds_lgb) / 2
        predictions[predictions < 0] = 0

        submission_df = pd.DataFrame({
            'street_name': test_df_orig['Street Name'],
            'violation_type': test_df_orig['Violation Description'],
            'predicted_count': predictions.round().astype(int)
        })
        submission_df.to_csv("submission.csv", index=False)
        print("submission.csv created successfully.")

        if has_ground_truth:
            test_rmse = np.sqrt(mean_squared_error(test_df_orig['violation_count'], predictions))
            print(f"RMSE on provided test set: {test_rmse}")

    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations.")
    
    # File path arguments with defaults pointing to './input/' directory [8, 11]
    parser.add_argument(
        "--train-path", type=str,
        default="./input/violations_per_street_2022.csv",
        help="Path to the training data CSV."
    )
    parser.add_argument(
        "--test-path", type=str,
        default=None,
        help="Path to the test data CSV (optional)."
    )
    parser.add_argument(
        "--boroughs-path", type=str,
        default="./input/street_borough_mapping.csv", # Corrected filename
        help="Path to the street-borough mapping CSV."
    )
    
    args = parser.parse_args()
    main(args)
