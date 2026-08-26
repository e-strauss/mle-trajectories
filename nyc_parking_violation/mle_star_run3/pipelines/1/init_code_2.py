
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    cols = df.columns
    new_cols = [c.lower().replace(' ', '_') for c in cols]
    df.columns = new_cols
    return df

def load_and_preprocess_data(file_path, augmentation_dfs, is_train=False, train_cols=None, train_cat_dtypes=None):
    """
    Loads the main data, merges with augmentation tables, and preprocesses it.

    Args:
        file_path (str): Path to the main data file (train or test).
        augmentation_dfs (dict): Dictionary of pre-loaded augmentation dataframes.
        is_train (bool): Flag indicating if we are processing the training data.
        train_cols (pd.Index): The columns from the training set, used to align test set.
        train_cat_dtypes (dict): Dictionary mapping categorical columns to their dtype from training.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
        np.ndarray or None: Ground truth target values if available.
        dict or None: Dictionary of categorical dtypes if is_train.
    """
    # Load main data
    df = pd.read_csv(file_path)
    df = clean_col_names(df)

    # Store ground truth if it exists and remove it from features
    ground_truth = None
    if 'violation_count' in df.columns:
        ground_truth = df['violation_count'].copy()
        # For training, this is the target. For testing, it's for scoring.
        # Don't drop it yet for training data, we need it for the target `y`.
        if not is_train:
            df = df.drop(columns=['violation_count'])

    # Rename columns for consistency in submission output
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)

    # Merge with augmentation tables
    df = pd.merge(df, augmentation_dfs['boroughs'], on='street_name', how='left')
    df = pd.merge(df, augmentation_dfs['physical'], on='street_name', how='left')

    # Feature Engineering & Missing Value Handling
    # Fill missing borough with 'Unknown'
    df['borough'].fillna('Unknown', inplace=True)
    
    # For numerical physical features, fill with median.
    for col in augmentation_dfs['physical'].columns:
        if col != 'street_name' and col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
            # Ensure all physical features are numeric
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    categorical_features = ['street_name', 'violation_type', 'borough']
    cat_dtypes = {}

    if is_train:
        # For training data, create and store the categorical dtypes
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
                cat_dtypes[col] = df[col].dtype
        return df, ground_truth, cat_dtypes
    else:
        # For test data, apply the learned dtypes from training
        if train_cat_dtypes:
            for col in categorical_features:
                if col in df.columns:
                    # New categories in test data that were not in train will become NaN
                    # XGBoost can handle NaNs in categorical features
                    df[col] = df[col].astype(train_cat_dtypes.get(col, 'category'), errors='ignore')
        
        # Align columns with training data
        if train_cols is not None:
            # Ensure all categorical columns exist before alignment
            for col in categorical_features:
                 if col in train_cols and col not in df.columns:
                     df[col] = pd.Series(dtype=train_cat_dtypes.get(col, 'category'))

            missing_cols = set(train_cols) - set(df.columns)
            for c in missing_cols:
                df[c] = 0 # default for missing numerical columns
            df = df[train_cols]

        return df, ground_truth, None


def main():
    """
    Main function to run the training and prediction pipeline.
    """
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction with XGBoost")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional. Path to the test/evaluation data CSV file.')
    args = parser.parse_args()

    # --- 1. Load Augmentation Data ---
    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError as e:
        print(f"Error loading augmentation data: {e}")
        print("Please ensure 'street_names_and_boroughs.csv' and 'physical_features_per_street.csv' are in the './input' directory.")
        return

    augmentation_dfs = {
        'boroughs': boroughs_df,
        'physical': physical_df,
    }

    # --- 2. Load and Preprocess Training Data ---
    print("Loading and preprocessing training data...")
    train_df, train_ground_truth, cat_dtypes = load_and_preprocess_data(
        args.train_path, augmentation_dfs, is_train=True
    )
    
    # --- 3. Feature and Target Preparation ---
    # Log-transform the target to handle skewness
    y = np.log1p(train_ground_truth)
    
    # Define features (X) by dropping the target
    X = train_df.drop(columns=['violation_count'])
    
    # Store feature names and dtypes for consistent use in test set
    train_cols = X.columns

    # --- 4. Validation Split ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Model Training ---
    print("Training XGBoost model...")
    # XGBoost can handle pandas category types natively if we enable it.
    # This is more memory-efficient than one-hot encoding.
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        enable_categorical=True,  # Enable native handling of 'category' dtype
        tree_method='hist',       # 'hist' is required for categorical support
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # --- 6. Validation Performance ---
    val_preds_log = xgb_model.predict(X_val)
    
    # Inverse transform predictions and clip at 0
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    # Inverse transform true validation target
    y_val_true = np.expm1(y_val)
    
    val_rmse = mean_squared_error(y_val_true, val_preds, squared=False)
    print(f"Final Validation Performance: {val_rmse:.4f}")

    # --- 7. Test Set Prediction (if provided) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        
        # Load and preprocess test data using artifacts from training
        test_df, test_ground_truth, _ = load_and_preprocess_data(
            args.test_path, 
            augmentation_dfs, 
            is_train=False, 
            train_cols=train_cols, 
            train_cat_dtypes=cat_dtypes
        )

        # Identify and report unseen keys
        # Ensure original data is available for key comparison
        original_train_df = pd.read_csv(args.train_path)
        original_train_df = clean_col_names(original_train_df).rename(columns={'violation_description':'violation_type'})
        train_keys = set(original_train_df['street_name'].astype(str) + "_" + original_train_df['violation_type'].astype(str))
        
        test_keys_df = test_df.copy() # Use the preprocessed test_df
        test_keys = set(test_keys_df['street_name'].astype(str) + "_" + test_keys_df['violation_type'].astype(str))
        
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} (street_name, violation_type) pairs in test set that were not in the training set.")
        print("These will be predicted using learned patterns from other features (like borough or physical street features).")

        # Predict on test data
        test_preds_log = xgb_model.predict(test_df[train_cols])
        
        # Inverse transform and clip predictions
        test_preds = np.expm1(test_preds_log)
        test_preds[test_preds < 0] = 0

        # Create submission file
        submission_df = pd.DataFrame({
            'street_name': test_df['street_name'],
            'violation_type': test_df['violation_type'],
            'predicted_count': test_preds
        })
        submission_df.to_csv('submission.csv', index=False)
        print("Generated submission.csv")

        # Score test set if ground truth is available
        if test_ground_truth is not None:
            test_rmse = mean_squared_error(test_ground_truth, test_preds, squared=False)
            print(f"Test Set RMSE: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
