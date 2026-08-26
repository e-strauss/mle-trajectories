
import pandas as pd
import numpy as np
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import os
import warnings

# Suppress CatBoost's verbose logging unless in debug mode
warnings.filterwarnings("ignore", category=UserWarning)

def clean_col_names(df):
    """
    Standardizes column names by lowercasing and replacing spaces with underscores.
    """
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def preprocess_data(file_path, augmentation_dfs, is_train=False, train_features_info=None):
    """
    Loads and preprocesses data for the CatBoost model. This function ensures
    that transformations applied to the training data (like median imputation)
    are consistently applied to the test data.

    Args:
        file_path (str): Path to the main data file (train or test).
        augmentation_dfs (dict): Dictionary of pre-loaded augmentation dataframes.
        is_train (bool): Flag indicating if processing training data.
        train_features_info (dict): State from training (e.g., column order, medians)
                                    to be applied to the test set.

    Returns:
        A tuple containing:
        - pd.DataFrame: The processed dataframe ready for the model.
        - np.ndarray or None: Ground truth target values if available.
        - dict or None: Information about training features (only if is_train=True).
    """
    df = pd.read_csv(file_path)
    df = clean_col_names(df)

    ground_truth = None
    if 'violation_count' in df.columns:
        ground_truth = df['violation_count'].copy()
        # For training, the target is handled later. For testing, we drop it
        # to ensure it's not used as a feature.
        if not is_train:
            df = df.drop(columns=['violation_count'])

    # Standardize column name for violation type for consistent merging
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)

    # --- Feature Engineering ---
    # Merge with augmentation data
    df = pd.merge(df, augmentation_dfs['boroughs'], on='street_name', how='left')
    df = pd.merge(df, augmentation_dfs['physical'], on='street_name', how='left')

    # Handle missing values
    df['borough'].fillna('Unknown', inplace=True)

    numerical_cols = [col for col in augmentation_dfs['physical'].columns if col != 'street_name']

    if is_train:
        # For training, calculate and store medians for imputation
        train_medians = {}
        for col in numerical_cols:
            if col in df.columns:
                median_val = df[col].median()
                train_medians[col] = median_val
                df[col].fillna(median_val, inplace=True)
        
        # Define categorical features and set dtype for memory efficiency
        categorical_features = ['street_name', 'violation_type', 'borough']
        for col in categorical_features:
            df[col] = df[col].astype('category')
            
        features = df.drop(columns=['violation_count']) if 'violation_count' in df.columns else df
        
        # Store all necessary info to replicate preprocessing on the test set
        train_features_info = {
            'columns': features.columns,
            'medians': train_medians,
            'categorical_features': categorical_features
        }
        return df, ground_truth, train_features_info
    else:
        # For testing, use stored medians and column info from training
        medians = train_features_info['medians']
        for col in numerical_cols:
            if col in df.columns:
                # Use stored median, with a fallback of 0 if column is new
                df[col].fillna(medians.get(col, 0), inplace=True)

        # Set categorical dtypes. CatBoost handles new/unseen categories.
        categorical_features = train_features_info['categorical_features']
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Align columns with the training set to ensure consistency
        train_cols = train_features_info['columns']
        # Add any missing columns that were in training but not in test
        missing_cols = set(train_cols) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        
        # Ensure the column order is identical to the training set
        df = df[train_cols]
        
        return df, ground_truth, None

def main():
    """
    Main function to run the training, validation, and prediction pipeline.
    """
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction with CatBoost")
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
        print(f"Error: Augmentation data not found. Please ensure all required CSVs are in ./input. Details: {e}")
        return

    augmentation_dfs = {'boroughs': boroughs_df, 'physical': physical_df}

    # --- 2. Load and Preprocess Training Data ---
    print("Loading and preprocessing training data...")
    train_df_processed, train_ground_truth, train_features_info = preprocess_data(
        args.train_path, augmentation_dfs, is_train=True
    )

    # --- 3. Feature and Target Preparation ---
    # Log-transform target to handle skewed distribution
    y = np.log1p(train_ground_truth)
    X = train_df_processed.drop(columns=['violation_count'])
    
    categorical_features = train_features_info['categorical_features']
    
    # --- 4. Validation Split ---
    # Create a hold-out set from 2022 data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Model Training ---
    print("Training CatBoost model...")
    cb_model = catboost.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0,
        cat_features=categorical_features,
        allow_writing_files=False # Prevents creation of catboost_info directory
    )

    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )

    # --- 6. Validation Performance ---
    val_preds_log = cb_model.predict(X_val)
    # Inverse transform predictions and clip at 0 for non-negativity
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0

    # Inverse transform the true validation target for scoring
    y_val_true = np.expm1(y_val)
    
    val_rmse = mean_squared_error(y_val_true, val_preds, squared=False)
    print(f"Final Validation Performance: {val_rmse:.4f}")

    # --- 7. Test Set Prediction (if provided) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")

        # Keep a raw copy of test data for key comparison and final submission file
        raw_test_df_for_output = clean_col_names(pd.read_csv(args.test_path))
        raw_test_df_for_output.rename(columns={'violation_description':'violation_type'}, inplace=True)

        # Preprocess test data using the info from the training run
        test_df, test_ground_truth, _ = preprocess_data(
            args.test_path,
            augmentation_dfs,
            is_train=False,
            train_features_info=train_features_info
        )

        # Report on unseen keys
        raw_train_df = pd.read_csv(args.train_path)
        raw_train_df = clean_col_names(raw_train_df).rename(columns={'violation_description':'violation_type'})
        train_keys = set(raw_train_df['street_name'].astype(str) + "_" + raw_train_df['violation_type'].astype(str))
        test_keys = set(raw_test_df_for_output['street_name'].astype(str) + "_" + raw_test_df_for_output['violation_type'].astype(str))
        
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} (street_name, violation_type) pairs in test set not present in training set.")
        print("CatBoost will handle these using its internal mechanisms for unseen categorical values.")

        # Predict on the test set
        test_preds_log = cb_model.predict(test_df)
        test_preds = np.expm1(test_preds_log)
        test_preds[test_preds < 0] = 0

        # Create submission file using original keys from the test file
        submission_df = pd.DataFrame({
            'street_name': raw_test_df_for_output['street_name'],
            'violation_type': raw_test_df_for_output['violation_type'],
            'predicted_count': test_preds.round() # Round predictions to nearest integer
        })
        submission_df.to_csv('submission.csv', index=False)
        print("Generated submission.csv")

        # Score test set if ground truth is available
        if test_ground_truth is not None:
            test_rmse = mean_squared_error(test_ground_truth, submission_df['predicted_count'], squared=False)
            print(f"Test Set RMSE: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
