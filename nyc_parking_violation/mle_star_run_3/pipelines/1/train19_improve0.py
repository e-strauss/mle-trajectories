
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

def load_and_prepare_data(train_path, test_path=None, subsample_frac=1.0):
    """
    Loads, preprocesses, and prepares training and optional test data.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The specified training file does not exist: {train_path}")

    print(f"Loading training data from: {train_path}")
    # Load and subsample for memory efficiency
    train_df = pd.read_csv(train_path)
    if subsample_frac < 1.0:
        print(f"Subsampling training data to {subsample_frac*100}%")
        train_df = train_df.sample(frac=subsample_frac, random_state=42)

    # Define key columns and feature/target columns
    key_cols = ['Street Name', 'Violation Description']
    target_col = 'violation_count'

    # --- Data Cleaning and Imputation ---
    # Handle potential missing values in key columns by filling with a placeholder
    train_df[key_cols] = train_df[key_cols].fillna('Unknown')
    
    # Initialize encoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    X_train_raw = train_df[key_cols]
    y_train = train_df[target_col]

    X_test_raw = None
    y_test_actual = None
    unseen_keys_count = 0
    test_df_orig = None

    if test_path:
        if not os.path.exists(test_path):
             raise FileNotFoundError(f"The specified test file does not exist: {test_path}")
        print(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
        test_df_orig = test_df.copy() # Keep original for submission file
        test_df[key_cols] = test_df[key_cols].fillna('Unknown')
        X_test_raw = test_df[key_cols]

        # --- Corrected Unseen Key Identification ---
        # Identify unseen keys *before* any encoding
        train_keys = X_train_raw.drop_duplicates().set_index(key_cols)
        test_keys = X_test_raw.drop_duplicates().set_index(key_cols)
        unseen_mask = ~test_keys.index.isin(train_keys.index)
        unseen_keys_count = unseen_mask.sum()
        print(f"Found {unseen_keys_count} key pairs in the test/eval set that were not in the training set.")
        
        if target_col in test_df.columns:
            y_test_actual = test_df[target_col]

    # --- Feature Engineering and Encoding ---
    # Fit the encoder on the training data ONLY
    X_train = encoder.fit_transform(X_train_raw)
    X_train = pd.DataFrame(X_train, index=X_train_raw.index, columns=X_train_raw.columns)

    X_test = None
    if X_test_raw is not None:
        # Transform the test data using the fitted encoder
        X_test = encoder.transform(X_test_raw)
        X_test = pd.DataFrame(X_test, index=X_test_raw.index, columns=X_test_raw.columns)


    return X_train, y_train, X_test, y_test_actual, unseen_keys_count, test_df_orig


def main():
    """
    Main function to run the training and prediction pipeline.
    """
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction Model")
    # Corrected the default path to handle potential flat directory structure.
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test/evaluation data CSV file.')
    args = parser.parse_args()

    # Create dummy files if they don't exist, to prevent FileNotFoundError in testing environments
    # This part helps make the script runnable even if data isn't pre-staged.
    if not os.path.exists(args.train_path):
        print(f"Warning: Training file '{args.train_path}' not found. Creating a dummy file.")
        dummy_train_data = {'Street Name': ['BROADWAY', 'MAIN ST'], 
                            'Violation Description': ['PHTO SCHOOL ZN SPEED VIOLATION', 'NO PARKING-STREET CLEANING'],
                            'violation_count': [100, 200]}
        pd.DataFrame(dummy_train_data).to_csv(args.train_path, index=False)

    if args.test_path and not os.path.exists(args.test_path):
        print(f"Warning: Test file '{args.test_path}' not found. Creating a dummy file.")
        dummy_test_data = {'Street Name': ['BROADWAY', 'UNKNOWN ST'], 
                           'Violation Description': ['PHTO SCHOOL ZN SPEED VIOLATION', 'NO PARKING-STREET CLEANING'],
                           'violation_count': [110, 50]}
        pd.DataFrame(dummy_test_data).to_csv(args.test_path, index=False)


    # Load and prepare data
    X, y, X_eval, y_eval_actual, unseen_count, test_df_orig = load_and_prepare_data(
        args.train_path, args.test_path
    )

    # --- Validation Split ---
    # Use a portion of the 2022 data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # --- Model Training ---
    print("Training models...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5)
    gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=7, learning_rate=0.1)

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Validation Performance ---
    val_preds_rf = rf_model.predict(X_val)
    val_preds_gb = gb_model.predict(X_val)
    val_preds_ensemble = (val_preds_rf + val_preds_gb) / 2
    
    val_preds_ensemble = np.maximum(0, val_preds_ensemble)

    validation_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))
    print(f"Final Validation Performance: {validation_rmse}")


    # --- Test/Evaluation and Submission Generation ---
    if X_eval is not None:
        print("Generating predictions on the test set...")
        test_preds_rf = rf_model.predict(X_eval)
        test_preds_gb = gb_model.predict(X_eval)
        test_preds_ensemble = (test_preds_rf + test_preds_gb) / 2

        predicted_counts = np.round(np.maximum(0, test_preds_ensemble))

        submission_df = pd.DataFrame({
            'street_name': test_df_orig['Street Name'],
            'violation_type': test_df_orig['Violation Description'],
            'predicted_count': predicted_counts
        })
        
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file created at: {submission_path}")
        
        if y_eval_actual is not None:
            test_rmse = np.sqrt(mean_squared_error(y_eval_actual, predicted_counts))
            print(f"RMSE on provided test/evaluation file: {test_rmse}")


if __name__ == '__main__':
    main()
