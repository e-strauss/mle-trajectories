
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# Define global constants for column names
TARGET_COL = 'violation_count'
STREET_NAME_COL = 'Street Name'
VIOLATION_DESC_COL = 'Violation Description'
CATEGORICAL_COLS = [STREET_NAME_COL, VIOLATION_DESC_COL]

class TargetEncoder:
    """
    A target encoder that uses K-fold cross-validation to prevent target leakage
    for the training set and applies a globally trained encoder for the test set.
    """
    def __init__(self, cols_to_encode, n_splits=5, random_state=42):
        self.cols_to_encode = cols_to_encode
        self.n_splits = n_splits
        self.random_state = random_state
        self.encoders = {}
        self.global_means = {}

    def fit_transform(self, df):
        """
        Fits the target encoder on the training data and transforms it.
        Uses K-Fold scheme to prevent leakage.
        """
        encoded_df = df.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in self.cols_to_encode:
            # Store global mean of the target for handling unseen values
            self.global_means[col] = df[TARGET_COL].mean()
            encoded_col_name = f'{col}_encoded'
            encoded_df[encoded_col_name] = np.nan

            for train_idx, val_idx in kf.split(df):
                train_fold, val_fold = df.iloc[train_idx], df.iloc[val_idx]
                # Calculate encoding map from the training fold
                encoder = train_fold.groupby(col)[TARGET_COL].mean()
                # Apply to the validation fold
                encoded_df.iloc[val_idx, encoded_df.columns.get_loc(encoded_col_name)] = val_fold[col].map(encoder)

            # Fill any remaining NaNs (rare categories) with the global mean
            encoded_df[encoded_col_name].fillna(self.global_means[col], inplace=True)

            # Train a final encoder on all data to be used for the test set
            self.encoders[col] = df.groupby(col)[TARGET_COL].mean()

        return encoded_df

    def transform(self, df):
        """
        Transforms a new DataFrame (e.g., validation or test set) using the fitted encoders.
        """
        transformed_df = df.copy()
        for col in self.cols_to_encode:
            encoded_col_name = f'{col}_encoded'
            transformed_df[encoded_col_name] = transformed_df[col].map(self.encoders.get(col, pd.Series()))
            # Fill unseen categories with the global mean
            transformed_df[encoded_col_name].fillna(self.global_means.get(col, 0), inplace=True)
        return transformed_df

def load_and_preprocess_data(train_path, test_path=None):
    """
    Loads training and optional test data, handling file existence and basic preprocessing.
    """
    print(f"Loading training data from: {train_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at '{train_path}'. Please check the path.")
    
    train_df = pd.read_csv(train_path)

    test_df = None
    if test_path:
        print(f"Loading test data from: {test_path}")
        if not os.path.exists(test_path):
            print(f"Warning: Test file not found at '{test_path}'. Cannot generate predictions.")
        else:
            test_df = pd.read_csv(test_path)
    
    # Normalize column names
    for df in [train_df, test_df]:
        if df is not None:
            df.columns = [col.strip() for col in df.columns]

    return train_df, test_df

def add_features(df, base_path):
    """Adds supplementary features to the dataframe."""
    
    # Feature 1: Camera on street
    camera_path = os.path.join(base_path, 'dot_camera_locations.csv')
    if os.path.exists(camera_path):
        print("Loading camera location data.")
        camera_df = pd.read_csv(camera_path)
        camera_df.columns = [col.strip() for col in camera_df.columns]
        # Normalize street names for a more reliable join
        camera_streets = set(camera_df[STREET_NAME_COL].str.upper().str.strip())
        df['has_camera'] = df[STREET_NAME_COL].str.upper().str.strip().isin(camera_streets).astype(int)
    else:
        print("Warning: Camera location data not found. Skipping 'has_camera' feature.")
        df['has_camera'] = 0

    return df

def main():
    """
    Main function to run the training and prediction pipeline.
    """
    parser = argparse.ArgumentParser(description='Predict NYC Parking Violations.')
    parser.add_argument('--train-path', type=str, default='input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data CSV file for prediction.')
    args = parser.parse_args()

    try:
        train_df, test_df = load_and_preprocess_data(args.train_path, args.test_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Add supplementary features
    base_data_dir = os.path.dirname(args.train_path)
    train_df = add_features(train_df, base_data_dir)
    if test_df is not None:
        test_df = add_features(test_df, base_data_dir)

    # --- Validation Setup ---
    # Split 2022 data into a training set and a final validation set
    dev_train_df, dev_val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"Training on {len(dev_train_df)} samples, validating on {len(dev_val_df)} samples.")

    # --- Feature Engineering ---
    encoder = TargetEncoder(cols_to_encode=CATEGORICAL_COLS, n_splits=5)
    
    # Fit on the development training set and transform it
    dev_train_encoded = encoder.fit_transform(dev_train_df)
    
    # Transform the development validation set
    dev_val_encoded = encoder.transform(dev_val_df)

    feature_cols = [col for col in dev_train_encoded.columns if col.endswith('_encoded') or col == 'has_camera']
    
    X_train = dev_train_encoded[feature_cols]
    y_train = dev_train_encoded[TARGET_COL]
    
    X_val = dev_val_encoded[feature_cols]
    y_val = dev_val_encoded[TARGET_COL]

    # --- Model Training ---
    print("Training model...")
    model = HistGradientBoostingRegressor(random_state=42, max_iter=500, learning_rate=0.05)
    
    # Using early stopping on a validation set is implicitly handled by HistGradientBoostingRegressor
    # if you provide an evaluation set. Here we train on the full dev_train set.
    model.fit(X_train, y_train)

    # --- Validation Performance ---
    val_preds = model.predict(X_val)
    val_preds = np.maximum(0, val_preds) # Ensure predictions are non-negative
    
    final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Final Validation Performance: {final_validation_score}")

    # --- Test Set Prediction (if provided) ---
    if test_df is not None:
        print("Generating predictions for the test set...")
        
        # Check for unseen keys
        train_keys = set(zip(train_df[STREET_NAME_COL], train_df[VIOLATION_DESC_COL]))
        test_keys = set(zip(test_df[STREET_NAME_COL], test_df[VIOLATION_DESC_COL]))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Test set contains {unseen_keys_count} (street, violation) pairs not seen in training data.")
        
        # Transform test data using the encoder fitted on the full 2022 dataset
        # We refit the encoder on the full dataset to use all available information
        print("Refitting encoder on full training data for test set transformation...")
        full_encoder = TargetEncoder(cols_to_encode=CATEGORICAL_COLS)
        train_encoded_full = full_encoder.fit_transform(train_df)
        test_encoded = full_encoder.transform(test_df)
        
        X_test = test_encoded[feature_cols]
        
        # Refit the model on the full 2022 dataset
        print("Refitting model on full training data for final predictions...")
        X_train_full = train_encoded_full[feature_cols]
        y_train_full = train_encoded_full[TARGET_COL]
        model.fit(X_train_full, y_train_full)
        
        test_preds = model.predict(X_test)
        test_preds = np.maximum(0, test_preds).round() # Clip and round to integer

        # Generate submission file
        submission_df = pd.DataFrame({
            'street_name': test_df[STREET_NAME_COL],
            'violation_type': test_df[VIOLATION_DESC_COL],
            'predicted_count': test_preds
        })
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file created at: {submission_path}")

        # If test set has ground truth, report RMSE
        if TARGET_COL in test_df.columns:
            test_rmse = np.sqrt(mean_squared_error(test_df[TARGET_COL], test_preds))
            print(f"Test set RMSE (post-hoc): {test_rmse}")

if __name__ == '__main__':
    main()
