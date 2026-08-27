
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import argparse
import os
import sys

def clean_col_names(df):
    """Standardizes column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.replace(' ', '_').replace('-', '_').lower() for col in cols]
    df.columns = new_cols
    return df

def feature_engineer(df, is_train, train_artifacts=None):
    """
    Applies feature engineering to the dataframe.
    - If is_train=True, it calculates aggregates and fits encoders.
    - If is_train=False, it uses the provided train_artifacts.
    """
    input_dir = './input'
    try:
        codes_df = pd.read_csv(os.path.join(input_dir, 'dof_parking_violation_codes.csv'))
        geo_df = pd.read_csv(os.path.join(input_dir, 'physical_id_to_address_name.csv'))
    except FileNotFoundError as e:
        print(f"Error: Augmentation file not found. Ensure '{e.filename}' is in the './input' directory.", file=sys.stderr)
        return None, None

    df = clean_col_names(df)
    codes_df = clean_col_names(codes_df)
    geo_df = clean_col_names(geo_df)

    # Standardize key column names for merging
    df = df.rename(columns={'street_name': 'street_name', 'violation_description': 'violation_description'})

    # 1. Merge violation code information (fine amount)
    df = df.merge(codes_df[['violation_description', 'all_other_areas_(fine_amt)']],
                  on='violation_description', how='left')
    
    # 2. Merge geographical information (Borough)
    if not geo_df.empty and 'st_name' in geo_df.columns and 'borocode' in geo_df.columns:
        boro_map = geo_df.groupby('st_name')['borocode'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1).reset_index()
        df = df.merge(boro_map, left_on='street_name', right_on='st_name', how='left')
        df.drop('st_name', axis=1, inplace=True)
        df['borocode'].fillna(-1, inplace=True)
        df['borocode'] = df['borocode'].astype(str)
    else:
        df['borocode'] = '-1'

    # --- 3. Create/Apply Aggregate & Categorical Features ---
    categorical_features = ['street_name', 'violation_description', 'borocode']
    
    if is_train:
        train_artifacts = {}
        
        # Fill fine amount NaNs based on training data median
        fine_amt_median = df['all_other_areas_(fine_amt)'].median()
        train_artifacts['fine_amt_median'] = fine_amt_median
        df['all_other_areas_(fine_amt)'].fillna(fine_amt_median, inplace=True)

        # Create Aggregate Features
        street_aggs = df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_aggs.columns = [f'street_{agg}' for agg in street_aggs.columns]
        train_artifacts['street_aggs'] = street_aggs.reset_index()

        violation_aggs = df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        violation_aggs.columns = [f'violation_{agg}' for agg in violation_aggs.columns]
        train_artifacts['violation_aggs'] = violation_aggs.reset_index()
        
        # Fit Ordinal Encoder
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
        df[categorical_features] = encoder.fit_transform(df[categorical_features])
        train_artifacts['encoder'] = encoder
        
    else: # is_test or is_validation
        # Use artifacts from training
        df['all_other_areas_(fine_amt)'].fillna(train_artifacts['fine_amt_median'], inplace=True)
        df[categorical_features] = train_artifacts['encoder'].transform(df[categorical_features])

    # Merge aggregates into the dataframe
    df = df.merge(train_artifacts['street_aggs'], on='street_name', how='left')
    df = df.merge(train_artifacts['violation_aggs'], on='violation_description', how='left')

    # Fill NaNs that result from merges (e.g., unseen keys in test set)
    # This is the primary handling for unseen keys: their aggregate features are set to a default value.
    for col in df.columns:
        if '_std' in col:
            df[col].fillna(0, inplace=True) # Std of a single point is 0
        elif df[col].dtype == 'float64':
            # For other stats like mean, sum, count fill with 0, assuming unseen means no history
            df[col].fillna(0, inplace=True)

    return df, train_artifacts


def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations using HistGradientBoostingRegressor.")
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data file relative to ./input/.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data file relative to ./input/.')
    args, _ = parser.parse_known_args()

    # --- Load and Prepare Training Data ---
    train_file_path = os.path.join('./input', args.train_path)
    try:
        train_df_raw = pd.read_csv(train_file_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at '{train_file_path}'", file=sys.stderr)
        return

    # Create a dev validation set from 2022 data
    train_data, val_data = train_test_split(train_df_raw, test_size=0.2, random_state=42)

    # Feature engineer training data (fit transformers)
    train_df_processed, train_artifacts = feature_engineer(train_data.copy(), is_train=True)
    if train_df_processed is None:
        return

    # Feature engineer validation data (using training transformers)
    val_df_processed, _ = feature_engineer(val_data.copy(), is_train=False, train_artifacts=train_artifacts)
    
    # --- Model Training ---
    # Define features and target
    # Exclude original un-encoded categorical columns if they still exist by chance
    features = [col for col in train_df_processed.columns if col not in ['violation_count', 'Street Name', 'Violation Description']]
    target = 'violation_count'
    
    X_train = train_df_processed[features]
    y_train = train_df_processed[target]
    
    X_val = val_df_processed[features]
    y_val = val_df_processed[target]

    # Initialize the model with Poisson loss for count data
    hgb_poisson = HistGradientBoostingRegressor(loss='poisson', random_state=42, categorical_features=None)
    
    # Train the model
    hgb_poisson.fit(X_train, y_train)

    # --- Validation ---
    val_preds = hgb_poisson.predict(X_val)
    val_preds = np.maximum(0, val_preds) # Ensure predictions are non-negative

    validation_rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f'Final Validation Performance: {validation_rmse}')

    # --- Inference on Test Set (if provided) ---
    if args.test_path:
        print(f"\n--- Running Inference on {args.test_path} ---")
        test_file_path = os.path.join('./input', args.test_path)
        try:
            test_df_raw = pd.read_csv(test_file_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at '{test_file_path}'", file=sys.stderr)
            return

        submission_keys = test_df_raw[['Street Name', 'Violation Description']].copy()
        submission_keys = submission_keys.rename(columns={'Street Name': 'street_name', 'Violation Description': 'violation_type'})

        # Report on unseen keys
        train_keys = set(zip(train_df_raw['Street Name'], train_df_raw['Violation Description']))
        test_keys = set(zip(submission_keys['street_name'], submission_keys['violation_type']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Number of (street_name, violation_type) pairs in test set not seen in training: {unseen_keys_count}")
        print("Handling for unseen keys: Aggregate features filled with 0. Unseen categorical values encoded as -1.")

        # Feature engineer the test set using artifacts from the training set
        test_df_processed, _ = feature_engineer(test_df_raw.copy(), is_train=False, train_artifacts=train_artifacts)
        if test_df_processed is None:
            return

        # Ensure test set columns match training set features, in the correct order
        X_test = test_df_processed[features]

        # Generate predictions
        test_preds = hgb_poisson.predict(X_test)
        test_preds = np.maximum(0, test_preds) # Clip at zero

        # Create submission file
        submission_df = submission_keys
        submission_df['predicted_count'] = np.round(test_preds).astype(int)
        submission_df.to_csv('submission.csv', index=False)
        print("Generated submission.csv")

        # Score if ground truth is available in the test file
        if 'violation_count' in clean_col_names(test_df_raw.copy()).columns:
            test_ground_truth = clean_col_names(test_df_raw.copy())['violation_count']
            test_rmse = mean_squared_error(test_ground_truth, test_preds, squared=False)
            print(f"RMSE on test file '{args.test_path}': {test_rmse}")


if __name__ == '__main__':
    main()
