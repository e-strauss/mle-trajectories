
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import os

def clean_col_names(df):
    """Standardizes column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.replace(' ', '_').replace('-', '_').lower() for col in cols]
    df.columns = new_cols
    return df

def feature_engineer(df, train_aggregates=None):
    """
    Applies feature engineering to the dataframe.
    If train_aggregates is provided, it uses them for the test set.
    Otherwise, it calculates them from the input df (training set).
    """
    input_dir = './input'
    try:
        codes_df = pd.read_csv(os.path.join(input_dir, 'dof_parking_violation_codes.csv'))
        geo_df = pd.read_csv(os.path.join(input_dir, 'physical_id_to_address_name.csv'))
    except FileNotFoundError as e:
        print(f"Error: Augmentation file not found. Ensure '{e.filename}' is in the './input' directory.")
        # Return None to signal a fatal error in loading data.
        return None, None

    df = clean_col_names(df)
    codes_df = clean_col_names(codes_df)
    geo_df = clean_col_names(geo_df)

    # Make sure column names are consistent
    df = df.rename(columns={'street_name': 'street_name', 'violation_description': 'violation_description'})


    # 1. Merge violation code information (fine amount)
    df = df.merge(codes_df[['violation_description', 'all_other_areas_(fine_amt)']],
                  on='violation_description', how='left')
    df['all_other_areas_(fine_amt)'].fillna(df['all_other_areas_(fine_amt)'].median(), inplace=True)

    # 2. Merge geographical information (Borough)
    if not geo_df.empty and 'st_name' in geo_df.columns and 'borocode' in geo_df.columns:
        # Aggregate borocode by street name, taking the mode (most frequent value)
        boro_map = geo_df.groupby('st_name')['borocode'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1).reset_index()
        df = df.merge(boro_map, left_on='street_name', right_on='st_name', how='left')
        df.drop('st_name', axis=1, inplace=True)
        # Fill streets not found in geo data with a special value (-1)
        df['borocode'].fillna(-1, inplace=True)
        # Convert borocode to a categorical type for the model
        df['borocode'] = df['borocode'].astype(str)
    else:
        df['borocode'] = '-1' # Add borocode column if geo data is missing

    # --- 3. Create Aggregate Features ---
    # We will only calculate aggregates if we are in "training" mode
    is_train = train_aggregates is None
    if is_train:
        # Group by street name
        street_aggs = df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_aggs.columns = [f'street_{agg}' for agg in street_aggs.columns]
        street_aggs.reset_index(inplace=True)

        # Group by violation description
        violation_aggs = df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        violation_aggs.columns = [f'violation_{agg}' for agg in violation_aggs.columns]
        violation_aggs.reset_index(inplace=True)

        train_aggregates = {'street': street_aggs, 'violation': violation_aggs}

    # Merge aggregates into the dataframe
    df = df.merge(train_aggregates['street'], on='street_name', how='left')
    df = df.merge(train_aggregates['violation'], on='violation_description', how='left')

    # Fill NaNs that result from merges (e.g., unseen keys in test set)
    # This is the primary handling for unseen keys: their aggregate features are set to 0.
    agg_cols = [col for col in df.columns if 'street_' in col or 'violation_' in col]
    for col in agg_cols:
        if 'std' in col:
            df[col].fillna(0, inplace=True) # Std of a single point is 0
        else:
            df[col].fillna(df[col].median(), inplace=True) # Fill with median for other stats for robustness

    # Ensure all feature columns are of a type CatBoost can handle
    df['street_name'] = df['street_name'].astype(str)
    df['violation_description'] = df['violation_description'].astype(str)

    return df, train_aggregates


def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations using CatBoost Regressor.")
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data file relative to ./input/.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data file relative to ./input/.')
    # Use parse_known_args to avoid errors with unknown arguments (e.g., in Jupyter)
    args, _ = parser.parse_known_args()

    # --- Load Training Data ---
    train_file_path = os.path.join('./input', args.train_path)
    try:
        train_df_raw = pd.read_csv(train_file_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at '{train_file_path}'")
        return

    # --- Feature Engineering on Training Data ---
    train_df, train_aggs = feature_engineer(train_df_raw.copy())
    if train_df is None: # Check if feature engineering failed
        return

    # --- Model Training ---
    categorical_features = ['street_name', 'violation_description', 'borocode']
    features = [col for col in train_df.columns if col not in ['violation_count']]
    target = 'violation_count'

    X = train_df[features]
    y = train_df[target]

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the CatBoost model
    cat = cb.CatBoostRegressor(random_state=42,
                               verbose=0,
                               cat_features=categorical_features,
                               loss_function='RMSE', # Directly optimize for the competition metric
                               iterations=500, # A reasonable number of iterations
                               early_stopping_rounds=50) # Prevent overfitting

    # Train the model
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    # --- Validation ---
    val_preds = cat.predict(X_val)
    # Ensure predictions are non-negative
    val_preds[val_preds < 0] = 0

    validation_rmse = mean_squared_error(y_val, val_preds, squared=False)
    print(f'Final Validation Performance: {validation_rmse}')

    # --- Inference on Test Set (if provided) ---
    if args.test_path:
        print(f"\n--- Running Inference on {args.test_path} ---")
        test_file_path = os.path.join('./input', args.test_path)
        try:
            test_df_raw = pd.read_csv(test_file_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at '{test_file_path}'")
            return

        submission_keys = test_df_raw[['Street Name', 'Violation Description']].copy()
        submission_keys.columns = ['street_name', 'violation_type']

        # Report on unseen keys
        train_keys = set(zip(train_df_raw['Street Name'], train_df_raw['Violation Description']))
        test_keys = set(zip(submission_keys['street_name'], submission_keys['violation_type']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Number of (street_name, violation_type) pairs in test set not seen in training: {unseen_keys_count}")
        print("Handling for unseen keys: Aggregate features filled with median values. Unseen categorical values handled by CatBoost.")

        # Feature engineer the test set using aggregates from the training set
        test_df, _ = feature_engineer(test_df_raw.copy(), train_aggregates=train_aggs)
        if test_df is None:
            return

        # Ensure test set columns match training set features
        X_test = test_df[features]

        # Generate predictions
        test_preds = cat.predict(X_test)
        # Ensure predictions are non-negative
        test_preds[test_preds < 0] = 0

        # Create submission file
        submission_df = submission_keys
        submission_df['predicted_count'] = np.round(test_preds).astype(int) # Round to nearest integer count
        submission_df.to_csv('submission.csv', index=False)
        print("Generated submission.csv")

        # Score if ground truth is available in the test file
        if 'violation_count' in clean_col_names(test_df_raw.copy()).columns:
            test_rmse = mean_squared_error(test_df['violation_count'], test_preds, squared=False)
            print(f"RMSE on test file '{args.test_path}': {test_rmse}")


if __name__ == '__main__':
    main()
