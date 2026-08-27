
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads data from a CSV file."""
    logging.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def discover_and_load_aux_data(base_path='./input'):
    """Discovers and loads auxiliary CSV files from the input directory."""
    aux_data = {}
    logging.info(f"Discovering auxiliary data in {base_path}")
    if not os.path.exists(base_path):
        logging.warning(f"Auxiliary data path not found: {base_path}")
        return aux_data
    for item in os.listdir(base_path):
        if item.endswith('.csv') and 'violations_per_street' not in item:
            file_path = os.path.join(base_path, item)
            df_name = item.replace('.csv', '')
            logging.info(f"Found and loading auxiliary file: {item}")
            aux_data[df_name] = load_data(file_path)
    return aux_data

def feature_engineering(df, aux_data):
    """Creates new features for the model."""
    logging.info("Starting feature engineering...")
    df['street_name_len'] = df['Street Name'].str.len().fillna(0)
    df['violation_desc_len'] = df['Violation Description'].str.len().fillna(0)

    # Interaction features
    df['street_violation_interaction'] = df['Street Name'].astype(str) + "_" + df['Violation Description'].astype(str)

    # Merging auxiliary data
    if 'nyc_cscl' in aux_data and aux_data['nyc_cscl'] is not None:
        cscl = aux_data['nyc_cscl'].copy()
        
        # Standardize column names to lowercase for robust matching
        cscl.columns = [col.lower() for col in cscl.columns]
        
        # Find the street name column from a list of possibilities
        possible_street_cols = ['full_stree', 'st_name', 'streetname', 'street_name', 'street']
        street_col_found = None
        for col in possible_street_cols:
            if col in cscl.columns:
                street_col_found = col
                break
        
        # Define aggregations, checking for column existence
        agg_dict = {}
        if 'objectid' in cscl.columns:
            agg_dict['cscl_feature_count'] = ('objectid', 'count')
        if 'st_width' in cscl.columns:
            agg_dict['st_width_avg'] = ('st_width', 'mean')

        if street_col_found and agg_dict:
            logging.info(f"Found street column '{street_col_found}' in nyc_cscl.csv. Aggregating on: {list(agg_dict.keys())}")
            cscl.rename(columns={street_col_found: 'Street Name'}, inplace=True)
            
            cscl_agg = cscl.groupby('Street Name').agg(**agg_dict).reset_index()
            df = pd.merge(df, cscl_agg, on='Street Name', how='left')
            
            # Fill NaNs for the newly merged columns
            if 'cscl_feature_count' in df.columns:
                df['cscl_feature_count'] = df['cscl_feature_count'].fillna(0)
            if 'st_width_avg' in df.columns:
                # Fill NaNs with the mean of the column, but handle case where mean is also NaN
                mean_width = df['st_width_avg'].mean()
                df['st_width_avg'] = df['st_width_avg'].fillna(mean_width if pd.notna(mean_width) else 0)

        else:
            logging.warning("Could not find required columns in nyc_cscl.csv to create features. Skipping merge.")

    # Create dummy columns if they weren't created by the merge or if cscl data was not available
    if 'cscl_feature_count' not in df.columns:
        df['cscl_feature_count'] = 0
    if 'st_width_avg' not in df.columns:
        df['st_width_avg'] = 0

    logging.info("Finished feature engineering.")
    return df

def handle_unseen_keys(train_df, test_df):
    """
    Identifies keys in test_df not present in train_df and provides a strategy.
    The main strategy is handled by the model's native handling of unknown categories.
    """
    train_keys = set(zip(train_df['Street Name'], train_df['Violation Description']))
    test_keys = set(zip(test_df['Street Name'], test_df['Violation Description']))

    unseen_keys = test_keys - train_keys
    num_unseen = len(unseen_keys)

    logging.info(f"Number of key pairs in test/eval: {len(test_keys)}")
    logging.info(f"Number of key pairs in training: {len(train_keys)}")
    logging.info(f"Number of unseen key pairs in test/eval: {num_unseen}")
    if num_unseen > 0:
        logging.info("Handling of unseen keys: The model will treat these as unknown/missing categories.")
    return num_unseen

def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="NYC Parking Violations Prediction")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help="Path to the training data file.")
    parser.add_argument('--test-path', type=str, help="Optional: Path to the test/evaluation data file.")
    parser.add_argument('--subsample', type=float, default=None, help="Fraction of data to use for quick runs.")

    args = parser.parse_args()

    # Load data
    train_df_full = load_data(args.train_path)
    if train_df_full is None:
        return

    # Standardize column names
    train_df_full.rename(columns={
        col: col.strip().replace(' ', '_') for col in train_df_full.columns
    }, inplace=True)
    train_df_full.rename(columns={
        'Street_Name': 'Street Name',
        'Violation_Description': 'Violation Description'
    }, inplace=True)


    # Subsample for faster development if requested
    if args.subsample:
        logging.info(f"Subsampling data to {args.subsample*100}%")
        train_df_full = train_df_full.sample(frac=args.subsample, random_state=42)

    # Load auxiliary data
    aux_data = discover_and_load_aux_data()

    # Feature Engineering on the full training data
    train_df_full = feature_engineering(train_df_full, aux_data)

    # Define features and target
    X = train_df_full.drop('violation_count', axis=1)
    y = train_df_full['violation_count']

    # Use a grouped holdout for validation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_inds, val_inds = next(gss.split(X, y, groups=X['Street Name']))

    X_train, X_val = X.iloc[train_inds], X.iloc[val_inds]
    y_train, y_val = y.iloc[train_inds], y.iloc[val_inds]

    logging.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Let the regressor handle categoricals directly.
    categorical_features_model = ['Street Name', 'Violation Description']
    features_for_model = ['Street Name', 'Violation Description', 'street_name_len', 'violation_desc_len', 'cscl_feature_count', 'st_width_avg']

    # Convert categorical columns to 'category' dtype for HistGradientBoostingRegressor
    for col in categorical_features_model:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')

    # Get integer indices for categorical features relative to the final feature list
    categorical_feature_indices = [i for i, col in enumerate(features_for_model) if col in categorical_features_model]

    model = HistGradientBoostingRegressor(
        categorical_features=categorical_feature_indices,
        random_state=42,
        # Use low level of memory to prevent OOM
        max_bins=128
    )

    # Train the model
    logging.info("Training the model...")
    model.fit(X_train[features_for_model], y_train)
    logging.info("Model training complete.")

    # Validate the model
    val_predictions = model.predict(X_val[features_for_model])
    val_predictions[val_predictions < 0] = 0 # Enforce non-negativity
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    logging.info(f"Validation RMSE (2022 holdout): {rmse}")
    print(f"Final Validation Performance: {rmse}")

    # If a test path is provided, run predictions and evaluate
    if args.test_path:
        logging.info(f"Processing test file: {args.test_path}")
        test_df = load_data(args.test_path)
        if test_df is not None:
            # Preserve original columns for submission file
            original_test_df = test_df.copy()

            # Standardize test column names to match training
            test_df.rename(columns={
                col: col.strip().replace(' ', '_') for col in test_df.columns
            }, inplace=True)
            test_df.rename(columns={
                'Street_Name': 'Street Name',
                'Violation_Description': 'Violation Description'
            }, inplace=True)

            # Feature engineering on test data
            test_df = feature_engineering(test_df, aux_data)

            # Check for unseen keys
            handle_unseen_keys(train_df_full, test_df)

            # Convert to category dtype for prediction
            for col in categorical_features_model:
                # Use categories from the training set to handle unseen values
                train_cats = X_train[col].cat.categories
                test_df[col] = pd.Categorical(test_df[col], categories=train_cats, ordered=False)


            # Predict
            logging.info("Generating predictions on the test set...")
            test_predictions = model.predict(test_df[features_for_model])
            test_predictions[test_predictions < 0] = 0 # Enforce non-negativity

            # Create submission file
            submission_df = pd.DataFrame({
                'street_name': original_test_df['Street Name'],
                'violation_type': original_test_df['Violation Description'],
                'predicted_count': test_predictions.round().astype(int)
            })
            submission_df.to_csv('submission.csv', index=False)
            logging.info("Submission file 'submission.csv' created.")

            # If the test file has ground truth, score it
            if 'violation_count' in test_df.columns:
                test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], test_predictions))
                logging.info(f"Test set RMSE: {test_rmse}")

if __name__ == '__main__':
    main()
