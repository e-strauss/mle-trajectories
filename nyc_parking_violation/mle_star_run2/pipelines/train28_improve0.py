
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # Standardize column names
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        sys.exit(1)

def feature_engineer(df, train_df=None):
    """Engineers features for the model."""
    logging.info("Starting feature engineering...")
    
    # Use the provided training data for building aggregation features
    if train_df is None:
        train_df = df

    # Ensure column names are standardized
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    if 'street_name' not in df.columns or 'violation_description' not in df.columns:
         raise ValueError("Input DataFrame must contain 'Street Name' and 'Violation Description' columns.")

    # Create interaction features if they don't exist
    df['street_violation'] = df['street_name'].astype(str) + "_" + df['violation_description'].astype(str)

    # Aggregations based on training data
    # Street Name aggregations
    street_agg = train_df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    street_agg.columns = ['street_name', 'street_mean', 'street_sum', 'street_std', 'street_key_count']
    street_agg['street_std'] = street_agg['street_std'].fillna(0) # std of a single value is NaN
    df = df.merge(street_agg, on='street_name', how='left')

    # Violation Description aggregations
    violation_type_agg = train_df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    violation_type_agg.columns = ['violation_description', 'violation_type_mean', 'violation_type_sum', 'violation_type_std', 'violation_type_key_count']
    violation_type_agg['violation_type_std'] = violation_type_agg['violation_type_std'].fillna(0) # std of a single value is NaN
    df = df.merge(violation_type_agg, on='violation_description', how='left')

    # Fill NaNs for keys that were not in the training set
    agg_cols = ['street_mean', 'street_sum', 'street_std', 'street_key_count', 
                'violation_type_mean', 'violation_type_sum', 'violation_type_std', 'violation_type_key_count']
    for col in agg_cols:
        # Filling with median as a robust choice for unseen keys
        median_val = train_df['violation_count'].median() if train_df is not None else df['violation_count'].median()
        if col.endswith('_mean') or col.endswith('_sum') or col.endswith('_std'):
             df[col] = df[col].fillna(median_val)
        else: # for key_count
             df[col] = df[col].fillna(1)


    logging.info("Feature engineering complete.")
    return df

def train_model(X, y, cat_features):
    """Trains a CatBoost Regressor model."""
    logging.info("Starting model training...")
    
    # Using KFold for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(X.shape[0])
    models = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        logging.info(f"Training Fold {fold+1}/5...")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=10,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=200,
            cat_features=cat_features,
            early_stopping_rounds=50,
            # Use less memory
            thread_count=-1,
            bootstrap_type='MVS' # Using MVS for subsampling
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        oof_predictions[val_index] = model.predict(X_val)
        models.append(model)
        
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    logging.info(f"OOF RMSE across all folds: {oof_rmse}")
    print(f'Final Validation Performance: {oof_rmse}') # Required output line

    # Train final model on all data
    logging.info("Training final model on all data...")
    final_model = CatBoostRegressor(
        iterations=1200, # Slightly more iterations for the final model
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        random_seed=42,
        verbose=200,
        cat_features=cat_features,
        thread_count=-1,
        bootstrap_type='MVS'
    )
    final_model.fit(X, y)

    return final_model

def main():
    """Main function to run the prediction pipeline."""
    parser = argparse.ArgumentParser(description='NYC Parking Violation Prediction')
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv', help='Path to the training data file.')
    parser.add_argument('--test-path', type=str, help='(Optional) Path to the test/evaluation data file.')
    args = parser.parse_args()

    # Load and prepare training data
    train_df = load_data(args.train_path)
    
    # Keep original street and violation names for submission file
    train_df['original_street_name'] = train_df['street_name']
    train_df['original_violation_description'] = train_df['violation_description']

    # Subsampling to manage memory, if needed.
    if len(train_df) > 500000:
        logging.info(f"Subsampling training data from {len(train_df)} to 500000 rows.")
        train_df = train_df.sample(n=500000, random_state=42)

    # Feature engineering on a copy of the training data
    # This train_full_df is used to generate features for both training and test sets
    train_full_df = feature_engineer(train_df.copy(), train_df=train_df)

    # Define features and target
    categorical_features = ['street_name', 'violation_description', 'street_violation']
    features = [col for col in train_full_df.columns if col not in ['violation_count', 'original_street_name', 'original_violation_description']]
    
    # Verify required feature columns exist
    required_features = ['street_mean', 'street_sum', 'street_key_count', 'street_std', 'violation_type_mean', 'violation_type_sum', 'violation_type_key_count', 'violation_type_std']
    for col in required_features:
        if col not in features:
            raise ValueError(f"Feature column '{col}' not found after feature engineering.")

    # Label encode categorical features
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        train_full_df[col] = le.fit_transform(train_full_df[col].astype(str))
        encoders[col] = le

    X = train_full_df[features]
    y = train_full_df['violation_count']

    # Train model (reports validation RMSE)
    model = train_model(X, y, cat_features=categorical_features)

    # Handle test/evaluation set if provided
    if args.test_path:
        logging.info(f"Processing test file: {args.test_path}")
        test_df = load_data(args.test_path)
        
        # Keep original names for submission file
        test_df['original_street_name'] = test_df['street_name']
        test_df['original_violation_description'] = test_df['violation_description']

        has_target = 'violation_count' in test_df.columns

        # Feature engineering for the test set
        # Important: use the original training data (train_df) to build features for the test set
        test_full_df = feature_engineer(test_df.copy(), train_df=train_df)

        # Handle unseen keys
        unseen_streets = test_df[~test_df['street_name'].isin(train_df['street_name'])]['street_name'].nunique()
        unseen_violations = test_df[~test_df['violation_description'].isin(train_df['violation_description'])]['violation_description'].nunique()
        logging.info(f"Test set contains {unseen_streets} street names not seen in training.")
        logging.info(f"Test set contains {unseen_violations} violation descriptions not seen in training.")
        logging.info("These are handled by filling with median/default values from the training set aggregations.")
        
        # Label encode test set using fitted encoders
        for col in categorical_features:
            le = encoders[col]
            # Handle unseen labels in test data by assigning a new category
            test_full_df[col] = test_full_df[col].astype(str).map(lambda s: s if s in le.classes_ else '<unknown>')
            le.classes_ = np.append(le.classes_, '<unknown>')
            test_full_df[col] = le.transform(test_full_df[col])


        X_test = test_full_df[features]

        # Predict
        predictions = model.predict(X_test)
        predictions[predictions < 0] = 0 # Ensure predictions are non-negative

        test_full_df['predicted_count'] = predictions.round().astype(int)

        # Score if target is available
        if has_target:
            test_rmse = np.sqrt(mean_squared_error(test_full_df['violation_count'], test_full_df['predicted_count']))
            logging.info(f"RMSE on provided test set: {test_rmse}")

        # Generate submission file
        submission_df = test_full_df[['original_street_name', 'original_violation_description', 'predicted_count']]
        submission_df.columns = ['street_name', 'violation_type', 'predicted_count']
        submission_df.to_csv('submission.csv', index=False)
        logging.info("submission.csv has been generated.")

if __name__ == '__main__':
    main()
