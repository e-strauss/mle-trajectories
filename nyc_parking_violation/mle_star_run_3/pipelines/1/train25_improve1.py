
import pandas as pd
import numpy as np
import argparse
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os

# Function to load and prepare data
def load_and_prepare_data(train_path, test_path=None):
    """Loads, merges, and preprocesses the data."""
    # Define file paths
    base_dir = "./input" if os.path.exists("./input") else "."
    violations_2022_path = os.path.join(base_dir, train_path)
    street_details_path = os.path.join(base_dir, "street_details.csv")
    pavement_maintenance_path = os.path.join(base_dir, "pavement_maintenance_history.csv")
    
    # Load core training data
    train_df = pd.read_csv(violations_2022_path)
    train_df.rename(columns={
        'Street Name': 'street_name',
        'Violation Description': 'violation_type',
        'violation_count': 'target'
    }, inplace=True)
    train_df['is_train'] = 1

    # Load test data if provided
    test_df = None
    if test_path:
        test_df_path = os.path.join(base_dir, test_path)
        if os.path.exists(test_df_path):
            test_df = pd.read_csv(test_df_path)
            test_df.rename(columns={
                'Street Name': 'street_name',
                'Violation Description': 'violation_type'
            }, inplace=True)
            if 'violation_count' in test_df.columns:
                test_df.rename(columns={'violation_count': 'target'}, inplace=True)
            else:
                test_df['target'] = np.nan # Target is not available
            test_df['is_train'] = 0
            full_df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            print(f"Test path not found: {test_df_path}")
            full_df = train_df
    else:
        full_df = train_df

    # Load and merge augmentation data
    if os.path.exists(street_details_path):
        street_details = pd.read_csv(street_details_path)
        full_df = pd.merge(full_df, street_details, on='street_name', how='left')

    if os.path.exists(pavement_maintenance_path):
        pavement_maintenance = pd.read_csv(pavement_maintenance_path)
        # Pre-aggregation for pavement maintenance
        pavement_maintenance['last_maintenance_year'] = pd.to_datetime(pavement_maintenance['last_maintenance_date'], errors='coerce').dt.year
        pavement_agg = pavement_maintenance.groupby('street_name').agg({
            'last_maintenance_year': 'max',
            'pavement_condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).reset_index()
        full_df = pd.merge(full_df, pavement_agg, on='street_name', how='left')

    # Feature Engineering
    full_df['street_name_len'] = full_df['street_name'].astype(str).apply(len)
    full_df['violation_type_len'] = full_df['violation_type'].astype(str).apply(len)
    
    # Handle cyclical features if any (example with year)
    if 'last_maintenance_year' in full_df.columns:
        current_year = 2023
        full_df['years_since_maintenance'] = current_year - full_df['last_maintenance_year']
        full_df.drop('last_maintenance_year', axis=1, inplace=True)

    # Impute missing values
    for col in full_df.columns:
        if full_df[col].isnull().any():
            if full_df[col].dtype == 'object':
                mode_val = full_df[col].mode()
                if not mode_val.empty:
                    full_df[col] = full_df[col].fillna(mode_val[0])
                else:
                    full_df[col] = full_df[col].fillna("missing")
            elif pd.api.types.is_numeric_dtype(full_df[col]):
                median_val = full_df[col].median()
                full_df[col] = full_df[col].fillna(median_val)
            else:
                mode_val = full_df[col].mode()
                if not mode_val.empty:
                    full_df[col] = full_df[col].fillna(mode_val[0])
                else:
                    full_df[col] = full_df[col].fillna("missing")

    # Encode Categorical Features
    categorical_features = ['street_name', 'violation_type', 'borough', 'pavement_condition']
    # Ensure all categorical columns exist before encoding
    categorical_features_exist = [c for c in categorical_features if c in full_df.columns]

    for col in categorical_features_exist:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(str))

    return full_df, categorical_features_exist

# Main function to run the pipeline
def main():
    parser = argparse.ArgumentParser(description="Parking Violation Prediction")
    parser.add_argument("--train-path", type=str, default="violations_per_street_2022.csv", help="Path to the training data.")
    parser.add_argument("--test-path", type=str, help="Path to the test data (optional).")
    args = parser.parse_args()

    # Load and prepare data
    full_df, categorical_features = load_and_prepare_data(args.train_path, args.test_path)

    # Separate train, validation, and test sets
    train_val_df = full_df[full_df['is_train'] == 1].copy()
    
    # Store original keys for submission file if test path is provided
    submission_keys = None
    if args.test_path:
        base_dir = "./input" if os.path.exists("./input") else "."
        test_file_path = os.path.join(base_dir, args.test_path)
        if os.path.exists(test_file_path):
            submission_keys = pd.read_csv(test_file_path)[['Street Name', 'Violation Description']]
            submission_keys.rename(columns={'Street Name': 'street_name', 'Violation Description': 'violation_type'}, inplace=True)
    
    # Get original training keys before encoding
    train_keys_orig = set(zip(train_val_df['street_name'].astype(str), train_val_df['violation_type'].astype(str)))

    X = train_val_df.drop(columns=['target', 'is_train'])
    y = train_val_df['target']

    # Create a validation set from the 2022 data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical feature indices for CatBoost
    cat_features_indices = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train]

    # Initialize and train the CatBoost model
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features_indices)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features_indices)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type="CPU" # Use CPU to avoid potential GPU memory issues
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate on the validation set
    val_preds = model.predict(X_val)
    val_preds = np.maximum(0, val_preds) # Ensure predictions are non-negative
    final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Final Validation Performance: {final_validation_score}")

    # If a test path is provided, generate predictions
    if args.test_path and 'is_train' in full_df.columns:
        test_df_from_full = full_df[full_df['is_train'] == 0].copy()
        
        if not test_df_from_full.empty:
            X_test = test_df_from_full.drop(columns=['target', 'is_train'])

            # Handle unseen keys
            if submission_keys is not None:
                test_keys_orig = set(zip(submission_keys['street_name'].astype(str), submission_keys['violation_type'].astype(str)))
                unseen_keys_count = len(test_keys_orig - train_keys_orig)
                print(f"Number of unseen (street_name, violation_type) pairs in test set: {unseen_keys_count}")
            
            predictions = model.predict(X_test)
            predictions = np.maximum(0, predictions) # Clip at zero

            # Create submission file
            submission_df = submission_keys.copy()
            submission_df['predicted_count'] = predictions
            
            submission_df.to_csv("submission.csv", index=False)
            print("submission.csv created successfully.")

            # If ground truth is available in the test file, score it
            if 'target' in test_df_from_full.columns and not test_df_from_full['target'].isnull().all():
                test_rmse = np.sqrt(mean_squared_error(test_df_from_full['target'], predictions))
                print(f"RMSE on the provided test set: {test_rmse}")
        else:
            print("No test data found to generate predictions.")

if __name__ == "__main__":
    main()
