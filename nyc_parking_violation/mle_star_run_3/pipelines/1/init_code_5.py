
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, Pool
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

"""
This script implements a parking violation prediction pipeline.

Goal:
Predict the number of parking violations in 2023 for each (street_name, violation_type) pair, 
using 2022 data and additional augmentation tables.

Core Logic:
1.  Data Loading and Augmentation: Loads the core 2022 violation data and enriches it with 
    external datasets (street segments, violation codes, parking meters).
2.  Feature Engineering: Creates features from the merged data, including lengths, meter counts, 
    and encoded categorical variables.
3.  Model Training: Uses a CatBoostRegressor, which is well-suited for handling categorical features.
4.  Validation: Implements a grouped hold-out validation strategy based on street names to simulate 
    predicting for unseen streets, providing a more realistic performance estimate.
5.  Inference:
    - If a test file is provided, it generates predictions for every row.
    - It carefully handles cases where street names or violation types in the test set were not 
      seen during training, using a fallback strategy based on group means.
    - If the test file includes the target column, it calculates and reports the RMSE.
6.  Output: Produces a 'submission.csv' file with the predictions.

Design Choices:
-   Configurable Paths: Uses `argparse` to allow users to specify input file paths.
-   Log Transformation: The target variable ('violation_count') is log-transformed (log1p) before 
    training to handle its skewed distribution and stabilize the model. Predictions are transformed back.
-   Handling Unseen Keys: A specific strategy is implemented to deal with records in the test set 
    that don't have a corresponding entry in the training data. It uses the mean violation count for the 
    known violation type, or the global mean if both are unknown. This is crucial for robustness.
-   Memory Management: CatBoost's `Pool` object is used for efficient memory handling, which helps 
    prevent Out-Of-Memory (OOM) errors on large datasets.
"""

def load_and_merge_data(train_path):
    """Loads all data sources and merges them into a single DataFrame."""
    base_dir = './input'
    
    # Load core training data
    df_train = pd.read_csv(os.path.join(base_dir, train_path))
    
    # Standardize column names
    df_train.rename(columns={
        'Street Name': 'street_name',
        'Violation Description': 'violation_type',
        'violation_count': 'violation_count'
    }, inplace=True)

    # Load augmentation data
    try:
        df_streets = pd.read_csv(os.path.join(base_dir, 'street_segment_data.csv'))
        df_violations = pd.read_csv(os.path.join(base_dir, 'violation_codes.csv'))
        df_meters = pd.read_csv(os.path.join(base_dir, 'parking_meters.csv'))

        # Feature Engineering on augmentation tables
        # Aggregate street segment data to get total length per street
        df_streets['street_length'] = df_streets.groupby('street')['shape_len'].transform('sum')
        df_streets = df_streets[['street', 'street_length']].drop_duplicates().rename(columns={'street': 'street_name'})
        
        # Aggregate meter data to get count per street
        meter_counts = df_meters['STREET_NAME'].value_counts().reset_index()
        meter_counts.columns = ['street_name', 'meter_count']

        # Merge features
        df_train = pd.merge(df_train, df_streets, on='street_name', how='left')
        df_train = pd.merge(df_train, meter_counts, on='street_name', how='left')
        df_train = pd.merge(df_train, df_violations, left_on='violation_type', right_on='Violation Description', how='left')

        # Fill NaNs created by merges
        df_train['street_length'].fillna(df_train['street_length'].median(), inplace=True)
        df_train['meter_count'].fillna(0, inplace=True)
        # For violation code features, fill with a placeholder
        for col in ['Violation Category', 'Violation Code']:
            if col in df_train.columns:
                df_train[col].fillna('Unknown', inplace=True)
        
    except FileNotFoundError:
        print("Warning: One or more augmentation files not found. Proceeding with core data only.")

    return df_train

def create_features(df):
    """Creates new features from the existing data."""
    # Basic date/time features (if applicable, here we use them as categorical)
    # Example: extract month/day from a hypothetical date column if it existed
    # For this dataset, we can create interaction features if needed
    
    # Feature interactions
    df['street_violation'] = df['street_name'].astype(str) + '_' + df['violation_type'].astype(str)
    
    # Label encode high-cardinality features for the model
    for col in ['street_name', 'violation_type', 'street_violation']:
        if col in df.columns:
            # Using .astype('category').cat.codes handles new values in test set by assigning -1
            df[col + '_encoded'] = df[col].astype('category').cat.codes
    
    return df

def main(args):
    """Main function to run the training and prediction pipeline."""
    
    # --- 1. Load and Prepare Data ---
    print("Loading and preparing data...")
    df_train_full = load_and_merge_data(args.train_path)
    df_train_full = create_features(df_train_full)

    # Define features and target
    categorical_features = ['street_name', 'violation_type', 'Violation Category', 'Violation Code']
    # Filter out features that might not exist if augmentation fails
    categorical_features = [f for f in categorical_features if f in df_train_full.columns]
    
    numerical_features = ['street_length', 'meter_count']
    numerical_features = [f for f in numerical_features if f in df_train_full.columns]

    features = categorical_features + numerical_features
    target = 'violation_count'
    
    # Handle missing values in numerical features again after all merges
    for col in numerical_features:
        median_val = df_train_full[col].median()
        df_train_full[col].fillna(median_val, inplace=True)

    # Target transformation
    df_train_full['log_target'] = np.log1p(df_train_full[target])

    # --- 2. Validation Split ---
    # Use GroupShuffleSplit to validate on unseen streets, which is more robust
    print("Creating validation split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_train_full, groups=df_train_full['street_name']))
    
    train_data = df_train_full.iloc[train_idx]
    val_data = df_train_full.iloc[val_idx]

    X_train = train_data[features]
    y_train = train_data['log_target']
    X_val = val_data[features]
    y_val = val_data['log_target']
    
    # --- 3. Model Training ---
    print("Training CatBoost model...")
    # Using Pool for memory efficiency
    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=categorical_features)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        # Allow model to use all cores
        thread_count=-1
    )
    
    model.fit(train_pool, eval_set=val_pool)

    # --- 4. Validation Performance ---
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log).clip(0) # Inverse transform and clip at 0
    final_validation_score = np.sqrt(mean_squared_error(val_data[target], val_preds))
    print(f"Final Validation Performance: {final_validation_score}")

    # --- 5. Full Data Retraining (Optional but good practice) ---
    print("Retraining model on full 2022 dataset...")
    full_train_pool = Pool(data=df_train_full[features], label=df_train_full['log_target'], cat_features=categorical_features)
    model.fit(full_train_pool)
    
    # --- 6. Inference on Test Set ---
    if args.test_path:
        print(f"Generating predictions for {args.test_path}...")
        
        # Load test data
        try:
            df_test = pd.read_csv(os.path.join("./input", args.test_path))
            df_test.rename(columns={
                'Street Name': 'street_name',
                'Violation Description': 'violation_type'
            }, inplace=True)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}")
            return
        
        # Pre-computation for handling unseen keys
        global_mean_pred = np.expm1(y_train.mean())
        violation_type_mean_map = np.expm1(train_data.groupby('violation_type')['log_target'].mean())

        # Apply same feature engineering
        # Reload augmentation data to merge with test data
        base_dir = './input'
        try:
            df_streets = pd.read_csv(os.path.join(base_dir, 'street_segment_data.csv'))
            df_violations = pd.read_csv(os.path.join(base_dir, 'violation_codes.csv'))
            df_meters = pd.read_csv(os.path.join(base_dir, 'parking_meters.csv'))

            df_streets['street_length'] = df_streets.groupby('street')['shape_len'].transform('sum')
            df_streets = df_streets[['street', 'street_length']].drop_duplicates().rename(columns={'street': 'street_name'})
            
            meter_counts = df_meters['STREET_NAME'].value_counts().reset_index()
            meter_counts.columns = ['street_name', 'meter_count']

            df_test = pd.merge(df_test, df_streets, on='street_name', how='left')
            df_test = pd.merge(df_test, meter_counts, on='street_name', how='left')
            df_test = pd.merge(df_test, df_violations, left_on='violation_type', right_on='Violation Description', how='left')
            
            df_test['street_length'].fillna(train_data['street_length'].median(), inplace=True)
            df_test['meter_count'].fillna(0, inplace=True)
            for col in ['Violation Category', 'Violation Code']:
                if col in df_test.columns:
                    df_test[col].fillna('Unknown', inplace=True)

        except FileNotFoundError:
            # If augmentation fails for test, create empty columns
            if 'street_length' not in df_test: df_test['street_length'] = train_data['street_length'].median()
            if 'meter_count' not in df_test: df_test['meter_count'] = 0
            if 'Violation Category' not in df_test: df_test['Violation Category'] = 'Unknown'
            if 'Violation Code' not in df_test: df_test['Violation Code'] = 'Unknown'
        
        for col in numerical_features:
            df_test[col].fillna(df_train_full[col].median(), inplace=True)

        # Identify seen and unseen keys
        trained_streets = set(train_data['street_name'])
        trained_violations = set(train_data['violation_type'])
        
        df_test['is_seen'] = df_test.apply(
            lambda row: row['street_name'] in trained_streets and row['violation_type'] in trained_violations,
            axis=1
        )
        
        seen_test = df_test[df_test['is_seen']]
        unseen_test = df_test[~df_test['is_seen']]
        
        print(f"Test set contains {len(df_test)} rows.")
        print(f"Found {len(unseen_test)} rows with unseen (street_name, violation_type) pairs.")

        # Predict for seen keys using the model
        if not seen_test.empty:
            X_test_seen = seen_test[features]
            log_preds_seen = model.predict(X_test_seen)
            preds_seen = np.expm1(log_preds_seen).clip(0)
            seen_test['predicted_count'] = preds_seen

        # Predict for unseen keys using heuristics
        if not unseen_test.empty:
            unseen_test['predicted_count'] = unseen_test['violation_type'].map(violation_type_mean_map).fillna(global_mean_pred)
            unseen_test['predicted_count'] = unseen_test['predicted_count'].clip(0)

        # Combine predictions
        submission_df = pd.concat([seen_test, unseen_test], ignore_index=True)[['street_name', 'violation_type', 'predicted_count']]
        
        # Ensure submission has the same order as original test file
        original_order_df = df_test[['street_name', 'violation_type']]
        submission_df = pd.merge(original_order_df, submission_df, on=['street_name', 'violation_type'], how='left').drop_duplicates()

        # Save submission file
        submission_df.to_csv('submission.csv', index=False)
        print("submission.csv created successfully.")

        # If ground truth is available in the test set, score the predictions
        if 'violation_count' in df_test.columns:
            test_truth = pd.merge(submission_df, df_test, on=['street_name', 'violation_type'], how='left')
            test_rmse = np.sqrt(mean_squared_error(test_truth['violation_count'], test_truth['predicted_count']))
            print(f"RMSE on provided test file: {test_rmse}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parking Violation Prediction Pipeline")
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='(Optional) Path to the test data CSV file for prediction.')
    
    args = parser.parse_args()
    main(args)
