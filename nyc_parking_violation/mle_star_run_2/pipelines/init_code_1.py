
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
import os
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def feature_engineer(df, train_stats_df=None):
    """
    Engineers features for the model.
    - If train_stats_df is provided, it uses stats from it to engineer features for df.
    - If train_stats_df is None, it calculates stats from df itself (training phase).
    """
    # Work on a copy to avoid modifying the original dataframe
    df_engineered = df.copy()

    # --- 1. Standardize Column Names ---
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- 2. Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    try:
        cscl = pd.read_csv(cscl_path)
        # Keep only relevant columns and remove duplicates
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()

        # Merge borough information
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    except FileNotFoundError:
        df_engineered['boroname'] = 'Unknown'

    # --- 3. Create Aggregate Features ---
    # Determine the source for calculating statistics
    if train_stats_df is None:
        source_df = df_engineered.copy()
    else:
        source_df = train_stats_df.copy()
        source_df.columns = [c.replace(' ', '_').lower() for c in source_df.columns]
        # Must also add borough info to the stats df if not present
        if 'boroname' not in source_df.columns:
            try:
                cscl = pd.read_csv(cscl_path)
                cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
                cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
                source_df['street_name_upper'] = source_df['street_name'].str.upper()
                source_df = pd.merge(source_df, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
                source_df['boroname'].fillna('Unknown', inplace=True)
                source_df.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
            except FileNotFoundError:
                 source_df['boroname'] = 'Unknown'


    # Aggregates by street_name
    street_agg = source_df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
    street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']

    # Aggregates by violation_description
    violation_agg = source_df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
    violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std', 'violation_key_count']

    # Aggregates by boroname
    boro_agg = source_df.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
    boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std', 'boro_key_count']

    # Merge aggregates into the engineered dataframe
    df_engineered = pd.merge(df_engineered, street_agg, on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, violation_agg, on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, boro_agg, on='boroname', how='left')

    # Fill NaNs created by left merges (for unseen keys in test data)
    # Also fill NaNs from std calculation (where count is 1)
    df_engineered.fillna(0, inplace=True)

    return df_engineered


def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations using LightGBM.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='(Optional) Path to the test data CSV file for prediction.')
    args = parser.parse_args()

    # --- 1. Load and Prepare Training Data ---
    print(f"Loading training data from {args.train_path}...")
    try:
        train_df_original = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}. Exiting.")
        return

    print("Engineering features for the training data...")
    train_df_featured = feature_engineer(train_df_original)

    # --- 2. Define Features and Target ---
    TARGET = 'violation_count'
    categorical_features = ['street_name', 'violation_description', 'boroname']
    
    # Convert categorical columns to 'category' dtype for LightGBM
    for col in categorical_features:
        train_df_featured[col] = train_df_featured[col].astype('category')

    features = [col for col in train_df_featured.columns if col != TARGET]
    
    # --- 3. Validation Split (Grouped by Street Name) ---
    print("Splitting data for validation...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(train_df_featured, groups=train_df_featured['street_name']))
    
    X_train = train_df_featured.iloc[train_idx][features]
    y_train = train_df_featured.iloc[train_idx][TARGET]
    X_val = train_df_featured.iloc[val_idx][features]
    y_val = train_df_featured.iloc[val_idx][TARGET]

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # --- 4. Model Training ---
    print("Training LightGBM model...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',  # MAE is often more robust to outliers
        metric='rmse',
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=40,
        max_depth=10,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=categorical_features
    )

    # --- 5. Validation Performance ---
    print("Evaluating model on the hold-out validation set...")
    val_preds = lgbm.predict(X_val)
    val_preds[val_preds < 0] = 0  # Ensure predictions are non-negative
    final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds))
    
    print(f'Final Validation Performance: {final_validation_score}')

    # --- 6. Test Prediction (if requested) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_original = pd.read_csv(args.test_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}. Skipping prediction.")
            return

        # Preserve original columns for submission file
        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()
        
        has_target = 'violation_count' in [c.lower().replace(' ', '_') for c in test_df_original.columns]
        if has_target:
            test_ground_truth = test_df_original['violation_count']

        # Engineer features for the test set using stats from the full training set
        print("Engineering features for the test data...")
        test_df_featured = feature_engineer(test_df_original, train_stats_df=train_df_original)

        # Handle categorical features, ensuring consistency with training data
        for col in categorical_features:
            train_cats = train_df_featured[col].cat.categories
            test_df_featured[col] = pd.Categorical(test_df_featured[col], categories=train_cats)
        
        # Align columns
        X_test = test_df_featured[features]

        # Report on unseen keys
        train_keys = set(zip(train_df_original['Street Name'], train_df_original['Violation Description']))
        test_keys = set(zip(test_df_original['Street Name'], test_df_original['Violation Description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} (street_name, violation_type) pairs in test set not present in training data.")
        
        print("Generating predictions on the test set...")
        test_preds = lgbm.predict(X_test)
        test_preds[test_preds < 0] = 0 # Ensure predictions are non-negative
        test_preds = np.round(test_preds).astype(int) # Round to nearest integer count

        # Create submission file
        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = test_preds
        submission_df.to_csv('submission.csv', index=False)
        print("Successfully created submission.csv")

        # If test set has labels, report performance
        if has_target:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_preds))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
