
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
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
        # Returning None to signal failure
        return None, None

    # Use a copy to avoid SettingWithCopyWarning
    df = df.copy()

    codes_df = clean_col_names(codes_df)
    geo_df = clean_col_names(geo_df)

    # Merge violation code information (fine amount)
    df = df.merge(codes_df[['violation_description', 'all_other_areas_(fine_amt)']],
                  on='violation_description', how='left')
    df['all_other_areas_(fine_amt)'].fillna(df['all_other_areas_(fine_amt)'].median(), inplace=True)

    # Merge geographical information
    # Aggregate borocode by street name, taking the mode (most frequent value)
    boro_map = geo_df.groupby('st_name')['borocode'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1).reset_index()
    df = df.merge(boro_map, left_on='street_name', right_on='st_name', how='left')
    df.drop('st_name', axis=1, inplace=True)
    df['borocode'].fillna(-1, inplace=True) # Fill streets not found in geo data with -1

    # --- Create Aggregate Features ---
    is_train = train_aggregates is None
    if is_train:
        street_aggs = df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_aggs.columns = [f'street_{agg}' for agg in street_aggs.columns]
        street_aggs.reset_index(inplace=True)

        violation_aggs = df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        violation_aggs.columns = [f'violation_{agg}' for agg in violation_aggs.columns]
        violation_aggs.reset_index(inplace=True)
        
        train_aggregates = {'street': street_aggs, 'violation': violation_aggs}

    # Merge aggregates
    df = df.merge(train_aggregates['street'], on='street_name', how='left')
    df = df.merge(train_aggregates['violation'], on='violation_description', how='left')

    # Fill NaNs that result from merges (e.g., unseen keys in test set)
    for col in df.columns:
        if 'street_' in col or 'violation_' in col:
            # Fill NaNs with 0 for aggregate features
            df[col].fillna(0, inplace=True)

    return df, train_aggregates

def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations.")
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data file relative to ./input/.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data file relative to ./input/.')
    args = parser.parse_args()

    # --- Load Training Data ---
    train_file_path = os.path.join('./input', args.train_path)
    try:
        train_df_raw = pd.read_csv(train_file_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at '{train_file_path}'")
        return

    train_df_raw = clean_col_names(train_df_raw)

    # --- Feature Engineering on Training Data ---
    train_df, train_aggs = feature_engineer(train_df_raw)
    if train_df is None: # Check if feature engineering failed
        return

    # --- Prepare Data for Models ---
    categorical_features = ['street_name', 'violation_description', 'borocode']
    for col in categorical_features:
        train_df[col] = train_df[col].astype('category')

    features = [col for col in train_df.columns if col not in ['violation_count']]
    target = 'violation_count'
    
    X = train_df[features]
    y = train_df[target]

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model 1: LightGBM Training ---
    print("Training LightGBM model...")
    y_train_log = np.log1p(y_train)
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train_log, categorical_feature=categorical_features, feature_name=features)

    # --- Model 2: XGBoost Training ---
    print("Training XGBoost model...")
    xgbr = xgb.XGBRegressor(objective='count:poisson',
                            random_state=42,
                            enable_categorical=True,
                            n_estimators=100)
    xgbr.fit(X_train, y_train)

    # --- Validation ---
    # Predict with LightGBM
    val_preds_lgbm_log = lgbm.predict(X_val)
    val_preds_lgbm = np.expm1(val_preds_lgbm_log)
    val_preds_lgbm[val_preds_lgbm < 0] = 0

    # Predict with XGBoost
    val_preds_xgb = xgbr.predict(X_val)
    val_preds_xgb[val_preds_xgb < 0] = 0

    # Ensemble predictions (simple average)
    ensemble_val_preds = (val_preds_lgbm + val_preds_xgb) / 2.0
    
    validation_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
    
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
            
        test_df_raw = clean_col_names(test_df_raw)
        submission_keys = test_df_raw[['street_name', 'violation_description']].copy()
        
        # Report on unseen keys
        train_keys = set(zip(train_df_raw['street_name'], train_df_raw['violation_description']))
        test_keys = set(zip(test_df_raw['street_name'], test_df_raw['violation_description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Number of (street_name, violation_type) pairs in test set not seen in training: {unseen_keys_count}")
        print("Handling for unseen keys: Aggregate features will be filled with 0.")

        # Feature engineer the test set using aggregates from the training set
        test_df, _ = feature_engineer(test_df_raw, train_aggregates=train_aggs)
        if test_df is None:
            return

        # Align categorical features with training data for both models
        for col in categorical_features:
            train_categories = X_train[col].cat.categories
            test_df[col] = pd.Categorical(test_df[col], categories=train_categories, ordered=False)
            if test_df[col].isnull().any():
                most_frequent_cat_code = X_train[col].cat.codes.mode()[0]
                test_df[col].fillna(X_train[col].cat.categories[most_frequent_cat_code], inplace=True)

        X_test = test_df[features]

        # Generate predictions from both models
        # LGBM
        test_preds_lgbm_log = lgbm.predict(X_test)
        test_preds_lgbm = np.expm1(test_preds_lgbm_log)
        test_preds_lgbm[test_preds_lgbm < 0] = 0

        # XGBoost
        test_preds_xgb = xgbr.predict(X_test)
        test_preds_xgb[test_preds_xgb < 0] = 0
        
        # Ensemble predictions
        ensemble_test_preds = (test_preds_lgbm + test_preds_xgb) / 2.0
        
        # Create submission file
        submission_df = submission_keys
        submission_df['predicted_count'] = ensemble_test_preds
        submission_df.to_csv('submission.csv', index=False)
        print("Generated submission.csv")

        # Score if ground truth is available in the test file
        if 'violation_count' in test_df_raw.columns:
            test_rmse = np.sqrt(mean_squared_error(test_df_raw['violation_count'], ensemble_test_preds))
            print(f"RMSE on test file '{args.test_path}': {test_rmse}")

if __name__ == '__main__':
    main()
