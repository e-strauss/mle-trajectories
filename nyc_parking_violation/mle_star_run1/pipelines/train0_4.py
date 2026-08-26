
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import argparse
import os
import sys

def clean_col_names(df):
    """Standardizes column names to be Python-friendly by making them lowercase,
    replacing spaces and dashes with underscores, and removing parentheses."""
    cols = df.columns
    new_cols = [col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').lower() for col in cols]
    df.columns = new_cols
    return df

def feature_engineer(df, is_train, train_artifacts=None):
    """
    Applies feature engineering using a robust train/test artifact pattern.
    - If is_train=True, it calculates aggregates and fits an OrdinalEncoder.
    - If is_train=False, it uses the provided train_artifacts.
    - This version is an integration of the base and reference solution approaches.
    """
    input_dir = './input'
    # Use a copy to avoid SettingWithCopyWarning and clean names at the start
    df = clean_col_names(df.copy())

    try:
        codes_df = pd.read_csv(os.path.join(input_dir, 'dof_parking_violation_codes.csv'))
        geo_df = pd.read_csv(os.path.join(input_dir, 'physical_id_to_address_name.csv'))
    except FileNotFoundError as e:
        print(f"Error: Augmentation file not found. Ensure '{e.filename}' is in the './input' directory.", file=sys.stderr)
        return None, None

    codes_df = clean_col_names(codes_df)
    geo_df = clean_col_names(geo_df)

    # 1. Merge violation code information (fine amount) - using reference solution's robust logic
    df = pd.merge(df, codes_df[['definition', 'all_other_areas']],
                  left_on='violation_description', right_on='definition', how='left')
    df.drop('definition', axis=1, inplace=True)
    df = df.rename(columns={'all_other_areas': 'fine_amt'})

    # 2. Merge geographical information (Borough)
    boro_map = geo_df.groupby('st_name')['borocode'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1).reset_index()
    df = df.merge(boro_map, left_on='street_name', right_on='st_name', how='left')
    df.drop('st_name', axis=1, inplace=True)
    df['borocode'].fillna(-1, inplace=True)
    df['borocode'] = df['borocode'].astype(str)

    categorical_features = ['street_name', 'violation_description', 'borocode']

    if is_train:
        train_artifacts = {}

        # a) Calculate and store median for imputation
        fine_amt_median = df['fine_amt'].median()
        train_artifacts['fine_amt_median'] = fine_amt_median

        # b) Calculate and store aggregate features
        street_aggs = df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_aggs.columns = [f'street_{agg}' for agg in street_aggs.columns]
        train_artifacts['street_aggs'] = street_aggs.reset_index()

        violation_aggs = df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        violation_aggs.columns = [f'violation_{agg}' for agg in violation_aggs.columns]
        train_artifacts['violation_aggs'] = violation_aggs.reset_index()

        # c) Create and fit the OrdinalEncoder
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
        encoder.fit(df[categorical_features])
        train_artifacts['encoder'] = encoder

    # Apply transformations using artifacts
    # Impute fine amount
    df['fine_amt'].fillna(train_artifacts['fine_amt_median'], inplace=True)

    # Merge aggregates (on string columns, before encoding)
    df = df.merge(train_artifacts['street_aggs'], on='street_name', how='left')
    df = df.merge(train_artifacts['violation_aggs'], on='violation_description', how='left')

    # Fill NaNs that result from merges for unseen keys in test/val
    agg_cols = [col for col in df.columns if 'street_' in col or 'violation_' in col]
    for col in agg_cols:
        if '_std' in col:
            df[col].fillna(0, inplace=True)
        else: # For mean, sum, count, 0 is a reasonable default for unseen items
            df[col].fillna(0, inplace=True)

    # Apply Ordinal Encoding to categorical features
    df[categorical_features] = train_artifacts['encoder'].transform(df[categorical_features])

    return df, train_artifacts

def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations.")
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data file relative to ./input/.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data file relative to ./input/.')
    args, _ = parser.parse_known_args()

    # --- Load Training Data ---
    train_file_path = os.path.join('./input', args.train_path)
    try:
        train_df_raw = pd.read_csv(train_file_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at '{train_file_path}'", file=sys.stderr)
        return

    # --- Split Data before Feature Engineering to prevent leakage ---
    # Clean names on the raw data before splitting to ensure consistency for key matching later
    train_df_raw_clean = clean_col_names(train_df_raw.copy())
    train_data_raw, val_data_raw = train_test_split(train_df_raw_clean, test_size=0.2, random_state=42)

    # --- Feature Engineering ---
    train_df_processed, train_artifacts = feature_engineer(train_data_raw, is_train=True)
    if train_df_processed is None:
        return
    val_df_processed, _ = feature_engineer(val_data_raw, is_train=False, train_artifacts=train_artifacts)

    # --- Prepare Data for Models ---
    target = 'violation_count'
    features = [col for col in train_df_processed.columns if col != target]

    X_train = train_df_processed[features]
    y_train = train_df_processed[target]
    X_val = val_df_processed[features]
    y_val = val_df_processed[target]

    # Identify indices of categorical features for models that require them
    categorical_feature_names = ['street_name', 'violation_description', 'borocode']
    categorical_feature_indices = [features.index(col) for col in categorical_feature_names]

    # --- Model Training ---
    # --- Model 1: LightGBM Training ---
    print("Training LightGBM model...")
    y_train_log = np.log1p(y_train)
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train_log, categorical_feature=categorical_feature_indices, feature_name=features)

    # --- Model 2: XGBoost Training ---
    print("Training XGBoost model...")
    xgbr = xgb.XGBRegressor(objective='count:poisson', random_state=42, n_estimators=100)
    xgbr.fit(X_train, y_train)

    # --- Model 3: CatBoost Training ---
    print("Training CatBoost model...")
    cat = cb.CatBoostRegressor(random_state=42, verbose=0, cat_features=categorical_feature_indices,
                               loss_function='RMSE', iterations=500, early_stopping_rounds=50)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    # --- Model 4: HistGradientBoostingRegressor ---
    print("Training HistGradientBoostingRegressor model...")
    hgb = HistGradientBoostingRegressor(loss='poisson', random_state=42, categorical_features=categorical_feature_indices)
    hgb.fit(X_train, y_train)

    # --- Model 5: RandomForestRegressor (from reference solution) ---
    print("Training RandomForestRegressor model...")
    rf_poisson = RandomForestRegressor(n_estimators=100, criterion='poisson', random_state=42, n_jobs=-1)
    rf_poisson.fit(X_train, y_train)

    # --- Validation ---
    val_preds_lgbm = np.maximum(0, np.expm1(lgbm.predict(X_val)))
    val_preds_xgb = np.maximum(0, xgbr.predict(X_val))
    val_preds_cat = np.maximum(0, cat.predict(X_val))
    val_preds_hgb = np.maximum(0, hgb.predict(X_val))
    val_preds_rf = np.maximum(0, rf_poisson.predict(X_val))

    # Ensemble predictions (simple average of 5 models)
    ensemble_val_preds = (val_preds_lgbm + val_preds_xgb + val_preds_cat + val_preds_hgb + val_preds_rf) / 5.0
    validation_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))

    print(f'Final Validation Performance: {validation_rmse}')

    # --- Inference on Test Set (if provided) ---
    if args.test_path:
        print(f"\n--- Running Inference on {args.test_path} ---")
        try:
            test_df_raw = pd.read_csv(os.path.join('./input', args.test_path))
        except FileNotFoundError:
            print(f"Error: Test file not found at '{os.path.join('./input', args.test_path)}'", file=sys.stderr)
            return

        test_df_clean = clean_col_names(test_df_raw.copy())
        submission_keys = test_df_clean[['street_name', 'violation_description']].copy()

        # Report on unseen keys
        train_keys = set(zip(train_df_raw_clean['street_name'], train_df_raw_clean['violation_description']))
        test_keys = set(zip(test_df_clean['street_name'], test_df_clean['violation_description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Number of (street_name, violation_description) pairs in test set not seen in training: {unseen_keys_count}")
        print("Handling for unseen keys: Aggregate features filled with 0. Unseen categorical values encoded as -1.")

        # Feature engineer the test set using artifacts from the training set
        test_df_processed, _ = feature_engineer(test_df_clean, is_train=False, train_artifacts=train_artifacts)
        if test_df_processed is None:
            return

        X_test = test_df_processed[features]

        # Generate predictions from all models
        test_preds_lgbm = np.maximum(0, np.expm1(lgbm.predict(X_test)))
        test_preds_xgb = np.maximum(0, xgbr.predict(X_test))
        test_preds_cat = np.maximum(0, cat.predict(X_test))
        test_preds_hgb = np.maximum(0, hgb.predict(X_test))
        test_preds_rf = np.maximum(0, rf_poisson.predict(X_test))

        # Ensemble predictions
        ensemble_test_preds = (test_preds_lgbm + test_preds_xgb + test_preds_cat + test_preds_hgb + test_preds_rf) / 5.0

        # Create submission file
        submission_df = submission_keys
        submission_df['predicted_count'] = ensemble_test_preds
        submission_df.to_csv('submission.csv', index=False)
        print("Generated submission.csv")

        # Score if ground truth is available in the test file
        if 'violation_count' in test_df_clean.columns:
            test_rmse = np.sqrt(mean_squared_error(test_df_clean['violation_count'], ensemble_test_preds))
            print(f"RMSE on test file '{args.test_path}': {test_rmse}")

if __name__ == '__main__':
    if not os.path.exists('./input'):
        print("Error: './input' directory not found. Please ensure it exists and contains the required data files.", file=sys.stderr)
    else:
        main()
