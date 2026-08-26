
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import gc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(file_path, is_train=False):
    """Loads data, standardizes column names, and renames target column."""
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

    original_columns = df.columns.tolist()
    
    # Standardize column names (e.g., 'Street Name' -> 'street_name')
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Define a map for renaming to ensure consistency
    rename_map = {}
    
    # Identify and map core columns based on expected names after cleaning
    if 'street_name' in df.columns:
        rename_map['street_name'] = 'street_name'
    if 'violation_description' in df.columns:
        rename_map['violation_description'] = 'violation_type'
    
    # Crucial fix: Identify and rename the target column 'violation_count' to 'target'
    if 'violation_count' in df.columns:
        rename_map['violation_count'] = 'target'
        
    df.rename(columns=rename_map, inplace=True)
    
    print(f"Original columns: {original_columns}")
    print(f"Cleaned and renamed columns: {df.columns.tolist()}")

    # For training data, the target column is essential.
    if is_train and 'target' not in df.columns:
        raise KeyError(f"Target column ('violation_count') not found in training file: {file_path}. Please check the CSV header.")

    # Fill missing key columns with a placeholder string
    for col in ['street_name', 'violation_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('unknown')

    return df

def feature_engineering(df, train_features_info=None):
    """Creates features for the model and handles categorical encoding."""
    print("Starting feature engineering...")
    
    # Ensure key columns have string type
    df['street_name'] = df['street_name'].astype(str)
    df['violation_type'] = df['violation_type'].astype(str)

    # Basic text-based features
    df['street_name_len'] = df['street_name'].apply(len)
    df['violation_type_len'] = df['violation_type'].apply(len)
    df['street_name_words'] = df['street_name'].apply(lambda x: len(x.split()))

    # Interaction feature
    df['street_violation'] = df['street_name'] + "_" + df['violation_type']

    categorical_cols = ['street_name', 'violation_type', 'street_violation']
    
    if train_features_info is not None:
        # --- Applying transformations to test/validation data ---
        encoders = train_features_info['encoders']
        for col in categorical_cols:
            le = encoders[f'{col}_le']
            # Map unseen categories to a special 'unseen' token
            df[col] = df[col].map(lambda s: s if s in le.classes_ else 'unseen')
            df[f'{col}_encoded'] = le.transform(df[col])
        print("Applied label encoding from training data.")
    else:
        # --- Fitting transformations on training data ---
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Include 'unseen' in the fit to handle future unseen values gracefully
            all_values = pd.concat([df[col], pd.Series(['unseen'])]).unique()
            le.fit(all_values)
            df[f'{col}_encoded'] = le.transform(df[col])
            encoders[f'{col}_le'] = le
        train_features_info = {'encoders': encoders}
        print("Fitted label encoders on training data.")

    # We don't need the original text columns for the model itself
    df.drop(columns=categorical_cols, inplace=True, errors='ignore')
    
    print("Feature engineering complete.")
    return df, train_features_info

def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv', help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None, help='(Optional) Path to the test data CSV file for prediction.')
    
    args = parser.parse_args()

    # --- 1. Load and Preprocess Training Data ---
    train_df = load_data(args.train_path, is_train=True)
    if train_df is None:
        return # Exit if data loading failed

    # --- 2. Create Aggregate Features ---
    # These are calculated on the full training set before splitting to provide a rich signal.
    print("Creating aggregate features from training data...")
    street_agg = train_df.groupby('street_name')['target'].agg(['mean', 'sum', 'std']).add_prefix('street_agg_')
    violation_agg = train_df.groupby('violation_type')['target'].agg(['mean', 'sum', 'std']).add_prefix('violation_agg_')

    train_df = pd.merge(train_df, street_agg, on='street_name', how='left')
    train_df = pd.merge(train_df, violation_agg, on='violation_type', how='left')
    
    # --- 3. Feature Engineering ---
    train_df_processed, feature_info = feature_engineering(train_df.copy())
    
    # --- 4. Model Training with Cross-Validation ---
    features = [col for col in train_df_processed.columns if col not in ['target']]
    target = train_df_processed['target']

    print(f"\nTraining with {len(features)} features: {features}")
    
    # Log-transform target to handle skewed distribution, adding 1 to avoid log(0)
    log_target = np.log1p(target)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(train_df_processed.shape[0])
    models = []
    
    print("Starting model training with 5-fold cross-validation...")
    for fold, (train_index, val_index) in enumerate(kf.split(train_df_processed)):
        print(f"--- Fold {fold+1}/5 ---")
        X_train, X_val = train_df_processed.loc[train_index, features], train_df_processed.loc[val_index, features]
        y_train, y_val = log_target.iloc[train_index], log_target.iloc[val_index]

        model = lgb.LGBMRegressor(random_state=42, n_estimators=1000)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        val_preds = model.predict(X_val)
        oof_predictions[val_index] = val_preds
        models.append(model)
        
    # Inverse transform to get predictions in original scale and clip at zero
    oof_predictions = np.expm1(oof_predictions)
    oof_predictions[oof_predictions < 0] = 0
    
    validation_rmse = np.sqrt(mean_squared_error(target, oof_predictions))
    print("\n--- Development Validation Performance ---")
    print(f"OOF RMSE (on 2022 data): {validation_rmse}")
    print(f'Final Validation Performance: {validation_rmse}') # Required output line
    
    del train_df, train_df_processed, log_target, oof_predictions
    gc.collect()

    # --- 5. Prediction on Test Set (if provided) ---
    if args.test_path:
        print(f"\n--- Generating Predictions for Test Set: {args.test_path} ---")
        test_df = load_data(args.test_path, is_train=False)
        if test_df is None:
            return

        original_test_df = test_df.copy()

        # Handle unseen keys by reporting them
        unseen_streets = ~test_df['street_name'].isin(street_agg.index)
        print(f"Found {unseen_streets.sum()} rows with street names not seen in training data. These will be handled by filling aggregate features with NaN/0.")

        # Merge pre-computed aggregate features
        test_df = pd.merge(test_df, street_agg, on='street_name', how='left')
        test_df = pd.merge(test_df, violation_agg, on='violation_type', how='left')
        
        # Apply feature engineering using fitted encoders from training
        test_df_processed, _ = feature_engineering(test_df, feature_info)
        
        # Align features with the training set
        for col in features:
            if col not in test_df_processed.columns:
                test_df_processed[col] = 0 # Or a more sophisticated fillna strategy
        test_df_processed = test_df_processed[features]

        # Predict using the ensemble of models
        test_predictions_log = np.zeros(test_df_processed.shape[0])
        for model in models:
            test_predictions_log += model.predict(test_df_processed)
        test_predictions_log /= len(models)
        
        # Inverse transform, clip, and round
        test_predictions = np.expm1(test_predictions_log)
        test_predictions[test_predictions < 0] = 0
        
        # Create submission file
        submission_df = pd.DataFrame({
            'street_name': original_test_df['street_name'],
            'violation_type': original_test_df['violation_type'],
            'predicted_count': test_predictions.round().astype(int)
        })
        submission_df.to_csv('submission.csv', index=False)
        print("\nGenerated submission.csv")
        
        # Score on test set if ground truth is available
        if 'target' in original_test_df.columns:
            test_rmse = np.sqrt(mean_squared_error(original_test_df['target'], test_predictions))
            print("\n--- Test Set Performance ---")
            print(f"RMSE on '{args.test_path}': {test_rmse}")

if __name__ == '__main__':
    main()
