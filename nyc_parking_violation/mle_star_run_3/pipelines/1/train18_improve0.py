
import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import gc

# Define a function to reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df


def load_and_prepare_data(violation_path, boro_path, physical_path):
    """Loads and merges violation data with augmentation tables."""
    print(f"Loading training data from: {violation_path}")
    try:
        df = pd.read_csv(violation_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {violation_path}")
        return None
        
    df = reduce_mem_usage(df)

    # Standardize column names for merging
    df.rename(columns={
        'Street Name': 'street_name',
        'Violation Description': 'violation_type',
        'violation_count': 'violation_count'
    }, inplace=True)
    
    # Load and merge borough information
    if boro_path and os.path.exists(boro_path):
        print(f"Loading borough info from: {boro_path}")
        boro_df = pd.read_csv(boro_path, usecols=['physicalid', 'boroname', 'st_name'])
        boro_df = reduce_mem_usage(boro_df)
        boro_df.rename(columns={'st_name': 'street_name'}, inplace=True)
        # To avoid duplicated street names, we'll keep the first borough entry
        boro_df = boro_df.drop_duplicates(subset=['street_name'], keep='first')
        df = pd.merge(df, boro_df[['street_name', 'boroname']], on='street_name', how='left')
    else:
        print(f"Warning: Borough info file not found at {boro_path}. Skipping merge.")
        df['boroname'] = 'Unknown'

    # Load and merge physical features
    if physical_path and os.path.exists(physical_path):
        print(f"Loading physical features from: {physical_path}")
        phys_df = pd.read_csv(physical_path)
        phys_df = reduce_mem_usage(phys_df)
        phys_df.rename(columns={'street_name': 'street_name'}, inplace=True)
        
        # Aggregate physical features to street level
        phys_agg = phys_df.groupby('street_name').agg({
            'StreetWidth': ['mean', 'std'],
            'traf_signal': 'any',
            'Number_of_Lanes':['mean', 'median']
        }).reset_index()
        phys_agg.columns = ['_'.join(col).strip() for col in phys_agg.columns.values]
        phys_agg.rename(columns={'street_name_':'street_name'}, inplace=True)
        
        df = pd.merge(df, phys_agg, on='street_name', how='left')
    else:
        print(f"Warning: Physical features file not found at {physical_path}. Skipping merge.")

    df['street_name'] = df['street_name'].astype('category')
    df['violation_type'] = df['violation_type'].astype('category')
    df['boroname'] = df['boroname'].astype('category')

    return df


def main():
    parser = argparse.ArgumentParser(description="Predict NYC Parking Violations.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv', help='Path to the training data file.')
    parser.add_argument('--test-path', type=str, default=None, help='Path to the test data file (optional).')
    parser.add_argument('--borough-info-path', type=str, default='./input/nyc_cscl_boro.csv', help='Path to borough information.')
    parser.add_argument('--physical-features-path', type=str, default='./input/physical_features.csv', help='Path to physical features data.')

    args = parser.parse_args()

    # --- 1. Load Data ---
    train_df = load_and_prepare_data(args.train-path, args.borough-info-path, args.physical-features-path)
    if train_df is None:
        return # Exit if training data failed to load

    # Feature Engineering
    categorical_features = ['street_name', 'violation_type', 'boroname']
    for col in categorical_features:
        train_df[col] = train_df[col].cat.codes

    # Target transformation
    train_df['violation_count_log1p'] = np.log1p(train_df['violation_count'])
    
    features = [col for col in train_df.columns if col not in ['violation_count', 'violation_count_log1p']]
    X = train_df[features]
    y = train_df['violation_count_log1p']
    
    # --- 2. Validation ---
    print("\nStarting validation using GroupKFold...")
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(train_df))
    
    # Use street_name codes as groups
    groups = train_df['street_name']

    models_cat = []
    models_lgb = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # CatBoost Model
        cat_model = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            loss_function='RMSE',
            random_seed=42,
            verbose=0,
            allow_writing_files=False
        )
        cat_model.fit(X_train, y_train,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=50,
                      verbose=0)
        
        # LightGBM Model
        lgb_model = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
        )
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(50, verbose=False)])

        # Ensemble predictions (average)
        cat_preds = cat_model.predict(X_val)
        lgb_preds = lgb_model.predict(X_val)
        fold_preds = (cat_preds + lgb_preds) / 2.0
        
        oof_preds[val_idx] = fold_preds
        models_cat.append(cat_model)
        models_lgb.append(lgb_model)
        del X_train, y_train, X_val, y_val, cat_model, lgb_model
        gc.collect()

    # Inverse transform and calculate final validation RMSE
    oof_preds[oof_preds < 0] = 0
    oof_preds_inv = np.expm1(oof_preds)
    final_validation_score = np.sqrt(mean_squared_error(train_df['violation_count'], oof_preds_inv))
    print(f"\nFinal Validation Performance: {final_validation_score}")

    # --- 3. Test Prediction (if test_path is provided) ---
    if args.test_path:
        print(f"\n--- Generating predictions for test file: {args.test_path} ---")
        try:
            test_df_orig = pd.read_csv(args.test-path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}")
            return
            
        test_df = test_df_orig.copy()
        
        # Save ground truth for scoring if it exists
        ground_truth = None
        if 'violation_count' in test_df.columns:
            ground_truth = test_df['violation_count']
        
        # Preprocess test data just like training data
        test_df.rename(columns={
            'Street Name': 'street_name',
            'Violation Description': 'violation_type',
        }, inplace=True)

        # Record original keys for submission file
        submission_keys = test_df[['street_name', 'violation_type']].copy()
        
        # Identify unseen keys
        train_keys = train_df[['street_name', 'violation_type']].set_index(['street_name', 'violation_type']).index
        test_keys = test_df[['street_name', 'violation_type']].set_index(['street_name', 'violation_type']).index
        unseen_keys_count = len(test_keys.difference(train_keys))
        print(f"Test set contains {unseen_keys_count} (street_name, violation_type) pairs not seen in training data.")

        # Apply same data merges and feature engineering
        test_df = load_and_prepare_data(args.test-path, args.borough-info-path, args.physical-features-path)
        
        for col in categorical_features:
            # Use the label encoders fitted on the training data
            le = LabelEncoder().fit(train_df_orig[col.replace('_', ' ').title()])
            
            # Handle unseen values in test set by adding them to the encoder
            new_labels = set(test_df[col].unique()) - set(le.classes_)
            if len(new_labels) > 0:
                le.classes_ = np.concatenate([le.classes_, sorted(list(new_labels))])
            
            test_df[col] = le.transform(test_df[col])

        X_test = test_df[features]
        
        # Average predictions from all fold models
        test_preds_cat = np.mean([model.predict(X_test) for model in models_cat], axis=0)
        test_preds_lgb = np.mean([model.predict(X_test) for model in models_lgb], axis=0)
        test_predictions_log = (test_preds_cat + test_preds_lgb) / 2.0
        
        # Post-processing
        test_predictions_log[test_predictions_log < 0] = 0
        test_predictions = np.expm1(test_predictions_log)

        # Create submission file
        submission_df = submission_keys
        submission_df['predicted_count'] = test_predictions.round().astype(int)
        submission_df.to_csv('submission.csv', index=False)
        print("submission.csv generated successfully.")
        
        # Score against ground truth if available
        if ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(ground_truth, test_predictions))
            print(f"Test RMSE (on provided file): {test_rmse}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\nAn unhandled error occurred:")
        traceback.print_exc()
