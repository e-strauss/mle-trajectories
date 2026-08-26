
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

def clean_col_names(df, rename_map=None):
    """Standardizes column names and applies specific renames."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df

def load_and_create_features(train_path, test_path=None):
    """
    Loads all data sources, merges them, and creates features for two models.
    """
    base_dir = './input'
    
    # --- 1. Load Core Data ---
    df_train = pd.read_csv(train_path)
    rename_map = {'street_name': 'street_name', 'violation_description': 'violation_type'}
    df_train = clean_col_names(df_train, rename_map)

    # --- 2. Load and Process ALL Augmentation Data ---
    
    # Base solution augmentation files
    try:
        boroughs_df = clean_col_names(pd.read_csv(os.path.join(base_dir, 'street_names_and_boroughs.csv')))
        physical_df = clean_col_names(pd.read_csv(os.path.join(base_dir, 'physical_features_per_street.csv')))
    except FileNotFoundError:
        print("Warning: Base solution augmentation data (boroughs/physical) not found.")
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])
        
    # Reference solution augmentation files
    try:
        df_streets = pd.read_csv(os.path.join(base_dir, 'street_segment_data.csv'))
        df_violations = pd.read_csv(os.path.join(base_dir, 'violation_codes.csv'))
        df_meters = pd.read_csv(os.path.join(base_dir, 'parking_meters.csv'))
        
        # Feature Engineering from reference solution
        df_streets['street_length'] = df_streets.groupby('street')['shape_len'].transform('sum')
        df_streets = df_streets[['street', 'street_length']].drop_duplicates().rename(columns={'street': 'street_name'})
        
        meter_counts = df_meters['STREET_NAME'].value_counts().reset_index()
        meter_counts.columns = ['street_name', 'meter_count']
    except FileNotFoundError:
        print("Warning: Reference solution augmentation data (segments/meters/codes) not found.")
        df_streets = pd.DataFrame(columns=['street_name', 'street_length'])
        df_violations = pd.DataFrame(columns=['Violation Description', 'Violation Category', 'Violation Code'])
        meter_counts = pd.DataFrame(columns=['street_name', 'meter_count'])

    # --- 3. Merge and Create Full DataFrame ---
    full_df = df_train.copy()
    
    # Merge features from both solutions
    full_df = pd.merge(full_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')
    full_df = pd.merge(full_df, df_streets, on='street_name', how='left')
    full_df = pd.merge(full_df, meter_counts, on='street_name', how='left')
    full_df = pd.merge(full_df, df_violations, left_on='violation_type', right_on='Violation Description', how='left')

    # --- 4. Imputation & Feature Creation ---
    imputation_values = {}

    # Impute base features
    full_df['borough'].fillna('Unknown', inplace=True)
    base_numerical_cols = [col for col in physical_df.columns if col != 'street_name']
    for col in base_numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            imputation_values[col] = median_val
            full_df[col].fillna(median_val, inplace=True)
            
    # Impute reference features
    if 'street_length' in full_df.columns:
        median_val = full_df['street_length'].median()
        imputation_values['street_length'] = median_val
        full_df['street_length'].fillna(median_val, inplace=True)
    if 'meter_count' in full_df.columns:
        imputation_values['meter_count'] = 0
        full_df['meter_count'].fillna(0, inplace=True)
    for col in ['Violation Category', 'Violation Code']:
        if col in full_df.columns:
            full_df[col].fillna('Unknown', inplace=True)

    # Create log-transformed target for the reference model
    full_df['log_target'] = np.log1p(full_df['violation_count'])
    
    # --- 5. Define Feature Sets ---
    cat_features_base = ['street_name', 'violation_type', 'borough']
    numerical_features_base = [col for col in base_numerical_cols if col in full_df.columns]
    features_base = cat_features_base + numerical_features_base
    
    cat_features_ref = ['street_name', 'violation_type', 'Violation Category', 'Violation Code']
    numerical_features_ref = ['street_length', 'meter_count']
    cat_features_ref = [f for f in cat_features_ref if f in full_df.columns]
    numerical_features_ref = [f for f in numerical_features_ref if f in full_df.columns]
    features_ref = cat_features_ref + numerical_features_ref

    all_cat_features = list(set(cat_features_base + cat_features_ref))
    for col in all_cat_features:
        full_df[col] = full_df[col].astype('category')
        
    # --- 6. Process Test Set ---
    test_df_processed = None
    test_ground_truth = None
    if test_path:
        test_df = pd.read_csv(test_path)
        test_df = clean_col_names(test_df, {'violation_description': 'violation_type'})
        if 'violation_count' in test_df.columns:
            test_ground_truth = test_df['violation_count'].copy()

        test_df_processed = test_df.copy()
        
        # Merge all features
        test_df_processed = pd.merge(test_df_processed, boroughs_df, on='street_name', how='left')
        test_df_processed = pd.merge(test_df_processed, physical_df, on='street_name', how='left')
        test_df_processed = pd.merge(test_df_processed, df_streets, on='street_name', how='left')
        test_df_processed = pd.merge(test_df_processed, meter_counts, on='street_name', how='left')
        test_df_processed = pd.merge(test_df_processed, df_violations, left_on='violation_type', right_on='Violation Description', how='left')

        # Impute missing values using values from TRAINING data
        test_df_processed['borough'].fillna('Unknown', inplace=True)
        for col, val in imputation_values.items():
            if col in test_df_processed.columns:
                test_df_processed[col].fillna(val, inplace=True)
            else:
                test_df_processed[col] = val
        
        if 'meter_count' in test_df_processed.columns: test_df_processed['meter_count'].fillna(0, inplace=True)
        for col in ['Violation Category', 'Violation Code']:
            if col in test_df_processed.columns:
                test_df_processed[col].fillna('Unknown', inplace=True)
        
        for col in all_cat_features:
             if col in test_df_processed.columns:
                 test_df_processed[col] = test_df_processed[col].astype('category')
    
    return full_df, test_df_processed, test_ground_truth, features_base, cat_features_base, features_ref, cat_features_ref

def main():
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction with Ensemble CatBoost")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv')
    parser.add_argument('--test-path', type=str, default=None)
    args = parser.parse_args()

    # --- 1. Data Preparation ---
    print("Loading and preparing data for both models...")
    train_data, test_data, test_ground_truth, features_base, cat_features_base, features_ref, cat_features_ref = \
        load_and_create_features(args.train_path, args.test_path)
    
    # --- 2. Validation Split (Grouped by Street to simulate unseen streets) ---
    print("Creating grouped validation split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(train_data, groups=train_data['street_name']))
    
    df_train = train_data.iloc[train_idx]
    df_val = train_data.iloc[val_idx]
    
    # Data for Base Model (linear target)
    X_train_base = df_train[features_base]
    y_train_base = df_train['violation_count']
    X_val_base = df_val[features_base]
    y_val_true = df_val['violation_count'] # Ground truth for final validation

    # Data for Reference Model (log target)
    X_train_ref = df_train[features_ref]
    y_train_ref = df_train['log_target']
    X_val_ref = df_val[features_ref]
    y_val_ref = df_val['log_target']

    # --- 3. Model Training ---
    
    # Model 1 (Base features, linear target)
    print("\nTraining Model 1 (Base features)...")
    model_1 = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=200,
        cat_features=cat_features_base,
        early_stopping_rounds=50,
        task_type='CPU',
    )
    model_1.fit(X_train_base, y_train_base, eval_set=(X_val_base, y_val_true), use_best_model=True)

    # Model 2 (Reference features, log target)
    print("\nTraining Model 2 (Reference features)...")
    train_pool = Pool(data=X_train_ref, label=y_train_ref, cat_features=cat_features_ref)
    val_pool = Pool(data=X_val_ref, label=y_val_ref, cat_features=cat_features_ref)
    
    model_2 = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    model_2.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # --- 4. Validation Performance & Ensembling ---
    print("\nEnsembling models and calculating validation performance...")
    
    val_preds_1 = model_1.predict(X_val_base)
    val_preds_2_log = model_2.predict(X_val_ref)
    val_preds_2 = np.expm1(val_preds_2_log) # Inverse transform predictions
    
    # Simple averaging ensemble
    ensemble_predictions = (val_preds_1 + val_preds_2) / 2.0
    ensemble_predictions = np.maximum(0, ensemble_predictions) # Clip at 0

    val_rmse = np.sqrt(mean_squared_error(y_val_true, ensemble_predictions))
    print(f"Final Validation Performance: {val_rmse:.4f}")

    # --- 5. Test Set Prediction ---
    if args.test_path and test_data is not None:
        print(f"\nProcessing test file and generating submission: {args.test_path}")

        test_data_base = test_data[features_base]
        test_data_ref = test_data[features_ref]
        
        test_preds_1 = model_1.predict(test_data_base)
        test_preds_2_log = model_2.predict(test_data_ref)
        test_preds_2 = np.expm1(test_preds_2_log)
        
        ensemble_test_preds = (test_preds_1 + test_preds_2) / 2.0
        
        submission_df = test_data[['street_name', 'violation_type']].copy()
        
        submission_df['predicted_count'] = np.maximum(0, ensemble_test_preds) # Clip at 0
        submission_df['predicted_count'] = submission_df['predicted_count'].round()

        submission_df.to_csv("submission.csv", index=False)
        print("Generated submission.csv")
        
        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, submission_df['predicted_count']))
            print(f"Test Set RMSE: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
