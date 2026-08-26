
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import argparse
import os
import sys

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path, is_train=True):
    """Loads and preprocesses the main violation data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
        
    df.rename(columns={
        'Street Name': 'street_name',
        'Violation Description': 'violation_type',
        'violation_count': 'violation_count'
    }, inplace=True)

    if 'street_name' not in df.columns or 'violation_type' not in df.columns:
        print(f"Error: Core columns 'Street Name' or 'Violation Description' not in {file_path}")
        return None

    # For test files, violation_count might be missing, which is fine.
    if is_train and 'violation_count' not in df.columns:
        print(f"Error: Training file {file_path} must contain 'violation_count' column.")
        return None

    return df

def load_augmentation_data(file_path):
    """Loads the CSCL augmentation data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Warning: Augmentation file {file_path} not found. Proceeding without it.")
        return None
    return df

def normalize_street_name(street_name):
    """Standardizes street names for joining."""
    if not isinstance(street_name, str):
        return None
    return street_name.upper().strip()

def main(train_path, test_path=None):
    """Main function to run the training and prediction pipeline."""
    
    # --- 1. Data Loading and Initial Preparation ---
    print("Loading data...")
    train_df = load_and_prepare_data(train_path, is_train=True)
    if train_df is None:
        # Replaced sys.exit(1)
        return

    # Subsample for memory efficiency
    if len(train_df) > 500000:
        print(f"Original training data size: {len(train_df)}. Subsampling to 500,000 rows.")
        train_df = train_df.sample(n=500000, random_state=42)

    # --- 2. Augmentation Data Integration ---
    print("Loading and integrating augmentation data...")
    cscl_path = './input/nyc_cscl.csv'
    cscl_df = load_augmentation_data(cscl_path)
    
    cscl_features = None # Initialize to handle case where file doesn't exist
    if cscl_df is not None:
        train_df['join_street_name'] = train_df['street_name'].apply(normalize_street_name)
        # Corrected the column name from 'FULL_STREE' to 'FULL_ST' which is a common standard in these datasets
        if 'FULL_ST' in cscl_df.columns:
            cscl_df['join_street_name'] = cscl_df['FULL_ST'].apply(normalize_street_name)
            cscl_features = cscl_df[['join_street_name', 'BOROUGH', 'ST_WIDTH', 'TRAFDIR', 'RW_TYPE']].drop_duplicates(subset=['join_street_name'])
            train_df = pd.merge(train_df, cscl_features, on='join_street_name', how='left')
            train_df.drop(columns=['join_street_name'], inplace=True)
            print(f"Successfully merged with CSCL data. {train_df['BOROUGH'].notna().sum() / len(train_df):.2%} of rows matched.")
        else:
            print("Warning: 'FULL_ST' column not found in CSCL data. Skipping CSCL augmentation.")
            cscl_df = None # Set to None to prevent later errors

    # --- 3. Feature Engineering ---
    print("Performing feature engineering...")
    
    # Store median for later use in test set
    st_width_median = train_df['ST_WIDTH'].median() if cscl_df is not None and 'ST_WIDTH' in train_df else 0

    if cscl_df is not None:
        for col in ['BOROUGH', 'TRAFDIR', 'RW_TYPE']:
            train_df[col] = train_df[col].fillna('Unknown')
        train_df['ST_WIDTH'] = train_df['ST_WIDTH'].fillna(st_width_median)
    
    train_df['street_name_len'] = train_df['street_name'].str.len().fillna(0)
    train_df['violation_type_len'] = train_df['violation_type'].str.len().fillna(0)
    train_df['street_violation_interaction'] = train_df['street_name'].astype(str) + "_" + train_df['violation_type'].astype(str)

    # Aggregate features (calculated from the entire training set)
    street_agg = train_df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std']).rename(columns=lambda x: f'street_{x}_viol').reset_index()
    violation_agg = train_df.groupby('violation_type')['violation_count'].agg(['mean', 'sum', 'std']).rename(columns=lambda x: f'viol_type_{x}_viol').reset_index()

    train_df = pd.merge(train_df, street_agg, on='street_name', how='left')
    train_df = pd.merge(train_df, violation_agg, on='violation_type', how='left')
    
    # Store aggregate feature means for test set imputation
    agg_feature_means = {}
    agg_cols_to_process = street_agg.columns.drop('street_name').tolist() + violation_agg.columns.drop('violation_type').tolist()
    for col in agg_cols_to_process:
         agg_feature_means[col] = train_df[col].mean()
         train_df[col].fillna(agg_feature_means[col], inplace=True)
    
    # --- 4. Data Splitting (Validation) ---
    print("Splitting data for validation...")
    
    X = train_df.drop('violation_count', axis=1)
    y = train_df['violation_count']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_inds, val_inds = next(gss.split(X, y, groups=X['street_violation_interaction']))
    
    X_train, X_val = X.iloc[train_inds], X.iloc[val_inds]
    y_train, y_val = y.iloc[train_inds], y.iloc[val_inds]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # --- 5. Modeling Pipeline ---
    print("Building and training the model...")
    
    categorical_features = ['street_name', 'violation_type', 'street_violation_interaction']
    if cscl_df is not None:
        categorical_features.extend(['BOROUGH', 'TRAFDIR', 'RW_TYPE'])
    
    numeric_features = [col for col in X_train.select_dtypes(include=np.number).columns if col not in ['violation_count']]
    
    # Create the preprocessor using ColumnTransformer.
    # This ensures consistent transformations for training and testing.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop', # Drop columns not specified in transformers
        n_jobs=-1
    )

    # Create the full model pipeline
    lgbm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42, n_jobs=-1))
    ])

    # Train the pipeline
    lgbm_pipeline.fit(X_train, y_train)

    # --- 6. Validation Performance ---
    print("Evaluating model on validation set...")
    val_predictions = lgbm_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    
    final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Final Validation Performance: {final_validation_score}")

    # --- 7. Test Set Prediction (if provided) ---
    if test_path:
        print(f"\nProcessing test file: {test_path}")
        test_df = load_and_prepare_data(test_path, is_train=False)
        
        if test_df is not None:
            submission_df = test_df[['street_name', 'violation_type']].copy()
            
            # --- Feature Engineering for Test Set ---
            print("Applying feature engineering to the test set...")
            if cscl_df is not None and cscl_features is not None:
                test_df['join_street_name'] = test_df['street_name'].apply(normalize_street_name)
                test_df = pd.merge(test_df, cscl_features, on='join_street_name', how='left')
                test_df.drop(columns=['join_street_name'], inplace=True)
                for col in ['BOROUGH', 'TRAFDIR', 'RW_TYPE']:
                    test_df[col] = test_df[col].fillna('Unknown')
                test_df['ST_WIDTH'] = test_df['ST_WIDTH'].fillna(st_width_median)

            test_df['street_name_len'] = test_df['street_name'].str.len().fillna(0)
            test_df['violation_type_len'] = test_df['violation_type'].str.len().fillna(0)
            test_df['street_violation_interaction'] = test_df['street_name'].astype(str) + "_" + test_df['violation_type'].astype(str)

            # Merge aggregates from training data
            test_df = pd.merge(test_df, street_agg, on='street_name', how='left')
            test_df = pd.merge(test_df, violation_agg, on='violation_type', how='left')

            unseen_count = test_df['street_mean_viol'].isnull().sum()
            if unseen_count > 0:
                print(f"Handling {unseen_count} rows with unseen street/violation keys.")
            
            # Impute missing aggregate features with the stored global means from training
            for col, mean_val in agg_feature_means.items():
                if col in test_df.columns:
                    test_df[col].fillna(mean_val, inplace=True)

            print("Generating predictions...")
            # Use the fitted pipeline to transform and predict
            test_predictions = lgbm_pipeline.predict(test_df)
            test_predictions[test_predictions < 0] = 0
            
            submission_df['predicted_count'] = test_predictions
            
            if 'violation_count' in test_df.columns:
                test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], test_predictions))
                print(f"RMSE on provided test set: {test_rmse}")

            submission_path = "submission.csv"
            submission_df.to_csv(submission_path, index=False)
            print(f"Submission file saved to {submission_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional: Path to the test data CSV file for prediction.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_path):
        print(f"Error: Training file not found at '{args.train_path}'")
    elif args.test_path and not os.path.exists(args.test_path):
        print(f"Warning: Test file not found at '{args.test_path}'. Skipping prediction.")
        main(args.train_path, None)
    else:
        main(args.train_path, args.test_path)
