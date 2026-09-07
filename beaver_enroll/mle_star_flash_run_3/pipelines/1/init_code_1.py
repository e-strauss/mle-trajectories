
import pandas as pd
import numpy as np
import os
import sys
import subprocess

# Install lightgbm if not present
try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
        import lightgbm as lgb
        print("lightgbm installed successfully.")
    except Exception as e:
        print(f"Failed to install lightgbm: {e}")
        # If installation fails, re-raise or handle appropriately.
        # For this context, we will assume a successful installation is critical.
        raise

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --- Configuration ---
# All input data is stored in "./input" directory.
TRAIN_DATA_DIR = "./input/table_splits/train"
TEST_DATA_DIR = "./input/test"  # Replaced __TEST_DATA_DIR__ with actual path
GOLD_LABELS_PATH = "./input/eval/gold_enrollment_train.csv"

# --- Data Loading and Preprocessing ---
def load_data_from_dir(data_dir):
    """Loads and merges relevant data files from a given directory."""
    print(f"Loading data from: {data_dir}")
    summary_path = os.path.join(data_dir, 'summary.csv')
    course_features_path = os.path.join(data_dir, 'course_features.csv')
    instructor_features_path = os.path.join(data_dir, 'instructor_features.csv')
    cross_listings_path = os.path.join(data_dir, 'cross_listings.csv')

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Required file not found: {summary_path}")

    df_summary = pd.read_csv(summary_path)
    df = df_summary.copy()

    # Heuristic for merging course features:
    # Assuming `SUBJECT_ID_SORT` in summary.csv might be offering-specific (e.g., 'CS-101-001')
    # and `course_features.csv` contains features for the general course (e.g., 'CS-101').
    # We attempt to derive a common `COURSE_IDENTIFIER_FOR_MERGE`.
    df['COURSE_IDENTIFIER_FOR_MERGE'] = df['SUBJECT_ID_SORT'].apply(
        lambda x: x.rsplit('-', 1)[0] if isinstance(x, str) and '-' in x and x.count('-') > 1 else x
    )

    if os.path.exists(course_features_path):
        df_course_features = pd.read_csv(course_features_path)
        # Assuming course_features has a column that serves as a course identifier
        # Try to find a common identifier, prioritizing 'SUBJECT_ID_SORT' if it matches the derived format
        course_features_merge_key = None
        if 'SUBJECT_ID_SORT' in df_course_features.columns:
            # Check if `SUBJECT_ID_SORT` in course features looks like a course code (e.g., 'CS-101')
            sample_val = df_course_features['SUBJECT_ID_SORT'].dropna().iloc[0] if not df_course_features['SUBJECT_ID_SORT'].dropna().empty else ''
            if isinstance(sample_val, str) and sample_val.count('-') <= 1: # Assumes course code has 0 or 1 dash (e.g., 'MATH101' or 'CS-101')
                course_features_merge_key = 'SUBJECT_ID_SORT'
        elif 'COURSE_CODE' in df_course_features.columns:
            course_features_merge_key = 'COURSE_CODE'
        elif 'COURSE_ID' in df_course_features.columns:
            course_features_merge_key = 'COURSE_ID'

        if course_features_merge_key:
            df_course_features_renamed = df_course_features.rename(
                columns={course_features_merge_key: 'COURSE_IDENTIFIER_FOR_MERGE'}
            )
            df = pd.merge(df, df_course_features_renamed, on='COURSE_IDENTIFIER_FOR_MERGE', how='left', suffixes=('', '_course'))
        else:
            print("Warning: Could not determine a clear merge key for course_features. Skipping merge.")
    
    df.drop(columns=['COURSE_IDENTIFIER_FOR_MERGE'], errors='ignore', inplace=True) # Drop the temporary merge column


    # Merge instructor features
    if os.path.exists(instructor_features_path):
        df_instructor_features = pd.read_csv(instructor_features_path)
        # Assuming instructor features are per-term and per-instructor
        # Need INSTRUCTOR_ID in summary.csv to merge
        if 'INSTRUCTOR_ID' in df.columns:
            df = pd.merge(df, df_instructor_features, on=['TERM_CODE', 'INSTRUCTOR_ID'], how='left', suffixes=('', '_instructor'))
        else:
            print("Warning: 'INSTRUCTOR_ID' not found in summary.csv, skipping instructor features merge.")
    else:
        print(f"Warning: Instructor features file not found at {instructor_features_path}. Skipping.")


    # Merge cross-listings
    if os.path.exists(cross_listings_path):
        df_cross_listings = pd.read_csv(cross_listings_path)
        # For simplicity, count cross listings per (TERM_CODE, SUBJECT_ID_SORT)
        cross_list_counts = df_cross_listings.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).size().reset_index(name='NUM_CROSS_LISTINGS')
        df = pd.merge(df, cross_list_counts, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        df['NUM_CROSS_LISTINGS'] = df['NUM_CROSS_LISTINGS'].fillna(0).astype(int)
        df['IS_CROSS_LISTED'] = (df['NUM_CROSS_LISTINGS'] > 0).astype(int)
    else:
        print(f"Warning: Cross listings file not found at {cross_listings_path}. Skipping.")
        df['NUM_CROSS_LISTINGS'] = 0
        df['IS_CROSS_LISTED'] = 0

    # Feature Engineering (basic examples)
    df['TERM_YEAR'] = df['TERM_CODE'] // 100
    df['TERM_SEASON'] = df['TERM_CODE'] % 100

    # Fill NaNs for numerical features with a sentinel value or mean/median
    # For categorical features, fill with 'missing'
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1) # Use -1 as a sentinel for numerical NaNs

    return df

def get_features_and_target(df, gold_labels=None, is_training=True):
    """Prepares features and target for the model."""
    
    if is_training:
        df = pd.merge(df, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        target_column = 'HIGH_ENROLLMENT'
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found after merging gold labels.")
    else:
        target_column = None # No target for test data prediction

    # Identify potential features. Avoid using IDs and the target directly.
    # Exclude identifiers that are not features or are used for merging/keys
    exclude_cols = [
        'TERM_CODE', 
        'SUBJECT_ID_SORT', 
        'INSTRUCTOR_ID', # High cardinality, often better to exclude unless target encoded
        'HIGH_ENROLLMENT' # Target column itself
    ]
    
    # Dynamically find columns to exclude based on availability
    actual_exclude_cols = [col for col in exclude_cols if col in df.columns]
    
    features = [col for col in df.columns if col not in actual_exclude_cols]

    # Identify and convert categorical features for LightGBM
    categorical_features = []
    for col in features:
        if df[col].dtype == 'object' or (df[col].nunique() < df.shape[0] * 0.1 and df[col].nunique() > 1 and df[col].dtype not in ['int64', 'float64']):
            # Convert object types to category type for LightGBM
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Ensure numerical columns are numeric, coercing if necessary
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1) # Fill NaNs after coercion

    if is_training:
        X = df[features]
        y = df[target_column]
        return X, y, categorical_features, df[['TERM_CODE', 'SUBJECT_ID_SORT']]
    else:
        X = df[features]
        return X, categorical_features, df[['TERM_CODE', 'SUBJECT_ID_SORT']]


# --- Main Script ---
if __name__ == "__main__":
    print("Starting enrollment prediction script...")

    # Load gold labels
    gold_labels = pd.read_csv(GOLD_LABELS_PATH)
    print(f"Loaded gold labels: {gold_labels.shape[0]} rows.")

    # Load and preprocess training data
    train_df_raw = load_data_from_dir(TRAIN_DATA_DIR)
    
    # Prepare features and target for training and validation
    X_full, y_full, categorical_features, train_keys = get_features_and_target(train_df_raw, gold_labels, is_training=True)

    print(f"Full training data shape: {X_full.shape}")
    print(f"Number of features: {len(X_full.columns)}")
    print(f"Categorical features identified: {categorical_features}")

    # Time-based validation split: Use the latest terms for validation
    # Sort data by TERM_CODE to ensure chronological split
    train_data_for_split = X_full.copy()
    train_data_for_split['TERM_CODE'] = train_keys['TERM_CODE']
    train_data_for_split['HIGH_ENROLLMENT'] = y_full

    train_data_for_split = train_data_for_split.sort_values(by='TERM_CODE').reset_index(drop=True)
    
    # Determine unique terms and split point
    unique_terms = train_data_for_split['TERM_CODE'].unique()
    unique_terms.sort() # Ensure terms are sorted chronologically

    if len(unique_terms) < 2:
        print("Warning: Not enough unique terms for a time-based train-validation split. Training on full data, no validation set created.")
        X_train = X_full.copy()
        y_train_train = y_full.copy()
        X_val = pd.DataFrame() # Empty validation set
        y_train_val = pd.Series() # Empty validation target
        categorical_features_in_X_train = [col for col in categorical_features if col in X_train.columns]
    else:
        # Use the latest 20% of unique terms for validation, ensure at least one term but not all
        validation_terms_count = max(1, min(int(len(unique_terms) * 0.2), len(unique_terms) - 1))
        validation_terms = unique_terms[-validation_terms_count:]

        X_val_data = train_data_for_split[train_data_for_split['TERM_CODE'].isin(validation_terms)]
        X_train_data = train_data_for_split[~train_data_for_split['TERM_CODE'].isin(validation_terms)]

        y_train_val = X_val_data['HIGH_ENROLLMENT']
        y_train_train = X_train_data['HIGH_ENROLLMENT']

        # Drop the temporary TERM_CODE and target column from feature sets
        X_val = X_val_data.drop(columns=['TERM_CODE', 'HIGH_ENROLLMENT'])
        X_train = X_train_data.drop(columns=['TERM_CODE', 'HIGH_ENROLLMENT'])

        # Ensure categorical features are actually present in the split datasets before passing to LightGBM
        categorical_features_in_X_train = [col for col in categorical_features if col in X_train.columns]

        print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
        print(f"Number of validation terms used: {len(validation_terms)}, latest terms: {validation_terms}")

    # --- Model Training ---
    print("Training LightGBM model...")
    lgbm = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31)
    
    fit_params = {
        'categorical_feature': categorical_features_in_X_train,
    }

    if not X_val.empty and not y_train_val.empty:
        fit_params['eval_set'] = [(X_val, y_train_val)]
        fit_params['eval_metric'] = 'binary_f1'
        fit_params['callbacks'] = [lgb.early_stopping(100, verbose=False)]
    else:
        print("No valid validation set available, training without early stopping on eval_set.")
        # Ensure eval_metric and callbacks are not passed if no eval_set
        fit_params.pop('eval_metric', None)
        fit_params.pop('callbacks', None)

    lgbm.fit(X_train, y_train_train, **fit_params)

    # --- Evaluation ---
    print("Evaluating on validation set...")
    if not X_val.empty and not y_train_val.empty:
        y_pred_val = lgbm.predict(X_val)
        macro_f1 = f1_score(y_train_val, y_pred_val, average='macro')
        print(f"Final Validation Performance: {macro_f1}")
    else:
        # As per requirement, we must print 'Final Validation Performance: {final_validation_score}'
        # If no validation set, report 0.0 or N/A
        macro_f1 = 0.0 # Default score if no validation is performed.
        print(f"Final Validation Performance: {macro_f1}")
    
    # The task asks to report macro F1 on a time-based validation slice within train.
    # Prediction on the test split is for later evaluation, not required for this task.
    # The code below is commented out as it's not strictly part of the current debugging/reporting task.
    
    # # --- Prediction on Test Data (Optional for this task, but common next step) ---
    # print("\nLoading and predicting on test data...")
    # test_df_raw = load_data_from_dir(TEST_DATA_DIR)
    # X_test, test_categorical_features, test_keys = get_features_and_target(test_df_raw, is_training=False)
    
    # # Align columns between train and test datasets
    # missing_cols_in_test = set(X_train.columns) - set(X_test.columns)
    # for c in missing_cols_in_test:
    #     X_test[c] = -1 # Fill missing columns with a default value (e.g., -1 for numerical, 'missing' for categorical)
    # # Ensure the order of columns in X_test matches X_train
    # X_test = X_test[X_train.columns] 

    # y_pred_test = lgbm.predict(X_test)
    
    # # Create submission DataFrame
    # results_df = pd.DataFrame({
    #     'TERM_CODE': test_keys['TERM_CODE'],
    #     'SUBJECT_ID_SORT': test_keys['SUBJECT_ID_SORT'],
    #     'HIGH_ENROLLMENT': y_pred_test
    # })
    # # print(results_df.head())
    # # results_df.to_csv('submission.csv', index=False)
    # print("Test predictions generated. (Not saved to file as per task description)")

print("Script finished.")
