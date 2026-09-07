
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import subprocess
import sys

# Install xgboost if not already installed
try:
    import xgboost as xgb
except ImportError:
    print("xgboost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    print("xgboost installed successfully.")


# --- Configuration ---
# Data paths for training and testing data.
# TRAIN_DATA_DIR points to the historical data.
# TEST_DATA_DIR is a placeholder that will be replaced by the evaluation system
# with the path to the future terms data.
TRAIN_DATA_DIR = "input/table_splits/train"
TEST_DATA_DIR = "input/table_splits/test" # Placeholder for final evaluation

# --- Data Loading ---
def load_data(data_dir, is_train=True):
    """
    Loads and merges core data tables from the specified directory.
    """
    df_summaries = pd.read_csv(f"{data_dir}/subject_summaries.csv")
    df_catalog = pd.read_csv(f"{data_dir}/course_catalog.csv")
    df_terms = pd.read_csv(f"{data_dir}/terms.csv")
    df_enrollment = pd.read_csv(f"{data_dir}/enrollment_data.csv")

    # Merge subject summaries with course catalog for course details
    df = pd.merge(df_summaries, df_catalog, on=['SUBJECT_CODE', 'COURSE_NUMBER'], how='left', suffixes=('_summary', '_catalog'))
    
    # Merge with terms data for term-specific information and chronological sorting
    df = pd.merge(df, df_terms, on='TERM_CODE', how='left')

    # Aggregate enrollment data from 'enrollment_data.csv' to get total enrollment
    # per course offering (TERM_CODE, SUBJECT_ID_SORT)
    agg_enrollment = df_enrollment.groupby(['TERM_CODE', 'SUBJECT_ID_SORT'])['ENROLLMENT_COUNT'].sum().reset_index()
    agg_enrollment.rename(columns={'ENROLLMENT_COUNT': 'ACTUAL_ENROLLMENT'}, inplace=True)
    
    # Merge the aggregated actual enrollment back into the main DataFrame.
    # This 'ACTUAL_ENROLLMENT' will be used to derive lagged features, not as a direct feature itself.
    df = pd.merge(df, agg_enrollment, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if is_train:
        # For training data, load the gold labels and merge them
        df_gold = pd.read_csv("input/eval/gold_enrollment_train.csv")
        df = pd.merge(df, df_gold, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    return df

# --- Feature Engineering ---
def engineer_features(df):
    """
    Creates new features from the raw data and prepares existing ones.
    Includes time-based features and lagged enrollment.
    """
    df_copy = df.copy()

    # Convert TERM_BEGIN_DATE to datetime objects for reliable chronological sorting and feature extraction
    df_copy['TERM_BEGIN_DATE'] = pd.to_datetime(df_copy['TERM_BEGIN_DATE'], errors='coerce')
    
    # Sort data chronologically by course offering for proper lagged feature calculation
    df_copy = df_copy.sort_values(by=['SUBJECT_ID_SORT', 'TERM_BEGIN_DATE'])

    # Time-based features from TERM_BEGIN_DATE
    df_copy['TERM_YEAR'] = df_copy['TERM_BEGIN_DATE'].dt.year
    df_copy['TERM_MONTH'] = df_copy['TERM_BEGIN_DATE'].dt.month
    df_copy['TERM_DAY_OF_YEAR'] = df_copy['TERM_BEGIN_DATE'].dt.dayofyear
    
    # Course level feature: Extracting the first digit of COURSE_NUMBER (e.g., '100' for 1xxx-level, '200' for 2xxx-level)
    # Handle non-numeric course numbers by converting to numeric first
    df_copy['COURSE_NUMBER_NUM'] = pd.to_numeric(df_copy['COURSE_NUMBER'], errors='coerce')
    # Integer division by 1000 to get the first digit (e.g., 199 -> 0, 1000 -> 1, 4999 -> 4)
    df_copy['COURSE_LEVEL'] = (df_copy['COURSE_NUMBER_NUM'] // 1000).astype('Int64') 
    
    # Lagged enrollment features:
    # 1. Previous term's actual enrollment for the same course (SUBJECT_ID_SORT)
    df_copy['PREV_TERM_ENROLLMENT'] = df_copy.groupby('SUBJECT_ID_SORT')['ACTUAL_ENROLLMENT'].shift(1)
    
    # 2. Average enrollment for the department (SUBJECT_CODE) in all previous terms.
    # Using expanding().mean() after shifting ensures no look-ahead bias.
    # The shift(1) ensures we're only using enrollment data from *before* the current term.
    # The expanding().mean() then takes the cumulative average of those shifted values.
    df_copy['DEPT_PREV_TERM_AVG_ENROLLMENT'] = df_copy.groupby('SUBJECT_CODE')['ACTUAL_ENROLLMENT'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Fill NaN values created by shifts (e.g., first term for a course/department, or missing historical data)
    # Using 0 for previous term enrollment if no history, and overall mean for department if no prior departmental data.
    df_copy['PREV_TERM_ENROLLMENT'] = df_copy['PREV_TERM_ENROLLMENT'].fillna(0) 
    df_copy['DEPT_PREV_TERM_AVG_ENROLLMENT'] = df_copy['DEPT_PREV_TERM_AVG_ENROLLMENT'].fillna(
        df_copy['DEPT_PREV_TERM_AVG_ENROLLMENT'].mean()
    )

    return df_copy

# --- Main Script Execution ---
if __name__ == "__main__":
    # Load and engineer features for the training data
    print("Loading training data...")
    train_df = load_data(TRAIN_DATA_DIR, is_train=True)
    print("Engineering features for training data...")
    train_df_fe = engineer_features(train_df)

    # Define the features to be used in the model
    # 'ACTUAL_ENROLLMENT' is not included as it represents enrollment for the current term and would be leakage.
    features = [
        'CREDITS', 
        'MAX_ENROLL',         # Maximum enrollment capacity
        'TERM_YEAR', 
        'TERM_MONTH', 
        'TERM_DAY_OF_YEAR',
        'COURSE_LEVEL',
        'PREV_TERM_ENROLLMENT',         # Lagged enrollment for the same course
        'DEPT_PREV_TERM_AVG_ENROLLMENT',# Lagged average enrollment for the department
        'SUBJECT_CODE',       # Categorical: Department code
        'COURSE_NUMBER',      # Categorical: Full course number (e.g., '101', '520')
        'INSTRUCTOR_ID'       # Categorical: Identifier for the instructor
    ]
    
    # Filter out rows where the target variable 'HIGH_ENROLLMENT' is missing (should only be in test set, but good practice)
    train_df_fe_labeled = train_df_fe.dropna(subset=['HIGH_ENROLLMENT'])

    # Time-based validation split:
    # Sort the entire labeled training data chronologically by TERM_BEGIN_DATE
    train_df_fe_labeled = train_df_fe_labeled.sort_values(by='TERM_BEGIN_DATE').reset_index(drop=True)
    
    # Use the last 20% of the chronologically sorted data as the validation set.
    # This simulates predicting on future terms using past terms for training.
    split_row_idx = int(len(train_df_fe_labeled) * 0.8)

    X_train = train_df_fe_labeled.iloc[:split_row_idx][features]
    y_train = train_df_fe_labeled.iloc[:split_row_idx]['HIGH_ENROLLMENT']
    X_val = train_df_fe_labeled.iloc[split_row_idx:][features]
    y_val = train_df_fe_labeled.iloc[split_row_idx:]['HIGH_ENROLLMENT']

    print(f"Training set size: {len(X_train)} samples, Validation set size: {len(X_val)} samples")
    print(f"Training terms: from {X_train['TERM_YEAR'].min()} to {X_train['TERM_YEAR'].max()}")
    print(f"Validation terms: from {X_val['TERM_YEAR'].min()} to {X_val['TERM_YEAR'].max()}")

    # --- Preprocessing Pipeline ---
    # Define numerical and categorical features for separate processing
    numerical_features = ['CREDITS', 'MAX_ENROLL', 'TERM_YEAR', 'TERM_MONTH', 'TERM_DAY_OF_YEAR', 
                          'COURSE_LEVEL', 'PREV_TERM_ENROLLMENT', 'DEPT_PREV_TERM_AVG_ENROLLMENT']
    categorical_features = ['SUBJECT_CODE', 'COURSE_NUMBER', 'INSTRUCTOR_ID']
    
    # Numerical transformer: Impute missing values with the mean, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer: Impute missing values with 'missing' constant, then one-hot encode
    # `handle_unknown='ignore'` prevents errors if new categories appear in the test set.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not explicitly specified in features lists
    )

    # --- Model Training ---
    # Create the full pipeline: preprocessing followed by XGBoost Classifier
    # 'use_label_encoder=False' suppresses a deprecation warning in newer XGBoost versions.
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', xgb.XGBClassifier(objective='binary:logistic', 
                                                                       eval_metric='logloss', 
                                                                       use_label_encoder=False, 
                                                                       random_state=42, 
                                                                       n_estimators=500, 
                                                                       learning_rate=0.05, 
                                                                       max_depth=6))])
    
    print("Training XGBoost model...")
    model_pipeline.fit(X_train, y_train)

    # --- Evaluation on Validation Set ---
    print("Evaluating on validation set...")
    y_pred_val = model_pipeline.predict(X_val)
    validation_f1_macro = f1_score(y_val, y_pred_val, average='macro')
    
    # Print the final validation performance
    print(f"Final Validation Performance: {validation_f1_macro}")

    # --- Prediction on Test Data (for completeness, though only validation score is requested) ---
    # Load and engineer features for the held-out test data
    print("Loading and engineering features for test data...")
    test_df = load_data(TEST_DATA_DIR, is_train=False)
    test_df_fe = engineer_features(test_df)

    # Ensure the test features match the training features order and names
    X_test = test_df_fe[features]

    # Make predictions on the test set
    # y_pred_test = model_pipeline.predict(X_test)
    # y_pred_test_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # The task only requires printing the validation metric, so no further test-specific output is needed.
