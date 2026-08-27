
# Required libraries will be automatically checked by the system and installed if missing.
# If you encounter ModuleNotFoundError locally, you would typically run:
# pip install pandas scikit-learn numpy

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np # Imported for potential NaN handling and general numeric operations

# Configuration
# All input data is stored in "./input" directory.
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_FILE = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

# --- Data Loading ---
print("Loading data...")

try:
    gold_enrollment_df = pd.read_csv(GOLD_ENROLLMENT_FILE)
    print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_df)} rows.")

    # Identify potential feature files in TRAIN_DATA_DIR
    # These are assumed file names based on common academic data structures.
    # The error handling below will gracefully skip files that don't exist.
    
    # Store loaded dataframes in a dictionary for easier merging later
    additional_dfs = {}
    
    # List of expected feature files and their keys for merging.
    # 'cols' are columns to select from the auxiliary file, in addition to the 'key' columns.
    feature_files_info = {
        "course_details.csv": {"key": ["TERM_CODE", "SUBJECT_ID_SORT"], "cols": ['COURSE_NUMBER', 'CREDITS', 'COURSE_LEVEL']},
        "subject_details.csv": {"key": ["SUBJECT_ID_SORT"], "cols": ['DEPARTMENT_ID', 'SUBJECT_NAME']}
    }

    for filename, info in feature_files_info.items():
        file_path = os.path.join(TRAIN_DATA_DIR, filename) #
        if os.path.exists(file_path):
            try:
                temp_df = pd.read_csv(file_path)
                additional_dfs[filename] = temp_df
                print(f"Loaded {filename} with {len(temp_df)} rows.")
            except Exception as e:
                print(f"Error loading {filename}: {e}. Skipping this file.")
        else:
            print(f"Warning: {filename} not found in {TRAIN_DATA_DIR}. Skipping this feature source.")

except FileNotFoundError as e:
    print(f"Critical Error: Missing required input file. Ensure {GOLD_ENROLLMENT_FILE} and other data files are present. Error: {e}")
    raise # Re-raise to stop execution if essential files are missing
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    raise

# --- Feature Engineering ---
print("Performing feature engineering...")

df = gold_enrollment_df.copy()

# Add a 'TERM_YEAR' column from TERM_CODE for time-based splitting
# Assuming TERM_CODE is typically YYYYTT (e.g., 202310 for Fall 2023)
try:
    # Convert TERM_CODE to string to handle cases where it might be loaded as numeric
    df['TERM_YEAR'] = df['TERM_CODE'].astype(str).str[:4].astype(int)
    # Also extract TERM_SEASON for potential categorical feature
    df['TERM_SEASON'] = df['TERM_CODE'].astype(str).str[4:]
    print("Extracted TERM_YEAR and TERM_SEASON from TERM_CODE.")
except Exception as e:
    print(f"Warning: Could not extract TERM_YEAR or TERM_SEASON from TERM_CODE. Error: {e}. Proceeding without these features derived this way.")
    df['TERM_YEAR'] = df['TERM_CODE'] # Fallback, might not be usable for chronological split
    df['TERM_SEASON'] = 'UNKNOWN' # Default category


# Merge with additional feature dataframes
for filename, info in feature_files_info.items():
    if filename in additional_dfs:
        merge_key = info["key"]
        
        # Ensure only columns that exist in the auxiliary dataframe are used for merging and selection
        actual_merge_key = [col for col in merge_key if col in additional_dfs[filename].columns]
        actual_cols_to_select = [col for col in (info["cols"] + merge_key) if col in additional_dfs[filename].columns]

        if not actual_merge_key or len(actual_merge_key) != len(merge_key):
            print(f"Warning: Missing one or more intended merge keys {merge_key} in {filename}. Skipping merge for this file.")
            continue
        
        # Select only relevant columns from the auxiliary dataframe and drop duplicates
        temp_df_to_merge = additional_dfs[filename][actual_cols_to_select].drop_duplicates(subset=actual_merge_key)
        
        # Perform a left merge to keep all rows from the gold_enrollment_df
        initial_rows = len(df)
        df = pd.merge(df, temp_df_to_merge, on=actual_merge_key, how='left', suffixes=('', '_drop'))
        # Drop columns created by suffixes if any conflicts (e.g., if gold_enrollment_df had a 'CREDITS' column)
        df.drop(columns=[col for col in df.columns if '_drop' in col], inplace=True)
        print(f"Merged with {filename}. Rows remain: {len(df)}. (Expected: {initial_rows})")

# Identify categorical and numerical features for processing
# Start with known categorical features from the gold file and derived ones
categorical_features = ['SUBJECT_ID_SORT', 'TERM_SEASON']
numerical_features = []

# Populate categorical and numerical features based on remaining columns
for col in df.columns:
    if col not in ['TERM_CODE', 'TERM_YEAR', 'HIGH_ENROLLMENT', 'HIGH_ENROLLMENT_ENCODED', 'SUBJECT_ID_SORT', 'TERM_SEASON']:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_features.append(col)

# Handle potential NaN values and convert to string for categorical features
# Ensure features are only added if they exist in the DataFrame
final_categorical_features = []
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].fillna('UNKNOWN_CATEGORY').astype(str)
        final_categorical_features.append(col)
    else:
        print(f"Warning: Categorical feature '{col}' not found in DataFrame. Skipping.")

# One-hot encode categorical features that are present
if final_categorical_features:
    df = pd.get_dummies(df, columns=final_categorical_features, drop_first=True)
    print(f"DataFrame shape after one-hot encoding: {df.shape}")
else:
    print("No relevant categorical features found or processed for one-hot encoding.")


# Fill numerical NaNs with a sensible default (e.g., 0 or mean)
# For simplicity, using 0 is often safe for count-like or missing numerical data.
final_numerical_features = []
for col in numerical_features:
    if col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0) # Using 0 as a default imputation strategy
            print(f"Filled NaN in numerical column '{col}' with 0.")
        final_numerical_features.append(col)
    else:
        print(f"Warning: Numerical feature '{col}' not found in DataFrame. Skipping.")


# Convert target variable to numerical (Y/N to 1/0)
label_encoder = LabelEncoder()
df['HIGH_ENROLLMENT_ENCODED'] = label_encoder.fit_transform(df['HIGH_ENROLLMENT'])
print("Encoded 'HIGH_ENROLLMENT' target variable.")

# --- Prepare for Modeling ---
# Identify features and target
# Exclude original TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT, and temporary TERM_YEAR from features
features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT', 'HIGH_ENROLLMENT_ENCODED', 'TERM_YEAR']

# Filter out non-feature columns from X
X = df.drop(columns=[col for col in features_to_exclude if col in df.columns], errors='ignore')
y = df['HIGH_ENROLLMENT_ENCODED']

# Handle cases where X might be empty or only contain one column (e.g., only OHE features)
if X.empty:
    raise ValueError("Feature DataFrame X is empty after processing. Check data loading and feature engineering.")
if X.shape[1] == 0:
    raise ValueError("Feature DataFrame X has no columns after processing. Check feature engineering steps.")


# Time-based validation split: latest years for validation
# This is crucial for evaluating performance on future, unseen data.
if 'TERM_YEAR' not in df.columns or df['TERM_YEAR'].nunique() < 2:
    print("Warning: Insufficient unique TERM_YEARs for time-based split or 'TERM_YEAR' column missing or invalid.")
    print("Falling back to a standard stratified train-test split for validation.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    latest_year = df['TERM_YEAR'].max()
    # Define validation as the latest year's data, training as all previous years.
    validation_terms_df = df[df['TERM_YEAR'] == latest_year]
    training_terms_df = df[df['TERM_YEAR'] < latest_year]

    if training_terms_df.empty:
        print("Warning: Training set is empty after chronological split (all data from latest year).")
        print("Falling back to a standard stratified train-test split for validation.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    elif validation_terms_df.empty:
        print("Warning: Validation set is empty after chronological split (no data for the latest year).")
        print("Falling back to a standard stratified train-test split for validation.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Ensure that X_train, X_val only contain feature columns present in X
        # This handles cases where original X might have been filtered, or OHE resulted in different columns
        # (though OHE on full DF then split mitigates this for validation)
        X_train = training_terms_df[X.columns]
        y_train = training_terms_df['HIGH_ENROLLMENT_ENCODED']
        X_val = validation_terms_df[X.columns]
        y_val = validation_terms_df['HIGH_ENROLLMENT_ENCODED']
        print(f"Time-based split: Training on years < {latest_year}, Validating on year {latest_year}.")

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Ensure consistency in feature columns between training and validation sets
# This is important if OHE was applied differently, or if data splits caused missing columns
train_cols = set(X_train.columns)
val_cols = set(X_val.columns)

missing_in_val = list(train_cols - val_cols)
for col in missing_in_val:
    X_val[col] = 0 # Add missing columns to validation set, fill with 0

missing_in_train = list(val_cols - train_cols)
for col in missing_in_train:
    X_train[col] = 0 # Add missing columns to training set, fill with 0

# Reorder columns to ensure they are identical
X_val = X_val[X_train.columns]


# --- Model Training ---
print("Training model...")
# Using RandomForestClassifier as a robust baseline model
# 'class_weight='balanced'' helps handle potential target imbalance by
# adjusting weights inversely proportional to class frequencies.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluation ---
print("Evaluating model...")
y_pred = model.predict(X_val)
# Calculate macro F1 score as per evaluation criteria
final_validation_score = f1_score(y_val, y_pred, average='macro')
print(f"Final Validation Performance: {final_validation_score}")

# The prompt specifically asks to print this line exactly.
print(f"Final Validation Performance: {final_validation_score}")

