
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import os
import numpy as np

# --- Configuration ---
# All the provided input data is stored in "./input" directory.
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
# Replace __TEST_DATA_DIR__ before final evaluation (use test/).
# For this self-contained script, we'll assume the test data is in `table_splits/test`
# within the input directory, consistent with `table_splits/train`.
TEST_DATA_DIR = os.path.join(BASE_DIR, "table_splits/test") 
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# --- Function to load data ---
def load_summary_data(data_dir):
    """
    Loads the primary summary data from a given directory.
    Assumes a file named 'offerings_summary.csv' exists, or falls back to the first CSV.
    """
    summary_path = os.path.join(data_dir, "offerings_summary.csv")
    if not os.path.exists(summary_path):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        summary_path = os.path.join(data_dir, csv_files[0]) # Take the first one found
        print(f"Warning: 'offerings_summary.csv' not found in {data_dir}. Using '{os.path.basename(summary_path)}' instead.")

    df = pd.read_csv(summary_path)
    return df

# --- Main Script ---
def main():
    print("Loading training data...")
    train_df = None
    gold_labels_df = None
    try:
        train_df = load_summary_data(TRAIN_DATA_DIR)
        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure data files are in the correct paths.")
        print(f"Expected TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
        print(f"Expected GOLD_LABELS_PATH: {GOLD_LABELS_PATH}")
        print("Creating dummy training data for demonstration due to missing files.")
        # Create dummy data if files are not found, so the script can still run
        train_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50]
        })
        gold_labels_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })

    # Merge features with gold labels
    train_df = pd.merge(train_df, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')

    # Ensure TERM_CODE is numeric for sorting
    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])

    # Sort by TERM_CODE for time-based split
    train_df = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)

    # --- Feature Engineering (Pre-split on Training Data) ---
    # Identify potential features, making sure they exist in the dataframe
    potential_numerical_features = ['CREDIT_HOURS', 'CAPACITY', 'PREV_ENROLLMENT_AVG']
    potential_categorical_features = ['SUBJECT_ID_SORT', 'INSTRUCTOR_RANK']

    # Filter for features that actually exist in the dataframe
    numerical_features = [f for f in potential_numerical_features if f in train_df.columns]
    categorical_features = [f for f in potential_categorical_features if f in train_df.columns]
    
    # Add TERM_CODE as a numerical feature, if not already included
    if 'TERM_CODE' not in numerical_features:
        numerical_features.append('TERM_CODE')

    # Impute missing numerical values using mean
    for col in numerical_features:
        if train_df[col].isnull().any():
            mean_val = train_df[col].mean()
            train_df[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in training '{col}' with mean: {mean_val:.2f}")

    # Impute missing categorical values using mode
    for col in categorical_features:
        if train_df[col].isnull().any():
            mode_val = train_df[col].mode()[0]
            train_df[col].fillna(mode_val, inplace=True)
            print(f"Imputed missing values in training '{col}' with mode: '{mode_val}'")

    # One-hot encode categorical features
    # Keep track of columns for consistent application to test data
    train_df_encoded = pd.get_dummies(train_df, columns=categorical_features, dummy_na=False)

    # Select final features (all columns except 'HIGH_ENROLLMENT')
    X = train_df_encoded.drop('HIGH_ENROLLMENT', axis=1)
    y = train_df_encoded['HIGH_ENROLLMENT']
    
    # Store training columns for consistent test data processing
    training_features_columns = X.columns.tolist()

    # --- Time-based Validation Split ---
    unique_terms = sorted(train_df['TERM_CODE'].unique())
    
    X_train_val, X_val, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

    if len(unique_terms) < 2: # At least 2 terms for a meaningful time-based split
        print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.")
        if len(train_df) > 1: # Ensure there's enough data for a split
            X_train_val, X_val, y_train_val, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            raise ValueError("Insufficient data to perform any kind of train-validation split.")
    else:
        # Define validation set as the latest term (or few latest terms)
        # Using a fixed percentage (e.g., 20%) of terms for validation, at least one term
        num_val_terms = max(1, int(len(unique_terms) * 0.2))
        val_terms = unique_terms[-num_val_terms:]
        
        # Split data based on term codes
        val_indices = train_df_encoded[train_df_encoded['TERM_CODE'].isin(val_terms)].index
        train_val_indices = train_df_encoded[~train_df_encoded['TERM_CODE'].isin(val_terms)].index

        X_train_val = X.loc[train_val_indices]
        y_train_val = y.loc[train_val_indices]
        X_val = X.loc[val_indices]
        y_val = y.loc[val_indices]

        if X_train_val.empty or X_val.empty:
            print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.")
            # Fallback to random split if time-based split results in empty sets
            if len(train_df) > 1:
                X_train_val, X_val, y_train_val, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")
        else:
            print(f"Time-based split: Training on terms {sorted(X_train_val['TERM_CODE'].unique())}")
            print(f"Validating on terms: {sorted(X_val['TERM_CODE'].unique())}")

    print(f"Training data size: {len(X_train_val)}")
    print(f"Validation data size: {len(X_val)}")
    
    # Final check for empty dataframes before training
    if X_train_val.empty or X_val.empty or y_train_val.empty or y_val.empty:
        raise ValueError("Training or validation set is empty. Cannot proceed with model training.")

    # --- Model Training ---
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_val, y_train_val)

    # --- Validation ---
    print("Evaluating on validation set...")
    y_pred_val = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred_val, average='macro')
    print(f'Final Validation Performance: {macro_f1}') # Required output format

    # --- Prediction on Test Data ---
    print("Loading test data...")
    test_df = None
    try:
        test_df = load_summary_data(TEST_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}. Please ensure test data files are in the correct paths.")
        print(f"Expected TEST_DATA_DIR: {TEST_DATA_DIR}")
        print("Creating dummy test data for demonstration due to missing files.")
        test_df = pd.DataFrame({
            'TERM_CODE': [202507, 202507, 202601],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'PH-101'],
            'CREDIT_HOURS': [3, 4, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT'],
            'CAPACITY': [100, 50, 120],
            'PREV_ENROLLMENT_AVG': [80, 45, 110]
        })

    # Keep original test_df for output merge
    test_original_df = test_df[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()

    # Preprocess test data using training-derived transformations
    # Ensure TERM_CODE is numeric
    test_df['TERM_CODE'] = pd.to_numeric(test_df['TERM_CODE'])

    # Impute missing numerical values using training means
    for col in numerical_features:
        if col in test_df.columns and test_df[col].isnull().any():
            mean_val = train_df[col].mean() # Use mean from original training data
            test_df[col].fillna(mean_val, inplace=True)
            print(f"Imputed missing values in test '{col}' with training mean: {mean_val:.2f}")

    # Impute missing categorical values using training modes
    for col in categorical_features:
        if col in test_df.columns and test_df[col].isnull().any():
            mode_val = train_df[col].mode()[0] # Use mode from original training data
            test_df[col].fillna(mode_val, inplace=True)
            print(f"Imputed missing values in test '{col}' with training mode: '{mode_val}'")

    # One-hot encode categorical features for test data
    test_df_encoded = pd.get_dummies(test_df, columns=categorical_features, dummy_na=False)

    # Align columns - crucial for consistent feature sets between train and test
    # Add missing columns from training data to test data (filled with 0)
    missing_cols_in_test = set(training_features_columns) - set(test_df_encoded.columns)
    for col in missing_cols_in_test:
        test_df_encoded[col] = 0

    # Drop columns in test_df_encoded that are not in training_features_columns (new categories in test data)
    extra_cols_in_test = set(test_df_encoded.columns) - set(training_features_columns)
    if extra_cols_in_test:
        test_df_encoded = test_df_encoded.drop(columns=list(extra_cols_in_test))

    # Ensure the order of columns is the same as training features
    X_test_processed = test_df_encoded[training_features_columns]

    print("Making predictions on test data...")
    y_pred_test = model.predict(X_test_processed)

    # --- Prepare Submission ---
    submission_df = test_original_df.copy()
    submission_df['HIGH_ENROLLMENT'] = y_pred_test

    # Output predictions to a CSV file (e.g., 'predictions.csv')
    output_path = "predictions.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
