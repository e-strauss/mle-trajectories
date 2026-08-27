
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Define paths
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")

# --- Helper Functions ---

def load_data(directory):
    """Loads all CSV files from a given directory into a dictionary of DataFrames."""
    data = {}
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return data
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            table_name = filename.replace(".csv", "")
            filepath = os.path.join(directory, filename)
            try:
                data[table_name] = pd.read_csv(filepath)
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return data

def preprocess_and_feature_engineer(gold_df, raw_data_dict):
    """
    Preprocesses raw data and engineers features.
    Merges relevant tables, calculates enrollment stats, and creates new features.
    """
    df = gold_df.copy()

    # Create a 'DEPARTMENT_CODE' by extracting alphabetic characters from 'SUBJECT_ID_SORT'
    # This assumes 'SUBJECT_ID_SORT' is like 'CS101', 'MATH200', etc.
    df['DEPARTMENT_CODE'] = df['SUBJECT_ID_SORT'].astype(str).apply(lambda x: ''.join(filter(str.isalpha, x)).upper())
    # Fill any empty department codes with 'UNKNOWN'
    df['DEPARTMENT_CODE'] = df['DEPARTMENT_CODE'].replace('', 'UNKNOWN')

    # --- Feature Engineering from sections.csv (if available) ---
    if 'sections' in raw_data_dict:
        sections_df = raw_data_dict['sections'].copy()
        # Ensure correct column types, coerce errors will turn non-numeric into NaN
        for col in ['TERM_CODE', 'ENROLLMENT', 'MAX_ENROLLMENT']:
            if col in sections_df.columns:
                sections_df[col] = pd.to_numeric(sections_df[col], errors='coerce')
        if 'SUBJECT_ID_SORT' in sections_df.columns:
            sections_df['SUBJECT_ID_SORT'] = sections_df['SUBJECT_ID_SORT'].astype(str)

        # Drop rows where essential merge keys are missing
        sections_df.dropna(subset=['TERM_CODE', 'SUBJECT_ID_SORT'], inplace=True)

        # Handle potential missing ENROLLMENT values
        sections_df['ENROLLMENT'] = sections_df['ENROLLMENT'].fillna(0)
        sections_df['MAX_ENROLLMENT'] = sections_df['MAX_ENROLLMENT'].fillna(0)

        # Aggregate section data to course offering level (TERM_CODE, SUBJECT_ID_SORT)
        course_enrollment_stats = sections_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            num_sections=('SECTION_NUMBER', 'nunique'),
            total_enrollment=('ENROLLMENT', 'sum'),
            avg_section_enrollment=('ENROLLMENT', 'mean'),
            max_section_enrollment=('ENROLLMENT', 'max'),
            total_max_enrollment=('MAX_ENROLLMENT', 'sum')
        ).reset_index()

        df = pd.merge(df, course_enrollment_stats, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

        # Calculate enrollment ratio
        df['enrollment_ratio'] = df['total_enrollment'] / df['total_max_enrollment']
        df['enrollment_ratio'] = df['enrollment_ratio'].replace([np.inf, -np.inf], np.nan) # Handle division by zero
        df['enrollment_ratio'] = df['enrollment_ratio'].fillna(0) # If max_enrollment is 0 or NaN, ratio is 0


    # --- Feature Engineering from courses.csv (if available) ---
    if 'courses' in raw_data_dict:
        courses_df = raw_data_dict['courses'].copy()
        # Ensure correct column types for merging
        for col in ['TERM_CODE', 'CREDIT_HOURS']:
            if col in courses_df.columns:
                courses_df[col] = pd.to_numeric(courses_df[col], errors='coerce')
        if 'SUBJECT_ID_SORT' in courses_df.columns:
            courses_df['SUBJECT_ID_SORT'] = courses_df['SUBJECT_ID_SORT'].astype(str)

        # Drop rows where essential merge keys are missing
        courses_df.dropna(subset=['TERM_CODE', 'SUBJECT_ID_SORT'], inplace=True)

        # Extract course level from SUBJECT_ID_SORT or COURSE_NUMBER if available
        def extract_course_level(s):
            if pd.isna(s): return np.nan
            s = str(s)
            numbers = ''.join(filter(str.isdigit, s))
            if numbers:
                level = int(numbers) // 100
                return min(level, 4) # Cap at typical 4-year undergraduate level, adjust as needed
            return np.nan

        if 'COURSE_NUMBER' in courses_df.columns:
            courses_df['COURSE_LEVEL'] = courses_df['COURSE_NUMBER'].astype(str).apply(extract_course_level)
        else: # Fallback to SUBJECT_ID_SORT if COURSE_NUMBER not available
            courses_df['COURSE_LEVEL'] = courses_df['SUBJECT_ID_SORT'].apply(extract_course_level)

        # Select relevant columns from courses_df to merge
        courses_cols_to_merge = ['TERM_CODE', 'SUBJECT_ID_SORT']
        if 'CREDIT_HOURS' in courses_df.columns:
            courses_cols_to_merge.append('CREDIT_HOURS')
        if 'COURSE_LEVEL' in courses_df.columns:
            courses_cols_to_merge.append('COURSE_LEVEL')
        # Ensure only unique 'TERM_CODE', 'SUBJECT_ID_SORT' combinations are merged
        courses_df_unique = courses_df[courses_cols_to_merge].drop_duplicates(subset=['TERM_CODE', 'SUBJECT_ID_SORT'])

        df = pd.merge(df, courses_df_unique, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')


    # --- General Features ---
    # Convert TERM_CODE to numerical features (e.g., year)
    # Assuming TERM_CODE is like YYYYMM, e.g., 202310 for Fall 2023
    df['TERM_YEAR'] = df['TERM_CODE'] // 100
    df['TERM_SEASON'] = df['TERM_CODE'] % 100 # e.g., 10 for Fall, 20 for Spring, 30 for Summer

    # Fill NaN values for numerical features before scaling
    numerical_cols = [
        'num_sections', 'total_enrollment', 'avg_section_enrollment',
        'max_section_enrollment', 'total_max_enrollment', 'enrollment_ratio',
        'CREDIT_HOURS', 'COURSE_LEVEL', 'TERM_YEAR', 'TERM_SEASON'
    ]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median()) # Using median for robustness
            if df[col].isnull().any(): # If median is also NaN (e.g., all values are NaN)
                df[col] = df[col].fillna(0) # Fallback to 0


    # --- Categorical Feature Encoding ---
    # One-hot encode DEPARTMENT_CODE and TERM_SEASON
    categorical_cols = ['DEPARTMENT_CODE']
    # If TERM_SEASON is treated as categorical (e.g., Spring vs Fall are distinct categories)
    # rather than a continuous number
    if 'TERM_SEASON' in df.columns and len(df['TERM_SEASON'].unique()) > 1:
        categorical_cols.append('TERM_SEASON')

    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)


    # Handle the target variable
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    return df

def train_validate_model(data):
    """Trains a RandomForestClassifier and evaluates it using a time-based validation split."""

    # Sort by TERM_CODE for time-based split
    data = data.sort_values(by='TERM_CODE').reset_index(drop=True)

    # Define features (X) and target (y)
    # Exclude identifier columns and the original target
    X = data.drop(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'], errors='ignore')
    y = data['HIGH_ENROLLMENT']

    # Identify and scale numerical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # Exclude dummy variables if they were created from numerical columns like TERM_SEASON
    numerical_features = [
        f for f in numerical_features
        if not f.startswith('DEPARTMENT_CODE_') and not f.startswith('TERM_SEASON_')
    ]

    scaler = StandardScaler()
    if numerical_features:
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Time-based split: Use the latest terms for validation
    unique_terms = sorted(data['TERM_CODE'].unique())
    if len(unique_terms) < 2: # Need at least 2 terms for any split
        raise ValueError("Not enough unique terms for any train-validation split.")
    elif len(unique_terms) < 5: # Not enough terms for a robust time split, use one latest term
        print("Warning: Few unique terms. Using the single latest term for validation.")
        validation_terms = [unique_terms[-1]]
    else:
        # Use the latest 20% of terms for validation
        split_term_idx = int(len(unique_terms) * 0.8)
        validation_terms = unique_terms[split_term_idx:]

    train_indices = data[~data['TERM_CODE'].isin(validation_terms)].index
    val_indices = data[data['TERM_CODE'].isin(validation_terms)].index

    X_train, X_val = X.loc[train_indices], X.loc[val_indices]
    y_train, y_val = y.loc[train_indices], y.loc[val_indices]

    if X_train.empty:
        raise ValueError("Time-based split resulted in an empty training set. Adjust split logic or data.")
    if X_val.empty:
        # If validation set is empty, it means the latest terms might not have enough data.
        # Fallback to a different strategy, e.g., using a smaller percentage or checking earlier terms.
        # For simplicity here, we will raise an error, but a more robust system might re-split.
        raise ValueError("Time-based split resulted in an empty validation set. "
                         "Consider if there's enough data for the chosen validation terms or adjust the split.")

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    # Ensure all columns in X_train and X_val are numeric. Drop any remaining non-numeric.
    X_train = X_train.select_dtypes(include=np.number)
    X_val = X_val.select_dtypes(include=np.number)

    # Align columns in train and validation sets, in case one-hot encoding created different columns
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)

    missing_in_val = list(train_cols - val_cols)
    for col in missing_in_val:
        X_val[col] = 0

    missing_in_train = list(val_cols - train_cols)
    for col in missing_in_train:
        X_train[col] = 0

    X_val = X_val[X_train.columns] # Ensure column order is the same

    # Initialize and train the model
    # Adjust class_weight for imbalanced datasets if necessary
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Evaluate the model
    f1_macro = f1_score(y_val, y_pred, average='macro')
    print(f"Validation Macro F1 Score: {f1_macro:.4f}")

    return model, f1_macro

# --- Main execution block ---
if __name__ == "__main__":
    try:
        # 1. Load gold labels
        if not os.path.exists(GOLD_ENROLLMENT_TRAIN_PATH):
            raise FileNotFoundError(f"Gold enrollment labels file not found at {GOLD_ENROLLMENT_TRAIN_PATH}")
        gold_enrollment_df = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        print("Gold enrollment labels loaded.")
        if gold_enrollment_df.empty:
            raise ValueError("Gold enrollment labels file is empty.")

        # 2. Load raw data from TRAIN_DATA_DIR
        raw_data = load_data(TRAIN_DATA_DIR)
        if not raw_data:
            print("No raw data tables were loaded. Proceeding with gold labels only if possible or raising error.")


        # 3. Preprocess and Feature Engineer
        # Ensure that 'sections' and 'courses' are actually present if we rely on them
        if 'sections' not in raw_data:
            print("Warning: 'sections.csv' not found. Enrollment statistics (num_sections, total_enrollment, etc.) will be limited or imputed.")
        if 'courses' not in raw_data:
            print("Warning: 'courses.csv' not found. Course details (credit hours, level) will be limited or imputed.")

        processed_df = preprocess_and_feature_engineer(gold_enrollment_df, raw_data)
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Processed data columns: {processed_df.columns.tolist()}")

        # Final check for NaN values in features before training
        # This is a robust final fill to ensure no NaNs go into the model.
        # It's after feature engineering because some NaNs might be intentional or part of feature creation
        # (e.g., if a course has no sections, num_sections might be NaN then filled with median/0).
        for col in processed_df.columns:
            if processed_df[col].dtype == object or processed_df[col].dtype == bool:
                # One-hot encoded columns are typically numeric (0/1). Categorical object columns should have been handled.
                # If any object columns remain, they could cause issues with model training.
                # For this problem, we expect all features to be numeric after processing.
                pass # Already handled categorical via get_dummies
            elif processed_df[col].isnull().any():
                if processed_df[col].dtype == np.number:
                    processed_df[col] = processed_df[col].fillna(0) # Default fill for any remaining numeric NaNs
                else:
                    # This case should ideally not be reached if all feature engineering is correct.
                    # It implies an unexpected non-numeric NaN column.
                    processed_df[col] = processed_df[col].fillna('MISSING')


        # Check if the dataframe for training is not empty and has the target
        if processed_df.empty or 'HIGH_ENROLLMENT' not in processed_df.columns or not processed_df['HIGH_ENROLLMENT'].nunique() > 1:
            raise ValueError("Processed DataFrame is empty, missing 'HIGH_ENROLLMENT' target column, or target has only one unique value. Cannot proceed with training.")
        if processed_df.drop(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'], errors='ignore').empty:
             raise ValueError("Processed DataFrame has no features for training after dropping identifiers and target.")

        # 4. Train and Validate Model
        trained_model, final_validation_score = train_validate_model(processed_df)

        # Print the required performance line
        print(f'Final Validation Performance: {final_validation_score}')

        # The trained_model can be saved here for later use with test data

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data is in the '{TRAIN_DATA_DIR}' directory and specified filenames are correct.")
    except ValueError as e:
        print(f"Data Processing or Model Training Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

