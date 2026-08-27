
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    import subprocess
    import sys
    print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# Define paths
# Assuming the script is run from the root directory where 'input' is present.
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")
TEST_DATA_DIR = None # Not available for training phase

print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")

import pandas as pd

# Custom exception for data quality issues
class DataQualityError(Exception):
    """Custom exception for significant data quality issues."""
    pass

# --- 1. Load Gold Labels ---
try:
    gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train)} rows.")
    if gold_enrollment_train.empty:
        raise ValueError("gold_enrollment_train.csv is empty.")
    required_cols = ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']
    if not all(col in gold_enrollment_train.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in gold_enrollment_train.columns]
        raise ValueError(f"gold_enrollment_train.csv missing required columns: {', '.join(missing_cols)}.")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
    # Create a dummy dataframe for development purposes if file is missing or invalid.
    # Added 'BAD_TERM', 'X', 'Z' to simulate problematic data for demonstration.
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', 'BAD_TERM', '202102', '202103'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CH', 'BIO', 'ART'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'X', 'Y', 'Z']
    })
    print("Using dummy gold_enrollment_train data.")

# --- 2. Implement strict data type validation and conversion ---
print("Applying strict data type validation and conversion to gold_enrollment_train.")

original_row_count = len(gold_enrollment_train)
problematic_rows_indices = pd.Index([]) # Collect indices of all rows to be removed
warnings_logged = []

# Define a data quality threshold (e.g., 5% of original rows)
DATA_QUALITY_THRESHOLD_PERCENT = 5
data_quality_max_problematic_rows = int(original_row_count * DATA_QUALITY_THRESHOLD_PERCENT / 100) if original_row_count > 0 else 0

# Validate and identify problematic rows for 'TERM_CODE' (should be convertible to int)
if 'TERM_CODE' in gold_enrollment_train.columns:
    temp_term_code = pd.to_numeric(gold_enrollment_train['TERM_CODE'], errors='coerce')
    invalid_term_code_indices = temp_term_code[temp_term_code.isna()].index

    if not invalid_term_code_indices.empty:
        problematic_terms = gold_enrollment_train.loc[invalid_term_code_indices, 'TERM_CODE'].unique()
        # Display up to 5 unique problematic values for the warning message
        sample_problematic_terms = problematic_terms[:5].tolist() if problematic_terms.size > 0 else []
        warnings_logged.append(
            f"Warning: {len(invalid_term_code_indices)} rows have invalid 'TERM_CODE' values. "
            f"First 5 unique problematic values: {sample_problematic_terms}. These rows will be removed."
        )
        problematic_rows_indices = problematic_rows_indices.union(invalid_term_code_indices)
else:
    warnings_logged.append("Warning: 'TERM_CODE' column not found for validation.")

# Validate and identify problematic rows for 'HIGH_ENROLLMENT' (only 'Y' or 'N' allowed)
if 'HIGH_ENROLLMENT' in gold_enrollment_train.columns:
    valid_high_enrollment_values = ['Y', 'N']
    invalid_high_enrollment_indices = gold_enrollment_train[
        ~gold_enrollment_train['HIGH_ENROLLMENT'].isin(valid_high_enrollment_values)
    ].index

    if not invalid_high_enrollment_indices.empty:
        problematic_enrollments = gold_enrollment_train.loc[invalid_high_enrollment_indices, 'HIGH_ENROLLMENT'].unique()
        # Display up to 5 unique problematic values for the warning message
        sample_problematic_enrollments = problematic_enrollments[:5].tolist() if problematic_enrollments.size > 0 else []
        warnings_logged.append(
            f"Warning: {len(invalid_high_enrollment_indices)} rows have unexpected 'HIGH_ENROLLMENT' values. "
            f"First 5 unique problematic values: {sample_problematic_enrollments}. These rows will be removed."
        )
        problematic_rows_indices = problematic_rows_indices.union(invalid_high_enrollment_indices)
else:
    warnings_logged.append("Warning: 'HIGH_ENROLLMENT' column not found for validation.")

# Perform row removals if any problematic rows were identified
if not problematic_rows_indices.empty:
    problematic_rows_count = len(problematic_rows_indices)
    gold_enrollment_train.drop(index=problematic_rows_indices, inplace=True)

    for warning_msg in warnings_logged:
        print(warning_msg) # Log all warnings collected

    # Raise error if too many rows were affected, only if original data was not empty
    if original_row_count > 0 and problematic_rows_count > data_quality_max_problematic_rows:
        removed_percentage = (problematic_rows_count / original_row_count) * 100
        raise DataQualityError(
            f"Too many rows ({problematic_rows_count} out of {original_row_count}, {removed_percentage:.2f}%) "
            f"removed due to data quality issues (exceeds {DATA_QUALITY_THRESHOLD_PERCENT}% threshold). "
            "Further investigation into raw data quality is required."
        )
    print(f"Cleaned gold_enrollment_train: Removed {problematic_rows_count} problematic rows. Remaining rows: {len(gold_enrollment_train)}.")
else:
    print("No problematic rows found during initial data validation.")

# Now, apply the required type conversions on the cleaned dataframe
current_row_count = len(gold_enrollment_train)

# Convert 'TERM_CODE' to integer
if 'TERM_CODE' in gold_enrollment_train.columns and current_row_count > 0:
    gold_enrollment_train['TERM_CODE'] = gold_enrollment_train['TERM_CODE'].astype(int)
    print("Converted 'TERM_CODE' to integer type.")
elif 'TERM_CODE' not in gold_enrollment_train.columns:
    print("Warning: 'TERM_CODE' column not found after cleaning, skipping final conversion.")

# Convert 'HIGH_ENROLLMENT' to numerical binary (1 for 'Y', 0 for 'N')
if 'HIGH_ENROLLMENT' in gold_enrollment_train.columns and current_row_count > 0:
    gold_enrollment_train['HIGH_ENROLLMENT'] = gold_enrollment_train['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0}).astype(int)
    print("Mapped 'HIGH_ENROLLMENT' to numerical binary (1/0) and converted to integer type.")
elif 'HIGH_ENROLLMENT' not in gold_enrollment_train.columns:
    print("Warning: 'HIGH_ENROLLMENT' column not found after cleaning, skipping final conversion.")

# Cast 'SUBJECT_ID_SORT' to categorical type
if 'SUBJECT_ID_SORT' in gold_enrollment_train.columns and current_row_count > 0:
    gold_enrollment_train['SUBJECT_ID_SORT'] = gold_enrollment_train['SUBJECT_ID_SORT'].astype('category')
    print("Converted 'SUBJECT_ID_SORT' to categorical type.")
elif 'SUBJECT_ID_SORT' not in gold_enrollment_train.columns:
    print("Warning: 'SUBJECT_ID_SORT' column not found after cleaning, skipping final conversion.")

print(f"Final gold_enrollment_train summary after validation and conversion:")
# Print data types to verify the conversions
print(gold_enrollment_train.dtypes)

# --- 2. Load Features from TRAIN_DATA_DIR ---
# Helper function to load a table if it exists
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        print(f"Loading {filename} from {directory}")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        print(f"Warning: {filename} not found at {filepath}. Skipping.")
        return pd.DataFrame() # Return empty DataFrame if file not found

# Load potential feature tables
# Assuming common academic data tables
terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')
# You could load more tables like 'courses.csv', 'subjects.csv' here and merge as needed.

# Create a base dataframe for merging features, starting with gold labels
data = gold_enrollment_train.copy()

# Add features from offerings_df if available and has required columns
if not offerings_df.empty:
    if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
        # Ensure 'ACTUAL_ENROLLMENT' and 'CAPACITY' are numeric
        offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

        # Aggregate offerings data per (TERM_CODE, SUBJECT_ID_SORT)
        agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            max_capacity=('CAPACITY', 'max'),
            num_offerings=('TERM_CODE', 'count'),
            sum_capacity=('CAPACITY', 'sum')
        ).reset_index()
        data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings data. Data shape: {data.shape}")
    else:
        print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_df is empty. Proceeding with limited features.")

# Add features from terms_df if available and has required columns
if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms data. Data shape: {data.shape}")
    else:
        print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

# --- 3. Feature Engineering ---
# Convert target to numeric
data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

# Extract features from TERM_CODE
data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

# Label Encode SUBJECT_ID_SORT
le_subject = LabelEncoder()
data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

# Define features and target
features = ['TERM_YEAR', 'TERM_SEMESTER', 'SUBJECT_ID_SORT_encoded']

# Dynamically add aggregated features if they exist after merging
if 'avg_enrollment' in data.columns:
    features.append('avg_enrollment')
if 'max_capacity' in data.columns:
    features.append('max_capacity')
if 'num_offerings' in data.columns:
    features.append('num_offerings')
if 'sum_capacity' in data.columns:
    features.append('sum_capacity')
if 'YEAR' in data.columns: # If 'YEAR' was merged from terms_df
    features.append('YEAR')


target = 'HIGH_ENROLLMENT_TARGET'

# Drop rows with NaN in features or target
initial_rows = data.shape[0]
data.dropna(subset=features + [target], inplace=True)
if data.shape[0] < initial_rows:
    print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

# Check if there's enough data after dropping NaNs
if data.empty:
    print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
    print("Final Validation Performance: 0.0")
else:
    X = data[features]
    y = data[target]

    print(f"Features used: {features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- 4. Data Splitting (Time-based validation) ---
    # Use the latest year in the training data for validation
    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        train_df = data[data['TERM_YEAR'] < latest_train_year]
        val_df = data[data['TERM_YEAR'] == latest_train_year]

        if val_df.empty and len(sorted_years) > 1:
            # Fallback if the latest year created an empty validation set
            second_latest_train_year = sorted_years[-2]
            print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df = data[data['TERM_YEAR'] < second_latest_train_year]
            val_df = data[data['TERM_YEAR'] == second_latest_train_year]
        elif val_df.empty:
             print("Warning: Only one or two years of data available, and time-based split created empty validation. Using random split.")
             train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        else:
            print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
    else:
        print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)


    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class. Cannot proceed with model training.")
        final_validation_score = 0.0 # Default score if training is not possible
    else:
        # --- 5. Model Training ---
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')

    # --- 7. Print Final Validation Performance ---
    print(f"Final Validation Performance: {final_validation_score}")
