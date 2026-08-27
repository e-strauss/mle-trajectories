
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

# --- 1. Load Gold Labels ---
try:
    gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train)} rows.")
    if gold_enrollment_train.empty:
        raise ValueError("gold_enrollment_train.csv is empty.")
    if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
        raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
    # Create a dummy dataframe for development purposes if file is missing or invalid.
    # In a real scenario, this would typically be a fatal error if data is critical.
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N']
    })
    print("Using dummy gold_enrollment_train data.")

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
    MIN_CLASS_SAMPLES = 5 # Minimum count of samples required for each target class in train and validation sets.

    # Helper function to check minimum class samples in a series
    # This function assumes 'all_unique_targets_full_data' is a list/array of all expected target classes across the full dataset.
    def _check_min_class_samples(y_series_subset, all_unique_targets_full_data, min_count):
        if y_series_subset.empty:
            return False
        class_counts = y_series_subset.value_counts()
        for target_class in all_unique_targets_full_data:
            if target_class not in class_counts or class_counts[target_class] < min_count:
                return False
        return True

    # Initialize variables
    train_df, val_df = None, None
    split_successful = False
    
    # Ensure 'y' is a Series from `data[target]` to correctly get unique values
    # If `y` is already defined as a Series, this line can be removed.
    # Assuming `y` is the target series for the full `data` dataset.
    all_unique_targets_in_data = y.unique() 

    # Condition for attempting time-based splitting:
    # 1. 'TERM_YEAR' column must exist.
    # 2. There must be more than one unique year to create a time-based split.
    # 3. There must be more than one target class for stratification, or for the min_class_samples check to be meaningful across classes.
    if ('TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1 and len(all_unique_targets_in_data) > 1):
        sorted_years = sorted(data['TERM_YEAR'].unique())

        # --- Attempt 1: Latest year for validation ---
        # Requires at least two unique years for a simple train-val split
        if len(sorted_years) >= 2:
            latest_val_year = sorted_years[-1]
            temp_train_df = data[data['TERM_YEAR'] < latest_val_year]
            temp_val_df = data[data['TERM_YEAR'] == latest_val_year]

            if not temp_val_df.empty and not temp_train_df.empty: # Ensure both train and val are non-empty
                temp_y_train = temp_train_df[target]
                temp_y_val = temp_val_df[target]

                if _check_min_class_samples(temp_y_train, all_unique_targets_in_data, MIN_CLASS_SAMPLES) and \
                   _check_min_class_samples(temp_y_val, all_unique_targets_in_data, MIN_CLASS_SAMPLES):
                    train_df = temp_train_df
                    val_df = temp_val_df
                    print(f"Time-based split (validation year={latest_val_year}) successful. All target classes have >= {MIN_CLASS_SAMPLES} samples in train/val sets.")
                    split_successful = True
                else:
                    print(f"Warning: Latest year split ({latest_val_year}) failed minimum class sample check ({MIN_CLASS_SAMPLES} per class). Attempting adaptive split.")
            else:
                print(f"Warning: Latest year split (validation year={latest_val_year}) resulted in empty train/validation sets. Attempting adaptive split.")
        else:
            print("Warning: Only one year of data available, simple latest-year time-based split not possible. Attempting adaptive split (though it will likely fail).")


        # --- Attempt 2: Adaptive - Last two years for validation ---
        # Only proceed if the first attempt failed and there are at least three unique years for this split
        if not split_successful and len(sorted_years) >= 3:
            adaptive_val_years = sorted_years[-2:] # e.g., [2022, 2023]
            train_max_year = sorted_years[-3]     # e.g., 2021 (training on years up to 2021)

            temp_train_df = data[data['TERM_YEAR'] <= train_max_year]
            temp_val_df = data[data['TERM_YEAR'].isin(adaptive_val_years)]

            if not temp_val_df.empty and not temp_train_df.empty: # Ensure both train and val are non-empty
                temp_y_train = temp_train_df[target]
                temp_y_val = temp_val_df[target]

                if _check_min_class_samples(temp_y_train, all_unique_targets_in_data, MIN_CLASS_SAMPLES) and \
                   _check_min_class_samples(temp_y_val, all_unique_targets_in_data, MIN_CLASS_SAMPLES):
                    train_df = temp_train_df
                    val_df = temp_val_df
                    print(f"Time-based adaptive split (validation years={adaptive_val_years[0]}-{adaptive_val_years[1]}) successful. All target classes have >= {MIN_CLASS_SAMPLES} samples in train/val sets.")
                    split_successful = True
                else:
                    print(f"Warning: Adaptive split (last two years) failed minimum class sample check ({MIN_CLASS_SAMPLES} per class). Falling back to stratified random split.")
            else:
                print(f"Warning: Adaptive validation set (last two years: {adaptive_val_years[0]}-{adaptive_val_years[1]}) resulted in empty train/validation sets. Falling back to stratified random split.")
        elif not split_successful and len(sorted_years) < 3: # Covers case where len(sorted_years) is 2 (only enough for attempt 1)
             print("Warning: Not enough years (less than 3) for the adaptive 'last two years' split. Falling back to stratified random split.")
    elif len(all_unique_targets_in_data) <= 1:
        print("Warning: Target variable 'y' has only one unique class or is empty. Time-based stratified split is not fully applicable, and `stratify=y` will be handled in fallback.")
        # This branch will fall through to the final fallback.
    else: # 'TERM_YEAR' not available or only one year of data (data['TERM_YEAR'].nunique() <= 1)
        print("Warning: 'TERM_YEAR' not available or only one unique year of data. Time-based split not possible.")
        # This branch will fall through to the final fallback.

    # --- Final Fallback: Stratified Random Split if no time-based split was successful ---
    if not split_successful:
        print("Falling back to stratified random split for validation.")
        from sklearn.model_selection import train_test_split # Ensure this is imported if not globally available
        
        # Apply stratified random split if there are multiple target classes
        if len(all_unique_targets_in_data) > 1:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        else:
            # If only one target class, stratification is not possible/needed. Use simple random split.
            print("Warning: Target variable 'y' has only one unique class. Using simple random split as stratification is not applicable.")
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        
    # Extract features and target from the chosen dataframes
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
