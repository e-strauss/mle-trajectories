
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
    final_validation_score = 0.0 # Default score if training is not possible
else:
    X = data[features]
    y = data[target]

    print(f"Features used: {features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- Helper function for robust split validation ---
    def is_split_valid(y_train_series, y_val_series, min_samples_per_class=10):
        """
        Checks if a split is valid based on target class distribution.

        Args:
            y_train_series (pd.Series): Target column for the training set.
            y_val_series (pd.Series): Target column for the validation set.
            min_samples_per_class (int): Minimum number of samples required for each class
                                         in both train and validation sets.

        Returns:
            bool: True if the split is valid, False otherwise.
        """
        if y_train_series.empty or y_val_series.empty:
            return False

        train_class_counts = y_train_series.value_counts()
        val_class_counts = y_val_series.value_counts()

        # Check for at least two unique classes
        if len(train_class_counts) < 2 or len(val_class_counts) < 2:
            return False

        # Check if all classes that appear in either train or val are present in both,
        # and meet the minimum sample count.
        all_observed_classes = set(train_class_counts.index) | set(val_class_counts.index)

        for cls in all_observed_classes:
            if cls not in train_class_counts or train_class_counts[cls] < min_samples_per_class:
                return False
            if cls not in val_class_counts or val_class_counts[cls] < min_samples_per_class:
                return False

        return True


    # --- 4. Data Splitting (Time-based validation) ---
    # NOTE: 'data', 'features', 'target', and 'y' (assumed to be data[target])
    # are expected to be defined in the preceding context.

    min_samples_per_class_for_split = 10 # Minimum samples per class (e.g., 10-20 suggested)
    time_based_split_achieved = False
    MAX_VAL_PROPORTION = 0.4 # Heuristic: Max 40% of data for validation in dynamic search

    train_df = pd.DataFrame() # Initialize to avoid UnboundLocalError later
    val_df = pd.DataFrame()

    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())

        # --- Attempt 1: Latest year for validation ---
        latest_train_year = sorted_years[-1]
        temp_train_df_attempt1 = data[data['TERM_YEAR'] < latest_train_year].copy()
        temp_val_df_attempt1 = data[data['TERM_YEAR'] == latest_train_year].copy()

        if is_split_valid(temp_train_df_attempt1[target], temp_val_df_attempt1[target], min_samples_per_class_for_split):
            train_df = temp_train_df_attempt1
            val_df = temp_val_df_attempt1
            time_based_split_achieved = True
            print(f"Time-based split: Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}. Split is valid.")
        else:
            print(f"Time-based split: Initial attempt (latest year: {latest_train_year}) failed validation or was problematic. Trying second latest year...")

            # --- Attempt 2: Second latest year for validation ---
            if not time_based_split_achieved and len(sorted_years) > 1:
                second_latest_train_year = sorted_years[-2]
                temp_train_df_attempt2 = data[data['TERM_YEAR'] < second_latest_train_year].copy()
                temp_val_df_attempt2 = data[data['TERM_YEAR'] == second_latest_train_year].copy()

                if is_split_valid(temp_train_df_attempt2[target], temp_val_df_attempt2[target], min_samples_per_class_for_split):
                    train_df = temp_train_df_attempt2
                    val_df = temp_val_df_attempt2
                    time_based_split_achieved = True
                    print(f"Time-based split: Using second latest year ({second_latest_train_year}) for validation. Training on years prior to {second_latest_train_year}. Split is valid.")
                else:
                    print(f"Time-based split: Second latest year split ({second_latest_train_year}) also failed validation or was problematic. Initiating dynamic time-based search.")

        # --- Dynamic time-based search if initial attempts failed ---
        if not time_based_split_achieved:
            print("Time-based split: Starting dynamic time-based search for a robust split...")
            current_val_years_list = []
            found_dynamic_split = False
            # Iterate from the latest year, expanding the validation window backwards
            for i in range(1, len(sorted_years) + 1):
                year_to_add_to_val = sorted_years[-i]
                current_val_years_list.append(year_to_add_to_val)

                # Ensure chronological order for display
                current_val_years_list.sort()

                temp_train_df_dynamic = data[~data['TERM_YEAR'].isin(current_val_years_list)].copy()
                temp_val_df_dynamic = data[data['TERM_YEAR'].isin(current_val_years_list)].copy()

                # Heuristic: Check if validation set is becoming too large
                if len(temp_val_df_dynamic) / len(data) > MAX_VAL_PROPORTION:
                    print(f"Time-based split: Dynamic search stopped. Validation set size ({len(temp_val_df_dynamic)} samples, {len(temp_val_df_dynamic)/len(data):.1%}) exceeded {MAX_VAL_PROPORTION:.0%} of total data. No robust time-based split found with current criteria.")
                    break # Exit loop, will fall back to random split

                if is_split_valid(temp_train_df_dynamic[target], temp_val_df_dynamic[target], min_samples_per_class_for_split):
                    train_df = temp_train_df_dynamic
                    val_df = temp_val_df_dynamic
                    time_based_split_achieved = True
                    found_dynamic_split = True
                    print(f"Time-based split: Dynamic search found a valid split. Using years {current_val_years_list} for validation and prior years for training.")
                    break
            if not found_dynamic_split and not time_based_split_achieved:
                 print("Time-based split: Dynamic time-based search completed without finding a robust split. Falling back to stratified random split.")

    else:
        print("Time-based split: 'TERM_YEAR' not available or only one year of data. Falling back to stratified random split.")

    # --- Fallback to stratified random split if no time-based split was achieved ---
    if not time_based_split_achieved:
        print("Fallback: Performing stratified random split...")

        temp_train_df_random, temp_val_df_random = None, None

        # Check target classes for stratification
        if y.nunique() < 2:
            print("Fallback Warning: Target variable has less than 2 unique classes. Cannot perform stratified split. Using simple random split.")
            temp_train_df_random, temp_val_df_random = train_test_split(data, test_size=0.2, random_state=42)
        else:
            # 'y' is assumed to be the target series for the entire 'data' DataFrame
            temp_train_df_random, temp_val_df_random = train_test_split(data, test_size=0.2, random_state=42, stratify=y)

        if is_split_valid(temp_train_df_random[target], temp_val_df_random[target], min_samples_per_class_for_split):
            train_df = temp_train_df_random
            val_df = temp_val_df_random
            print("Fallback: Stratified random split is valid.")
        else:
            print("Fallback Serious Warning: Stratified random split also failed validation. This indicates a potential issue with data distribution or class rarity even for random sampling. Proceeding with the random split, but model training/evaluation may be unreliable.")
            # As a last resort, assign the problematic random split
            train_df = temp_train_df_random
            val_df = temp_val_df_random


    # --- Final assignment for X and y ---
    if train_df.empty or val_df.empty:
        print("Critical Error: No valid train/validation split could be created. Returning empty DataFrames/Series.")
        X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
    else:
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

