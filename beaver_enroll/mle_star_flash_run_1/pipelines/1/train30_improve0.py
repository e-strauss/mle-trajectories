
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

    initial_split_strategy_used = "none" # Tracks which initial strategy was attempted
    train_df_candidate = pd.DataFrame() # Initialize as empty
    val_df_candidate = pd.DataFrame()

    # Attempt time-based split first
    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        initial_split_strategy_used = "time_based"
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        temp_train_df = data[data['TERM_YEAR'] < latest_train_year]
        temp_val_df = data[data['TERM_YEAR'] == latest_train_year]

        if temp_val_df.empty and len(sorted_years) > 1:
            # Fallback if the latest year created an empty validation set
            second_latest_train_year = sorted_years[-2]
            print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df_candidate = data[data['TERM_YEAR'] < second_latest_train_year]
            val_df_candidate = data[data['TERM_YEAR'] == second_latest_train_year]
        elif not temp_val_df.empty:
            print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
            train_df_candidate = temp_train_df
            val_df_candidate = temp_val_df
        else:
            # This else implies temp_val_df is empty and len(sorted_years) <= 1, meaning not enough years for a meaningful time split.
            print(f"Warning: Time-based split could not form a valid validation set (empty). Falling back to robust stratified random split.")
            initial_split_strategy_used = "time_based_failed_to_produce_candidates" # Indicate time-split didn't even get valid initial candidates
    else:
        print("Warning: 'TERM_YEAR' not available or only one year of data. Falling back to robust stratified random split.")
        initial_split_strategy_used = "no_time_based_possible"

    # Initialize X_train, y_train, X_val, y_val as empty DataFrames/Series in case no valid split is found
    X_train, y_train = pd.DataFrame(), pd.Series(dtype=y.dtype)
    X_val, y_val = pd.DataFrame(), pd.Series(dtype=y.dtype)

    # Helper to check validity of a split
    def is_split_valid(X_t, y_t, X_v, y_v):
        return (not X_t.empty and not X_v.empty and
                len(np.unique(y_t)) >= 2 and len(np.unique(y_v)) >= 2)

    split_successful = False

    # If time-based split produced candidates, check their validity
    if initial_split_strategy_used == "time_based":
        # Ensure candidates are not empty before attempting to access features/target
        if not train_df_candidate.empty and not val_df_candidate.empty:
            X_train_cand, y_train_cand = train_df_candidate[features], train_df_candidate[target]
            X_val_cand, y_val_cand = val_df_candidate[features], val_df_candidate[target]

            if is_split_valid(X_train_cand, y_train_cand, X_val_cand, y_val_cand):
                X_train, y_train = X_train_cand, y_train_cand
                X_val, y_val = X_val_cand, y_val_cand
                split_successful = True
                print("Time-based split resulted in a valid training and validation set.")
            else:
                print("Time-based split resulted in an invalid split (empty sets or insufficient target classes). Proceeding to robust stratified random split.")
        else:
            print("Time-based split produced empty dataframes. Proceeding to robust stratified random split.")

    # If time-based split was not successful or not applicable, perform robust stratified random split
    if not split_successful:
        print("Attempting robust stratified random split...")
        
        # Check target class diversity for the entire dataset upfront
        if len(np.unique(y)) < 2:
            print(f"Error: Target variable 'y' has only {len(np.unique(y))} unique class(es) in the entire dataset. Cannot perform a split with two classes.")
            # X_train, y_train, X_val, y_val are already empty, so the final check will catch this.
        else:
            max_stratified_retries = 20 # Increased retries to find a valid split
            for i in range(max_stratified_retries):
                current_random_state = 42 + i # Use incrementing random states
                try:
                    temp_train_df, temp_val_df = train_test_split(data, test_size=0.2, random_state=current_random_state, stratify=y)
                    X_train_cand, y_train_cand = temp_train_df[features], temp_train_df[target]
                    X_val_cand, y_val_cand = temp_val_df[features], temp_val_df[target]

                    if is_split_valid(X_train_cand, y_train_cand, X_val_cand, y_val_cand):
                        X_train, y_train = X_train_cand, y_train_cand
                        X_val, y_val = X_val_cand, y_val_cand
                        split_successful = True
                        print(f"Robust stratified random split successful after {i+1} attempt(s) with random_state={current_random_state}.")
                        break
                    else:
                        print(f"Stratified split attempt {i+1} with random_state={current_random_state} failed validity check (empty sets or insufficient target classes). Retrying...")
                except ValueError as e:
                    # This might happen if 'stratify=y' cannot be performed due to very small groups
                    print(f"Stratified split attempt {i+1} with random_state={current_random_state} encountered an error during split: {e}. Retrying...")

            if not split_successful:
                print(f"Error: All {max_stratified_retries} attempts for robust stratified random split failed to produce a valid split. Cannot proceed with model training.")
                # X_train, y_train, X_val, y_val remain empty from initialization, so the final check will catch this.

    # Final check and model training
    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class after all splitting attempts. Cannot proceed with model training.")
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
