
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

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # --- 4. Data Splitting (Robust Validation Strategy) ---

    MIN_VAL_SAMPLES = 20 # Minimum number of samples required in the validation set
    MIN_VAL_CLASSES = 2  # Minimum number of unique classes required in the validation set target

    train_df = None
    val_df = None
    split_method_used = "None"
    
    # Helper function to check if a split's validation set is robust
    def is_split_robust(df_val_check, target_col, min_samples, min_classes, was_stratify_intended):
        if df_val_check.empty:
            return False
        if len(df_val_check) < min_samples:
            return False
        # Only check for minimum classes if stratification was intended (i.e., global target has >= min_classes)
        # Otherwise, if the global target itself has < min_classes, this check is not applicable.
        if was_stratify_intended and df_val_check[target_col].nunique() < min_classes:
            return False
        return True

    # Determine if stratification is generally possible for the entire dataset
    global_target_nunique = y.nunique()
    if global_target_nunique < MIN_VAL_CLASSES:
        print(f"Warning: Global target variable '{target}' has less than {MIN_VAL_CLASSES} unique classes ({global_target_nunique}). Stratified split may not be fully effective, or not possible.")
        stratify_possible_globally = False
        stratify_series = None # Cannot stratify if only one class in entire dataset
    else:
        stratify_possible_globally = True
        stratify_series = y # Use y for stratification if possible

    # --- Attempt 1: Time-based Split ---
    temp_train_df_time, temp_val_df_time = pd.DataFrame(), pd.DataFrame()
    time_split_attempted = False

    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        time_split_attempted = True
        sorted_years = sorted(data['TERM_YEAR'].unique())
        
        # Try with the latest year
        latest_train_year = sorted_years[-1]
        attempt_train_df = data[data['TERM_YEAR'] < latest_train_year]
        attempt_val_df = data[data['TERM_YEAR'] == latest_train_year]

        if not attempt_val_df.empty:
            temp_train_df_time, temp_val_df_time = attempt_train_df, attempt_val_df
            print(f"Attempting time-based split with latest year ({latest_train_year}) for validation.")
        elif len(sorted_years) > 1:
            # Fallback if the latest year created an empty validation set
            second_latest_train_year = sorted_years[-2]
            attempt_train_df = data[data['TERM_YEAR'] < second_latest_train_year]
            attempt_val_df = data[data['TERM_YEAR'] == second_latest_train_year]
            if not attempt_val_df.empty:
                temp_train_df_time, temp_val_df_time = attempt_train_df, attempt_val_df
                print(f"Validation from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation.")
            else:
                print("Warning: Both latest and second-latest year time-based splits resulted in empty validation sets.")
        else:
            print("Warning: 'TERM_YEAR' available but only one unique year, or latest year split was empty.")

        if is_split_robust(temp_val_df_time, target, MIN_VAL_SAMPLES, MIN_VAL_CLASSES, stratify_possible_globally):
            train_df = temp_train_df_time
            val_df = temp_val_df_time
            split_method_used = "Time-based"
            print(f"Time-based split successful and robust: Val samples={len(val_df)}, Val classes={val_df[target].nunique()}.")
        else:
            print(f"Time-based split not robust enough (Val samples: {len(temp_val_df_time)}, Val classes: {temp_val_df_time[target].nunique()}).")
    
    if not time_split_attempted:
        print("Warning: 'TERM_YEAR' not available or not suitable for time-based splitting.")
        
    # --- Attempt 2: Robust Stratified Random Split (if time-based failed or not applicable) ---
    if split_method_used == "None":
        print("Performing robust random split (stratified if possible).")
        test_sizes = [0.2, 0.3, 0.15] # Common ratios to try
        
        for test_size in test_sizes:
            # Ensure enough samples for both train and validation based on test_size and MIN_VAL_SAMPLES
            if len(data) * (1 - test_size) < MIN_VAL_SAMPLES or len(data) * test_size < MIN_VAL_SAMPLES:
                print(f"Skipping test_size={test_size} as it would lead to too few samples for train or val ({len(data)*(1-test_size):.0f}/{len(data)*test_size:.0f}).")
                continue

            try:
                current_stratify = stratify_series if stratify_possible_globally else None
                temp_train_df_rand, temp_val_df_rand = train_test_split(data, test_size=test_size, random_state=42, stratify=current_stratify)
                
                if is_split_robust(temp_val_df_rand, target, MIN_VAL_SAMPLES, MIN_VAL_CLASSES, stratify_possible_globally):
                    train_df = temp_train_df_rand
                    val_df = temp_val_df_rand
                    split_method_used = "Random (Stratified)" if stratify_possible_globally else "Random (Non-Stratified)"
                    print(f"Robust random split successful with test_size={test_size}: Val samples={len(val_df)}, Val classes={val_df[target].nunique()}.")
                    break # Exit loop if successful
                else:
                    print(f"Random split with test_size={test_size} not robust enough (Val samples: {len(temp_val_df_rand)}, Val classes: {temp_val_df_rand[target].nunique()}). Trying next test_size...")

            except ValueError as e:
                # This usually happens if stratification is impossible for a given test_size due to rare classes.
                print(f"Warning: Stratified random split attempt with test_size={test_size} failed (error: {e}). Trying non-stratified split for this test_size.")
                # Fallback to non-stratified random split for the current test_size
                try:
                    temp_train_df_rand, temp_val_df_rand = train_test_split(data, test_size=test_size, random_state=42)
                    # For non-stratified, we cannot guarantee MIN_VAL_CLASSES, so was_stratify_intended=False
                    if is_split_robust(temp_val_df_rand, target, MIN_VAL_SAMPLES, MIN_VAL_CLASSES, False):
                        train_df = temp_train_df_rand
                        val_df = temp_val_df_rand
                        split_method_used = "Random (Non-Stratified Fallback)"
                        print(f"Robust non-stratified random split successful with test_size={test_size}: Val samples={len(val_df)}.")
                        break
                    else:
                        print(f"Non-stratified random split with test_size={test_size} also not robust enough. Trying next test_size...")
                except Exception as ex:
                    print(f"Further random split attempt failed for test_size={test_size}: {ex}")
        
        # If still no robust split found after all random attempts,
        # proceed with a default random split, accepting potential non-robustness.
        if train_df is None or val_df is None:
            print("CRITICAL WARNING: No robust split could be achieved with specified criteria after multiple attempts. Proceeding with a basic default random split which may not be ideal.")
            final_stratify = stratify_series if stratify_possible_globally else None
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=final_stratify)
            split_method_used = "Default Random (potentially non-robust)"
            
            if train_df.empty or val_df.empty: # Final safeguard for truly empty splits
                 print("ERROR: Final default random split still resulted in empty train/validation sets. Data might be too small.")


    # --- Final Assignment ---
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape} (Split method: {split_method_used})")

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
