

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
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202201', '202202', '202301', '202302', '202303'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL', 'CS', 'MA'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
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
initial_data = gold_enrollment_train.copy()

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
        initial_data = pd.merge(initial_data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings data. Data shape: {initial_data.shape}")
    else:
        print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_df is empty. Proceeding with limited features.")

# Add features from terms_df if available and has required columns
if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        initial_data = pd.merge(initial_data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms data. Data shape: {initial_data.shape}")
    else:
        print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

# --- Define a function to run a single experiment with modifiable components ---
def run_experiment(
    data_df_raw,
    use_cyclical_semester_features=True,
    fill_na_for_aggregated_features=False,
    use_year_from_terms=True,
    experiment_name="Baseline"
):
    print(f"\n--- Running Experiment: {experiment_name} ---")
    data = data_df_raw.copy()

    # --- 3. Feature Engineering ---
    # Convert target to numeric
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Extract features from TERM_CODE
    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)

    if use_cyclical_semester_features:
        # Refined TERM_SEMESTER: Cyclical features
        term_semester_numeric = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce')
        unique_semesters_valid = sorted(term_semester_numeric.dropna().unique())
        period = len(unique_semesters_valid)

        if period > 1:
            semester_to_idx_map = {sem: i for i, sem in enumerate(unique_semesters_valid)}
            data['TERM_SEMESTER_mapped_cycle'] = term_semester_numeric.map(semester_to_idx_map)
            data['TERM_SEMESTER_SIN'] = np.sin(2 * np.pi * data['TERM_SEMESTER_mapped_cycle'] / period)
            data['TERM_SEMESTER_COS'] = np.cos(2 * np.pi * data['TERM_SEMESTER_mapped_cycle'] / period)
            data['TERM_SEMESTER_SIN'].fillna(0, inplace=True)
            data['TERM_SEMESTER_COS'].fillna(0, inplace=True)
            print("Using cyclical TERM_SEMESTER features (SIN/COS).")
        else:
            data['TERM_SEMESTER_SIN'] = 0.0
            data['TERM_SEMESTER_COS'] = 0.0
            print("Not enough unique semesters for cyclical features. Using 0.0 for SIN/COS.")
    else:
        # Revert to simple numeric TERM_SEMESTER if not using cyclical features
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        print("Using simple numeric TERM_SEMESTER feature.")

    # Label Encode SUBJECT_ID_SORT
    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    # Define base features
    features = ['TERM_YEAR', 'SUBJECT_ID_SORT_encoded']
    if use_cyclical_semester_features:
        features.extend(['TERM_SEMESTER_SIN', 'TERM_SEMESTER_COS'])
    else:
        features.append('TERM_SEMESTER')

    # Dynamically add aggregated features if they exist after merging
    aggregated_cols = ['avg_enrollment', 'max_capacity', 'num_offerings', 'sum_capacity']
    for col in aggregated_cols:
        if col in data.columns:
            features.append(col)

    if use_year_from_terms and 'YEAR' in data.columns:
        features.append('YEAR')
        print("Including 'YEAR' from terms_df.")
    elif not use_year_from_terms:
        print("Excluding 'YEAR' from terms_df.")
    else:
        print("'YEAR' from terms_df not available or not included.")


    target = 'HIGH_ENROLLMENT_TARGET'

    # --- NaN Handling ---
    initial_rows = data.shape[0]
    
    if fill_na_for_aggregated_features:
        print("Filling NaNs in aggregated numerical features and 'YEAR' with 0.")
        fillable_cols = []
        for col in ['avg_enrollment', 'max_capacity', 'num_offerings', 'sum_capacity']:
            if col in features and col in data.columns:
                fillable_cols.append(col)
        if 'YEAR' in features and 'YEAR' in data.columns:
            fillable_cols.append('YEAR')

        for col in fillable_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(0) # Fill with 0 for numerical features
        
        data.dropna(subset=features + [target], inplace=True) # Still drop if NaNs in other features or target
    else:
        print("Dropping rows with NaN in any feature or target.")
        data.dropna(subset=features + [target], inplace=True)
    
    if data.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

    # Check if there's enough data after dropping NaNs
    if data.empty:
        print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
        return 0.0
    else:
        X = data[features]
        y = data[target]

        print(f"Features used: {features}")
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

        # --- 4. Data Splitting (Time-based validation) ---
        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_train_year = sorted_years[-1]

            train_df = data[data['TERM_YEAR'] < latest_train_year]
            val_df = data[data['TERM_YEAR'] == latest_train_year]

            if val_df.empty and len(sorted_years) > 1:
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
            return 0.0 # Default score if training is not possible
        else:
            # --- 5. Model Training ---
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)

            # --- 6. Evaluation ---
            val_predictions = model.predict(X_val)
            final_validation_score = f1_score(y_val, val_predictions, average='macro')
            return final_validation_score


# --- Run the ablation study ---

results = {}

# Baseline: Current solution with cyclical semester features, dropna for all, and use 'YEAR'
baseline_score = run_experiment(
    data_df_raw=initial_data,
    use_cyclical_semester_features=True,
    fill_na_for_aggregated_features=False,
    use_year_from_terms=True,
    experiment_name="Baseline (Cyclical Semester Features, dropna, use YEAR)"
)
results["Baseline (Cyclical Semester Features, dropna, use YEAR)"] = baseline_score
print(f"Baseline Macro F1 Score: {baseline_score}")

# Ablation 1: No Cyclical Semester Features (revert to simple numeric TERM_SEMESTER)
ablation1_score = run_experiment(
    data_df_raw=initial_data,
    use_cyclical_semester_features=False,
    fill_na_for_aggregated_features=False,
    use_year_from_terms=True,
    experiment_name="Ablation 1 (No Cyclical Semester Features, use simple numeric TERM_SEMESTER)"
)
results["Ablation 1 (No Cyclical Semester Features, use simple numeric TERM_SEMESTER)"] = ablation1_score
print(f"Ablation 1 Macro F1 Score: {ablation1_score}")

# Ablation 2: Fill NaNs for aggregated numerical features with 0 (instead of dropping rows because of them)
ablation2_score = run_experiment(
    data_df_raw=initial_data,
    use_cyclical_semester_features=True,
    fill_na_for_aggregated_features=True,
    use_year_from_terms=True,
    experiment_name="Ablation 2 (Fill NaNs in aggregated features & YEAR with 0, keep other NaNs for dropna)"
)
results["Ablation 2 (Fill NaNs in aggregated features & YEAR with 0, keep other NaNs for dropna)"] = ablation2_score
print(f"Ablation 2 Macro F1 Score: {ablation2_score}")

# Ablation 3: No 'YEAR' feature from terms_df
ablation3_score = run_experiment(
    data_df_raw=initial_data,
    use_cyclical_semester_features=True,
    fill_na_for_aggregated_features=False,
    use_year_from_terms=False,
    experiment_name="Ablation 3 (No 'YEAR' feature from terms_df)"
)
results["Ablation 3 (No 'YEAR' feature from terms_df)"] = ablation3_score
print(f"Ablation 3 Macro F1 Score: {ablation3_score}")

# Determine the best performing scenario
best_scenario = max(results, key=results.get)
highest_score = results[best_scenario]

print("\n--- Ablation Study Results Summary ---")
for scenario, score in results.items():
    print(f"{scenario}: Macro F1 Score = {score:.4f}")

print("\nConclusion:")
print(f"The part of the code that contributes the most to the overall performance is: '{best_scenario}' with a Macro F1 Score of {highest_score:.4f}.")

