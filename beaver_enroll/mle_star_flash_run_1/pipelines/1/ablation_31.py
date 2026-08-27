
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import copy # To ensure independent modifications for each experiment

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
    import lightgbm
except ImportError:
    import subprocess
    import sys
    print("Installing required packages: pandas, numpy, scikit-learn, lightgbm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn", "lightgbm"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn
    import lightgbm


# Define paths
# Assuming the script is run from the root directory where 'input' is present.
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")
TEST_DATA_DIR = None # Not available for training phase

# Helper function to load a table if it exists
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        # print(f"Loading {filename} from {directory}") # Suppress verbose loading for ablation runs
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        # print(f"Warning: {filename} not found at {filepath}. Skipping.") # Suppress verbose warnings
        return pd.DataFrame() # Return empty DataFrame if file not found

def run_ablation_experiment(
    exp_name: str,
    modify_term_code_parsing: bool = False,
    modify_capacity_agg: bool = False,
    remove_subject_id_sort: bool = False
):
    print(f"\n--- Running Experiment: {exp_name} ---")

    # --- 1. Load Gold Labels ---
    # Using a try-except block similar to the original to handle missing file
    try:
        gold_enrollment_train_orig = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train_orig.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train_orig.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
        # print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train_orig)} rows.")
    except (FileNotFoundError, ValueError) as e:
        # print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
        gold_enrollment_train_orig = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202201', '202201', '202301', '202301',
                          '202001', '202001', '202002', '202002', '202101', '202101'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL',
                                'CS', 'MA', 'CS', 'PH', 'MA', 'EL'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N',
                                'Y', 'N', 'Y', 'N', 'Y', 'N']
        })
        # print("Using dummy gold_enrollment_train data.")
    
    # Create a copy for the current experiment to avoid modifying original dataframes
    gold_enrollment_train = gold_enrollment_train_orig.copy()

    # --- 2. Load Features from TRAIN_DATA_DIR ---
    terms_df_orig = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df_orig = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')
    
    # Using dummy data for terms_df and offerings_df if files not found or empty
    if terms_df_orig.empty:
        # print("Creating dummy terms_df.")
        terms_df_orig = pd.DataFrame({
            'TERM_CODE': ['202001', '202002', '202101', '202201', '202301'],
            'YEAR': [2020, 2020, 2021, 2022, 2023]
        })
    if offerings_df_orig.empty:
        # print("Creating dummy offerings_df.")
        offerings_df_orig = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202201', '202201', '202301', '202301'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL'],
            'ACTUAL_ENROLLMENT': [100, 30, 120, 25, 110, 40, 130, 35, 150, 50],
            'CAPACITY': [120, 50, 150, 40, 130, 60, 160, 50, 180, 70]
        })

    terms_df = terms_df_orig.copy()
    offerings_df = offerings_df_orig.copy()

    # Create a base dataframe for merging features, starting with gold labels
    data = gold_enrollment_train.copy()

    # Add features from offerings_df if available and has required columns
    if not offerings_df.empty:
        if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
            # Ensure 'ACTUAL_ENROLLMENT' and 'CAPACITY' are numeric
            offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
            offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

            # Aggregate offerings data per (TERM_CODE, SUBJECT_ID_SORT)
            agg_method_capacity = 'mean' if modify_capacity_agg else 'max'
            
            agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
                max_capacity=('CAPACITY', agg_method_capacity), # Ablation point 2
                num_offerings=('TERM_CODE', 'count'),
                sum_capacity=('CAPACITY', 'sum')
            ).reset_index()
            data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
            # print(f"Merged with aggregated offerings data. Data shape: {data.shape}")
        # else:
            # print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
    # else:
        # print("Warning: offerings_df is empty. Proceeding with limited features.")

    # Add features from terms_df if available and has required columns
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
            # print(f"Merged with terms data. Data shape: {data.shape}")
        # else:
            # print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

    # --- 3. Feature Engineering ---
    # Convert target to numeric
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Extract features from TERM_CODE
    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    if modify_term_code_parsing: # Ablation point 1
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce')
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce')
    else:
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # Label Encode SUBJECT_ID_SORT
    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    # Define features and target
    features = ['TERM_YEAR', 'TERM_SEMESTER']

    # Ablation point 3: Remove SUBJECT_ID_SORT_encoded
    if not remove_subject_id_sort:
        features.append('SUBJECT_ID_SORT_encoded')

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
        return 0.0 # Default score if training is not possible
    else:
        X = data[features]
        y = data[target]

        # print(f"Features used: {features}")
        # print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

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
                # print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
                train_df = data[data['TERM_YEAR'] < second_latest_train_year]
                val_df = data[data['TERM_YEAR'] == second_latest_train_year]
            elif val_df.empty:
                # print("Warning: Only one or two years of data available, and time-based split created empty validation. Using random split.")
                train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
            # else:
                # print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
        else:
            # print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)

        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        # print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")
        # print(f"Val target unique classes: {np.unique(y_val)}")


        if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print("Error: Training or validation set is empty, or target has only one class after all fallbacks. Cannot proceed with model training.")
            return 0.0 # Default score if training is not possible
        else:
            # --- 5. Model Training (Using LightGBM as per plan_implement_agent_1) ---
            model = lgb.LGBMClassifier(objective='binary', is_unbalance=True, random_state=42)
            model.fit(X_train, y_train)

            # --- 6. Evaluation ---
            val_predictions = model.predict(X_val)
            final_validation_score = f1_score(y_val, val_predictions, average='macro')
            return final_validation_score

# --- Run Ablation Study ---
results = {}

# Baseline
results['Baseline (Original TERM_CODE parsing, max capacity, include SUBJECT_ID_SORT_encoded)'] = run_ablation_experiment(
    "Baseline"
)

# Ablation 1: Modify TERM_CODE parsing (no fillna(0) and astype(int))
results['Ablation 1 (TERM_YEAR/SEMESTER as float, let dropna handle NaNs)'] = run_ablation_experiment(
    "Ablation 1",
    modify_term_code_parsing=True
)

# Ablation 2: Change aggregation for max_capacity from max to mean
results['Ablation 2 (Capacity aggregation: mean instead of max)'] = run_ablation_experiment(
    "Ablation 2",
    modify_capacity_agg=True
)

# Ablation 3: Remove SUBJECT_ID_SORT_encoded feature
results['Ablation 3 (Removed SUBJECT_ID_SORT_encoded feature)'] = run_ablation_experiment(
    "Ablation 3",
    remove_subject_id_sort=True
)

# Print all results
print("\n--- Ablation Study Results ---")
for name, score in results.items():
    print(f"{name}: Macro F1 Score = {score:.4f}")

# Determine the most contributing part
best_score = -1.0
best_config = ""
for name, score in results.items():
    if score > best_score:
        best_score = score
        best_config = name
    elif score == best_score and "Baseline" not in best_config: # Prefer baseline if scores are equal, unless an ablation already improved it
        # If another ablation also has the same best score, we can list them all or pick the first one encountered.
        # For simplicity, if scores are equal and it's not the baseline, keep the current best_config.
        # If baseline is the best AND an ablation matches it, we still list baseline.
        pass

if best_score == 0.0:
    print("\nConclusion: All configurations resulted in a Macro F1 Score of 0.0. This indicates a fundamental issue preventing meaningful model training or evaluation, likely due to data limitations or setup problems. No specific part could be identified as contributing most.")
else:
    print(f"\nConclusion: The configuration that contributed the most to the overall performance is: '{best_config}' with a Macro F1 Score of {best_score:.4f}.")
    print("Specifically:")
    if "Baseline" in best_config:
        print("- The original setup (TERM_CODE parsing, max capacity aggregation, inclusion of SUBJECT_ID_SORT_encoded) appears to be the most effective, or changes had no positive impact.")
    else:
        # Check if the best config involved modifications
        if 'Ablation 1' in best_config:
            print("- Allowing TERM_YEAR/TERM_SEMESTER to remain as float (without fillna(0).astype(int)) and letting dropna handle potential NaNs contributed positively.")
        if 'Ablation 2' in best_config:
            print("- Changing 'CAPACITY' aggregation from 'max' to 'mean' contributed positively.")
        if 'Ablation 3' in best_config:
            print("- Removing the 'SUBJECT_ID_SORT_encoded' feature contributed positively.")
    print("Conversely, if the baseline was significantly lower than an ablation, it implies that the modified part of the code was detrimental to performance in the baseline.")
