
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

# --- Setup from original script (executed once) ---
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

# Ensure 'input' directory exists for dummy data creation if necessary
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
if not os.path.exists(TRAIN_DATA_DIR):
    os.makedirs(TRAIN_DATA_DIR)

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
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202201', '202202', '202301'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'EL', 'MA'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
    })
    # Add more dummy data for better split in some cases
    gold_enrollment_train = pd.concat([gold_enrollment_train, pd.DataFrame({
        'TERM_CODE': ['202001', '202002', '202101', '202201', '202301', '202301'],
        'SUBJECT_ID_SORT': ['BIO', 'CHEM', 'PHY', 'MATH', 'ENG', 'HIST'],
        'HIGH_ENROLLMENT': ['N', 'Y', 'N', 'Y', 'N', 'Y']
    })], ignore_index=True)
    print("Using expanded dummy gold_enrollment_train data.")


# --- 2. Load Features from TRAIN_DATA_DIR ---
def load_table_if_exists(directory, filename, dummy_data_func=None):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        print(f"Loading {filename} from {directory}")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        print(f"Warning: {filename} not found at {filepath}.")
        if dummy_data_func:
            print(f"Creating dummy data for {filename}.")
            return dummy_data_func()
        return pd.DataFrame()

# Dummy data functions for missing files
def create_dummy_terms_df():
    return pd.DataFrame({
        'TERM_CODE': ['202001', '202002', '202101', '202102', '202201', '202202', '202301', '202302'],
        'YEAR': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023]
    })

def create_dummy_offerings_df():
    return pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202201', '202202', '202301',
                      '202001', '202002', '202101', '202201', '202301', '202301'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'EL', 'MA', 'BIO', 'CHEM', 'PHY', 'MATH', 'ENG', 'HIST'],
        'ACTUAL_ENROLLMENT': [50, 30, 55, 20, 35, 25, 60, 22, 30, 40, 25, 45, 30, 50, 30, 40],
        'CAPACITY': [60, 40, 60, 30, 40, 30, 70, 30, 40, 50, 30, 50, 40, 60, 40, 50]
    })


terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv', create_dummy_terms_df)
offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv', create_dummy_offerings_df)

# Create a base dataframe for merging features, starting with gold labels
base_data = gold_enrollment_train.copy()

if not offerings_df.empty:
    if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
        offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')
        agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            max_capacity=('CAPACITY', 'max'),
            num_offerings=('TERM_CODE', 'count'),
            sum_capacity=('CAPACITY', 'sum')
        ).reset_index()
        base_data = pd.merge(base_data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings data. Base data shape: {base_data.shape}")
    else:
        print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_df is empty. Proceeding with limited features.")

if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        base_data = pd.merge(base_data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms data. Base data shape: {base_data.shape}")
    else:
        print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")


def run_experiment(
    data_df: pd.DataFrame,
    experiment_name: str,
    use_term_code_label_encoded: bool = False,
    include_subject_id_sort: bool = True,
    rf_n_estimators: int = 100,
    random_state: int = 42
) -> float:
    print(f"\n--- Running Experiment: {experiment_name} ---")
    data = data_df.copy() # Work on a copy to avoid side effects

    # --- 3. Feature Engineering (dynamic based on flags) ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # TERM_CODE encoding
    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    if use_term_code_label_encoded:
        le_term = LabelEncoder()
        data['TERM_CODE_encoded'] = le_term.fit_transform(data['TERM_CODE_str'])
        features = ['TERM_CODE_encoded']
    else:
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        features = ['TERM_YEAR', 'TERM_SEMESTER']

    # SUBJECT_ID_SORT encoding
    if include_subject_id_sort:
        le_subject = LabelEncoder()
        data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])
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
    if 'YEAR' in data.columns:
        features.append('YEAR')

    target = 'HIGH_ENROLLMENT_TARGET'

    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

    if data.empty:
        print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
        return 0.0

    X = data[features]
    y = data[target]

    print(f"Features used: {features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- Improved Data Splitting (from plan_implement_agent_1) ---
    y_global = data[target]
    class_counts = y_global.value_counts()
    single_instance_classes = class_counts[class_counts == 1].index.tolist()

    data_for_split = data
    single_instance_samples_df = pd.DataFrame(columns=data.columns)

    if single_instance_classes:
        # print(f"Detected {len(single_instance_classes)} classes with only one instance globally: {single_instance_classes}")
        single_instance_samples_df = data[y_global.isin(single_instance_classes)]
        data_for_split = data[~y_global.isin(single_instance_classes)]
        # print(f"Extracted {len(single_instance_samples_df)} single-instance samples. Remaining data for split: {len(data_for_split)}")

    y_for_split = data_for_split[target]

    train_df = pd.DataFrame(columns=data.columns)
    val_df = pd.DataFrame(columns=data.columns)
    split_method_chosen = "none"

    if not data_for_split.empty and 'TERM_YEAR' in data_for_split.columns and data_for_split['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data_for_split['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        temp_train_df = data_for_split[data_for_split['TERM_YEAR'] < latest_train_year]
        temp_val_df = data_for_split[data_for_split['TERM_YEAR'] == latest_train_year]

        if temp_val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df = data_for_split[data_for_split['TERM_YEAR'] < second_latest_train_year]
            val_df = data_for_split[data_for_split['TERM_YEAR'] == second_latest_train_year]
            split_method_chosen = "time-based (fallback to second latest)"
        elif temp_val_df.empty:
             print("Warning: Only one or two years of data available, and time-based split created empty validation. Falling back to stratified random split.")
        else:
            print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
            train_df = temp_train_df
            val_df = temp_val_df
            split_method_chosen = "time-based"

    if split_method_chosen == "none" or (split_method_chosen == "time-based (fallback to second latest)" and (train_df.empty or val_df.empty)):
        if not data_for_split.empty and y_for_split.nunique() > 1:
            print("Performing stratified random split on remaining data (without single-instance classes).")
            try:
                train_df, val_df = train_test_split(data_for_split, test_size=0.2, random_state=random_state, stratify=y_for_split)
                split_method_chosen = "stratified random"
            except ValueError as e:
                print(f"Error during stratified random split: {e}. Falling back to non-stratified random split if possible.")
                train_df, val_df = train_test_split(data_for_split, test_size=0.2, random_state=random_state)
                split_method_chosen = "non-stratified random (fallback)"
        elif not data_for_split.empty and y_for_split.nunique() == 1:
            print("Warning: Remaining data for split has only one class. Cannot perform stratified random split. Performing non-stratified random split.")
            train_df, val_df = train_test_split(data_for_split, test_size=0.2, random_state=random_state)
            split_method_chosen = "non-stratified random (single class)"
        else:
            print("Warning: No sufficient data remaining for splitting even after handling single-instance classes.")

    if not single_instance_samples_df.empty:
        if not train_df.empty:
            train_df = pd.concat([train_df, single_instance_samples_df], ignore_index=True)
            # print(f"Added {len(single_instance_samples_df)} single-instance samples to the training set.")
        else:
            train_df = single_instance_samples_df
            print("Training set was empty after splits; single-instance samples now form the training set.")

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")
    print(f"Train target unique classes: {np.unique(y_train) if not y_train.empty else 'N/A'}")
    print(f"Val target unique classes: {np.unique(y_val) if not y_val.empty else 'N/A'}")

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class after all fallbacks. Cannot proceed with model training.")
        return 0.0
    else:
        model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=random_state, class_weight='balanced')
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
        return final_validation_score

# --- Main Ablation Study Execution ---
results = {}

# Baseline
results['Baseline (TERM_YEAR/SEMESTER, SUBJECT_ID_SORT, n_estimators=100)'] = run_experiment(
    base_data,
    'Baseline',
    use_term_code_label_encoded=False,
    include_subject_id_sort=True,
    rf_n_estimators=100
)

# Ablation 1: TERM_CODE as Label Encoded
results['Ablation 1 (TERM_CODE Label Encoded)'] = run_experiment(
    base_data,
    'Ablation 1',
    use_term_code_label_encoded=True,
    include_subject_id_sort=True,
    rf_n_estimators=100
)

# Ablation 2: No SUBJECT_ID_SORT_encoded feature
results['Ablation 2 (No SUBJECT_ID_SORT_encoded)'] = run_experiment(
    base_data,
    'Ablation 2',
    use_term_code_label_encoded=False,
    include_subject_id_sort=False,
    rf_n_estimators=100
)

# Ablation 3: Reduced n_estimators for RandomForest
results['Ablation 3 (RF n_estimators=50)'] = run_experiment(
    base_data,
    'Ablation 3',
    use_term_code_label_encoded=False,
    include_subject_id_sort=True,
    rf_n_estimators=50
)

print("\n--- Ablation Study Results ---")
best_score = -1.0
best_config = ""
for config, score in results.items():
    print(f"{config}: Macro F1 Score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_config = config

print(f"\nThe configuration that contributed the most to overall performance is: {best_config} with a Macro F1 Score of {best_score:.4f}.")
if best_config != 'Baseline (TERM_YEAR/SEMESTER, SUBJECT_ID_SORT, n_estimators=100)':
    print(f"This represents an improvement over the Baseline score of {results['Baseline (TERM_YEAR/SEMESTER, SUBJECT_ID_SORT, n_estimators=100)']:.4f}.")

