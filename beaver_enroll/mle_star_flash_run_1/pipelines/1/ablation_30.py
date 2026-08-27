

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# LabelEncoder is part of the original script, but plan_implement_agent_1 uses TargetEncoder and pd.get_dummies
# from sklearn.preprocessing import LabelEncoder 

# Imports for plan_implement_agent_1
from category_encoders import TargetEncoder

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
    import category_encoders # Check for category_encoders
except ImportError:
    import subprocess
    import sys
    print("Installing required packages: pandas, numpy, scikit-learn, category_encoders...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn", "category_encoders"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn
    from category_encoders import TargetEncoder


# Define paths
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
    # Increased dummy data rows and years for more robust splitting
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202102', '202201', '202201', '202202', '202202', '202301', '202301', '202302', '202302'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
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
terms_df_raw = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_df_raw = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

# Create dummy offerings_df and terms_df if not loaded, to ensure script execution
if offerings_df_raw.empty:
    print("Creating dummy offerings_df data.")
    offerings_df_raw = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202102', '202201', '202201', '202202', '202202', '202301', '202301', '202302', '202302'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH'],
        'ACTUAL_ENROLLMENT': [50, 30, 60, 20, 55, 35, 65, 25, 60, 30, 70, 40, 75, 45, 80, 50],
        'CAPACITY': [60, 40, 70, 30, 65, 45, 75, 35, 70, 40, 80, 50, 85, 55, 90, 60]
    })

if terms_df_raw.empty:
    print("Creating dummy terms_df data.")
    terms_df_raw = pd.DataFrame({
        'TERM_CODE': ['202001', '202002', '202101', '202102', '202201', '202202', '202301', '202302'],
        'YEAR': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023]
    })


def run_ablation_experiment(
    config_name,
    gold_enrollment_train_df,
    terms_df_param,
    offerings_df_param,
    capacity_agg_method='max', # Ablation 1: Change CAPACITY aggregation method
    impute_nans=False,         # Ablation 2: Replace dropna with mean imputation for numerical features
    rf_n_estimators=100        # Ablation 3: Reduce RandomForestClassifier n_estimators
):
    print(f"\n--- Running Experiment: {config_name} ---")
    data = gold_enrollment_train_df.copy()
    offerings_df_local = offerings_df_param.copy()
    terms_df_local = terms_df_param.copy()

    # Add features from offerings_df if available and has required columns
    if not offerings_df_local.empty:
        if all(col in offerings_df_local.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
            offerings_df_local['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df_local['ACTUAL_ENROLLMENT'], errors='coerce')
            offerings_df_local['CAPACITY'] = pd.to_numeric(offerings_df_local['CAPACITY'], errors='coerce')

            agg_dict = {
                'avg_enrollment': ('ACTUAL_ENROLLMENT', 'mean'),
                'num_offerings': ('TERM_CODE', 'count'),
                'sum_capacity': ('CAPACITY', 'sum')
            }
            # Dynamically add the capacity aggregation based on method
            if capacity_agg_method == 'max':
                agg_dict['max_capacity'] = ('CAPACITY', 'max')
            elif capacity_agg_method == 'mean':
                agg_dict['mean_capacity'] = ('CAPACITY', 'mean')
            elif capacity_agg_method == 'median':
                agg_dict['median_capacity'] = ('CAPACITY', 'median')

            agg_features = offerings_df_local.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(**agg_dict).reset_index()
            data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
            print(f"Merged with aggregated offerings data. Data shape: {data.shape} (Capacity agg method: {capacity_agg_method})")
        else:
            print("Warning: offerings_df_local missing expected columns for aggregation. Skipping merge.")
    else:
        print("Warning: offerings_df_local is empty. Proceeding with limited features.")

    # Add features from terms_df if available and has required columns
    if not terms_df_local.empty:
        if 'TERM_CODE' in terms_df_local.columns and 'YEAR' in terms_df_local.columns:
            data = pd.merge(data, terms_df_local[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
            print(f"Merged with terms data. Data shape: {data.shape}")
        else:
            print("Warning: terms_df_local missing expected columns (YEAR). Skipping merge.")

    # --- 3. Feature Engineering (incorporating plan_implement_agent_1 modifications) ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    # Using 'TERM_SEMESTER_numeric' to avoid conflict if 'TERM_SEMESTER' is used elsewhere
    data['TERM_SEMESTER_numeric'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # One-Hot Encode TERM_SEMESTER
    term_semester_dummies = pd.get_dummies(data['TERM_SEMESTER_numeric'].astype(str), prefix='TERM_SEMESTER', drop_first=False)
    data = pd.concat([data, term_semester_dummies], axis=1)

    # Define initial features list as per plan_implement_agent_1 baseline
    current_features = ['TERM_YEAR']

    # Dynamically add aggregated features based on their presence and selected aggregation method
    if 'avg_enrollment' in data.columns:
        current_features.append('avg_enrollment')
    
    # Add the specific capacity feature name based on the method
    if capacity_agg_method == 'max' and 'max_capacity' in data.columns:
        current_features.append('max_capacity')
    elif capacity_agg_method == 'mean' and 'mean_capacity' in data.columns:
        current_features.append('mean_capacity')
    elif capacity_agg_method == 'median' and 'median_capacity' in data.columns:
        current_features.append('median_capacity')

    if 'num_offerings' in data.columns:
        current_features.append('num_offerings')
    if 'sum_capacity' in data.columns:
        current_features.append('sum_capacity')
    if 'YEAR' in data.columns:
        current_features.append('YEAR')
    
    current_features.extend(term_semester_dummies.columns.tolist()) # Add OHE semester features

    target = 'HIGH_ENROLLMENT_TARGET'
    
    # Drop rows with NaN in target
    initial_rows_before_target_nan_drop = data.shape[0]
    data.dropna(subset=[target], inplace=True)
    if data.shape[0] < initial_rows_before_target_nan_drop:
        print(f"Dropped {initial_rows_before_target_nan_drop - data.shape[0]} rows due to NaN in target.")
    
    # Filter features to only those that exist in `data` after merging and preliminary FE
    available_features = [f for f in current_features if f in data.columns]

    # --- NaN Handling for features ---
    if impute_nans:
        print("Performing mean imputation for numerical features.")
        for col in available_features:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
    else: # Default behavior from the original script: drop NaNs in features
        initial_rows_after_target_nan_drop = data.shape[0]
        data.dropna(subset=available_features, inplace=True)
        if data.shape[0] < initial_rows_after_target_nan_drop:
            print(f"Dropped {initial_rows_after_target_nan_drop - data.shape[0]} rows due to NaN in features (no imputation).")

    if data.empty:
        print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
        return 0.0
    
    # Prepare X and y for splitting, before target encoding SUBJECT_ID_SORT
    X_pre_te = data.copy()
    y_pre_te = data[target].copy()

    # --- 4. Data Splitting (Time-based validation with fallback) ---
    train_df, val_df = pd.DataFrame(), pd.DataFrame()
    
    if 'TERM_YEAR' in X_pre_te.columns and X_pre_te['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(X_pre_te['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        train_df = X_pre_te[X_pre_te['TERM_YEAR'] < latest_train_year].copy()
        val_df = X_pre_te[X_pre_te['TERM_YEAR'] == latest_train_year].copy()

        if val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df = X_pre_te[X_pre_te['TERM_YEAR'] < second_latest_train_year].copy()
            val_df = X_pre_te[X_pre_te['TERM_YEAR'] == second_latest_train_year].copy()
        elif val_df.empty: # Only one or two years of data, and current logic failed
             print("Warning: Only one or two years of data available, and time-based split created empty validation. Using random split.")
             if len(np.unique(y_pre_te)) < 2:
                 print("Error: Target has only one class, cannot perform stratified split. Using simple random split.")
                 train_df, val_df = train_test_split(X_pre_te, test_size=0.2, random_state=42)
             else:
                 train_df, val_df = train_test_split(X_pre_te, test_size=0.2, random_state=42, stratify=y_pre_te)
        else:
            print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
    else:
        print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
        if len(np.unique(y_pre_te)) < 2:
            print("Error: Target has only one class, cannot perform stratified split. Using simple random split.")
            train_df, val_df = train_test_split(X_pre_te, test_size=0.2, random_state=42)
        else:
            train_df, val_df = train_test_split(X_pre_te, test_size=0.2, random_state=42, stratify=y_pre_te)

    # --- Target Encode SUBJECT_ID_SORT after split to prevent leakage ---
    # Reinitialize TargetEncoder for each run to avoid state contamination
    te_subject = TargetEncoder(cols=['SUBJECT_ID_SORT'], handle_missing='value', handle_unknown='value')
    if 'SUBJECT_ID_SORT' in train_df.columns:
        train_df['SUBJECT_ID_SORT_target_encoded'] = te_subject.fit_transform(train_df['SUBJECT_ID_SORT'], train_df[target])
        val_df['SUBJECT_ID_SORT_target_encoded'] = te_subject.transform(val_df['SUBJECT_ID_SORT'])
        if 'SUBJECT_ID_SORT_target_encoded' not in available_features: # Ensure it's in feature list for model
            available_features.append('SUBJECT_ID_SORT_target_encoded')
    
    # Filter final features based on what's available and what should be used for model
    # Exclude the original 'SUBJECT_ID_SORT' categorical column if its target-encoded version is used
    final_features_for_model = [f for f in available_features if f in train_df.columns and f != 'SUBJECT_ID_SORT']

    X_train, y_train = train_df[final_features_for_model], train_df[target]
    X_val, y_val = val_df[final_features_for_model], val_df[target]
    
    print(f"Features used: {final_features_for_model}")
    print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
    print(f"Shape of X_val: {X_val.shape}, Shape of y_val: {y_val.shape}")
    print(f"Train target unique classes: {np.unique(y_train)}")
    print(f"Val target unique classes: {np.unique(y_val)}")


    final_validation_score = 0.0
    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class. Cannot proceed with model training.")
    else:
        # --- 5. Model Training ---
        model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')

    print(f"Final Validation Performance for {config_name}: {final_validation_score}")
    return final_validation_score


# --- Ablation Study ---
print("\n--- Starting Ablation Study ---")

results = {}

# Baseline Configuration (using settings from plan_implement_agent_1 and current train.py defaults)
# - Capacity aggregation: max
# - NaN handling: dropna
# - RF n_estimators: 100
baseline_score = run_ablation_experiment(
    "Baseline: Original Solution (after plan_implement_agent_1 updates)",
    gold_enrollment_train,
    terms_df_raw,
    offerings_df_raw,
    capacity_agg_method='max',
    impute_nans=False,
    rf_n_estimators=100
)
results["Baseline: Original Solution (after plan_implement_agent_1 updates)"] = baseline_score

# Ablation 1: Change CAPACITY aggregation from max to median
ablation1_score = run_ablation_experiment(
    "Ablation 1: Capacity Aggregation Method (median instead of max)",
    gold_enrollment_train,
    terms_df_raw,
    offerings_df_raw,
    capacity_agg_method='median', # MODIFICATION
    impute_nans=False,
    rf_n_estimators=100
)
results["Ablation 1: Capacity Aggregation Method (median instead of max)"] = ablation1_score

# Ablation 2: Replace dropna with mean imputation for numerical features
ablation2_score = run_ablation_experiment(
    "Ablation 2: Mean Imputation for NaNs (instead of dropping rows)",
    gold_enrollment_train,
    terms_df_raw,
    offerings_df_raw,
    capacity_agg_method='max',
    impute_nans=True, # MODIFICATION
    rf_n_estimators=100
)
results["Ablation 2: Mean Imputation for NaNs (instead of dropping rows)"] = ablation2_score

# Ablation 3: Reduce RandomForestClassifier n_estimators to 10
ablation3_score = run_ablation_experiment(
    "Ablation 3: RF n_estimators=10 (instead of 100)",
    gold_enrollment_train,
    terms_df_raw,
    offerings_df_raw,
    capacity_agg_method='max',
    impute_nans=False,
    rf_n_estimators=10 # MODIFICATION
)
results["Ablation 3: RF n_estimators=10 (instead of 100)"] = ablation3_score


# --- Summarize Results and Determine Most Contributing Part ---
print("\n--- Ablation Study Summary ---")
for config, score in results.items():
    print(f"{config}: Macro F1 Score = {score:.4f}")

# Determine the best performing configuration
best_config = None
highest_score = -1.0 # Initialize with a value lower than any possible F1 score

for config, score in results.items():
    if score > highest_score:
        highest_score = score
        best_config = config
    elif score == highest_score:
        # If scores are equal, maintain the first encountered best_config or handle as tied
        # For simplicity, if scores are equal, the first one encountered as 'best' is kept.
        pass

print(f"\nThe part of the code that contributes the most to the overall performance is: {best_config} (Macro F1 Score: {highest_score:.4f})")
print("Note: If multiple configurations yielded the same highest score, the one listed first is reported as 'most contributing'.")

