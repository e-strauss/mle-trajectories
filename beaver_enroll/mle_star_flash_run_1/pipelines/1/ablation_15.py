

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import subprocess
import sys

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
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


# Helper function to load a table if it exists
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        # print(f"Loading {filename} from {directory}") # Suppress for cleaner ablation output
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        # print(f"Warning: {filename} not found at {filepath}. Skipping.") # Suppress for cleaner ablation output
        return pd.DataFrame() # Return empty DataFrame if file not found

def run_ablation_experiment(
    exp_name,
    include_term_derived_features=True,
    include_subject_id_sort_encoded=True,
    rf_n_estimators=100
):
    print(f"\n--- Running Experiment: {exp_name} ---")

    # --- 1. Load Gold Labels ---
    gold_enrollment_train = pd.DataFrame()
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        # print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train)} rows.")
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
        gold_enrollment_train = pd.DataFrame({
            'TERM_CODE': [202001, 202001, 202002, 202002, 202101, 202101, 202201, 202201, 202301, 202301, 202302, 202302],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'CS', 'PH', 'MA', 'EL'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
        })
        print("Using dummy gold_enrollment_train data.")


    # --- 2. Load Features from TRAIN_DATA_DIR ---
    terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

    data = gold_enrollment_train.copy()

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
            data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
            # print(f"Merged with aggregated offerings data. Data shape: {data.shape}")
        # else:
            # print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
    # else:
        # print("Warning: offerings_df is empty. Proceeding with limited features.")

    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
            # print(f"Merged with terms data. Data shape: {data.shape}")
        # else:
            # print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Initialize features list
    features = []

    if include_term_derived_features:
        data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        features.extend(['TERM_YEAR', 'TERM_SEMESTER'])
        if 'YEAR' in data.columns:
            features.append('YEAR')

    if include_subject_id_sort_encoded:
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

    target = 'HIGH_ENROLLMENT_TARGET'

    # Drop rows with NaN in features or target
    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

    if data.empty or not features or target not in data.columns:
        print("Error: No data or insufficient features/target after preprocessing. Cannot train model.")
        return 0.0 # Return 0.0 if training is not possible

    X = data[features]
    y = data[target]

    # print(f"Features used: {features}")
    # print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- 4. Data Splitting (Time-based validation with robustness checks) ---
    MIN_CLASS_SAMPLES = 2 # Minimum count of samples required for each target class in train and validation sets. Adjusted for dummy data.

    # Helper function to check minimum class samples in a series
    def _check_min_class_samples(y_series_subset, all_unique_targets_full_data, min_count):
        if y_series_subset.empty:
            return False
        class_counts = y_series_subset.value_counts()
        for target_class in all_unique_targets_full_data:
            if target_class not in class_counts or class_counts[target_class] < min_count:
                return False
        return True

    train_df, val_df = None, None
    split_successful = False
    
    all_unique_targets_in_data = y.unique() 

    if ('TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1 and len(all_unique_targets_in_data) > 1):
        sorted_years = sorted(data['TERM_YEAR'].unique())

        if len(sorted_years) >= 2:
            latest_val_year = sorted_years[-1]
            temp_train_df = data[data['TERM_YEAR'] < latest_val_year]
            temp_val_df = data[data['TERM_YEAR'] == latest_val_year]

            if not temp_val_df.empty and not temp_train_df.empty:
                temp_y_train = temp_train_df[target]
                temp_y_val = temp_val_df[target]

                if _check_min_class_samples(temp_y_train, all_unique_targets_in_data, MIN_CLASS_SAMPLES) and \
                   _check_min_class_samples(temp_y_val, all_unique_targets_in_data, MIN_CLASS_SAMPLES):
                    train_df = temp_train_df
                    val_df = temp_val_df
                    print(f"Time-based split (validation year={latest_val_year}) successful.")
                    split_successful = True
                # else:
                    # print(f"Warning: Latest year split ({latest_val_year}) failed minimum class sample check. Attempting adaptive split.")
            # else:
                # print(f"Warning: Latest year split (validation year={latest_val_year}) resulted in empty train/validation sets. Attempting adaptive split.")
        # else:
            # print("Warning: Only one year of data available, simple latest-year time-based split not possible. Attempting adaptive split.")

        if not split_successful and len(sorted_years) >= 3:
            adaptive_val_years = sorted_years[-2:]
            train_max_year = sorted_years[-3]

            temp_train_df = data[data['TERM_YEAR'] <= train_max_year]
            temp_val_df = data[data['TERM_YEAR'].isin(adaptive_val_years)]

            if not temp_val_df.empty and not temp_train_df.empty:
                temp_y_train = temp_train_df[target]
                temp_y_val = temp_val_df[target]

                if _check_min_class_samples(temp_y_train, all_unique_targets_in_data, MIN_CLASS_SAMPLES) and \
                   _check_min_class_samples(temp_y_val, all_unique_targets_in_data, MIN_CLASS_SAMPLES):
                    train_df = temp_train_df
                    val_df = temp_val_df
                    print(f"Time-based adaptive split (validation years={adaptive_val_years[0]}-{adaptive_val_years[1]}) successful.")
                    split_successful = True
                # else:
                    # print(f"Warning: Adaptive split (last two years) failed minimum class sample check. Falling back to stratified random split.")
            # else:
                # print(f"Warning: Adaptive validation set (last two years) resulted in empty train/validation sets. Falling back to stratified random split.")
        # elif not split_successful and len(sorted_years) < 3:
            # print("Warning: Not enough years (less than 3) for the adaptive 'last two years' split. Falling back to stratified random split.")
    # elif len(all_unique_targets_in_data) <= 1:
        # print("Warning: Target variable 'y' has only one unique class or is empty. Time-based stratified split is not fully applicable, and `stratify=y` will be handled in fallback.")
    # else:
        # print("Warning: 'TERM_YEAR' not available or only one unique year of data. Time-based split not possible.")

    if not split_successful:
        # print("Falling back to stratified random split for validation.")
        if len(all_unique_targets_in_data) > 1:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        else:
            # print("Warning: Target variable 'y' has only one unique class. Using simple random split as stratification is not applicable.")
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    # print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class. Cannot proceed with model training.")
        return 0.0
    else:
        # --- 5. Model Training ---
        model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
        return final_validation_score

# --- Main Ablation Study Execution ---

results = {}

# Baseline
results['Baseline (All features, n_estimators=100)'] = run_ablation_experiment(
    "Baseline (All features, n_estimators=100)",
    include_term_derived_features=True,
    include_subject_id_sort_encoded=True,
    rf_n_estimators=100
)

# Ablation 1: No TERM_CODE derived features
results['Ablation 1 (No TERM_CODE derived features)'] = run_ablation_experiment(
    "Ablation 1 (No TERM_CODE derived features)",
    include_term_derived_features=False,
    include_subject_id_sort_encoded=True,
    rf_n_estimators=100
)

# Ablation 2: No SUBJECT_ID_SORT_encoded feature
results['Ablation 2 (No SUBJECT_ID_SORT_encoded feature)'] = run_ablation_experiment(
    "Ablation 2 (No SUBJECT_ID_SORT_encoded feature)",
    include_term_derived_features=True,
    include_subject_id_sort_encoded=False,
    rf_n_estimators=100
)

# Ablation 3: Simpler RandomForest (n_estimators=10)
results['Ablation 3 (Simpler RandomForest: n_estimators=10)'] = run_ablation_experiment(
    "Ablation 3 (Simpler RandomForest: n_estimators=10)",
    include_term_derived_features=True,
    include_subject_id_sort_encoded=True,
    rf_n_estimators=10
)

print("\n--- Ablation Study Results ---")
baseline_score = results['Baseline (All features, n_estimators=100)']
print(f"Baseline F1 Score: {baseline_score:.4f}")

most_impactful_component = "None of the ablated components showed a significant impact or the data was too simple."
max_impact_change = 0
best_scenario_name = "Baseline"
best_scenario_score = baseline_score

for name, score in results.items():
    if name == 'Baseline (All features, n_estimators=100)':
        continue
    
    change = score - baseline_score
    print(f"{name}: F1 Score = {score:.4f} (Change from Baseline: {'+' if change >= 0 else ''}{change:.4f})")

    abs_change = abs(change)
    if abs_change > max_impact_change:
        max_impact_change = abs_change
        
        if score > baseline_score:
            most_impactful_component = f"Removing the {name.replace('Ablation X (No ', '').replace(' feature)', '')} was beneficial, improving performance by {change:.4f} F1 score."
        elif score < baseline_score:
            most_impactful_component = f"Removing the {name.replace('Ablation X (No ', '').replace(' feature)', '')} was detrimental, decreasing performance by {-change:.4f} F1 score."
        else:
            most_impactful_component = f"The {name.replace('Ablation X (', '').replace(')', '')} change showed no impact on performance."
        
    if score > best_scenario_score:
        best_scenario_score = score
        best_scenario_name = name

print("\n--- Conclusion ---")
if best_scenario_name != "Baseline":
    print(f"The best performing scenario was '{best_scenario_name}' with an F1 Score of {best_scenario_score:.4f}.")
    print(f"This indicates that the change in '{best_scenario_name}' (compared to baseline) contributes the most to the overall performance.")
else:
    print(f"The Baseline model performed best (or equally well), with an F1 Score of {baseline_score:.4f}.")
    print("This suggests that the ablated components are either beneficial, have no significant impact, or the chosen ablations were not impactful enough under the current data/setup.")
print(most_impactful_component)

