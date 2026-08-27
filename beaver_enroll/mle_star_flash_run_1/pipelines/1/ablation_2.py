

import pandas as pd
import numpy as np
import os
import sys
import subprocess
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
    print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# Define paths (relative to the script)
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

# --- Dummy Data Generation (if files don't exist) ---
# This ensures the script can run independently for the ablation study
def generate_dummy_data_for_ablation():
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    print("Generating dummy data for ablation study...")

    # Dummy terms.csv
    terms_data = {
        'TERM_CODE': ['202001', '202002', '202101', '202102', '202201', '202202', '202301'],
        'YEAR': [2020, 2020, 2021, 2021, 2022, 2022, 2023],
        'TERM_NAME': ['Fall 2020', 'Spring 2020', 'Fall 2021', 'Spring 2021', 'Fall 2022', 'Spring 2022', 'Fall 2023']
    }
    terms_df_dummy = pd.DataFrame(terms_data)
    terms_df_dummy.to_csv(os.path.join(TRAIN_DATA_DIR, 'terms.csv'), index=False)
    print("Generated dummy terms.csv")

    # Dummy offerings.csv
    offerings_data = []
    common_terms = ['202001', '202002', '202101', '202102', '202201', '202202', '202301']
    common_subjects = ['CS', 'MA', 'PH', 'EL', 'BI']
    for term in common_terms:
        for subject in common_subjects:
            num_classes = np.random.randint(1, 4)
            for _ in range(num_classes):
                offerings_data.append({
                    'TERM_CODE': term,
                    'SUBJECT_ID_SORT': subject,
                    'ACTUAL_ENROLLMENT': np.random.randint(10, 100),
                    'CAPACITY': np.random.randint(20, 120),
                    'COURSE_NUMBER': np.random.randint(100, 500)
                })
    offerings_df_dummy = pd.DataFrame(offerings_data)
    offerings_df_dummy.to_csv(os.path.join(TRAIN_DATA_DIR, 'offerings.csv'), index=False)
    print("Generated dummy offerings.csv")

    # Dummy gold_enrollment_train.csv
    gold_data = []
    for term in common_terms:
        for subject in common_subjects:
            # Simulate high/low enrollment
            if np.random.rand() > 0.5:
                gold_data.append({
                    'TERM_CODE': term,
                    'SUBJECT_ID_SORT': subject,
                    'HIGH_ENROLLMENT': 'Y' if np.random.rand() > 0.4 else 'N'
                })
    gold_enrollment_df_dummy = pd.DataFrame(gold_data)
    gold_enrollment_df_dummy.to_csv(GOLD_ENROLLMENT_TRAIN_PATH, index=False)
    print("Generated dummy gold_enrollment_train.csv")

# Check if dummy data is needed for the original script's expected files
if not os.path.exists(GOLD_ENROLLMENT_TRAIN_PATH) or \
   not os.path.exists(os.path.join(TRAIN_DATA_DIR, 'offerings.csv')) or \
   not os.path.exists(os.path.join(TRAIN_DATA_DIR, 'terms.csv')):
    generate_dummy_data_for_ablation()


# Main function to run the model with different configurations
def run_ablation_experiment(
    use_offerings_agg_features: bool = True,
    use_term_year_semester_features: bool = True,
    use_terms_year_feature: bool = True,
    force_random_split: bool = False
) -> float:

    # --- 1. Load Gold Labels ---
    gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    if gold_enrollment_train.empty or not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
        # Fallback for critical error, in ablation scenario, this is a fatal problem
        print("Error: gold_enrollment_train.csv is empty or missing required columns. Cannot proceed.")
        return 0.0

    # --- 2. Load Features from TRAIN_DATA_DIR ---
    # Helper function to load a table if it exists (internal to avoid global state issues in ablation)
    def load_table_if_exists_internal(directory, filename):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return pd.DataFrame()

    terms_df = load_table_if_exists_internal(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df = load_table_if_exists_internal(TRAIN_DATA_DIR, 'offerings.csv')

    data = gold_enrollment_train.copy()

    # Add features from offerings_df if enabled and available
    if use_offerings_agg_features and not offerings_df.empty:
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
    
    # Add features from terms_df if enabled and available
    if use_terms_year_feature and not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    features = []
    if use_term_year_semester_features:
        data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        features.extend(['TERM_YEAR', 'TERM_SEMESTER'])
    else: # If not using these, create dummy columns to prevent errors in subsequent merge
        data['TERM_YEAR'] = 0 # Placeholder for time-based split logic
        data['TERM_SEMESTER'] = 0

    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])
    features.append('SUBJECT_ID_SORT_encoded')

    # Dynamically add aggregated features if they exist and are enabled
    if use_offerings_agg_features and 'avg_enrollment' in data.columns:
        features.extend(['avg_enrollment', 'max_capacity', 'num_offerings', 'sum_capacity'])
    if use_terms_year_feature and 'YEAR' in data.columns:
        features.append('YEAR')

    target = 'HIGH_ENROLLMENT_TARGET'

    # Drop rows with NaN in features or target
    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        pass # Suppress verbose output for ablation runs

    if data.empty:
        return 0.0 # Not enough data for training

    X = data[features]
    y = data[target]

    if X.empty or y.empty or len(y.unique()) < 2:
        return 0.0 # Not enough data or target classes for training

    # --- 4. Data Splitting (Time-based validation or forced random) ---
    X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()

    if force_random_split:
        # Force random split for ablation
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
    else:
        # Original time-based split logic
        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_train_year = sorted_years[-1]

            train_df = data[data['TERM_YEAR'] < latest_train_year]
            val_df = data[data['TERM_YEAR'] == latest_train_year]

            if val_df.empty and len(sorted_years) > 1:
                second_latest_train_year = sorted_years[-2]
                train_df = data[data['TERM_YEAR'] < second_latest_train_year]
                val_df = data[data['TERM_YEAR'] == second_latest_train_year]
            elif val_df.empty: # Fallback if even latest year for val is empty, or only 1 year.
                 train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        else: # 'TERM_YEAR' not available or only one year of data.
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)

        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0 # Not enough data for a valid train/val split

    # --- 5. Model Training ---
    # Model with class_weight='balanced' as it was found to be impactful in previous study
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # --- 6. Evaluation ---
    val_predictions = model.predict(X_val)
    final_validation_score = f1_score(y_val, val_predictions, average='macro')
    return final_validation_score

# --- Run Ablation Experiments ---
results = {}

# Baseline
print("--- Running Baseline (Original Configuration) ---")
results['Baseline'] = run_ablation_experiment()
print(f"Baseline Macro F1 Score: {results['Baseline']:.4f}\n")

# Ablation 1: No Aggregated Offerings Features
print("--- Running Ablation: No Aggregated Offerings Features ---")
results['No Aggregated Offerings Features'] = run_ablation_experiment(
    use_offerings_agg_features=False
)
print(f"No Aggregated Offerings Features Macro F1 Score: {results['No Aggregated Offerings Features']:.4f}\n")

# Ablation 2: No TERM_YEAR and TERM_SEMESTER Features
print("--- Running Ablation: No TERM_YEAR and TERM_SEMESTER Features ---")
results['No TERM_YEAR/SEMESTER Features'] = run_ablation_experiment(
    use_term_year_semester_features=False
)
print(f"No TERM_YEAR/SEMESTER Features Macro F1 Score: {results['No TERM_YEAR/SEMESTER Features']:.4f}\n")

# Ablation 3: No YEAR feature from terms_df
print("--- Running Ablation: No YEAR feature from terms_df ---")
results['No Terms YEAR Feature'] = run_ablation_experiment(
    use_terms_year_feature=False
)
print(f"No Terms YEAR Feature Macro F1 Score: {results['No Terms YEAR Feature']:.4f}\n")

# Ablation 4: Force Random Split
print("--- Running Ablation: Force Random Split ---")
results['Force Random Split'] = run_ablation_experiment(
    force_random_split=True
)
print(f"Force Random Split Macro F1 Score: {results['Force Random Split']:.4f}\n")

# --- Summarize Results ---
print("\n--- Ablation Study Summary ---")
baseline_score = results['Baseline']
most_impactful_component = ""
max_drop = -1.0 # Initialize with a value less than any possible drop (scores are >=0)

for name, score in results.items():
    if name == 'Baseline':
        continue
    drop = baseline_score - score
    print(f"- {name}: Macro F1 Score = {score:.4f} (Change from Baseline: {drop:+.4f})")
    if drop > max_drop:
        max_drop = drop
        most_impactful_component = name

print(f"\nBaseline Macro F1 Score: {baseline_score:.4f}")
if most_impactful_component and max_drop > 0:
    print(f"The part of the code that contributes the most to the overall performance is: '{most_impactful_component}' (caused a drop of {max_drop:.4f} when removed/modified).")
elif max_drop <= 0:
    print("All ablated components either had no negative impact or slightly improved performance (unlikely for ablation on critical features).")

