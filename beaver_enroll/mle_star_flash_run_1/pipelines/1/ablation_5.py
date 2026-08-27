

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
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")
TEST_DATA_DIR = None # Not available for training phase

print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")

# --- Helper to generate dummy data for terms.csv and offerings.csv if not found ---
def generate_dummy_terms_df(gold_df):
    """Generates dummy terms data based on unique TERM_CODEs in gold_df."""
    unique_terms = gold_df['TERM_CODE'].unique()
    years = [int(str(t)[:4]) for t in unique_terms]
    return pd.DataFrame({
        'TERM_CODE': unique_terms,
        'YEAR': years,
        'DESCRIPTION': [f"Term {t}" for t in unique_terms]
    })

def generate_dummy_offerings_df(gold_df):
    """Generates dummy offerings data based on gold_df's TERM_CODE and SUBJECT_ID_SORT."""
    records = []
    # Use a fixed seed for reproducibility of dummy data
    np.random.seed(42)
    for _, row in gold_df.iterrows():
        term = row['TERM_CODE']
        subject = row['SUBJECT_ID_SORT']
        # Simulate multiple offerings per subject-term, with varying enrollments/capacities
        num_courses = np.random.randint(1, 5) # 1 to 4 courses per subject-term
        for i in range(num_courses):
            capacity = np.random.randint(20, 100)
            actual_enrollment = np.random.randint(10, capacity + 5) # Can exceed capacity slightly
            records.append({
                'TERM_CODE': term,
                'SUBJECT_ID_SORT': subject,
                'ACTUAL_ENROLLMENT': actual_enrollment,
                'CAPACITY': capacity,
                'COURSE_ID': f"{subject}{term}{i}"
            })
    return pd.DataFrame(records)


# Helper function to load a table or generate dummy if it does not exist
def load_table_or_generate_dummy(directory, filename, gold_df_for_dummy=None):
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
        if gold_df_for_dummy is not None:
            print(f"Generating dummy data for {filename}.")
            if filename == 'terms.csv':
                return generate_dummy_terms_df(gold_df_for_dummy)
            elif filename == 'offerings.csv':
                return generate_dummy_offerings_df(gold_df_for_dummy)
        print(f"No dummy generation logic for {filename}, returning empty DataFrame.")
        return pd.DataFrame() # Return empty DataFrame if file not found and no dummy generation


# Encapsulate the core logic in a function for ablation study
def run_ablation_experiment(
    exp_name: str,
    use_subject_id_encoded: bool,
    use_term_year_semester: bool,
    use_offerings_agg_features: bool
):
    print(f"\n--- Running Ablation Experiment: {exp_name} ---")

    # --- 1. Load Gold Labels ---
    # Use the sophisticated dummy data generation provided in the context, if file is not found.
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train)} rows.")
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading gold_enrollment_train.csv: {e}. Creating sophisticated dummy data for execution.")
        # Create a sophisticated dummy dataframe for development purposes with correlations and temporal trends.
        # This part is from the plan_implement_agent_1 context, directly copied.
        terms = []
        for year in range(2020, 2023):
            terms.append(f"{year}01")
            terms.append(f"{year}02")
        subjects = ['CS', 'MA', 'PH', 'EL', 'HI', 'PS', 'EN', 'AR']
        dummy_data_records = []
        # Use a fixed seed for reproducibility of dummy data
        np.random.seed(42)
        for term_code in terms:
            current_year = int(term_code[:4])
            term_type = term_code[4:]
            for subject_id in subjects:
                p_high = 0.45
                if subject_id == 'CS':
                    p_high += (current_year - 2020) * 0.08
                    p_high = min(p_high, 0.9)
                if subject_id == 'EN':
                    p_high -= (current_year - 2020) * 0.05
                    p_high = max(p_high, 0.15)
                if subject_id == 'MA' and term_type == '01':
                    p_high += 0.1
                elif subject_id == 'MA' and term_type == '02':
                    p_high -= 0.05
                if subject_id == 'PH' and term_type == '01' and current_year % 2 == 0:
                    p_high += 0.07
                if subject_id == 'AR':
                    p_high = 0.5
                if subject_id == 'HI':
                    if term_type == '02':
                        p_high = 0.4
                    else:
                        p_high = 0.55
                p_high = max(0.05, min(p_high, 0.95))
                high_enrollment_status = 'Y' if np.random.rand() < p_high else 'N'
                dummy_data_records.append({
                    'TERM_CODE': term_code,
                    'SUBJECT_ID_SORT': subject_id,
                    'HIGH_ENROLLMENT': high_enrollment_status
                })
        gold_enrollment_train = pd.DataFrame(dummy_data_records)
        print("Using sophisticated dummy gold_enrollment_train data with correlations and temporal trends for execution.")


    # --- 2. Load Features from TRAIN_DATA_DIR ---
    # Ensure dummy data is generated for terms and offerings if files are missing
    # Pass gold_enrollment_train to dummy data generators for consistent data generation
    terms_df = load_table_or_generate_dummy(TRAIN_DATA_DIR, 'terms.csv', gold_enrollment_train)
    offerings_df = load_table_or_generate_dummy(TRAIN_DATA_DIR, 'offerings.csv', gold_enrollment_train)

    data = gold_enrollment_train.copy()

    # Add features from offerings_df if available and has required columns
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
            print(f"Merged with aggregated offerings data. Data shape: {data.shape}")
        else:
            print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
    elif not use_offerings_agg_features:
        print("Ablation: Not using aggregated offerings features.")
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
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Define base features, which will be conditionally extended
    features = []

    # TERM_YEAR and TERM_SEMESTER
    if use_term_year_semester:
        data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        features.extend(['TERM_YEAR', 'TERM_SEMESTER'])
    else:
        print("Ablation: Not using TERM_YEAR and TERM_SEMESTER features.")

    # SUBJECT_ID_SORT_encoded
    if use_subject_id_encoded:
        le_subject = LabelEncoder()
        data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])
        features.append('SUBJECT_ID_SORT_encoded')
    else:
        print("Ablation: Not using SUBJECT_ID_SORT_encoded feature.")

    # Dynamically add aggregated features if they exist after merging
    if use_offerings_agg_features:
        if 'avg_enrollment' in data.columns:
            features.append('avg_enrollment')
        if 'max_capacity' in data.columns:
            features.append('max_capacity')
        if 'num_offerings' in data.columns:
            features.append('num_offerings')
        if 'sum_capacity' in data.columns:
            features.append('sum_capacity')

    # Add 'YEAR' from terms_df if it was merged
    if 'YEAR' in data.columns and 'YEAR' not in features: # Ensure no duplicates if TERM_YEAR is also used
        features.append('YEAR')


    target = 'HIGH_ENROLLMENT_TARGET'

    # Filter features to only include those that actually exist in the DataFrame
    existing_features = [f for f in features if f in data.columns]

    # Drop rows with NaN in features or target
    initial_rows = data.shape[0]
    data.dropna(subset=existing_features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

    # Check if there's enough data after dropping NaNs and if there are any features left
    if data.empty or not existing_features:
        print("Error: No data or no features remaining after processing. Cannot train model.")
        return 0.0 # Default score if training is not possible
    else:
        X = data[existing_features]
        y = data[target]

        print(f"Features used: {existing_features}")
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

        # --- 4. Data Splitting (Time-based validation with fallback) ---
        train_df, val_df = pd.DataFrame(), pd.DataFrame() # Initialize empty

        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_train_year = sorted_years[-1]

            train_df_attempt = data[data['TERM_YEAR'] < latest_train_year]
            val_df_attempt = data[data['TERM_YEAR'] == latest_train_year]

            # Adjust if latest year results in empty or single-class validation set
            if val_df_attempt.empty and len(sorted_years) > 1:
                second_latest_train_year = sorted_years[-2]
                print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
                train_df_attempt = data[data['TERM_YEAR'] < second_latest_train_year]
                val_df_attempt = data[data['TERM_YEAR'] == second_latest_train_year]

            # Final check for validity of time-based split
            if val_df_attempt.empty or len(np.unique(val_df_attempt[target])) < 2 or train_df_attempt.empty or len(np.unique(train_df_attempt[target])) < 2:
                print("Warning: Time-based split resulted in invalid train/val sets (empty or single class). Preparing for random split fallback.")
            else:
                train_df, val_df = train_df_attempt, val_df_attempt
                print(f"Using time-based split with latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
        else:
            print("Warning: 'TERM_YEAR' not available or only one year of data, or time-based split logic could not be applied. Preparing for random split fallback.")

        # Fallback to random stratified split if time-based split was not applied or was invalid
        if train_df.empty or val_df.empty:
            print("Falling back to random stratified split for validation.")
            if len(np.unique(y)) < 2: # If overall target has only one class, stratified split won't work
                print("Error: Target has only one class overall. Cannot perform stratified split.")
                return 0.0
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)


        X_train, y_train = train_df[existing_features], train_df[target]
        X_val, y_val = val_df[existing_features], val_df[target]

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

# --- Main Ablation Study Execution ---
results = {}

# Baseline: All chosen features included
results['Baseline'] = run_ablation_experiment(
    "Baseline (All features)",
    use_subject_id_encoded=True,
    use_term_year_semester=True,
    use_offerings_agg_features=True
)

# Ablation 1: No SUBJECT_ID_SORT_encoded feature
results['Ablation: No SUBJECT_ID_SORT_encoded'] = run_ablation_experiment(
    "Ablation: No SUBJECT_ID_SORT_encoded",
    use_subject_id_encoded=False,
    use_term_year_semester=True,
    use_offerings_agg_features=True
)

# Ablation 2: No TERM_YEAR and TERM_SEMESTER features
results['Ablation: No TERM_YEAR/SEMESTER'] = run_ablation_experiment(
    "Ablation: No TERM_YEAR and TERM_SEMESTER features",
    use_subject_id_encoded=True,
    use_term_year_semester=False,
    use_offerings_agg_features=True
)

# Ablation 3: No aggregated features from offerings_df
results['Ablation: No Offerings Aggregated Features'] = run_ablation_experiment(
    "Ablation: No Offerings Aggregated Features",
    use_subject_id_encoded=True,
    use_term_year_semester=True,
    use_offerings_agg_features=False
)

# --- Print Results ---
print("\n--- Ablation Study Results ---")
for exp, score in results.items():
    print(f"{exp}: Macro F1 Score = {score:.4f}")

# Determine the most impactful part
baseline_score = results['Baseline']
impacts = {}
for exp, score in results.items():
    if exp != 'Baseline':
        impact = baseline_score - score # Positive impact means score dropped when removed
        impacts[exp] = impact

if impacts:
    most_impactful_change = max(impacts, key=impacts.get)
    most_impactful_value = impacts[most_impactful_change]

    if most_impactful_value > 0.001: # Consider a change significant if > 0.001
        print(f"\nThe most impactful part (when removed, caused the largest drop in score) is: '{most_impactful_change}' with a performance drop of {most_impactful_value:.4f}.")
    elif most_impactful_value < -0.001: # If removing it improved the score
        least_impactful_change = min(impacts, key=impacts.get)
        least_impactful_value = impacts[least_impactful_change]
        print(f"\nRemoving '{least_impactful_change}' resulted in an improved score of {-least_impactful_value:.4f}, suggesting it might be detrimental or redundant with current dummy data.")
    else:
        print("\nNo single ablation caused a significant positive or negative change in performance. All ablated components seem to have negligible impact, or the dummy data is not sensitive enough.")
else:
    print("\nNo ablation experiments were conducted or evaluated.")

