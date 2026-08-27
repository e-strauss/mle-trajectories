

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import sys
import subprocess

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


def run_training_pipeline(
    use_subject_id_sort_encoded: bool = True,
    use_aggregated_offerings_features: bool = True,
    use_term_semester_feature: bool = True
):
    """
    Runs the full training pipeline with specified ablation configurations.
    Returns the Macro F1 Score on the validation set.
    """
    print(f"\n--- Running training pipeline with configuration: ---")
    print(f"  - Use SUBJECT_ID_SORT_encoded: {use_subject_id_sort_encoded}")
    print(f"  - Use aggregated offerings features: {use_aggregated_offerings_features}")
    print(f"  - Use TERM_SEMESTER feature: {use_term_semester_feature}")

    # --- 1. Load Gold Labels ---
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
        # Create a dummy dataframe for development purposes if file is missing or invalid.
        # Ensure enough data and years for time-based split and classification
        gold_enrollment_train = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202101', '202201', '202201', '202202', '202202', '202301', '202301', '202301'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'EL', 'PH', 'MA', 'CS', 'EL', 'MA'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
        })
        print("Using dummy gold_enrollment_train data.")

    # --- 2. Load Features from TRAIN_DATA_DIR ---
    def load_table_if_exists(directory, filename):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
                return pd.DataFrame()
        else:
            return pd.DataFrame() # Return empty DataFrame if file not found

    terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

    data = gold_enrollment_train.copy()

    # Add features from offerings_df if available and enabled by ablation config
    if use_aggregated_offerings_features and not offerings_df.empty:
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
        else:
            print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
    elif not use_aggregated_offerings_features:
        print("Skipping aggregated offerings features as per ablation configuration.")
    else:
        print("Warning: offerings_df is empty. Proceeding with limited features.")

    # Add features from terms_df if available
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        else:
            print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

    # --- 3. Feature Engineering ---
    # Convert target to numeric
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Extract features from TERM_CODE
    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    if use_term_semester_feature:
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # Label Encode SUBJECT_ID_SORT (if enabled)
    if use_subject_id_sort_encoded:
        le_subject = LabelEncoder()
        data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    # Define features for the model
    features = ['TERM_YEAR']
    if use_term_semester_feature:
        features.append('TERM_SEMESTER')
    if use_subject_id_sort_encoded:
        features.append('SUBJECT_ID_SORT_encoded')

    # Dynamically add aggregated features if they exist and are enabled
    if use_aggregated_offerings_features:
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
        return 0.0 # Return 0.0 if no data to train

    X = data[features]
    y = data[target]

    # --- 4. Data Splitting (Time-based validation) ---
    train_df, val_df = pd.DataFrame(), pd.DataFrame() # Initialize to avoid UnboundLocalError
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

# Dictionary to store ablation results
ablation_results = {}

# --- Baseline Experiment (Original `train.py` configuration) ---
print("\n--- Running Baseline Configuration ---")
baseline_score = run_training_pipeline(
    use_subject_id_sort_encoded=True,
    use_aggregated_offerings_features=True,
    use_term_semester_feature=True
)
ablation_results["Baseline (Original `train.py`)"] = baseline_score
print(f"Baseline Macro F1 Score: {baseline_score:.4f}")

# --- Ablation 1: Remove SUBJECT_ID_SORT_encoded feature ---
print("\n--- Running Ablation: Remove SUBJECT_ID_SORT_encoded feature ---")
ablation1_score = run_training_pipeline(
    use_subject_id_sort_encoded=False,
    use_aggregated_offerings_features=True,
    use_term_semester_feature=True
)
ablation_results["No SUBJECT_ID_SORT_encoded"] = ablation1_score
print(f"Ablation (No SUBJECT_ID_SORT_encoded) Macro F1 Score: {ablation1_score:.4f}")

# --- Ablation 2: Remove aggregated features from offerings_df ---
print("\n--- Running Ablation: Remove aggregated offerings features ---")
ablation2_score = run_training_pipeline(
    use_subject_id_sort_encoded=True,
    use_aggregated_offerings_features=False,
    use_term_semester_feature=True
)
ablation_results["No aggregated offerings features"] = ablation2_score
print(f"Ablation (No aggregated offerings features) Macro F1 Score: {ablation2_score:.4f}")

# --- Ablation 3: Remove TERM_SEMESTER feature ---
print("\n--- Running Ablation: Remove TERM_SEMESTER feature ---")
ablation3_score = run_training_pipeline(
    use_subject_id_sort_encoded=True,
    use_aggregated_offerings_features=True,
    use_term_semester_feature=False
)
ablation_results["No TERM_SEMESTER feature"] = ablation3_score
print(f"Ablation (No TERM_SEMESTER feature) Macro F1 Score: {ablation3_score:.4f}")

print("\n--- Ablation Study Summary ---")
for name, score in ablation_results.items():
    print(f"{name}: {score:.4f}")

# Determine the most impactful component
baseline = ablation_results["Baseline (Original `train.py`)"]

if baseline == 0.0:
    print("\nNote: Baseline performance is 0.0, which often indicates an issue with data availability or split for meaningful comparison, especially with dummy data.")
    print("Cannot confidently determine the most impactful component when baseline is 0.0.")
else:
    impacts = {}
    for name, score in ablation_results.items():
        if name != "Baseline (Original `train.py`)":
            impact = score - baseline # Positive means improvement from ablation, negative means degradation
            impacts[name] = impact

    if not impacts:
        print("\nNo ablation experiments were run beyond baseline.")
    else:
        # Find the ablation with the largest absolute impact
        most_impactful_change = max(impacts, key=lambda k: abs(impacts[k]))
        impact_value = impacts[most_impactful_change]

        if impact_value < 0:
            print(f"\nThe most impactful change was removing '{most_impactful_change}'. This decreased performance by {abs(impact_value):.4f} Macro F1 Score.")
            print(f"Therefore, the '{most_impactful_change.replace('No ', '')}' component contributes positively the most to the overall performance.")
        elif impact_value > 0:
            print(f"\nThe most impactful change was removing '{most_impactful_change}'. This increased performance by {impact_value:.4f} Macro F1 Score.")
            print(f"Therefore, the '{most_impactful_change.replace('No ', '')}' component contributes negatively the most (its removal is beneficial) to the overall performance.")
        else:
            print(f"\nRemoving '{most_impactful_change}' had no measurable impact on performance.")
            print("Based on this study, no single ablated component showed a significantly larger positive or negative contribution compared to others that resulted in a score of 0.0 or showed no change.")

