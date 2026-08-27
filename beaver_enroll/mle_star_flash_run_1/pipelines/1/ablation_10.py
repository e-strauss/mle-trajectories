
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

# Define paths - using dummy input for robust testing in ablation
INPUT_DIR = "./input" # This path is for context, actual files won't be read in this ablation if dummy data is used.
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")


# --- Helper function to load a table if it exists or use dummy ---
def load_table_robustly(filepath, dummy_df, required_cols, name):
    if os.path.exists(filepath):
        print(f"Loading {name} from {filepath}")
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print(f"Warning: {name} is empty. Using dummy data instead.")
                return dummy_df
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {name} missing required columns. Using dummy data instead.")
                return dummy_df
            return df
        except Exception as e:
            print(f"Error reading {name}: {e}. Using dummy data instead.")
            return dummy_df
    else:
        print(f"Warning: {name} not found at {filepath}. Using dummy data.")
        return dummy_df

# --- Define richer dummy data for robust ablation testing ---
gold_enrollment_train_dummy = pd.DataFrame({
    'TERM_CODE': ['202001', '202001', '202101', '202101', '202102', '202201', '202201', '202202', '202301', '202301', '202302', '202302'],
    'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'MA', 'PH', 'CS', 'EL', 'MA', 'EL', 'PH', 'CS', 'MA'],
    'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y']
})

offerings_df_dummy = pd.DataFrame({
    'TERM_CODE': ['202001', '202001', '202101', '202101', '202102', '202201', '202201', '202202', '202301', '202301', '202302', '202302',
                  '202001', '202101', '202102', '202201', '202201', '202301'], # Add some duplicates for aggregation
    'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'MA', 'PH', 'CS', 'EL', 'MA', 'EL', 'PH', 'CS', 'MA',
                        'CS', 'CS', 'PH', 'EL', 'MA', 'PH'],
    'ACTUAL_ENROLLMENT': [90, 30, 95, 35, 80, 40, 70, 30, 85, 45, 90, 80, 85, 92, 75, 65, 32, 40],
    'CAPACITY': [100, 50, 100, 50, 100, 50, 100, 50, 100, 50, 100, 100, 100, 100, 90, 80, 40, 50],
    'SOME_OTHER_COL': np.arange(18)
})

terms_df_dummy = pd.DataFrame({
    'TERM_CODE': ['202001', '202101', '202102', '202201', '202202', '202301', '202302'],
    'YEAR': [2020, 2021, 2021, 2022, 2022, 2023, 2023],
    'TERM_DESCRIPTION': ['Fall 2020', 'Spring 2021', 'Summer 2021', 'Fall 2022', 'Spring 2022', 'Summer 2023', 'Fall 2023']
})


# --- 1. Load Gold Labels ---
gold_enrollment_train = load_table_robustly(
    GOLD_ENROLLMENT_TRAIN_PATH,
    gold_enrollment_train_dummy,
    ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'],
    'gold_enrollment_train.csv'
)

# --- 2. Load Features from TRAIN_DATA_DIR ---
terms_df = load_table_robustly(os.path.join(TRAIN_DATA_DIR, 'terms.csv'), terms_df_dummy, ['TERM_CODE', 'YEAR'], 'terms.csv')
offerings_df = load_table_robustly(os.path.join(TRAIN_DATA_DIR, 'offerings.csv'), offerings_df_dummy, ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY'], 'offerings.csv')


# Create a base dataframe for merging features, starting with gold labels
data = gold_enrollment_train.copy()

# Add features from offerings_df if available and has required columns
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

# --- 3. Feature Engineering (Including new features from the provided snippet) ---
# Convert target to numeric
data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

# Extract features from TERM_CODE
data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

# Label Encode SUBJECT_ID_SORT
le_subject = LabelEncoder()
data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

# Create new temporal ordering feature: TERM_ORDER (NEW)
unique_terms_sorted = sorted(data['TERM_CODE'].astype(str).unique())
term_to_order = {term: i for i, term in enumerate(unique_terms_sorted)}
data['TERM_ORDER'] = data['TERM_CODE'].astype(str).map(term_to_order)
data['TERM_ORDER'].fillna(-1, inplace=True)

# Generate meaningful interaction features (NEW)
if 'avg_enrollment' in data.columns:
    data['SUBJECT_x_AVG_ENROLLMENT'] = data['SUBJECT_ID_SORT_encoded'] * data['avg_enrollment']
if 'max_capacity' in data.columns:
    data['SUBJECT_x_MAX_CAPACITY'] = data['SUBJECT_ID_SORT_encoded'] * data['max_capacity']

# Introduce ratio-based features (NEW)
if 'avg_enrollment' in data.columns and 'max_capacity' in data.columns:
    data['AVG_ENROLLMENT_RATIO_MAX_CAPACITY'] = data['avg_enrollment'] / data['max_capacity']
    data['AVG_ENROLLMENT_RATIO_MAX_CAPACITY'] = data['AVG_ENROLLMENT_RATIO_MAX_CAPACITY'].replace([np.inf, -np.inf], np.nan).fillna(0)
if 'avg_enrollment' in data.columns and 'num_offerings' in data.columns:
    data['AVG_ENROLLMENT_RATIO_NUM_OFFERINGS'] = data['avg_enrollment'] / data['num_offerings']
    data['AVG_ENROLLMENT_RATIO_NUM_OFFERINGS'] = data['AVG_ENROLLMENT_RATIO_NUM_OFFERINGS'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Define a full list of all possible features after engineering
# This list will be dynamically filtered for each ablation experiment
all_possible_features = [
    'TERM_YEAR', 'TERM_SEMESTER', 'SUBJECT_ID_SORT_encoded', 'TERM_ORDER',
    'avg_enrollment', 'max_capacity', 'num_offerings', 'sum_capacity', 'YEAR', # Merged features
    'SUBJECT_x_AVG_ENROLLMENT', 'SUBJECT_x_MAX_CAPACITY', # Interaction features
    'AVG_ENROLLMENT_RATIO_MAX_CAPACITY', 'AVG_ENROLLMENT_RATIO_NUM_OFFERINGS' # Ratio features
]

# Filter `all_possible_features` to only include columns that actually exist in `data`
initial_full_features = [f for f in all_possible_features if f in data.columns]
target = 'HIGH_ENROLLMENT_TARGET'

print(f"Data shape after full feature engineering: {data.shape}")
print(f"Initial full features available: {initial_full_features}")


# --- Ablation Study Function ---
def run_experiment(data_df, current_features, target_col, experiment_name):
    print(f"\n--- Running Experiment: {experiment_name} ---")
    
    # Drop rows with NaN in the *current* features or target for this experiment
    initial_rows = data_df.shape[0]
    data_filtered = data_df.dropna(subset=current_features + [target_col]).copy()
    if data_filtered.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data_filtered.shape[0]} rows due to NaN in features or target for this experiment.")

    if data_filtered.empty:
        print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
        return 0.0 # Return 0.0 if no data

    X = data_filtered[current_features]
    y = data_filtered[target_col]

    print(f"Features used in this experiment: {current_features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- 4. Data Splitting (Time-based validation) ---
    if 'TERM_YEAR' in data_filtered.columns and data_filtered['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data_filtered['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1] if len(sorted_years) > 0 else None

        train_df_val_split = data_filtered[data_filtered['TERM_YEAR'] < latest_train_year] if latest_train_year else pd.DataFrame()
        val_df_val_split = data_filtered[data_filtered['TERM_YEAR'] == latest_train_year] if latest_train_year else pd.DataFrame()

        if val_df_val_split.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df_val_split = data_filtered[data_filtered['TERM_YEAR'] < second_latest_train_year]
            val_df_val_split = data_filtered[data_filtered['TERM_YEAR'] == second_latest_train_year]
        elif val_df_val_split.empty:
             print("Warning: Only one or two years of data available, and time-based split created empty validation. Using random split.")
             train_df_val_split, val_df_val_split = train_test_split(data_filtered, test_size=0.01, random_state=42, stratify=y) # Small test_size for potentially limited dummy data
        else:
            print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
    else:
        print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
        # Ensure that stratify can be applied, if not, fallback to simple split
        if y.nunique() > 1:
            train_df_val_split, val_df_val_split = train_test_split(data_filtered, test_size=0.2, random_state=42, stratify=y)
        else: # Cannot stratify if only one class exists
            print("Cannot stratify target with only one class. Performing simple random split.")
            train_df_val_split, val_df_val_split = train_test_split(data_filtered, test_size=0.2, random_state=42)


    X_train, y_train = train_df_val_split[current_features], train_df_val_split[target_col]
    X_val, y_val = val_df_val_split[current_features], val_df_val_split[target_col]

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

# --- Define Ablation Groups ---
temporal_ordering_features = ['TERM_ORDER']
interaction_features = ['SUBJECT_x_AVG_ENROLLMENT', 'SUBJECT_x_MAX_CAPACITY']
ratio_features = ['AVG_ENROLLMENT_RATIO_MAX_CAPACITY', 'AVG_ENROLLMENT_RATIO_NUM_OFFERINGS']

# Filter ablation groups to only include features that actually exist in 'data'
temporal_ordering_features = [f for f in temporal_ordering_features if f in data.columns]
interaction_features = [f for f in interaction_features if f in data.columns]
ratio_features = [f for f in ratio_features if f in data.columns]


results = {}

# --- Baseline Experiment (all new features included) ---
baseline_features = initial_full_features
baseline_score = run_experiment(data, baseline_features, target, "Baseline (All New Features)")
results["Baseline"] = baseline_score

# --- Ablation 1: No Temporal Ordering Feature ---
features_no_temporal_ordering = [f for f in baseline_features if f not in temporal_ordering_features]
if len(temporal_ordering_features) > 0:
    score_no_temporal_ordering = run_experiment(data, features_no_temporal_ordering, target, "Ablation: No Temporal Ordering (TERM_ORDER)")
    results["No Temporal Ordering (TERM_ORDER)"] = score_no_temporal_ordering
else:
    print("\nSkipping 'No Temporal Ordering' ablation as feature 'TERM_ORDER' not created.")

# --- Ablation 2: No Interaction Features ---
features_no_interaction = [f for f in baseline_features if f not in interaction_features]
if len(interaction_features) > 0:
    score_no_interaction = run_experiment(data, features_no_interaction, target, "Ablation: No Interaction Features")
    results["No Interaction Features"] = score_no_interaction
else:
    print("\nSkipping 'No Interaction Features' ablation as no interaction features were created.")

# --- Ablation 3: No Ratio Features ---
features_no_ratio = [f for f in baseline_features if f not in ratio_features]
if len(ratio_features) > 0:
    score_no_ratio = run_experiment(data, features_no_ratio, target, "Ablation: No Ratio Features")
    results["No Ratio Features"] = score_no_ratio
else:
    print("\nSkipping 'No Ratio Features' ablation as no ratio features were created.")


print("\n--- Ablation Study Results ---")
for exp, score in results.items():
    print(f"{exp}: Macro F1 Score = {score:.4f}")

# Determine the most impactful part
if results:
    baseline_score = results["Baseline"]
    most_impactful_change = "None"
    max_impact_diff = 0.0

    # Find which ablation had the biggest positive or negative impact
    for exp, score in results.items():
        if exp == "Baseline":
            continue
        diff = abs(score - baseline_score)
        if diff > max_impact_diff:
            max_impact_diff = diff
            most_impactful_change = exp
    
    if most_impactful_change != "None":
        if results[most_impactful_change] > baseline_score:
            print(f"\nThe most impactful part contributing to overall performance is the *absence* of '{most_impactful_change}', as removing it improved the score by {results[most_impactful_change] - baseline_score:.4f}.")
        elif results[most_impactful_change] < baseline_score:
            print(f"\nThe most impactful part contributing to overall performance is '{most_impactful_change}', as its *presence* seems crucial (removing it decreased the score by {baseline_score - results[most_impactful_change]:.4f}).")
        else:
            print("\nNone of the ablated parts had a measurable impact on performance.")
    else:
        print("\nNone of the ablated parts had a measurable impact on performance.")
