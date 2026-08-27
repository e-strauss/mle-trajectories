
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
    # Increased dummy data rows and years for more robust splitting
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202201', '202201', '202202', '202301', '202301', '202302', '202302', '202401', '202401'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL', 'CS', 'MA', 'PH', 'EL', 'CS', 'MA'],
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
terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

# Create a base dataframe for merging features, starting with gold labels
base_data = gold_enrollment_train.copy()

# Add features from offerings_df if available and has required columns
if not offerings_df.empty:
    if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
        # Ensure 'ACTUAL_ENROLLMENT' and 'CAPACITY' are numeric
        offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

        # Aggregate offerings data per (TERM_CODE, SUBJECT_ID_SORT)
        agg_features_df = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            max_capacity=('CAPACITY', 'max'),
            num_offerings=('TERM_CODE', 'count'),
            sum_capacity=('CAPACITY', 'sum')
        ).reset_index()
        base_data = pd.merge(base_data, agg_features_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings data. Base data shape: {base_data.shape}")
    else:
        print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_df is empty. Proceeding with limited features.")

# Add features from terms_df if available and has required columns
if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        base_data = pd.merge(base_data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms data. Base data shape: {base_data.shape}")
    else:
        print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")


# --- Define the run_experiment function for ablation studies ---
def run_experiment(
    data_frame,
    experiment_name,
    use_missing_indicator_features=True,
    use_feature_imputation=True, # Controls whether to impute or dropna for features
    include_avg_enrollment=True,
    include_max_capacity=True,
    include_num_offerings=True,
    include_sum_capacity=True,
    include_term_semester=True,
    random_state=42
):
    print(f"\n--- Running Experiment: {experiment_name} ---")
    data = data_frame.copy()

    # --- Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Extract features from TERM_CODE
    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    
    if include_term_semester:
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # Label Encode SUBJECT_ID_SORT
    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    # Define initial features list based on parameters
    current_features = ['TERM_YEAR', 'SUBJECT_ID_SORT_encoded']
    if include_term_semester:
        current_features.append('TERM_SEMESTER')

    # Dynamically add aggregated features if they exist and are flagged to be included
    if 'avg_enrollment' in data.columns and include_avg_enrollment:
        current_features.append('avg_enrollment')
    if 'max_capacity' in data.columns and include_max_capacity:
        current_features.append('max_capacity')
    if 'num_offerings' in data.columns and include_num_offerings:
        current_features.append('num_offerings')
    if 'sum_capacity' in data.columns and include_sum_capacity:
        current_features.append('sum_capacity')
    if 'YEAR' in data.columns: # 'YEAR' from terms_df is always included if available from base merge
        current_features.append('YEAR')

    target = 'HIGH_ENROLLMENT_TARGET'

    # --- NaN Handling and Feature List Update (from plan_implement_agent_1) ---
    # Step 1: Drop rows with NaN in the target column
    initial_rows_target_dropna = data.shape[0]
    data.dropna(subset=[target], inplace=True)
    if data.shape[0] < initial_rows_target_dropna:
        print(f"Dropped {initial_rows_target_dropna - data.shape[0]} rows due to NaN in target column.")
    
    if data.empty:
        print("Error: No data remaining after dropping NaN in target. Cannot train model.")
        return 0.0

    features_for_nan_handling = list(current_features) # Copy for iteration
    new_indicator_features = [] # To store names of newly created indicator columns

    if use_feature_imputation:
        # Step 2: Implement imputation for features and add missing indicators
        for col in features_for_nan_handling:
            if col in data.columns: # Ensure column exists before checking for NaNs
                if data[col].isnull().any():
                    # Create a binary indicator column for missingness if enabled
                    if use_missing_indicator_features:
                        indicator_col_name = f'{col}_is_missing'
                        data[indicator_col_name] = data[col].isnull().astype(int)
                        new_indicator_features.append(indicator_col_name)

                    # Impute missing values based on data type
                    if pd.api.types.is_numeric_dtype(data[col]):
                        median_val = data[col].median()
                        data[col].fillna(median_val, inplace=True)
                    elif pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                        data[col].fillna('Missing', inplace=True)
        # Step 3: Update the main 'current_features' list with the new indicator columns
        current_features.extend(new_indicator_features)
    else: # If not using feature imputation, drop NaNs from features
        initial_rows_feature_dropna = data.shape[0]
        data.dropna(subset=current_features, inplace=True)
        if data.shape[0] < initial_rows_feature_dropna:
            print(f"Dropped {initial_rows_feature_dropna - data.shape[0]} rows due to NaN in features (no imputation).")
    
    # Filter features to only include those that actually exist in the DataFrame
    final_features_list = [f for f in current_features if f in data.columns]
    
    # Check if there's enough data after dropping NaNs
    if data.empty or len(final_features_list) == 0:
        print("Error: No data or no valid features remaining after NaN removal. Cannot train model.")
        return 0.0

    X = data[final_features_list]
    y = data[target]

    if X.empty or y.empty or len(y.unique()) < 2:
        print("Error: No data or target with single class after preprocessing. Cannot train model.")
        return 0.0

    print(f"Features used: {final_features_list}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- Data Splitting (Time-based validation with robust fallback) ---
    train_df, val_df = pd.DataFrame(), pd.DataFrame()

    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())
        
        # Attempt to use the latest year for validation
        for i in range(1, len(sorted_years)):
            latest_train_year_candidate = sorted_years[-i]
            train_df_candidate = data[data['TERM_YEAR'] < latest_train_year_candidate]
            val_df_candidate = data[data['TERM_YEAR'] == latest_train_year_candidate]

            if not val_df_candidate.empty and len(val_df_candidate[target].unique()) > 1 and \
               not train_df_candidate.empty and len(train_df_candidate[target].unique()) > 1:
                train_df, val_df = train_df_candidate, val_df_candidate
                print(f"Using year {latest_train_year_candidate} for validation. Training on years prior.")
                break
        
        if train_df.empty or val_df.empty: # Fallback to random split if time-based failed to produce valid sets
            print("Time-based split created problematic validation set after multiple attempts. Using random split.")
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=random_state, stratify=y)
    else:
        print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=random_state, stratify=y)


    X_train, y_train = train_df[final_features_list], train_df[target]
    X_val, y_val = val_df[final_features_list], val_df[target]

    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")
    print(f"Train target unique classes: {y_train.unique()}, Val target unique classes: {y_val.unique()}")

    final_validation_score = 0.0
    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class after all fallbacks. Cannot proceed with model training.")
    else:
        # --- Model Training ---
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        model.fit(X_train, y_train)

        # --- Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
    
    print(f"Macro F1 Score: {final_validation_score}")
    return final_validation_score

# --- Run Ablation Study ---
results = {}

# Baseline - using plan_implement_agent_1's full NaN handling
# (imputation + missing indicators) and all specified features
results['Baseline'] = run_experiment(
    base_data,
    "Baseline",
    use_missing_indicator_features=True,
    use_feature_imputation=True,
    include_avg_enrollment=True,
    include_max_capacity=True,
    include_num_offerings=True,
    include_sum_capacity=True,
    include_term_semester=True
)

# Ablation 1: No Missing Value Indicator Features
# Keep median imputation, but don't add _is_missing columns.
results['Ablation 1: No Missing Value Indicator Features'] = run_experiment(
    base_data,
    "Ablation 1: No Missing Value Indicator Features",
    use_missing_indicator_features=False, # <-- Change: Don't add indicator features
    use_feature_imputation=True,
    include_avg_enrollment=True,
    include_max_capacity=True,
    include_num_offerings=True,
    include_sum_capacity=True,
    include_term_semester=True
)

# Ablation 2: No Feature Imputation (revert to dropna for features)
# This means no imputation happens, and rows with NaNs in features are dropped.
# Missing indicators are also implicitly not used since no imputation is done.
results['Ablation 2: No Feature Imputation (dropna for features)'] = run_experiment(
    base_data,
    "Ablation 2: No Feature Imputation (dropna for features)",
    use_missing_indicator_features=False, # Implicitly False as imputation is off
    use_feature_imputation=False, # <-- Change: Turn off imputation, use dropna
    include_avg_enrollment=True,
    include_max_capacity=True,
    include_num_offerings=True,
    include_sum_capacity=True,
    include_term_semester=True
)

# Ablation 3: Simpler Aggregated Features (only num_offerings)
# Only include 'num_offerings' from offerings_df aggregations.
results['Ablation 3: Simpler Aggregated Features (only num_offerings)'] = run_experiment(
    base_data,
    "Ablation 3: Simpler Aggregated Features (only num_offerings)",
    use_missing_indicator_features=True,
    use_feature_imputation=True,
    include_avg_enrollment=False, # <-- Change: Exclude
    include_max_capacity=False, # <-- Change: Exclude
    include_num_offerings=True, # Keep this one
    include_sum_capacity=False, # <-- Change: Exclude
    include_term_semester=True
)


# --- Print Final Results and Conclusion ---
print("\n--- Ablation Study Results ---")
for name, score in results.items():
    print(f"{name}: Macro F1 Score = {score:.4f}")

best_scenario = max(results, key=results.get)
best_score = results[best_scenario]

print(f"\nConclusion: The part of the code that contributes the most to the overall performance is: {best_scenario} (Macro F1 Score: {best_score:.4f}).")
