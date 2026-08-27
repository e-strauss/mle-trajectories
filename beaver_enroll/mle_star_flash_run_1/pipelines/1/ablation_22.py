
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
TEST_DATA_DIR = None # Not available for training phase

print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")

# Helper function to load a table if it exists
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        # print(f"Loading {filename} from {directory}")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        # print(f"Warning: {filename} not found at {filepath}. Skipping.")
        return pd.DataFrame() # Return empty DataFrame if file not found

def run_ablation_experiment(
    gold_enrollment_train_initial,
    terms_df_initial,
    offerings_df_initial,
    use_cyclical_semester_features=True,
    nan_handling_strategy='dropna', # 'dropna' or 'mean_impute'
    use_class_weight='balanced', # 'balanced' or None
):
    # Create copies to ensure each experiment starts with clean data
    gold_enrollment_train = gold_enrollment_train_initial.copy()
    terms_df = terms_df_initial.copy()
    offerings_df = offerings_df_initial.copy()

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
        # else:
            # print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
    # else:
        # print("Warning: offerings_df is empty. Proceeding with limited features.")

    # Add features from terms_df if available and has required columns
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        # else:
            # print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # Label Encode SUBJECT_ID_SORT
    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    # Define features and target
    features = ['TERM_YEAR', 'SUBJECT_ID_SORT_encoded']

    if use_cyclical_semester_features:
        # Apply cyclical encoding to TERM_SEMESTER
        semester_period = 40 # Based on common semester codes like 10, 20, 30, 40
        data['TERM_SEMESTER_sin'] = np.sin(2 * np.pi * data['TERM_SEMESTER'] / semester_period)
        data['TERM_SEMESTER_cos'] = np.cos(2 * np.pi * data['TERM_SEMESTER'] / semester_period)
        features.extend(['TERM_SEMESTER_sin', 'TERM_SEMESTER_cos'])
    else:
        # Use raw TERM_SEMESTER if cyclical features are off
        features.append('TERM_SEMESTER')


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

    # Handle NaNs based on strategy
    initial_rows = data.shape[0]
    if nan_handling_strategy == 'dropna':
        data.dropna(subset=features + [target], inplace=True)
    elif nan_handling_strategy == 'mean_impute':
        for col in features:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else: # For other types like object, fill with mode or a placeholder
                data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'missing', inplace=True)

    if data.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target with strategy '{nan_handling_strategy}'.")


    if data.empty or target not in data.columns or not all(f in data.columns for f in features):
        print("Error: Not enough data or missing required columns after feature engineering/NaN handling.")
        return 0.0

    X = data[features]
    y = data[target]

    if X.empty or y.empty:
        print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
        return 0.0
    
    # Check for NaNs *after* imputation strategy and before splitting
    if X.isnull().sum().sum() > 0:
        print(f"Warning: NaNs still present in features after '{nan_handling_strategy}'. Dropping for safety.")
        combined_df = pd.concat([X, y], axis=1).dropna()
        X = combined_df[X.columns]
        y = combined_df[y.name]
        if X.empty or y.empty:
            print("Error: No data remaining after final NaN check. Cannot train model.")
            return 0.0

    # --- 4. Data Splitting (Time-based validation) ---
    val_df = pd.DataFrame()
    train_df = pd.DataFrame()
    
    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        train_df = data[data['TERM_YEAR'] < latest_train_year]
        val_df = data[data['TERM_YEAR'] == latest_train_year]

        if val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            # print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df = data[data['TERM_YEAR'] < second_latest_train_year]
            val_df = data[data['TERM_YEAR'] == second_latest_train_year]
        # else:
            # print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
    
    if val_df.empty or train_df.empty:
        # print("Warning: Time-based split created empty validation/train set or 'TERM_YEAR' not available/only one year of data. Using random split for validation.")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)


    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    # print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")
    # print(f"Train target unique classes: {np.unique(y_train)}, Val target unique classes: {np.unique(y_val)}")


    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        # print("Error: Training or validation set is empty, or target has only one class after all fallbacks. Cannot proceed with model training.")
        return 0.0 # Default score if training is not possible
    else:
        # --- 5. Model Training ---
        model_params = {'n_estimators': 100, 'random_state': 42}
        if use_class_weight == 'balanced':
            model_params['class_weight'] = 'balanced'
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
        return final_validation_score

# --- Main Ablation Study Execution ---

# Load initial data once for all experiments
try:
    gold_enrollment_train_initial = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train_initial)} rows.")
    if gold_enrollment_train_initial.empty:
        raise ValueError("gold_enrollment_train.csv is empty.")
    if not all(col in gold_enrollment_train_initial.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
        raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
    gold_enrollment_train_initial = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202102', '202201', '202201'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
    })
    print("Using dummy gold_enrollment_train data.")

terms_df_initial = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
if terms_df_initial.empty:
    print("Using dummy terms_df data.")
    terms_df_initial = pd.DataFrame({
        'TERM_CODE': ['202001', '202002', '202101', '202102', '202201'],
        'YEAR': [2020, 2020, 2021, 2021, 2022]
    })

offerings_df_initial = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')
if offerings_df_initial.empty:
    print("Using dummy offerings_df data.")
    offerings_df_initial = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202102', '202201', '202201'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL'],
        'ACTUAL_ENROLLMENT': [50, 20, 60, 25, 55, 15, 65, 30, 70, 20],
        'CAPACITY': [60, 30, 70, 35, 65, 25, 75, 40, 80, 30]
    })


results = {}

# Baseline
print("\n--- Running Baseline Experiment ---")
baseline_score = run_ablation_experiment(
    gold_enrollment_train_initial,
    terms_df_initial,
    offerings_df_initial,
    use_cyclical_semester_features=True,
    nan_handling_strategy='dropna',
    use_class_weight='balanced'
)
results['Baseline'] = baseline_score
print(f"Baseline F1 Score (Cyclical Semester, Dropna, Class Weight Balanced): {baseline_score}")

# Ablation 1: No Cyclical Encoding for TERM_SEMESTER (use raw TERM_SEMESTER)
print("\n--- Running Ablation 1: No Cyclical Encoding for TERM_SEMESTER ---")
ablation1_score = run_ablation_experiment(
    gold_enrollment_train_initial,
    terms_df_initial,
    offerings_df_initial,
    use_cyclical_semester_features=False, # Ablation
    nan_handling_strategy='dropna',
    use_class_weight='balanced'
)
results['No Cyclical Semester Encoding'] = ablation1_score
print(f"Ablation 1 F1 Score (Raw TERM_SEMESTER, Dropna, Class Weight Balanced): {ablation1_score}")

# Ablation 2: Mean Imputation for NaN Handling
print("\n--- Running Ablation 2: Mean Imputation for NaN Handling ---")
ablation2_score = run_ablation_experiment(
    gold_enrollment_train_initial,
    terms_df_initial,
    offerings_df_initial,
    use_cyclical_semester_features=True,
    nan_handling_strategy='mean_impute', # Ablation
    use_class_weight='balanced'
)
results['Mean Imputation NaN Handling'] = ablation2_score
print(f"Ablation 2 F1 Score (Cyclical Semester, Mean Impute, Class Weight Balanced): {ablation2_score}")

# Ablation 3: Remove class_weight='balanced' from RandomForestClassifier
print("\n--- Running Ablation 3: No Class Weight Balancing ---")
ablation3_score = run_ablation_experiment(
    gold_enrollment_train_initial,
    terms_df_initial,
    offerings_df_initial,
    use_cyclical_semester_features=True,
    nan_handling_strategy='dropna',
    use_class_weight=None # Ablation
)
results['No Class Weight Balancing'] = ablation3_score
print(f"Ablation 3 F1 Score (Cyclical Semester, Dropna, No Class Weight Balanced): {ablation3_score}")

print("\n--- Ablation Study Summary ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Determine the most impactful part
best_score = max(results.values())
if best_score == 0.0:
    print("\nConclusion: All experiments yielded an F1 score of 0.0. This indicates a fundamental issue preventing meaningful model training or evaluation, likely due to insufficient or problematic data. No specific part could be identified as contributing significantly.")
elif best_score == baseline_score:
    if len(set(results.values())) == 1:
        print("\nConclusion: All configurations, including baseline and ablations, yielded the same performance. This suggests that the ablated components do not have a measurable impact or the dataset is too limited to show differences.")
    else:
        # Check if removing any feature improved the score, if so, the baseline is not "the best contributor"
        improved_by_ablation = False
        for name, score in results.items():
            if name != 'Baseline' and score > baseline_score:
                improved_by_ablation = True
                break
        
        if improved_by_ablation:
            max_ablation_score = -1
            max_ablation_name = ""
            for name, score in results.items():
                if name != 'Baseline' and score > max_ablation_score:
                    max_ablation_score = score
                    max_ablation_name = name
            print(f"\nConclusion: The baseline configuration performed well, but some ablations led to an improvement. The best performing configuration is '{max_ablation_name}' with an F1 score of {max_ablation_score:.4f}. This suggests that the removed component ('{max_ablation_name.replace('No ', '').replace('Mean Imputation NaN Handling', 'Dropna NaN Handling').replace('No Class Weight Balancing', 'Class Weight Balancing')}') was detrimental or that the alternative ('{max_ablation_name}') was more effective.")
        else:
            print(f"\nConclusion: The baseline configuration, utilizing Cyclical Semester Features, Dropna NaN Handling, and Class Weight Balancing, contributed the most to the overall performance with an F1 score of {baseline_score:.4f}, as no ablation improved upon it.")
else:
    # An ablation performed better than the baseline
    best_config_name = max(results, key=results.get)
    # Determine if it was an "improvement by removal" or "improvement by alternative"
    if "No " in best_config_name: # e.g., 'No Cyclical Semester Encoding' or 'No Class Weight Balancing'
        original_component = best_config_name.replace("No ", "")
        print(f"\nConclusion: Removing '{original_component}' led to the highest performance. Therefore, '{original_component}' was detrimental to the model.")
    elif "Mean Imputation NaN Handling" in best_config_name:
        print(f"\nConclusion: Using 'Mean Imputation NaN Handling' for NaNs instead of 'dropna' contributed the most to the overall performance, achieving an F1 score of {best_score:.4f}.")
    else:
        print(f"\nConclusion: The configuration '{best_config_name}' contributed the most to the overall performance, achieving an F1 score of {best_score:.4f}.")

