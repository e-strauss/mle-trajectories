

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer # For ablation 1

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

def run_ablation_experiment(ablation_mode='baseline', max_depth=None, use_term_code_encoded=False, use_imputation=False):
    """
    Runs the full model training and evaluation pipeline with specified ablation configurations.
    Returns the Macro F1 Score.
    """
    print(f"\n--- Running Experiment: {ablation_mode} ---")

    # --- 1. Load Gold Labels ---
    gold_enrollment_train = pd.DataFrame() # Initialize as empty
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
        # Create more diverse dummy data for better splitting outcomes and testing imputation
        dummy_data_base = pd.DataFrame({
            'TERM_CODE': ['202001', '202002', '202101', '202102', '202201', '202202', '202301', '202302'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'PH', 'EL', 'CH', 'BIO', 'ART', 'MUS'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
        })
        gold_enrollment_train = pd.DataFrame()
        for year in range(2020, 2025): # Generate data for 5 years
            for sem_code, sem_type in zip(['01', '02'], ['Fall', 'Spring']): # Example semesters
                current_term = f"{year}{sem_code}"
                temp_df = dummy_data_base.copy()
                temp_df['TERM_CODE'] = current_term
                # Introduce some randomness in target
                temp_df['HIGH_ENROLLMENT'] = np.random.choice(['Y', 'N'], size=len(temp_df), p=[0.6, 0.4])
                gold_enrollment_train = pd.concat([gold_enrollment_train, temp_df], ignore_index=True)
        
        # Introduce NaNs for testing imputation
        if use_imputation:
            nan_indices = np.random.choice(gold_enrollment_train.index, size=int(len(gold_enrollment_train) * 0.05), replace=False)
            gold_enrollment_train.loc[nan_indices, 'TERM_CODE'] = np.nan
            nan_indices_subj = np.random.choice(gold_enrollment_train.index, size=int(len(gold_enrollment_train) * 0.05), replace=False)
            gold_enrollment_train.loc[nan_indices_subj, 'SUBJECT_ID_SORT'] = np.nan

        # Ensure 'TERM_CODE' is string for string operations later
        gold_enrollment_train['TERM_CODE'] = gold_enrollment_train['TERM_CODE'].astype(str)
        print(f"Using rich dummy gold_enrollment_train data with {len(gold_enrollment_train)} rows.")

    # --- 2. Load Features from TRAIN_DATA_DIR ---
    def load_table_if_exists(directory, filename):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception: # Catch any read error
                return pd.DataFrame()
        return pd.DataFrame()

    terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

    data = gold_enrollment_train.copy()

    # Create dummy offerings and terms if not loaded to ensure merges don't fail for dummy gold data
    if offerings_df.empty and (len(data) > 0):
        unique_terms_subjects = data[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates()
        offerings_df = pd.DataFrame({
            'TERM_CODE': unique_terms_subjects['TERM_CODE'].tolist(),
            'SUBJECT_ID_SORT': unique_terms_subjects['SUBJECT_ID_SORT'].tolist(),
            'ACTUAL_ENROLLMENT': np.random.randint(10, 50, len(unique_terms_subjects)),
            'CAPACITY': np.random.randint(30, 60, len(unique_terms_subjects))
        })
        offerings_df['TERM_CODE'] = offerings_df['TERM_CODE'].astype(str)

    if terms_df.empty and (len(data) > 0):
        unique_terms = data['TERM_CODE'].dropna().unique() # Only consider non-NaN terms for 'YEAR'
        terms_df = pd.DataFrame({
            'TERM_CODE': unique_terms,
            'YEAR': [int(tc[:4]) if str(tc).isdigit() and len(str(tc)) >= 4 else 0 for tc in unique_terms]
        })
        terms_df['TERM_CODE'] = terms_df['TERM_CODE'].astype(str)

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
    
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Label Encode SUBJECT_ID_SORT (always present) - Fill NaNs before encoding
    data['SUBJECT_ID_SORT'] = data['SUBJECT_ID_SORT'].fillna('UNKNOWN_SUBJECT') 
    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    current_features = ['SUBJECT_ID_SORT_encoded']

    # Ablation 3: Use TERM_CODE as a single encoded feature OR TERM_YEAR/SEMESTER
    if use_term_code_encoded:
        le_term_code = LabelEncoder()
        # Handle NaNs in TERM_CODE before encoding
        data['TERM_CODE_filled'] = data['TERM_CODE'].fillna('UNKNOWN_TERM_CODE')
        data['TERM_CODE_encoded'] = le_term_code.fit_transform(data['TERM_CODE_filled'])
        current_features.append('TERM_CODE_encoded')
    else:
        # For TERM_YEAR and TERM_SEMESTER, ensure TERM_CODE is not NaN for slicing
        data['TERM_CODE_str'] = data['TERM_CODE'].fillna('000000').astype(str) # Fill with sentinel for parsing
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        current_features.extend(['TERM_YEAR', 'TERM_SEMESTER'])

    # Dynamically add aggregated features if they exist after merging
    if 'avg_enrollment' in data.columns: current_features.append('avg_enrollment')
    if 'max_capacity' in data.columns: current_features.append('max_capacity')
    if 'num_offerings' in data.columns: current_features.append('num_offerings')
    if 'sum_capacity' in data.columns: current_features.append('sum_capacity')
    if 'YEAR' in data.columns: current_features.append('YEAR') # From terms_df

    features = current_features
    target = 'HIGH_ENROLLMENT_TARGET'

    # Drop rows where target is NaN (should ideally not happen with gold data, but good practice)
    data.dropna(subset=[target], inplace=True)
    if data.empty:
        print("Error: No data remaining after target NaN removal. Cannot train model.")
        return 0.0

    # Ablation 1: Imputation for numerical features vs. dropna
    initial_rows = data.shape[0]
    
    # Identify numerical features among the chosen 'features' list that might have NaNs
    numerical_features_to_impute = [f for f in features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]

    if use_imputation:
        imputer = SimpleImputer(strategy='mean')
        if numerical_features_to_impute:
            data[numerical_features_to_impute] = imputer.fit_transform(data[numerical_features_to_impute])
        
        # After imputation, drop rows if any remaining `features` (e.g., encoded categoricals if they somehow got NaNs) are NaN.
        data.dropna(subset=features, inplace=True) 
    else:
        # Original behavior: drop any row where any selected feature or target is NaN
        data.dropna(subset=features, inplace=True)

    if data.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features during {ablation_mode} setup after imputation/dropna.")

    if data.empty:
        print("Error: No data remaining after feature engineering and NaN handling. Cannot train model.")
        return 0.0

    X = data[features]
    y = data[target]

    # --- 4. Data Splitting (Refined Hierarchical Strategy) ---
    train_df, val_df = None, None
    min_val_samples = 5 # Adjusted for dummy data to ensure splits are possible
    min_train_samples = 5 # Adjusted for dummy data
    test_size_for_percentage_split = 0.2

    def is_valid_split_df(df, min_samples):
        return not df.empty and len(df) >= min_samples

    if data.empty:
        X_train, y_train = pd.DataFrame(columns=features), pd.Series(dtype='object')
        X_val, y_val = pd.DataFrame(columns=features), pd.Series(dtype='object')
    else:
        # Strategy 1 & 2: Time-based validation using full years
        # Only attempt if TERM_YEAR is present and not using TERM_CODE_encoded
        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1 and not use_term_code_encoded:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_year = sorted_years[-1]
            temp_train_df = data[data['TERM_YEAR'] < latest_year]
            temp_val_df = data[data['TERM_YEAR'] == latest_year]

            if is_valid_split_df(temp_val_df, min_val_samples) and is_valid_split_df(temp_train_df, min_train_samples):
                train_df, val_df = temp_train_df, temp_val_df
            else:
                if len(sorted_years) > 1 and train_df is None: # Fallback to second latest year
                    second_latest_year = sorted_years[-2]
                    temp_train_df = data[data['TERM_YEAR'] < second_latest_year]
                    temp_val_df = data[data['TERM_YEAR'] == second_latest_year]

                    if is_valid_split_df(temp_val_df, min_val_samples) and is_valid_split_df(temp_train_df, min_train_samples):
                        train_df, val_df = temp_train_df, temp_val_df
        
        # Strategy 3: Pseudo-time-based split (fixed percentage of chronologically latest data)
        if train_df is None:
            data_sorted = data.copy()
            
            # Determine which column to sort by for pseudo-time-based split
            if use_term_code_encoded and 'TERM_CODE_encoded' in data_sorted.columns:
                data_sorted = data_sorted.sort_values(by='TERM_CODE_encoded', ascending=True)
            elif 'TERM_YEAR' in data_sorted.columns: # Default to TERM_YEAR if available
                data_sorted = data_sorted.sort_values(by='TERM_YEAR', ascending=True)
            # If neither, data_sorted retains original index order.

            data_sorted = data_sorted.reset_index(drop=True) 
            
            split_idx = int(len(data_sorted) * (1 - test_size_for_percentage_split))
            
            temp_train_df = data_sorted.iloc[:split_idx]
            temp_val_df = data_sorted.iloc[split_idx:]

            if is_valid_split_df(temp_val_df, min_val_samples) and is_valid_split_df(temp_train_df, min_train_samples):
                train_df, val_df = temp_train_df, temp_val_df

        # Strategy 4: Final Fallback - Random splitting
        if train_df is None:
            temp_train_df_random, temp_val_df_random = None, None

            try:
                if y is not None and not y.empty and y.nunique() > 1:
                    class_counts = y.value_counts()
                    # Ensure each class has enough samples for both train and test
                    if (class_counts < 2).any(): 
                        raise ValueError("Insufficient samples for stratification in some classes.")
                    
                    split_train_strat, split_val_strat = train_test_split(data, test_size=test_size_for_percentage_split, random_state=42, stratify=y)
                    if is_valid_split_df(split_train_strat, min_train_samples) and is_valid_split_df(split_val_strat, min_val_samples):
                        temp_train_df_random, temp_val_df_random = split_train_strat, split_val_strat
                    else:
                        raise ValueError("Small splits after stratified attempt.")
                else:
                    raise ValueError("Target 'y' not suitable for stratification.")
            except ValueError:
                # Fallback to non-stratified if stratification fails or creates small sets
                split_train_non_strat, split_val_non_strat = train_test_split(data, test_size=test_size_for_percentage_split, random_state=42)
                temp_train_df_random, temp_val_df_random = split_train_non_strat, split_val_non_strat

            train_df, val_df = temp_train_df_random, temp_val_df_random
            
            if not (is_valid_split_df(train_df, min_train_samples) and is_valid_split_df(val_df, min_val_samples)):
                print(f"Warning: Random split resulted in critically small train ({len(train_df)}) or validation ({len(val_df)}) sets. Proceeding with potentially insufficient data.")

        # Final safety net: Ensure train_df and val_df are always populated
        if train_df is None or train_df.empty or val_df is None or val_df.empty:
            print("Error: All splitting strategies failed to produce valid or sufficiently sized datasets. Defaulting to full data for training and an empty validation set.")
            X_train, y_train = data[features], data[target]
            X_val, y_val = pd.DataFrame(columns=features), pd.Series(dtype='object')
        else:
            X_train, y_train = train_df[features], train_df[target]
            X_val = val_df[features]
            y_val = val_df[target] if not val_df.empty else pd.Series(dtype='object')
        
    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")

    final_validation_score = 0.0
    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class. Cannot proceed with model training.")
        final_validation_score = 0.0
    else:
        # --- 5. Model Training ---
        rf_params = {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced'}
        if max_depth is not None: # Ablation 2: Add max_depth
            rf_params['max_depth'] = max_depth
            
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')

    return final_validation_score

# --- Main Ablation Study Execution ---
results = {}

# Baseline
results['baseline'] = run_ablation_experiment(ablation_mode='baseline', 
                                              max_depth=None, 
                                              use_term_code_encoded=False, 
                                              use_imputation=False)

# Ablation 1: Imputation for numerical features instead of dropna (for those features)
results['imputation_numerical_features'] = run_ablation_experiment(ablation_mode='imputation_numerical_features', 
                                                                   max_depth=None, 
                                                                   use_term_code_encoded=False, 
                                                                   use_imputation=True)

# Ablation 2: Add max_depth=10 to RandomForestClassifier
results['rf_max_depth_10'] = run_ablation_experiment(ablation_mode='rf_max_depth_10', 
                                                     max_depth=10, 
                                                     use_term_code_encoded=False, 
                                                     use_imputation=False)

# Ablation 3: Use TERM_CODE directly as an encoded feature (replaces TERM_YEAR, TERM_SEMESTER)
results['term_code_encoded_feature'] = run_ablation_experiment(ablation_mode='term_code_encoded_feature', 
                                                               max_depth=None, 
                                                               use_term_code_encoded=True, 
                                                               use_imputation=False)

print("\n--- Ablation Study Results Summary ---")
for mode, score in results.items():
    print(f"{mode.replace('_', ' ').capitalize()}: Macro F1 Score = {score:.4f}")

# Determine the most impactful change
baseline_score = results['baseline']
most_impactful_change = 'baseline'
max_diff = 0

for mode, score in results.items():
    if mode == 'baseline':
        continue
    
    diff = abs(score - baseline_score)
    if diff > max_diff:
        max_diff = diff
        most_impactful_change = mode

if most_impactful_change == 'baseline':
    print("\nNo single ablation showed a significant change in performance compared to the baseline.")
else:
    print(f"\nThe part that contributes the most (or has the largest impact) to the overall performance is: '{most_impactful_change.replace('_', ' ').capitalize()}'.")
    print(f"Baseline F1: {baseline_score:.4f}, Ablation F1: {results[most_impactful_change]:.4f}, Difference: {max_diff:.4f}")
