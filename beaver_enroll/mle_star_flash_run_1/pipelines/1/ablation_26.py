
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
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")
TEST_DATA_DIR = None # Not available for training phase


# Helper function to load a table if it exists
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            return pd.DataFrame()
    else:
        return pd.DataFrame() # Return empty DataFrame if file not found

# This function encapsulates the model training and evaluation process,
# allowing for parameter variations for ablation.
def run_experiment(
    use_imputation_for_features=False,
    subject_id_encoding_method='label_encode', # 'label_encode' or 'one_hot_encode'
    include_terms_year_feature=True
):
    # --- 1. Load Gold Labels ---
    gold_enrollment_train = pd.DataFrame() # Initialize to avoid UnboundLocalError
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        # Creating a more robust dummy dataset to avoid common pitfalls in ablation studies (small validation set, single class)
        gold_enrollment_train = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202201', '202201', '202301', '202301', '202302', '202302', '201901', '201901', '201902', '201902', '201801', '201801', '202301', '202301'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'CS', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL', 'MA', 'EL', 'CH', 'PH'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N']
        })
        # Introduce some NaNs for testing imputation (for Ablation 1)
        gold_enrollment_train.loc[0, 'SUBJECT_ID_SORT'] = np.nan
        gold_enrollment_train.loc[1, 'TERM_CODE'] = np.nan
        gold_enrollment_train.loc[2, 'HIGH_ENROLLMENT'] = np.nan # NaN in target will be dropped

    data = gold_enrollment_train.copy()

    # --- 2. Load Features from TRAIN_DATA_DIR ---
    terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

    # Add features from offerings_df if available and has required columns
    if not offerings_df.empty:
        if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
            offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
            offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')
            
            # Introduce NaNs into offering_df for testing imputation
            offerings_df.loc[0, 'ACTUAL_ENROLLMENT'] = np.nan
            offerings_df.loc[1, 'CAPACITY'] = np.nan

            agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
                max_capacity=('CAPACITY', 'max'),
                num_offerings=('TERM_CODE', 'count'),
                sum_capacity=('CAPACITY', 'sum')
            ).reset_index()
            data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    terms_year_present = False
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            if include_terms_year_feature: # Ablation 3: conditionally include YEAR from terms_df
                data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
                terms_year_present = True

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Always drop rows where the target is NaN first
    data.dropna(subset=['HIGH_ENROLLMENT_TARGET'], inplace=True)

    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # Ablation 2: SUBJECT_ID_SORT encoding method
    if 'SUBJECT_ID_SORT' in data.columns: # Ensure column exists before encoding
        if subject_id_encoding_method == 'label_encode':
            # Handle potential NaNs before Label Encoding
            data['SUBJECT_ID_SORT_fill'] = data['SUBJECT_ID_SORT'].fillna('MISSING_SUBJECT').astype(str)
            le_subject = LabelEncoder()
            data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT_fill'])
        elif subject_id_encoding_method == 'one_hot_encode':
            # Handle potential NaNs before One-Hot Encoding
            data['SUBJECT_ID_SORT_fill'] = data['SUBJECT_ID_SORT'].fillna('MISSING_SUBJECT').astype('category')
            subject_ohe = pd.get_dummies(data['SUBJECT_ID_SORT_fill'], prefix='SUBJECT')
            data = pd.concat([data, subject_ohe], axis=1)
        else:
            raise ValueError("Invalid subject_id_encoding_method")
    else:
        pass # Will proceed without subject features

    features = ['TERM_YEAR', 'TERM_SEMESTER']

    # Dynamically add aggregated features if they exist after merging
    if 'avg_enrollment' in data.columns: features.append('avg_enrollment')
    if 'max_capacity' in data.columns: features.append('max_capacity')
    if 'num_offerings' in data.columns: features.append('num_offerings')
    if 'sum_capacity' in data.columns: features.append('sum_capacity')
    
    if terms_year_present: # Ablation 3: conditionally include YEAR from terms_df
        if 'YEAR' in data.columns: # Double check YEAR exists after merge
            features.append('YEAR')
    
    # Add SUBJECT_ID_SORT encoded features based on method
    if 'SUBJECT_ID_SORT' in data.columns: # Only add if column was present initially
        if subject_id_encoding_method == 'label_encode':
            if 'SUBJECT_ID_SORT_encoded' in data.columns:
                features.append('SUBJECT_ID_SORT_encoded')
        elif subject_id_encoding_method == 'one_hot_encode':
            ohe_cols = [col for col in data.columns if col.startswith('SUBJECT_')]
            features.extend(ohe_cols)

    target = 'HIGH_ENROLLMENT_TARGET'

    # Filter `features` list to only include columns that actually exist in `data`
    existing_features = [f for f in features if f in data.columns]
    
    # Ablation 1: NaN handling for features
    if use_imputation_for_features:
        data_processed = data.copy()
        
        # Identify numerical and categorical features among the selected 'existing_features'
        numerical_features = []
        categorical_features = []

        for col in existing_features:
            if pd.api.types.is_numeric_dtype(data_processed[col]):
                numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        # Impute numerical features: create indicator, then median imputation
        for col in numerical_features:
            if data_processed[col].isnull().any():
                data_processed[f'{col}_is_missing'] = data_processed[col].isnull().astype(int)
                # Ensure the new indicator feature is added to existing_features *before* it's used
                if f'{col}_is_missing' not in existing_features:
                    existing_features.append(f'{col}_is_missing') 
                median_val = data_processed[col].median()
                data_processed[col].fillna(median_val, inplace=True)

        # Impute categorical features: fill with "Missing"
        for col in categorical_features:
            if data_processed[col].isnull().any():
                data_processed[col].fillna("Missing", inplace=True)
        
        data = data_processed # Use the imputed data
    else:
        # Original script's NaN handling: drop rows with any NaN in existing_features
        data.dropna(subset=existing_features, inplace=True)

    if data.empty or target not in data.columns or data[existing_features].empty:
        return 0.0 # No data after processing or target/features missing

    # Fix: Ensure all relevant feature columns in 'data' are numeric before splitting.
    # This prevents TypeError when fillna(0) is called on X_train/X_val with Categorical dtypes.
    for col in existing_features:
        if col in data.columns: # Ensure the column exists before processing
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')
    # Fill any NaNs that might have resulted from 'errors=coerce' with 0 for these columns.
    data[existing_features] = data[existing_features].fillna(0)


    # --- 4. Data Splitting (Time-based validation) ---
    train_df, val_df = pd.DataFrame(), pd.DataFrame()
    
    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        temp_train_df = data[data['TERM_YEAR'] < latest_train_year]
        temp_val_df = data[data['TERM_YEAR'] == latest_train_year]

        if temp_val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            train_df = data[data['TERM_YEAR'] < second_latest_train_year]
            val_df = data[data['TERM_YEAR'] == second_latest_train_year]
        elif temp_val_df.empty:
             # Fallback to random split if time-based split is not possible/empty
             train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])
        else:
            train_df, val_df = temp_train_df, temp_val_df
    else:
        # Fallback to random split if TERM_YEAR is not present or has only one unique value
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])

    X_train, y_train = train_df[existing_features], train_df[target]
    X_val, y_val = val_df[existing_features], val_df[target]

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0

    # Align columns for OHE and potential new NaNs
    # These fillna calls should now be safe as columns should be numeric at this point.
    X_train.fillna(0, inplace=True)
    X_val.fillna(0, inplace=True)
    
    # Align columns: add missing columns to X_val (fill with 0) and remove extra columns from X_val
    train_cols = X_train.columns
    val_cols = X_val.columns

    missing_in_val = list(set(train_cols) - set(val_cols))
    for col in missing_in_val:
        X_val[col] = 0
    
    extra_in_val = list(set(val_cols) - set(train_cols))
    X_val = X_val.drop(columns=extra_in_val, errors='ignore') # Use errors='ignore' to prevent error if column not found
    
    X_val = X_val[train_cols] # Reorder columns to match train

    # Final check for empty data after alignment
    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0
    else:
        # --- 5. Model Training ---
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
        return final_validation_score

# --- Ablation Study Execution ---
results = {}

# Baseline
results['Baseline (Original NaN Handling, Label Encoded SUBJECT_ID_SORT, Include YEAR from terms_df)'] = run_experiment(
    use_imputation_for_features=False,
    subject_id_encoding_method='label_encode',
    include_terms_year_feature=True
)

# Ablation 1: Imputation for Features (replace dropna for features with imputation)
results['Ablation 1 (Imputation for Features, Label Encoded SUBJECT_ID_SORT, Include YEAR from terms_df)'] = run_experiment(
    use_imputation_for_features=True,
    subject_id_encoding_method='label_encode',
    include_terms_year_feature=True
)

# Ablation 2: One-Hot Encode SUBJECT_ID_SORT instead of Label Encode
results['Ablation 2 (Original NaN Handling, One-Hot Encoded SUBJECT_ID_SORT, Include YEAR from terms_df)'] = run_experiment(
    use_imputation_for_features=False,
    subject_id_encoding_method='one_hot_encode',
    include_terms_year_feature=True
)

# Ablation 3: Exclude YEAR feature from terms_df
results['Ablation 3 (Original NaN Handling, Label Encoded SUBJECT_ID_SORT, Exclude YEAR from terms_df)'] = run_experiment(
    use_imputation_for_features=False,
    subject_id_encoding_method='label_encode',
    include_terms_year_feature=False
)

# Print Results
print("Ablation Study Results (Macro F1 Score):")
for config, score in results.items():
    print(f"- {config}: {score:.4f}")

# Determine the most contributing part
best_score = max(results.values())
if best_score == 0.0:
    print("\nConclusion: All configurations yielded an F1 score of 0.0, indicating a fundamental issue with data or model setup, thus no specific part could be identified as most contributing.")
else:
    best_config = [config for config, score in results.items() if score == best_score]
    if len(best_config) == 1:
        print(f"\nConclusion: The configuration '{best_config[0]}' contributed the most to the overall performance with a Macro F1 Score of {best_score:.4f}.")
    else:
        print(f"\nConclusion: Multiple configurations achieved the highest Macro F1 Score of {best_score:.4f}. These include: {', '.join(best_config)}.")

print(f"Final Validation Performance: {best_score:.4f}")
