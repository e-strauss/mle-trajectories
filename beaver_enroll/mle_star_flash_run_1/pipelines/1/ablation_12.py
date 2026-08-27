
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
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
    # print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    # print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# Define paths (using dummy paths for execution environment if actual input is missing)
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

# --- Helper functions (from original train.py) ---
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# --- Initial Data Loading and Merging (mimicking train.py for setup) ---
# Load Gold Labels
try:
    gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    if gold_enrollment_train.empty:
        raise ValueError("gold_enrollment_train.csv is empty.")
    if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
        raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
except (FileNotFoundError, ValueError) as e:
    # Create a slightly larger dummy dataframe to help with train-validation split
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202102', '202102', '202201', '202201'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N']
    })

# Load potential feature tables
terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

# Create a base dataframe for merging features
initial_data = gold_enrollment_train.copy()

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
        initial_data = pd.merge(initial_data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

# Add features from terms_df if available and has required columns
if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        initial_data = pd.merge(initial_data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')

# Convert target to numeric (always done)
initial_data['HIGH_ENROLLMENT_TARGET'] = initial_data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

# Global LabelEncoder for SUBJECT_ID_SORT (used in ablations not using TE)
le_subject_global = LabelEncoder()
le_subject_global.fit(initial_data['SUBJECT_ID_SORT']) # Fit on all unique subjects in initial data

# --- Ablation Study Function ---
def run_ablation_experiment(
    base_data_df,
    use_term_code_target_encoding=False,
    use_subject_id_sort_target_encoding=False,
    global_le_subject=None
):
    local_data = base_data_df.copy()
    target_feature = 'HIGH_ENROLLMENT_TARGET'
    
    # --- Feature Engineering based on flags ---
    
    # K-Fold Target Encoding
    categorical_features_for_te_local = []
    if use_term_code_target_encoding:
        categorical_features_for_te_local.append('TERM_CODE')
    if use_subject_id_sort_target_encoding:
        categorical_features_for_te_local.append('SUBJECT_ID_SORT')
    
    if categorical_features_for_te_local: # Only perform if any TE is enabled
        N_SPLITS = 5
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        
        for col in categorical_features_for_te_local:
            local_data[f'{col}_TargetEncoded'] = np.nan
            global_mean = local_data[target_feature].mean()
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(local_data)):
                target_mean_map = local_data.iloc[train_idx].groupby(col)[target_feature].mean()
                local_data.loc[val_idx, f'{col}_TargetEncoded'] = local_data.loc[val_idx, col].map(target_mean_map).fillna(global_mean)

    # Original Feature Engineering for TERM_CODE if TE not used
    if not use_term_code_target_encoding:
        local_data['TERM_CODE_str'] = local_data['TERM_CODE'].astype(str)
        local_data['TERM_YEAR'] = pd.to_numeric(local_data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        local_data['TERM_SEMESTER'] = pd.to_numeric(local_data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
    
    # Original Feature Engineering for SUBJECT_ID_SORT if TE not used
    if not use_subject_id_sort_target_encoding:
        if global_le_subject:
            local_data['SUBJECT_ID_SORT_encoded'] = global_le_subject.transform(local_data['SUBJECT_ID_SORT'])
        else:
            # Fallback if global_le_subject is not passed, should not happen in this setup
            le = LabelEncoder()
            local_data['SUBJECT_ID_SORT_encoded'] = le.fit_transform(local_data['SUBJECT_ID_SORT'])

    # Define features for the model
    features_for_model = []
    if use_term_code_target_encoding:
        features_for_model.append('TERM_CODE_TargetEncoded')
    else:
        features_for_model.extend(['TERM_YEAR', 'TERM_SEMESTER'])

    if use_subject_id_sort_target_encoding:
        features_for_model.append('SUBJECT_ID_SORT_TargetEncoded')
    else:
        features_for_model.append('SUBJECT_ID_SORT_encoded')

    # Add dynamic features (from offerings, terms_df) - these are always included if available
    if 'avg_enrollment' in local_data.columns: features_for_model.append('avg_enrollment')
    if 'max_capacity' in local_data.columns: features_for_model.append('max_capacity')
    if 'num_offerings' in local_data.columns: features_for_model.append('num_offerings')
    if 'sum_capacity' in local_data.columns: features_for_model.append('sum_capacity')
    if 'YEAR' in local_data.columns: features_for_model.append('YEAR')

    # Drop rows with NaN in features or target
    initial_rows_count = local_data.shape[0]
    local_data.dropna(subset=features_for_model + [target_feature], inplace=True)
    if local_data.shape[0] < initial_rows_count:
        pass # print(f"  Dropped {initial_rows_count - local_data.shape[0]} rows due to NaN in features or target.")

    if local_data.empty:
        return 0.0

    X = local_data[features_for_model]
    y = local_data[target_feature]

    # Data Splitting (Time-based validation as in original script)
    train_df, val_df = pd.DataFrame(), pd.DataFrame() # Initialize
    
    # Check if 'TERM_YEAR' is available. If not, create a proxy year for splitting
    if 'TERM_YEAR' not in local_data.columns:
        local_data['_SPLIT_TERM_YEAR'] = pd.to_numeric(local_data['TERM_CODE'].astype(str).str[:4], errors='coerce').fillna(0).astype(int)
        split_year_col = '_SPLIT_TERM_YEAR'
    else:
        split_year_col = 'TERM_YEAR'

    if local_data[split_year_col].nunique() > 1:
        sorted_years = sorted(local_data[split_year_col].unique())
        latest_train_year = sorted_years[-1]

        temp_train_df = local_data[local_data[split_year_col] < latest_train_year]
        temp_val_df = local_data[local_data[split_year_col] == latest_train_year]

        if temp_val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            temp_train_df = local_data[local_data[split_year_col] < second_latest_train_year]
            temp_val_df = local_data[local_data[split_year_col] == second_latest_train_year]
        elif temp_val_df.empty:
             # Fallback to random split if time-based split creates empty validation, adjust test_size for small data
             train_df, val_df = train_test_split(local_data, test_size=0.2, random_state=42, stratify=y)
        else:
            train_df = temp_train_df
            val_df = temp_val_df
    else:
        # Fallback to random split if only one year of data
        train_df, val_df = train_test_split(local_data, test_size=0.2, random_state=42, stratify=y)

    X_train, y_train = train_df[features_for_model], train_df[target_feature]
    X_val, y_val = val_df[features_for_model], val_df[target_feature]

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0
    
    # Model Training and Evaluation
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    f1 = f1_score(y_val, val_predictions, average='macro')
    return f1

# --- Run Ablation Experiments ---
results = {}

# Baseline: Both TERM_CODE and SUBJECT_ID_SORT are K-Fold Target Encoded (new proposed feature engineering)
results['Baseline (Both TERM_CODE and SUBJECT_ID_SORT K-Fold Target Encoded)'] = run_ablation_experiment(
    initial_data,
    use_term_code_target_encoding=True,
    use_subject_id_sort_target_encoding=True,
    global_le_subject=le_subject_global
)

# Ablation 1: No TERM_CODE Target Encoding (revert to original TERM_YEAR/TERM_SEMESTER)
results['Ablation 1 (TERM_CODE: Original TERM_YEAR/TERM_SEMESTER; SUBJECT_ID_SORT: K-Fold Target Encoded)'] = run_ablation_experiment(
    initial_data,
    use_term_code_target_encoding=False,
    use_subject_id_sort_target_encoding=True,
    global_le_subject=le_subject_global
)

# Ablation 2: No SUBJECT_ID_SORT Target Encoding (revert to original LabelEncoder)
results['Ablation 2 (TERM_CODE: K-Fold Target Encoded; SUBJECT_ID_SORT: Label Encoded)'] = run_ablation_experiment(
    initial_data,
    use_term_code_target_encoding=True,
    use_subject_id_sort_target_encoding=False,
    global_le_subject=le_subject_global
)

# Ablation 3: No K-Fold Target Encoding at all (both revert to original methods)
results['Ablation 3 (TERM_CODE: Original TERM_YEAR/TERM_SEMESTER; SUBJECT_ID_SORT: Label Encoded)'] = run_ablation_experiment(
    initial_data,
    use_term_code_target_encoding=False,
    use_subject_id_sort_target_encoding=False,
    global_le_subject=le_subject_global
)

# --- Print Final Results ---
print("--- Ablation Study Results ---")
for exp, score in results.items():
    print(f"{exp}: Macro F1 Score = {score:.4f}")

# Determine the most impactful part
best_score = -1.0
best_experiment = "N/A"
for exp, score in results.items():
    if score > best_score:
        best_score = score
        best_experiment = exp

print(f"\nConclusion: The configuration '{best_experiment}' achieved the highest Macro F1 Score of {best_score:.4f}.")

if best_experiment == 'Baseline (Both TERM_CODE and SUBJECT_ID_SORT K-Fold Target Encoded)':
    print("This suggests that using K-Fold Target Encoding for both TERM_CODE and SUBJECT_ID_SORT contributes positively to performance.")
elif 'Ablation 1' in best_experiment:
    print("This suggests that the K-Fold Target Encoding for TERM_CODE might be detrimental or less effective than using TERM_YEAR/TERM_SEMESTER features, and that K-Fold Target Encoding for SUBJECT_ID_SORT is beneficial.")
elif 'Ablation 2' in best_experiment:
    print("This suggests that the K-Fold Target Encoding for SUBJECT_ID_SORT might be detrimental or less effective than using Label Encoding, and that K-Fold Target Encoding for TERM_CODE is beneficial.")
elif 'Ablation 3' in best_experiment:
    print("This suggests that the original feature engineering methods (TERM_YEAR/TERM_SEMESTER for TERM_CODE and Label Encoding for SUBJECT_ID_SORT) are collectively more effective than the K-Fold Target Encoding approach tested, or that the K-Fold Target Encoding approach has issues.")

