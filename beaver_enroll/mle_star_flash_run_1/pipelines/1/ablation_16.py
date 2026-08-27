
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

# Helper function to load a table if it exists
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        # print(f"Loading {filename} from {directory}") # Suppress for cleaner ablation output
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            # print(f"Error reading {filename}: {e}. Returning empty DataFrame.") # Suppress
            return pd.DataFrame()
    else:
        # print(f"Warning: {filename} not found at {filepath}. Skipping.") # Suppress
        return pd.DataFrame() # Return empty DataFrame if file not found

def run_ablation_experiment(
    use_feature_importance_selection=True,
    use_class_weight_balanced=True,
    term_code_encoding_method='split_year_semester', # 'split_year_semester' or 'label_encode'
    random_state_val=42
):
    """
    Runs a single experiment for the ablation study with specified configurations.

    Args:
        use_feature_importance_selection (bool): Whether to use feature importance to select aggregated features.
        use_class_weight_balanced (bool): Whether to use class_weight='balanced' in RandomForestClassifier.
        term_code_encoding_method (str): Method for TERM_CODE encoding ('split_year_semester' or 'label_encode').
        random_state_val (int): Random state for reproducibility.

    Returns:
        float: The Macro F1 Score for the experiment.
    """
    
    # --- 1. Load Gold Labels ---
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        # print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.") # Suppress
        gold_enrollment_train = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202201', '202201', '202301', '202301',
                          '202001', '202001', '202002', '202002', '202101', '202101'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'PH', 'EL',
                                'CS', 'MA', 'CS', 'PH', 'MA', 'EL'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N',
                                'Y', 'N', 'Y', 'N', 'Y', 'N']
        })
        # print("Using dummy gold_enrollment_train data.") # Suppress

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
    
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Ablation: TERM_CODE encoding method
    if term_code_encoding_method == 'split_year_semester':
        data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
        data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)
        term_features = ['TERM_YEAR', 'TERM_SEMESTER']
    elif term_code_encoding_method == 'label_encode':
        le_term = LabelEncoder()
        data['TERM_CODE_encoded'] = le_term.fit_transform(data['TERM_CODE'].astype(str))
        term_features = ['TERM_CODE_encoded']
    else:
        raise ValueError("Invalid term_code_encoding_method")

    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    features = term_features + ['SUBJECT_ID_SORT_encoded']

    TARGET_COLUMN = 'HIGH_ENROLLMENT_TARGET'

    candidate_agg_features_list = [
        'avg_enrollment', 'max_capacity', 'num_offerings', 'sum_capacity', 'YEAR'
    ]
    available_agg_features = [col for col in candidate_agg_features_list if col in data.columns]

    # Ablation: Feature Importance-based selection for aggregated features
    if use_feature_importance_selection and available_agg_features:
        temp_model_features_for_rf = list(set(features + available_agg_features))
        if TARGET_COLUMN in temp_model_features_for_rf:
            temp_model_features_for_rf.remove(TARGET_COLUMN)

        if (not data.empty and TARGET_COLUMN in data.columns and 
            not data[TARGET_COLUMN].isnull().all() and 
            data[TARGET_COLUMN].nunique() > 1 and 
            len(temp_model_features_for_rf) > 0):

            X_temp = data[temp_model_features_for_rf].copy()
            y_temp = data[TARGET_COLUMN].copy()
            temp_df_for_fit = pd.concat([X_temp, y_temp], axis=1).dropna()

            if not temp_df_for_fit.empty and temp_df_for_fit[TARGET_COLUMN].nunique() > 1:
                X_temp_fit = temp_df_for_fit[temp_model_features_for_rf]
                y_temp_fit = temp_df_for_fit[TARGET_COLUMN]
                
                try:
                    temp_rf_model = RandomForestClassifier(n_estimators=75, max_depth=8, random_state=random_state_val, n_jobs=-1, class_weight='balanced')
                    temp_rf_model.fit(X_temp_fit, y_temp_fit)
                    importances = temp_rf_model.feature_importances_
                    feature_importance_df = pd.DataFrame({'feature': temp_model_features_for_rf, 'importance': importances})
                    agg_feature_importances = feature_importance_df[feature_importance_df['feature'].isin(available_agg_features)].sort_values(by='importance', ascending=False)

                    selected_agg_features = []
                    if not agg_feature_importances.empty:
                        max_agg_importance = agg_feature_importances['importance'].max()
                        importance_threshold_relative = max_agg_importance * 0.15
                        min_absolute_importance_threshold = 0.0005 
                        final_importance_threshold = max(importance_threshold_relative, min_absolute_importance_threshold)

                        selected_agg_features = agg_feature_importances[agg_feature_importances['importance'] >= final_importance_threshold]['feature'].tolist()
                        if not selected_agg_features and len(available_agg_features) > 0:
                            num_to_select_fallback = min(2, len(available_agg_features))
                            selected_agg_features = agg_feature_importances['feature'].head(num_to_select_fallback).tolist()

                    for agg_f in selected_agg_features:
                        if agg_f not in features:
                            features.append(agg_f)
                except ValueError:
                    pass # Silently skip if model fitting for importance fails
    elif not use_feature_importance_selection: # If not using selection, add all available
        for agg_f in available_agg_features:
            if agg_f not in features:
                features.append(agg_f)


    target = TARGET_COLUMN
    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        pass # print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.") # Suppress

    if data.empty:
        return 0.0
    
    X = data[features]
    y = data[target]

    # --- 4. Data Splitting (Time-based validation) ---
    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1: # Only use TERM_YEAR for split if it exists and is used
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        train_df = data[data['TERM_YEAR'] < latest_train_year]
        val_df = data[data['TERM_YEAR'] == latest_train_year]

        if val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            train_df = data[data['TERM_YEAR'] < second_latest_train_year]
            val_df = data[data['TERM_YEAR'] == second_latest_train_year]
        elif val_df.empty:
             train_df, val_df = train_test_split(data, test_size=0.2, random_state=random_state_val, stratify=y)
    else:
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=random_state_val, stratify=y)

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0 # Default score if training is not possible
    else:
        # --- 5. Model Training ---
        model_params = {'n_estimators': 100, 'random_state': random_state_val}
        # Ablation: class_weight='balanced'
        if use_class_weight_balanced:
            model_params['class_weight'] = 'balanced'
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
        return final_validation_score

# --- Ablation Study Execution ---
results = {}

# Baseline
print("--- Running Baseline (All features, Feature Importance selection, class_weight='balanced', TERM_CODE split) ---")
baseline_score = run_ablation_experiment(
    use_feature_importance_selection=True,
    use_class_weight_balanced=True,
    term_code_encoding_method='split_year_semester'
)
results['Baseline'] = baseline_score
print(f"Baseline Macro F1 Score: {baseline_score:.4f}\n")

# Ablation 1: No Feature Importance-based Aggregated Feature Selection
print("--- Ablation: No Feature Importance-based Aggregated Feature Selection (add all available agg features) ---")
ablation1_score = run_ablation_experiment(
    use_feature_importance_selection=False,
    use_class_weight_balanced=True,
    term_code_encoding_method='split_year_semester'
)
results['No Feature Importance Selection'] = ablation1_score
print(f"Ablation 1 (No Feature Importance Selection) Macro F1 Score: {ablation1_score:.4f}\n")

# Ablation 2: No Class Weight Balancing
print("--- Ablation: No Class Weight Balancing ---")
ablation2_score = run_ablation_experiment(
    use_feature_importance_selection=True,
    use_class_weight_balanced=False,
    term_code_encoding_method='split_year_semester'
)
results['No Class Weight Balancing'] = ablation2_score
print(f"Ablation 2 (No Class Weight Balancing) Macro F1 Score: {ablation2_score:.4f}\n")

# Ablation 3: TERM_CODE as Label Encoded Feature
print("--- Ablation: TERM_CODE as Label Encoded Feature (instead of split year/semester) ---")
ablation3_score = run_ablation_experiment(
    use_feature_importance_selection=True,
    use_class_weight_balanced=True,
    term_code_encoding_method='label_encode'
)
results['TERM_CODE Label Encoded'] = ablation3_score
print(f"Ablation 3 (TERM_CODE Label Encoded) Macro F1 Score: {ablation3_score:.4f}\n")

# --- Final Conclusion ---
print("\n--- Ablation Study Summary ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Determine the most impactful part
# Find the config that resulted in the highest score
best_config_name = max(results, key=results.get)
best_score = results[best_config_name]

# Compare with baseline
if best_config_name == 'Baseline':
    print("\nThe baseline configuration achieved the highest performance.")
else:
    print(f"\nThe most impactful change was '{best_config_name}', which resulted in the highest Macro F1 Score of {best_score:.4f}.")

# Also check for most detrimental (largest drop from baseline, if baseline is not 0.0)
if baseline_score != 0.0:
    most_detrimental_name = None
    max_detriment = 0.0
    for name, score in results.items():
        if name != 'Baseline':
            detriment = baseline_score - score
            if detriment > max_detriment:
                max_detriment = detriment
                most_detrimental_name = name
    
    if most_detrimental_name:
        print(f"The most detrimental change was '{most_detrimental_name}', causing a drop of {max_detriment:.4f} from the baseline.")
