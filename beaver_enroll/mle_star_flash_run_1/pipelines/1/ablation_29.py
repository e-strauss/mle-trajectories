
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Constants for experiment
MIN_POSITIVE_TARGET_SAMPLES = 5 # Minimum positive samples required in validation set for meaningful F1

def run_experiment(config):
    """
    Runs the machine learning experiment with the given configuration.
    Returns the Macro F1 Score.
    """
    
    # --- 1. Load Gold Labels ---
    gold_enrollment_train = pd.DataFrame()
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        # Create a more robust dummy dataframe for development purposes
        gold_enrollment_train = pd.DataFrame({
            'TERM_CODE': [f'202{y}0{s}' for y in range(0,4) for s in [1,2] for _ in range(10)], # 4 years * 2 semesters * 10 entries = 80 rows
            'SUBJECT_ID_SORT': (['CS', 'MA', 'PH', 'EL', 'BI'] * 16),
            'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], size=80, p=[0.5, 0.5])
        })
        # Ensure sufficient 'Y' and 'N' targets for each year for time-based split robustness
        for year in gold_enrollment_train['TERM_CODE'].astype(str).str[:4].unique():
            year_mask = gold_enrollment_train['TERM_CODE'].astype(str).str[:4] == year
            if gold_enrollment_train.loc[year_mask, 'HIGH_ENROLLMENT'].str.contains('Y').sum() < 2: # At least 2 'Y's
                indices_to_change = gold_enrollment_train.loc[year_mask].sample(2, random_state=42).index
                gold_enrollment_train.loc[indices_to_change, 'HIGH_ENROLLMENT'] = 'Y'
            if gold_enrollment_train.loc[year_mask, 'HIGH_ENROLLMENT'].str.contains('N').sum() < 2: # At least 2 'N's
                indices_to_change = gold_enrollment_train.loc[year_mask].sample(2, random_state=42).index
                gold_enrollment_train.loc[indices_to_change, 'HIGH_ENROLLMENT'] = 'N'


    # --- 2. Load Features from TRAIN_DATA_DIR ---
    def load_table_if_exists(directory, filename):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

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

    # Ablation 2: Remove YEAR feature from terms_df merge
    if not config.get('remove_terms_year_merge', False) and not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)

    # Ablation 1: TERM_SEMESTER One-Hot Encoded (instead of numeric integer)
    if config.get('ohe_term_semester', False):
        data['TERM_SEMESTER_str'] = data['TERM_CODE_str'].str[4:].astype(str)
    else:
        data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    features = ['TERM_YEAR', 'SUBJECT_ID_SORT_encoded']
    
    if not config.get('ohe_term_semester', False):
        features.append('TERM_SEMESTER')
    else:
        features.append('TERM_SEMESTER_str')

    if 'avg_enrollment' in data.columns:
        features.append('avg_enrollment')
    if 'max_capacity' in data.columns:
        features.append('max_capacity')
    if 'num_offerings' in data.columns:
        features.append('num_offerings')
    if 'sum_capacity' in data.columns:
        features.append('sum_capacity')
    if 'YEAR' in data.columns and not config.get('remove_terms_year_merge', False):
        features.append('YEAR')

    target = 'HIGH_ENROLLMENT_TARGET'

    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    
    if data.empty:
        return 0.0

    X = data[features]
    y = data[target]

    # Preprocessing for OHE TERM_SEMESTER
    if config.get('ohe_term_semester', False):
        numeric_features = [f for f in features if f != 'TERM_SEMESTER_str']
        categorical_features = ['TERM_SEMESTER_str']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
        
        X_processed = preprocessor.fit_transform(X)
        X = pd.DataFrame(X_processed, index=X.index, columns=preprocessor.get_feature_names_out())
    
    if X.empty or y.empty:
        return 0.0

    # --- 4. Data Splitting (Time-based validation with robust logic from previous agent) ---
    data_for_split = data.copy()
    data_for_split = data_for_split.loc[X.index] # Ensure index matches X after preprocessing
    data_for_split[target] = y 

    X_train, y_train, X_val, y_val = None, None, None, None

    if 'TERM_YEAR' in data_for_split.columns and data_for_split['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data_for_split['TERM_YEAR'].unique())
        num_years = len(sorted_years)

        chosen_train_indices = None
        chosen_val_indices = None
        
        for i in range(1, num_years): 
            val_years = sorted_years[-i:] 
            train_end_year = val_years[0] 
            
            current_train_df_slice = data_for_split[data_for_split['TERM_YEAR'] < train_end_year]
            current_val_df_slice = data_for_split[data_for_split['TERM_YEAR'].isin(val_years)]

            if (not current_val_df_slice.empty and 
                current_train_df_slice[target].sum() > 0 and 
                current_val_df_slice[target].sum() >= MIN_POSITIVE_TARGET_SAMPLES):
                
                chosen_train_indices = current_train_df_slice.index
                chosen_val_indices = current_val_df_slice.index
                break 
        
        if chosen_train_indices is not None:
            X_train, y_train = X.loc[chosen_train_indices], y.loc[chosen_train_indices]
            X_val, y_val = X.loc[chosen_val_indices], y.loc[chosen_val_indices]
        else: 
            if len(y.unique()) < 2:
                return 0.0
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        if len(y.unique()) < 2:
            return 0.0
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return 0.0
    else:
        # --- 5. Model Training ---
        rf_params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        # Ablation 3: RandomForestClassifier min_samples_leaf
        if config.get('rf_min_samples_leaf') is not None:
            rf_params['min_samples_leaf'] = config['rf_min_samples_leaf']

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')
        return final_validation_score

# Main ablation study execution
if __name__ == "__main__":
    
    results = {}

    # Baseline Configuration
    baseline_config = {
        'description': "Baseline (TERM_SEMESTER numeric, includes YEAR from terms_df, RF default min_samples_leaf=1)",
        'ohe_term_semester': False,
        'remove_terms_year_merge': False,
        'rf_min_samples_leaf': 1
    }
    results['Baseline'] = run_experiment(baseline_config)

    # Ablation 1: TERM_SEMESTER One-Hot Encoded (instead of numeric integer)
    ablation_1_config = baseline_config.copy()
    ablation_1_config['description'] = "Ablation 1 (TERM_SEMESTER One-Hot Encoded)"
    ablation_1_config['ohe_term_semester'] = True
    results['Ablation 1'] = run_experiment(ablation_1_config)

    # Ablation 2: Remove YEAR feature from terms_df merge
    ablation_2_config = baseline_config.copy()
    ablation_2_config['description'] = "Ablation 2 (Removed YEAR feature from terms_df merge)"
    ablation_2_config['remove_terms_year_merge'] = True
    results['Ablation 2'] = run_experiment(ablation_2_config)

    # Ablation 3: RandomForestClassifier min_samples_leaf=5
    ablation_3_config = baseline_config.copy()
    ablation_3_config['description'] = "Ablation 3 (RandomForestClassifier min_samples_leaf=5)"
    ablation_3_config['rf_min_samples_leaf'] = 5
    results['Ablation 3'] = run_experiment(ablation_3_config)

    # Print results
    print("\n--- Ablation Study Results ---")
    all_configs = {
        'Baseline': baseline_config,
        'Ablation 1': ablation_1_config,
        'Ablation 2': ablation_2_config,
        'Ablation 3': ablation_3_config
    }
    for name, score in results.items():
        print(f"{all_configs[name]['description']} Macro F1 Score: {score:.4f}")

    # Determine the most contributing part
    max_score = -1.0
    best_config_name = ""
    for name, score in results.items():
        if score > max_score:
            max_score = score
            best_config_name = name
    
    if max_score == 0.0:
        print("\nConclusion: All configurations yielded a Macro F1 Score of 0.0000. This indicates a fundamental issue with data or setup, and no component can be identified as contributing meaningfully.")
    else:
        best_config = all_configs[best_config_name]
        print(f"\nConclusion: The configuration '{best_config['description']}' contributed the most to the overall performance with a Macro F1 Score of {max_score:.4f}.")
