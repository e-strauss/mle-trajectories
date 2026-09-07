
import os
import pandas as pd
import numpy as np
import shutil
import subprocess
import sys
import traceback
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Attempt to install and import required packages.
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
    from tabpfn import TabPFNClassifier
    from lightgbm import LGBMClassifier
except (ImportError, subprocess.CalledProcessError):
    print("Warning: Failed to install or import TabPFN/LightGBM. Using dummy classifiers.", file=sys.stderr)
    # Create dummy classes if installation fails
    class DummyClassifier:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y, *args, **kwargs): self.classes_, _ = np.unique(y, return_inverse=True)
        def predict_proba(self, X):
            probas = np.zeros((len(X), len(self.classes_)))
            if probas.shape[1] > 0: probas[:, 0] = 1.0
            return probas
    TabPFNClassifier = DummyClassifier
    LGBMClassifier = DummyClassifier

# Define constants
BASE_INPUT_DIR = './input_ablation'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

# --- Data Loading Functions (for ablation) ---

def load_data_parallel(train_dir):
    """Loads data using concurrent futures for parallel I/O."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }

    def _read_csv_worker(name, path):
        try:
            if os.path.exists(path):
                return name, pd.read_csv(path)
            return name, pd.DataFrame()
        except Exception:
            return name, pd.DataFrame()

    dataframes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paths)) as executor:
        futures = [executor.submit(_read_csv_worker, name, path) for name, path in paths.items()]
        for future in concurrent.futures.as_completed(futures):
            name, df = future.result()
            dataframes[name] = df
    return dataframes

def load_data_sequential(train_dir):
    """Loads data sequentially."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }
    dataframes = {}
    for name, path in paths.items():
        try:
            if os.path.exists(path):
                dataframes[name] = pd.read_csv(path)
            else:
                dataframes[name] = pd.DataFrame()
        except Exception:
            dataframes[name] = pd.DataFrame()
    return dataframes

# --- Core Modeling and Feature Engineering Functions ---

def feature_engineering(dfs):
    """Merges dataframes and creates features."""
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary.empty or gold_labels.empty:
        return pd.DataFrame()

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'),
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_ablation_study(config):
    """Runs a single experiment based on the provided configuration."""
    # 1. Data Loading (controlled by config)
    if config.get('use_parallel_loading', True):
        dataframes = load_data_parallel(BASE_INPUT_DIR)
    else:
        dataframes = load_data_sequential(BASE_INPUT_DIR)

    # 2. Feature Engineering
    data = feature_engineering(dataframes)
    if data.empty or data['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    # 3. Data Splitting (Time-based)
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    validation_term = sorted(data['TERM_CODE_INT'].unique())[-1]
    train_df = data[data['TERM_CODE_INT'] < validation_term]
    val_df = data[data['TERM_CODE_INT'] == validation_term]

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # 4. Model Training
    pipeline_cat_features = ['DEPARTMENT']
    pipeline_num_features = [c for c in data.select_dtypes(np.number).columns if c not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

    # RF Model
    pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42), pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    # LGBM Model
    pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42), pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm.fit(train_df, y_train)
    
    proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
    proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
    
    # Ensemble predictions
    if config.get('use_tabpfn', True):
        # TabPFN Model (simplified for this study)
        tabpfn_features = [c for c in ['NUM_COURSES', 'TOTAL_SEATS', 'NUM_FACULTY'] if c in train_df.columns]
        if not tabpfn_features:
            proba_tabpfn = np.zeros(len(val_df))
        else:
            X_train_tabpfn = train_df[tabpfn_features].fillna(0)
            X_val_tabpfn = val_df[tabpfn_features].fillna(0)
            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
            clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
            proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]
        
        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
    else:
        # Ensemble without TabPFN
        proba_ensemble = (proba_rf + proba_lgbm) / 2

    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)


def create_dummy_data():
    """Creates a more robust set of dummy data files."""
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)
    
    # Data params
    terms = [202101, 202102, 202201, 202202]
    subjects = [f'SUB-{i:02d}' for i in range(1, 21)]
    
    # subject_summary and gold_labels
    all_rows = []
    for term in terms:
        for sub in subjects:
            is_high = np.random.choice(['Y', 'N'], p=[0.4, 0.6])
            all_rows.append({'TERM_CODE': term, 'SUBJECT_ID_SORT': sub, 'HIGH_ENROLLMENT': is_high})
    
    full_df = pd.DataFrame(all_rows)
    subject_summary = full_df[['TERM_CODE', 'SUBJECT_ID_SORT']]
    gold_labels = full_df[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
    
    subject_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'subject_summary.csv'), index=False)
    gold_labels.to_csv(GOLD_LABELS_PATH, index=False)
    
    # course_attributes
    course_attrs = pd.DataFrame({
        'SUBJECT_ID_SORT': subjects,
        'COURSE_ACADEMIC_LEVEL_DESC': np.random.choice(['Undergraduate', 'Graduate'], len(subjects))
    })
    course_attrs.to_csv(os.path.join(BASE_INPUT_DIR, 'course_attributes.csv'), index=False)

    # course_summary
    course_summary_rows = []
    for term in terms:
        for sub in subjects:
            for i in range(np.random.randint(1, 5)):
                course_summary_rows.append({
                    'TERM_CODE': term, 'SUBJECT_ID_SORT': sub, 'COURSE_ID': f'{sub}-C{i}',
                    'MAX_ENROLLMENT': np.random.randint(10, 100)
                })
    pd.DataFrame(course_summary_rows).to_csv(os.path.join(BASE_INPUT_DIR, 'course_summary.csv'), index=False)

    # faculty_summary
    faculty_summary_rows = []
    for term in terms:
        for sub in subjects:
            for i in range(np.random.randint(1, 4)):
                faculty_summary_rows.append({
                    'TERM_CODE': term, 'SUBJECT_ID_SORT': sub, 'FACULTY_ID': f'F-{np.random.randint(1,50)}'
                })
    pd.DataFrame(faculty_summary_rows).to_csv(os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv'), index=False)

def main():
    """Main function to run the ablation study."""
    create_dummy_data()
    
    ablation_results = []
    
    # Baseline
    baseline_score = run_ablation_study(config={
        'use_parallel_loading': True,
        'use_tabpfn': True,
    })
    ablation_results.append({'name': 'Baseline', 'score': baseline_score, 'drop': 0.0})

    # Ablation 1: Sequential Data Loading
    score_no_parallel = run_ablation_study(config={
        'use_parallel_loading': False,
        'use_tabpfn': True,
    })
    ablation_results.append({'name': 'Ablation: Use Sequential Loading', 'score': score_no_parallel, 'drop': baseline_score - score_no_parallel})

    # Ablation 2: Missing Faculty Data
    faculty_file = os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv')
    faculty_backup = faculty_file + '.bak'
    os.rename(faculty_file, faculty_backup) # Simulate missing file
    score_no_faculty = run_ablation_study(config={
        'use_parallel_loading': True,
        'use_tabpfn': True,
    })
    os.rename(faculty_backup, faculty_file) # Restore file
    ablation_results.append({'name': 'Ablation: No Faculty Data', 'score': score_no_faculty, 'drop': baseline_score - score_no_faculty})

    # Ablation 3: No TabPFN in Ensemble
    score_no_tabpfn = run_ablation_study(config={
        'use_parallel_loading': True,
        'use_tabpfn': False,
    })
    ablation_results.append({'name': 'Ablation: No TabPFN Model', 'score': score_no_tabpfn, 'drop': baseline_score - score_no_tabpfn})

    # --- Print Results ---
    print("\n--- Ablation Study Results ---")
    print(f"{'Experiment':<35} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 80)
    for res in ablation_results:
        print(f"{res['name']:<35} | {res['score']:<20.4f} | {res['drop']:<20.4f}")

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    # Filter out baseline for finding max drop
    impactful_components = [res for res in ablation_results if res['name'] != 'Baseline' and res['drop'] > 0]
    
    if not impactful_components:
        print("No component removal resulted in a significant performance drop.")
    else:
        most_impactful = max(impactful_components, key=lambda x: x['drop'])
        component_map = {
            'Ablation: Use Sequential Loading': 'Parallel Data Loading (Integrity)',
            'Ablation: No Faculty Data': 'Faculty Summary Data',
            'Ablation: No TabPFN Model': 'TabPFN Model in Ensemble'
        }
        component_name = component_map.get(most_impactful['name'], 'Unknown Component')
        print(f"The component that contributes the most to the overall performance is: '{component_name}'")

    # Cleanup
    shutil.rmtree(BASE_INPUT_DIR)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
