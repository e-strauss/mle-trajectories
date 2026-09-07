
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback
import sys
import subprocess
from io import StringIO

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

# --- Data Loading and Feature Engineering Functions ---

def load_data(train_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'instructor_attributes': os.path.join(train_dir, 'instructor_attributes.csv'),
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

def feature_engineering(dfs, enable_lag_features=True):
    """Merges dataframes and creates features, with an option to disable lag features."""
    subject_summary = dfs.get('subject_summary', pd.DataFrame())
    gold_labels = dfs.get('gold_labels', pd.DataFrame())
    course_attributes = dfs.get('course_attributes', pd.DataFrame())
    course_summary = dfs.get('course_summary', pd.DataFrame())
    faculty_summary = dfs.get('faculty_summary', pd.DataFrame())

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files are missing.")

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if not df.empty and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'),
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'),
            AVG_SEATS=('MAX_ENROLLMENT', 'mean')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan

    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

    if enable_lag_features:
        data = data.sort_values(['SUBJECT_ID_SORT', 'TERM_CODE']).reset_index(drop=True)
        lag_features_to_create = ['TOTAL_SEATS', 'NUM_COURSES', 'NUM_FACULTY']
        for col in lag_features_to_create:
            if col in data.columns:
                data[f'LAG1_{col}'] = data.groupby('SUBJECT_ID_SORT')[col].shift(1)
                data[f'PCT_CHANGE_{col}'] = (data[col] - data[f'LAG1_{col}']) / data[f'LAG1_{col}']

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features, imputation_strategy='median'):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy=imputation_strategy)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)],
        remainder='drop'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_experiment(data, imputation_strategy='median', use_rf=True):
    """Runs a single experiment configuration and returns the F1 score."""
    # --- Data Splitting ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    if len(sorted_terms) < 2:
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term]
        val_df = data[data['TERM_CODE_INT'] == validation_term]
        if train_df.empty or val_df.empty:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

    if train_df.empty or val_df.empty or val_df['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- Feature and Model Setup ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    # LGBM
    lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbose=-1)
    pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features, imputation_strategy)
    pipeline_lgbm.fit(train_df, y_train)
    proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]

    # RF
    if use_rf:
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features, imputation_strategy)
        pipeline_rf.fit(train_df, y_train)
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
    else:
        proba_rf = 0

    # TabPFN
    tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in train_df.columns]
    tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in train_df.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfen_cat_features
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df[col])
        train_df[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df[col] = val_df[col].map(mapping).fillna(-1).astype(int)
    X_train_tabpfn = train_df[all_tabpfn_features].fillna(0).iloc[:,:100]
    X_val_tabpfn = val_df[all_tabpfn_features].fillna(0).iloc[:,:100]
    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
    proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

    # --- Ensembling and Evaluation ---
    if use_rf:
        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
    else:
        proba_ensemble = (proba_lgbm + proba_tabpfn) / 2
        
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)


def main():
    """Main function to execute the ablation study."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR)
    args = parser.parse_args()

    # --- 0. Dependency Installation & Data Loading ---
    try:
        # Redirect stdout to suppress installation messages
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        return
    finally:
        sys.stdout = original_stdout # Restore stdout

    dfs = load_data(args.train_data_dir)
    results = {}

    # --- Experiment 1: Baseline ---
    try:
        data_baseline = feature_engineering(dfs, enable_lag_features=True)
        results['Baseline (with Lag Features, RF, Median Impute)'] = run_experiment(data_baseline.copy(), imputation_strategy='median', use_rf=True)
    except Exception:
        results['Baseline (with Lag Features, RF, Median Impute)'] = 0.0

    # --- Experiment 2: No Lag Features ---
    try:
        data_no_lag = feature_engineering(dfs, enable_lag_features=False)
        results['Ablation: No Lag Features'] = run_experiment(data_no_lag.copy(), imputation_strategy='median', use_rf=True)
    except Exception:
        results['Ablation: No Lag Features'] = 0.0

    # --- Experiment 3: No RandomForest in Ensemble ---
    try:
        results['Ablation: No RandomForest Model'] = run_experiment(data_baseline.copy(), imputation_strategy='median', use_rf=False)
    except Exception:
        results['Ablation: No RandomForest Model'] = 0.0

    # --- Experiment 4: Use Mean Imputation ---
    try:
        results['Ablation: Use Mean Imputation'] = run_experiment(data_baseline.copy(), imputation_strategy='mean', use_rf=True)
    except Exception:
        results['Ablation: Use Mean Imputation'] = 0.0

    # --- Print Results and Conclusion ---
    print("\n--- Ablation Study Results ---")
    baseline_score = results.get('Baseline (with Lag Features, RF, Median Impute)', 0.0)
    
    perf_drops = {}
    for key, score in results.items():
        drop = baseline_score - score
        print(f"{key}: {score:.4f} (Performance Drop: {drop:.4f})")
        if 'Ablation' in key:
            perf_drops[key] = drop

    if not perf_drops:
        most_impactful = "N/A"
    else:
        most_impactful = max(perf_drops, key=perf_drops.get)
    
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: '{most_impactful.replace('Ablation: ', '')}'")


if __name__ == '__main__':
    try:
        # Re-importing for the main execution scope after installation
        from lightgbm import LGBMClassifier
        from tabpfn import TabPFNClassifier
        main()
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
