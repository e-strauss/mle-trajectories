
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
from sklearn.linear_model import LogisticRegression
import traceback
import sys
import subprocess
import io

# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data_files():
    """Creates minimal dummy data files required for the script to run."""
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)
    
    # Gold Labels
    gold_data = """TERM_CODE,SUBJECT_ID_SORT,HIGH_ENROLLMENT
202210,CS-101,Y
202210,MATH-203,N
202220,CS-101,N
202220,HIST-301,Y
202310,CS-101,Y
202310,MATH-203,Y
202310,HIST-301,N
"""
    with open(GOLD_LABELS_PATH, 'w') as f:
        f.write(gold_data)

    # Subject Summary
    subject_summary_data = """TERM_CODE,SUBJECT_ID_SORT
202210,CS-101
202210,MATH-203
202220,CS-101
202220,HIST-301
202310,CS-101
202310,MATH-203
202310,HIST-301
"""
    with open(os.path.join(DEFAULT_TRAIN_DIR, 'subject_summary.csv'), 'w') as f:
        f.write(subject_summary_data)

    # Course Attributes
    course_attr_data = """SUBJECT_ID_SORT,CAMPUS_ID_DESC,DEPARTMENT_ID_DESC
CS-101,Main Campus,Computer Science
MATH-203,Main Campus,Mathematics
HIST-301,Online,History
"""
    with open(os.path.join(DEFAULT_TRAIN_DIR, 'course_attributes.csv'), 'w') as f:
        f.write(course_attr_data)
        
    # Course Summary
    course_summary_data = """TERM_CODE,SUBJECT_ID_SORT,COURSE_ID,MAX_ENROLLMENT
202210,CS-101,CS-101-A,100
202210,CS-101,CS-101-B,120
202220,CS-101,CS-101-A,110
202310,CS-101,CS-101-A,150
202310,MATH-203,MATH-203-A,50
"""
    with open(os.path.join(DEFAULT_TRAIN_DIR, 'course_summary.csv'), 'w') as f:
        f.write(course_summary_data)

    # Faculty Summary (can be empty to test robustness)
    faculty_summary_data = """TERM_CODE,SUBJECT_ID_SORT,FACULTY_ID,NUM_COURSES_TAUGHT
202310,CS-101,prof_x,2
202310,MATH-203,prof_y,1
"""
    with open(os.path.join(DEFAULT_TRAIN_DIR, 'faculty_summary.csv'), 'w') as f:
        f.write(faculty_summary_data)
        
    # Instructor attributes (optional)
    with open(os.path.join(DEFAULT_TRAIN_DIR, 'instructor_attributes.csv'), 'w') as f:
        f.write("FACULTY_ID,TENURE_STATUS\nprof_x,Tenured")


def run_ablation_study(config):
    """
    Runs a single instance of the training pipeline with a specific configuration.
    This function encapsulates the logic from the original train.py script.
    """
    use_stacking_ensemble = config.get('use_stacking_ensemble', False)
    n_jobs = config.get('n_jobs', -1)
    omit_course_summary = config.get('omit_course_summary', False)

    # --- 1. Data Loading ---
    def load_data(train_dir):
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
                # Ablation specific logic
                if name == 'course_summary' and omit_course_summary:
                    dataframes[name] = pd.DataFrame()
                    continue
                
                if os.path.exists(path):
                    dataframes[name] = pd.read_csv(path)
                else:
                    dataframes[name] = pd.DataFrame()
            except Exception:
                dataframes[name] = pd.DataFrame()
        return dataframes

    # --- 2. Feature Engineering ---
    def feature_engineering(dfs):
        subject_summary = dfs.get('subject_summary')
        gold_labels = dfs.get('gold_labels')
        course_attributes = dfs.get('course_attributes')
        course_summary = dfs.get('course_summary')
        faculty_summary = dfs.get('faculty_summary')

        if subject_summary.empty or gold_labels.empty:
            raise ValueError("Core data files are missing.")

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
                TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'),
                AVG_SEATS=('MAX_ENROLLMENT', 'mean')
            ).reset_index()
            data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        else:
            data['NUM_COURSES'] = np.nan
            data['TOTAL_SEATS'] = np.nan
            data['AVG_SEATS'] = np.nan
        
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
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data['TERM_CODE_INT'] = data['TERM_CODE']
        data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
        data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
        return data

    # --- 3. Pipeline Definition ---
    def build_classifier_pipeline(classifier, numeric_features, categorical_features):
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

    # --- Execute Pipeline ---
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception:
        return 0.0

    dataframes = load_data(DEFAULT_TRAIN_DIR)
    try:
        data = feature_engineering(dataframes)
    except ValueError:
        return 0.0

    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    if len(sorted_terms) < 2:
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
        if train_df.empty or val_df.empty:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- Model Training ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=n_jobs)
    pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=n_jobs)
    pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm.fit(train_df, y_train)

    tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    tabpfn_num_features = ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY']
    tabpfn_cat_features = [f for f in tabpfn_cat_features if f in train_df.columns]
    tabpfn_num_features = [f for f in tabpfn_num_features if f in train_df.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

    train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df_tabpfn[col])
        train_df_tabpfn[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

    X_train_tabpfn, X_val_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0), val_df_tabpfn[all_tabpfn_features].fillna(0)
    if X_train_tabpfn.shape[1] > 100:
        X_train_tabpfn, X_val_tabpfn = X_train_tabpfn.iloc[:, :100], X_val_tabpfn.iloc[:, :100]

    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

    # --- Validation and Ensembling ---
    final_validation_score = 0.0
    if not y_val.empty:
        proba_rf_val = pipeline_rf.predict_proba(val_df)[:, 1]
        proba_lgbm_val = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn_val = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        if use_stacking_ensemble:
            # Stacking Ensemble
            proba_rf_train = pipeline_rf.predict_proba(train_df)[:, 1]
            proba_lgbm_train = pipeline_lgbm.predict_proba(train_df)[:, 1]
            proba_tabpfn_train = clf_tabpfn.predict_proba(X_train_tabpfn)[:, 1]
            
            X_train_meta = np.vstack([proba_rf_train, proba_lgbm_train, proba_tabpfn_train]).T
            X_val_meta = np.vstack([proba_rf_val, proba_lgbm_val, proba_tabpfn_val]).T
            
            meta_model = LogisticRegression(random_state=42)
            meta_model.fit(X_train_meta, y_train)
            pred_ensemble = meta_model.predict(X_val_meta)
        else:
            # Simple Averaging Ensemble
            proba_ensemble = (proba_rf_val + proba_lgbm_val + proba_tabpfn_val) / 3
            pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        final_validation_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)

    return final_validation_score

if __name__ == '__main__':
    # Set up dummy data for a self-contained run
    create_dummy_data_files()

    # Define ablation scenarios
    scenarios = {
        "Baseline (Simple Avg, Parallel, All Data)": {
            "use_stacking_ensemble": False,
            "n_jobs": -1,
            "omit_course_summary": False
        },
        "Ablation: Stacking Ensemble": {
            "use_stacking_ensemble": True,
            "n_jobs": -1,
            "omit_course_summary": False
        },
        "Ablation: No Parallelization": {
            "use_stacking_ensemble": False,
            "n_jobs": 1,
            "omit_course_summary": False
        },
        "Ablation: No Course Summary Data": {
            "use_stacking_ensemble": False,
            "n_jobs": -1,
            "omit_course_summary": True
        }
    }

    # Run study and store results
    results = {}
    original_stdout = sys.stdout
    sys.stdout = io.StringIO() # Suppress verbose output from libraries
    try:
        for name, config in scenarios.items():
            score = run_ablation_study(config)
            results[name] = score
    finally:
        sys.stdout = original_stdout # Restore stdout

    # Analyze and print results
    baseline_score = results["Baseline (Simple Avg, Parallel, All Data)"]
    performance_drops = {}

    print("--- Ablation Study Results ---")
    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
        if "Ablation" in name:
            component_name = name.split("Ablation: ")[1]
            performance_drops[component_name] = drop
    
    print("\n--- Conclusion ---")
    if not performance_drops or max(performance_drops.values()) <= 0:
        print("No component removal resulted in a significant performance drop.")
    else:
        most_impactful_component = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

