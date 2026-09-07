
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
import shutil

# --- Setup: Define constants and create dummy data for a self-contained run ---
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def setup_dummy_data():
    """Creates minimal dummy data files to ensure the script can run."""
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)
    
    # Gold Labels
    gold_data = {
        'TERM_CODE': [202201, 202201, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-303', 'MATH-202'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y']
    }
    pd.DataFrame(gold_data).to_csv(GOLD_LABELS_PATH, index=False)

    # Subject Summary
    subject_summary_data = {
        'TERM_CODE': [202201, 202201, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-303', 'MATH-202'],
        'CAMPUS_ID_DESC': ['Main Campus', 'Main Campus', 'Main Campus', 'Online', 'Main Campus'],
        'DEPARTMENT_ID_DESC': ['CompSci', 'Mathematics', 'CompSci', 'Fine Arts', 'Mathematics'],
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(DEFAULT_TRAIN_DIR, 'subject_summary.csv'), index=False)

    # Course Summary
    course_summary_data = {
        'TERM_CODE': [202201, 202201, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-303', 'MATH-202'],
        'COURSE_ID': [1, 2, 3, 4, 5],
        'MAX_ENROLLMENT': [100, 30, 120, 50, 40]
    }
    pd.DataFrame(course_summary_data).to_csv(os.path.join(DEFAULT_TRAIN_DIR, 'course_summary.csv'), index=False)

    # Faculty Summary
    faculty_summary_data = {
        'TERM_CODE': [202201, 202201, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-303', 'MATH-202'],
        'FACULTY_ID': [10, 20, 10, 30, 20],
        'NUM_COURSES_TAUGHT': [2, 1, 3, 2, 2]
    }
    pd.DataFrame(faculty_summary_data).to_csv(os.path.join(DEFAULT_TRAIN_DIR, 'faculty_summary.csv'), index=False)

    # Course Attributes
    course_attributes_data = {
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'ART-303'],
        'ATTRIBUTE': ['STEM', 'STEM', 'Humanities']
    }
    pd.DataFrame(course_attributes_data).to_csv(os.path.join(DEFAULT_TRAIN_DIR, 'course_attributes.csv'), index=False)


def cleanup_dummy_data():
    """Removes the dummy data directory."""
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)

# --- Core Modeling and Feature Engineering Functions ---

def load_data(train_dir):
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }
    dataframes = {name: pd.read_csv(path) if os.path.exists(path) else pd.DataFrame() for name, path in paths.items()}
    return dataframes

def feature_engineering(dfs, ablate_faculty_features=False):
    """
    Merges dataframes, creates a combined set of features, and engineers
    department-relative features for contextual understanding.
    The 'ablate_faculty_features' flag controls the inclusion of faculty-derived features.
    """
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files are missing or empty.")

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

    if not ablate_faculty_features:
        if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
            faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                NUM_FACULTY=('FACULTY_ID', 'nunique'),
                AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
            ).reset_index()
            data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        else:
            data['NUM_FACULTY'] = np.nan
            data['AVG_FACULTY_LOAD'] = np.nan
        data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']
    else:
        # Create empty columns if ablating to prevent downstream errors
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan
        data['COURSES_PER_FACULTY'] = np.nan

    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)],
        remainder='drop'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_single_pipeline(ablate_faculty_features=False, ablate_lgbm=False, use_lgbm_only=False):
    """
    Executes a single training and validation run with specified ablations.
    Returns the macro F1 score.
    """
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

    dataframes = load_data(DEFAULT_TRAIN_DIR)
    data = feature_engineering(dataframes, ablate_faculty_features=ablate_faculty_features)
    
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    validation_term = sorted_terms[-1]
    train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
    val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2: return 0.0

    y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']
    
    # Define features for pipeline models
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

    # Model 1: RandomForest
    pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    # Model 2: LightGBM
    pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm.fit(train_df, y_train)

    # Model 3: TabPFN
    tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in train_df.columns]
    tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in train_df.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

    train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df_tabpfn[col])
        train_df_tabpfn[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

    X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0)
    X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0)
    
    if X_train_tabpfn.shape[1] > 100:
        X_train_tabpfn, X_val_tabpfn = X_train_tabpfn.iloc[:, :100], X_val_tabpfn.iloc[:, :100]

    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

    # Validation and Ensembling with Ablation Logic
    proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
    proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
    proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

    if use_lgbm_only:
        final_proba = proba_lgbm
    elif ablate_lgbm:
        final_proba = (proba_rf + proba_tabpfn) / 2
    else: # Baseline
        final_proba = (proba_rf + proba_lgbm + proba_tabpfn) / 3

    final_pred = (final_proba >= 0.5).astype(int)
    return f1_score(y_val, final_pred, average='macro', zero_division=0)

def main():
    """Runs the ablation study and prints the results."""
    setup_dummy_data()
    results = {}
    print("--- Ablation Study Results ---")

    try:
        # Baseline
        baseline_score = run_single_pipeline(ablate_faculty_features=False, ablate_lgbm=False, use_lgbm_only=False)
        results['Baseline (Full Model)'] = baseline_score

        # Ablation 1: No Faculty-derived Features
        score_no_faculty = run_single_pipeline(ablate_faculty_features=True, ablate_lgbm=False, use_lgbm_only=False)
        results['Ablation: No Faculty Features'] = score_no_faculty

        # Ablation 2: No LGBM Model in Ensemble
        score_no_lgbm = run_single_pipeline(ablate_faculty_features=False, ablate_lgbm=True, use_lgbm_only=False)
        results['Ablation: No LGBM Model'] = score_no_lgbm
        
        # Ablation 3: LGBM Only (No Ensemble)
        score_lgbm_only = run_single_pipeline(ablate_faculty_features=False, ablate_lgbm=False, use_lgbm_only=True)
        results['Ablation: LGBM Only'] = score_lgbm_only
        
        # --- Print Performance ---
        for name, score in results.items():
            drop = baseline_score - score
            print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")

        # --- Conclusion ---
        performance_drops = {
            'Faculty-derived Features': baseline_score - score_no_faculty,
            'LGBM Model in Ensemble': baseline_score - score_no_lgbm,
            'Ensembling Strategy (vs LGBM only)': baseline_score - score_lgbm_only,
        }
        
        print("\n--- Conclusion ---")
        if not performance_drops or max(performance_drops.values()) <= 0.0001:
            print("No component removal resulted in a significant performance drop.")
        else:
            most_impactful_component = max(performance_drops, key=performance_drops.get)
            print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

    except Exception as e:
        print(f"An error occurred during the ablation study: {e}")
        print(traceback.format_exc())
    finally:
        cleanup_dummy_data()

if __name__ == '__main__':
    main()
