
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

# Define constants
BASE_INPUT_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data():
    """Creates dummy data files for a self-contained ablation study."""
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)

    subject_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202204, 202204, 202204],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-301', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'ART-301']
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'subject_summary.csv'), index=False)

    pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202204, 202204, 202204],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-301', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'ART-301'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N']
    }).to_csv(GOLD_LABELS_PATH, index=False)

    pd.DataFrame({
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'ART-301'],
        'CAMPUS_ID_DESC': ['Main', 'Main', 'Online'],
        'DEPARTMENT_ID_DESC': ['Engineering', 'Sciences', 'Arts']
    }).to_csv(os.path.join(BASE_INPUT_DIR, 'course_attributes.csv'), index=False)

    pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202204, 202204, 202204],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-301', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'ART-301'],
        'COURSE_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'MAX_ENROLLMENT': [100, 50, 120, 80, 90, 60, 110, 40, 70]
    }).to_csv(os.path.join(BASE_INPUT_DIR, 'course_summary.csv'), index=False)

    pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202204, 202204, 202204],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-301', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'ART-301'],
        'FACULTY_ID': [10, 20, 10, 30, 11, 21, 12, 22, 31],
        'NUM_COURSES_TAUGHT': [2, 1, 3, 2, 1, 2, 2, 1, 1]
    }).to_csv(os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv'), index=False)

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
            if os.path.exists(path): dataframes[name] = pd.read_csv(path)
            else: dataframes[name] = pd.DataFrame()
        except Exception: dataframes[name] = pd.DataFrame()
    return dataframes

def feature_engineering(dfs):
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files missing.")

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), left_on='SUBJECT_ID_SORT', right_on='SUBJECT_ID_SORT_attr', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'), AVG_SEATS=('MAX_ENROLLMENT', 'mean')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'), AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
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

def run_experiment(LGBMClassifier, TabPFNClassifier, use_unified_features=False, use_single_term_split=False, use_lgbm_only=False):
    """Runs a single experiment with specific ablation settings."""
    try:
        dataframes = load_data(BASE_INPUT_DIR)
        data = feature_engineering(dataframes)

        # --- Data Splitting Ablation ---
        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        sorted_terms = sorted(data['TERM_CODE_INT'].unique())
        
        if len(sorted_terms) < 2:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
        else:
            if use_single_term_split: # Ablation: Simple split
                validation_term = sorted_terms[-1]
                train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
                val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
            else: # Baseline: Robust split
                validation_start_term = sorted_terms[-2] if len(sorted_terms) >= 3 else sorted_terms[-1]
                train_df = data[data['TERM_CODE_INT'] < validation_start_term].copy()
                val_df = data[data['TERM_CODE_INT'] >= validation_start_term].copy()
            
            if train_df.empty or val_df.empty or val_df['HIGH_ENROLLMENT'].nunique() < 2:
                train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

        if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
            return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        # --- Feature Set Ablation ---
        tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC_attr', 'DEPARTMENT_ID_DESC_attr'] if f in train_df.columns]
        tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in train_df.columns]
        
        if use_unified_features: # Ablation: use restricted feature set for all
            pipeline_cat_features = tabpfn_cat_features
            pipeline_num_features = tabpfn_num_features
        else: # Baseline: use broad feature set for pipelines
            pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
            pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

        # --- Model Training ---
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]

        # --- Ensemble Ablation ---
        if use_lgbm_only: # Ablation: LGBM only
            proba_ensemble = proba_lgbm
        else: # Baseline: Full Ensemble
            rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
            pipeline_rf.fit(train_df, y_train)
            proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
            
            all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features
            train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
            for col in tabpfn_cat_features:
                codes, uniques = pd.factorize(train_df_tabpfn[col])
                train_df_tabpfn[col] = codes
                val_df_tabpfn[col] = val_df_tabpfn[col].map({label: i for i, label in enumerate(uniques)}).fillna(-1).astype(int)
            X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
            X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]

            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
            proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3

        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    except Exception:
        traceback.print_exc()
        return 0.0

def main():
    """Main function to run the script."""
    try:
        # Fix for 'pkg_resources' error by ensuring setuptools is installed.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        return

    create_dummy_data()

    # --- Run Ablation Study ---
    results = {}
    print("--- Running Ablation Study ---")
    
    # Baseline
    baseline_score = run_experiment(LGBMClassifier, TabPFNClassifier)
    print(f"Final Validation Performance: {baseline_score}")
    results['Baseline (Full Model)'] = {'score': baseline_score, 'drop': 0.0}

    # Ablation 1: Simple Time Split
    score_simple_split = run_experiment(LGBMClassifier, TabPFNClassifier, use_single_term_split=True)
    results['Ablation: No Robust Temporal Split'] = {'score': score_simple_split, 'drop': baseline_score - score_simple_split}
    
    # Ablation 2: Unified Feature Set
    score_unified_features = run_experiment(LGBMClassifier, TabPFNClassifier, use_unified_features=True)
    results['Ablation: No Broad Feature Set'] = {'score': score_unified_features, 'drop': baseline_score - score_unified_features}

    # Ablation 3: No Ensemble
    score_lgbm_only = run_experiment(LGBMClassifier, TabPFNClassifier, use_lgbm_only=True)
    results['Ablation: No Ensemble (LGBM Only)'] = {'score': score_lgbm_only, 'drop': baseline_score - score_lgbm_only}

    # --- Print Results ---
    print("\n--- Ablation Study Results ---")
    print(f"{'Configuration':<35} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 80)
    for name, result in results.items():
        print(f"{name:<35} | {result['score']:.4f}{'':<16} | {result['drop']:.4f}")

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    if baseline_score == 0:
        print("Study was inconclusive as baseline performance was zero.")
    else:
        # Find the component with the largest performance drop
        ablation_results = {k: v for k, v in results.items() if k != 'Baseline (Full Model)'}
        if not ablation_results:
             print("No ablation studies were run.")
             return
             
        max_drop_item = max(ablation_results.items(), key=lambda item: item[1]['drop'])
        max_drop_component = max_drop_item[0].replace('Ablation: No ', '')
        
        if max_drop_item[1]['drop'] > 0:
            print(f"The component that contributes the most to the overall performance is: '{max_drop_component}'")
        else:
            print("No single component removal resulted in a significant performance drop.")

    # Clean up dummy data
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)

if __name__ == '__main__':
    main()
