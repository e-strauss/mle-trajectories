
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
import shutil

# Define constants
BASE_INPUT_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data():
    """Creates dummy data files for a self-contained ablation study."""
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)

    # subject_summary.csv
    subject_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'ART-101', 'CS-202']
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'subject_summary.csv'), index=False)

    # gold_enrollment_train.csv
    gold_labels_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'ART-101', 'CS-202'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'N', 'Y']
    }
    pd.DataFrame(gold_labels_data).to_csv(GOLD_LABELS_PATH, index=False)

    # course_attributes.csv
    course_attributes_data = {
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'ART-101', 'CS-202'],
        'CAMPUS_ID_DESC': ['Main Campus', 'Main Campus', 'South Campus', 'Main Campus'],
        'DEPARTMENT_ID_DESC': ['Engineering', 'Sciences', 'Humanities', 'Engineering']
    }
    pd.DataFrame(course_attributes_data).to_csv(os.path.join(BASE_INPUT_DIR, 'course_attributes.csv'), index=False)

    # course_summary.csv
    course_summary_data = {
        'TERM_CODE': [202201, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101', 'MATH-101'],
        'COURSE_ID': [1, 2, 3, 4],
        'MAX_ENROLLMENT': [100, 110, 120, 50]
    }
    pd.DataFrame(course_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'course_summary.csv'), index=False)
    
    # faculty_summary.csv
    faculty_summary_data = {
        'TERM_CODE': [202201, 202202, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'MATH-101'],
        'FACULTY_ID': [10, 10, 20],
        'NUM_COURSES_TAUGHT': [2, 3, 1]
    }
    pd.DataFrame(faculty_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv'), index=False)


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
            if os.path.exists(path):
                dataframes[name] = pd.read_csv(path)
            else:
                dataframes[name] = pd.DataFrame()
        except Exception:
            dataframes[name] = pd.DataFrame()
    return dataframes

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
    
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features, use_cat_imputer=True):
    numeric_transformer = SimpleImputer(strategy='median')
    
    if use_cat_imputer:
        cat_steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    else:
        # Without the imputer, NaNs will be encoded as all-zero vectors by OneHotEncoder
        cat_steps = [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]

    categorical_transformer = Pipeline(steps=cat_steps)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_experiment(use_stacking=True, use_tabpfn_in_stack=True, use_cat_imputer=True):
    """Runs a single experiment with a specific configuration."""
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

        dataframes = load_data(BASE_INPUT_DIR)
        data = feature_engineering(dataframes)

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

        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        # --- Model Training ---
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features, use_cat_imputer)
        pipeline_rf.fit(train_df, y_train)

        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced')
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features, use_cat_imputer)
        pipeline_lgbm.fit(train_df, y_train)
        
        tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
        tabpfn_num_features = [f for f in data.select_dtypes(include=np.number).columns if f not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        tabpfn_cat_features = [f for f in tabpfn_cat_features if f in train_df.columns]
        tabpfn_num_features = [f for f in tabpfn_num_features if f in train_df.columns]
        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

        train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

        # --- Validation and Ensembling ---
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        if use_stacking:
            # Stacking Ensemble
            X_train_meta = np.stack([
                pipeline_rf.predict_proba(train_df)[:, 1],
                pipeline_lgbm.predict_proba(train_df)[:, 1],
                clf_tabpfn.predict_proba(X_train_tabpfn)[:, 1]
            ], axis=1)

            base_probas = [proba_rf, proba_lgbm]
            if use_tabpfn_in_stack:
                base_probas.append(proba_tabpfn)
            
            X_val_meta = np.stack(base_probas, axis=1)

            meta_learner = LogisticRegression(random_state=42)
            meta_learner.fit(X_train_meta, y_train)
            pred_ensemble = meta_learner.predict(X_val_meta)
        else:
            # Simple Averaging Ensemble
            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
            pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 0.0

def main():
    """Main function to conduct the ablation study."""
    # Install dependencies quietly
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "scikit-learn==1.3.0", "--quiet"])
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        # This is a critical failure, no point in continuing
        return

    create_dummy_data()

    results = {}
    print("--- Running Ablation Study ---")

    # Baseline
    print("Running Baseline (Stacking Ensemble, All Features, Categorical Imputer)...")
    results['Baseline'] = run_experiment(use_stacking=True, use_tabpfn_in_stack=True, use_cat_imputer=True)

    # Ablation 1: Replace Stacking with Simple Averaging
    print("Running Ablation: No Stacking (Simple Average)...")
    results['No Stacking'] = run_experiment(use_stacking=False, use_tabpfn_in_stack=True, use_cat_imputer=True)

    # Ablation 2: Remove TabPFN from Stacking inputs
    print("Running Ablation: No TabPFN in Stack...")
    results['No TabPFN in Stack'] = run_experiment(use_stacking=True, use_tabpfn_in_stack=False, use_cat_imputer=True)

    # Ablation 3: Remove categorical imputer
    print("Running Ablation: No Categorical Imputer...")
    results['No Categorical Imputer'] = run_experiment(use_stacking=True, use_tabpfn_in_stack=True, use_cat_imputer=False)
    
    print("\n--- Ablation Study Results ---")
    baseline_score = results.get('Baseline', 0.0)
    
    performance_drops = {}
    
    print(f"{'Configuration':<25} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 75)
    
    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name:<25} | {score:<20.4f} | {drop:<20.4f}")
        if name != 'Baseline':
            performance_drops[name] = drop

    print("\n--- Conclusion ---")
    if not performance_drops:
        print("Could not run any ablations.")
    elif all(v <= 0 for v in performance_drops.values()):
         print("No component removal resulted in a significant performance drop.")
    else:
        most_impactful = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful}'")

if __name__ == '__main__':
    main()

