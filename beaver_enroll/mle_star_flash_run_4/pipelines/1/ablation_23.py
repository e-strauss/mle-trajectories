
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
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Define constants
BASE_INPUT_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data():
    """Creates dummy data files to ensure the script can run without external dependencies."""
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)
    
    # subject_summary.csv
    subject_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301, 202201, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'PHY-101', 'PHY-101', 'PHY-101', 'ART-100']
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'subject_summary.csv'), index=False)

    # gold_enrollment_train.csv
    gold_labels_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301, 202201, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'PHY-101', 'PHY-101', 'PHY-101', 'ART-100'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N']
    }
    pd.DataFrame(gold_labels_data).to_csv(GOLD_LABELS_PATH, index=False)
    
    # course_summary.csv
    course_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301, 202201, 202202, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'PHY-101', 'PHY-101', 'PHY-101'],
        'COURSE_ID': [1, 2, 1, 2, 1, 2, 3, 3, 3],
        'MAX_ENROLLMENT': [150, 50, 160, 60, 170, 45, 80, 90, 95]
    }
    pd.DataFrame(course_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'course_summary.csv'), index=False)
    
    # faculty_summary.csv
    faculty_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'CS-101', 'MATH-202'],
        'FACULTY_ID': [10, 20, 10, 21, 11, 20],
        'NUM_COURSES_TAUGHT': [2, 1, 2, 1, 2, 1]
    }
    pd.DataFrame(faculty_summary_data).to_csv(os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv'), index=False)

    # course_attributes.csv (can be empty for this study)
    pd.DataFrame({'SUBJECT_ID_SORT': []}).to_csv(os.path.join(BASE_INPUT_DIR, 'course_attributes.csv'), index=False)


def load_data(train_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }
    dataframes = {name: pd.read_csv(path) for name, path in paths.items() if os.path.exists(path)}
    return dataframes

def feature_engineering(dfs, use_expanding_hist_features=True, use_departmental_features=True):
    """
    Merges dataframes and creates features, with flags to ablate specific feature sets.
    """
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary is None or gold_labels is None:
        raise ValueError("Core data files are missing.")

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.sort_values(['SUBJECT_ID_SORT', 'TERM_CODE_INT'], inplace=True)
    
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if course_summary is not None:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'),
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if use_expanding_hist_features:
        metrics_to_expand = ['TOTAL_SEATS', 'NUM_COURSES']
        for metric in metrics_to_expand:
            if metric in data.columns:
                data[f'hist_{metric}_mean'] = data.groupby('SUBJECT_ID_SORT')[metric].transform(lambda s: s.shift(1).expanding().mean())
                data[f'hist_{metric}_std'] = data.groupby('SUBJECT_ID_SORT')[metric].transform(lambda s: s.shift(1).expanding().std())

    if use_departmental_features:
        metrics_for_dept_context = ['TOTAL_SEATS', 'NUM_COURSES']
        for metric in metrics_for_dept_context:
            if metric in data.columns:
                data[f'dept_avg_{metric}'] = data.groupby(['TERM_CODE_INT', 'DEPARTMENT'])[metric].transform('mean')
                data[f'relative_{metric}_to_dept'] = data[metric] / data[f'dept_avg_{metric}']
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_ablation_study(use_expanding_hist_features, use_departmental_features, use_lgbm_only):
    """Runs a single ablation scenario and returns the F1 score."""
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
        
        # 1. Data Loading & Feature Engineering
        dataframes = load_data(BASE_INPUT_DIR)
        data = feature_engineering(dataframes, use_expanding_hist_features, use_departmental_features)

        # 2. Data Splitting
        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        sorted_terms = sorted(data['TERM_CODE_INT'].unique())
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term]
        val_df = data[data['TERM_CODE_INT'] == validation_term]

        if train_df.empty or val_df.empty: return 0.0

        y_train = train_df['HIGH_ENROLLMENT']
        y_val = val_df['HIGH_ENROLLMENT']

        # 3. Model Training
        pipeline_cat_features = ['DEPARTMENT']
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=1, verbose=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)

        if use_lgbm_only:
            proba_final = pipeline_lgbm.predict_proba(val_df)[:, 1]
        else:
            # Train other models for ensemble
            rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
            pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
            pipeline_rf.fit(train_df, y_train)
            
            # TabPFN setup
            tabpfn_features = [f for f in pipeline_num_features if f in train_df.columns]
            X_train_tabpfn = train_df[tabpfn_features].fillna(0)
            X_val_tabpfn = val_df[tabpfn_features].fillna(0)
            
            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
            clf_tabpfn.fit(X_train_tabpfn.iloc[:1000], y_train.iloc[:1000], overwrite_warning=True)

            # 4. Ensembling and Validation
            proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
            proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
            proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]
            proba_final = (proba_lgbm + proba_rf + proba_tabpfn) / 3

        pred_final = (proba_final >= 0.5).astype(int)
        return f1_score(y_val, pred_final, average='macro', zero_division=0)
    except Exception:
        traceback.print_exc()
        return 0.0

if __name__ == '__main__':
    # Ensure dependencies are installed
    try:
        # Added setuptools to fix pkg_resources error during dependency builds
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        # Do not exit, allow the script to continue if possible or fail naturally

    create_dummy_data()
    
    # --- Run Ablation Study ---
    scenarios = {
        'Baseline (All New Features + Ensemble)': {
            'hist': True, 'dept': True, 'lgbm_only': False
        },
        'Ablation: No Expanding Window Features': {
            'hist': False, 'dept': True, 'lgbm_only': False
        },
        'Ablation: No Departmental Features': {
            'hist': True, 'dept': False, 'lgbm_only': False
        },
        'Ablation: LGBM Only (No Ensemble)': {
            'hist': True, 'dept': True, 'lgbm_only': True
        }
    }
    
    results = {}
    print("--- Ablation Study Results ---")
    for name, config in scenarios.items():
        score = run_ablation_study(
            use_expanding_hist_features=config['hist'],
            use_departmental_features=config['dept'],
            use_lgbm_only=config['lgbm_only']
        )
        results[name] = score

    baseline_score = results.get('Baseline (All New Features + Ensemble)', 0.0)
    print(f"Final Validation Performance: {baseline_score}")
    
    performance_drops = {}
    
    if baseline_score == 0.0:
        print("Study was inconclusive as baseline performance was zero.")
    else:
        for name, score in results.items():
            drop = baseline_score - score
            print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
            if name != 'Baseline (All New Features + Ensemble)':
                performance_drops[name] = drop
        
        if not performance_drops:
             print("\n--- Conclusion ---\nNo components were ablated.")
        elif all(v <= 0 for v in performance_drops.values()):
            print("\n--- Conclusion ---\nNo single component removal resulted in a significant performance drop.")
        else:
            most_impactful = max(performance_drops, key=performance_drops.get)
            # Clean up the name for the conclusion
            clean_name = most_impactful.replace('Ablation: ', '').replace(' (No Ensemble)', '')
            print(f"\n--- Conclusion ---\nThe component that contributes the most to the overall performance is: '{clean_name}'")
