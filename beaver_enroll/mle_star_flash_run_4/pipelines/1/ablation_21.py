
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- Function to Create Dummy Data ---
def create_dummy_data(base_dir='./input'):
    """Creates dummy CSV files for a self-contained ablation study."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Gold Labels (Target)
    gold_labels_data = {
        'TERM_CODE': [202110, 202210, 202310, 202110, 202210, 202310],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101', 'MATH-202', 'MATH-202', 'MATH-202'],
        'HIGH_ENROLLMENT': ['N', 'Y', 'Y', 'Y', 'N', 'Y']
    }
    pd.DataFrame(gold_labels_data).to_csv(os.path.join(base_dir, 'gold_enrollment_train.csv'), index=False)

    # Subject Summary
    subject_summary_data = {
        'TERM_CODE': [202110, 202210, 202310, 202110, 202210, 202310],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101', 'MATH-202', 'MATH-202', 'MATH-202'],
        'TOTAL_ENROLLMENT': [150, 250, 280, 200, 180, 210]
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(base_dir, 'subject_summary.csv'), index=False)

    # Course Summary
    course_summary_data = {
        'TERM_CODE': [202110, 202210, 202310, 202110, 202210, 202310],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101', 'MATH-202', 'MATH-202', 'MATH-202'],
        'COURSE_ID': [1, 2, 3, 4, 5, 6],
        'MAX_ENROLLMENT': [80, 130, 150, 100, 90, 110]
    }
    pd.DataFrame(course_summary_data).to_csv(os.path.join(base_dir, 'course_summary.csv'), index=False)

    # Course Attributes (for composition features)
    course_attributes_data = {
        'COURSE_ID': [1, 2, 3, 4, 5, 6],
        'COURSE_NUMBER': ['101', '404', '450', '202', '299', '501'],
        'COURSE_INSTRUCTION_TYPE_CODE': ['LEC', 'LEC', 'LAB', 'LEC', 'SEM', 'LEC']
    }
    pd.DataFrame(course_attributes_data).to_csv(os.path.join(base_dir, 'course_attributes.csv'), index=False)

    # Faculty Summary
    faculty_summary_data = {
        'TERM_CODE': [202110, 202210, 202310, 202110, 202210, 202310],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101', 'MATH-202', 'MATH-202', 'MATH-202'],
        'FACULTY_ID': [10, 11, 11, 20, 21, 20],
        'NUM_COURSES_TAUGHT': [1, 2, 2, 1, 1, 2]
    }
    pd.DataFrame(faculty_summary_data).to_csv(os.path.join(base_dir, 'faculty_summary.csv'), index=False)

# --- Core Modeling and Feature Engineering Logic ---

def load_data(train_dir, gold_path):
    """Loads all necessary CSV files."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': gold_path
    }
    dataframes = {name: pd.read_csv(path) for name, path in paths.items() if os.path.exists(path)}
    return dataframes

def feature_engineering(dfs, use_composition_features=True, use_lag_features=True):
    """
    Merges dataframes and creates features, with flags to ablate specific components.
    """
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    for df in [subject_summary, gold_labels, course_summary, faculty_summary, course_attributes]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    # --- Ablation Target 1: Course Composition Features ---
    course_agg = pd.DataFrame()
    if not course_summary.empty:
        enriched_course_summary = course_summary.copy()
        composition_aggs = {}
        
        if use_composition_features and not course_attributes.empty and 'COURSE_ID' in course_attributes.columns:
            enriched_course_summary = pd.merge(enriched_course_summary, course_attributes, on='COURSE_ID', how='left')
            if 'COURSE_NUMBER' in enriched_course_summary.columns:
                course_levels = pd.to_numeric(enriched_course_summary['COURSE_NUMBER'].astype(str).str[0], errors='coerce')
                enriched_course_summary['IS_LOWER_DIVISION'] = (course_levels < 3).astype(float)
                composition_aggs['LOWER_DIVISION_COURSES'] = ('IS_LOWER_DIVISION', 'sum')
            if 'COURSE_INSTRUCTION_TYPE_CODE' in enriched_course_summary.columns:
                composition_aggs['DISTINCT_COURSE_TYPES'] = ('COURSE_INSTRUCTION_TYPE_CODE', 'nunique')
        
        agg_dict = {'NUM_COURSES': ('COURSE_ID', 'nunique'), 'TOTAL_SEATS': ('MAX_ENROLLMENT', 'sum'), **composition_aggs}
        course_agg = enriched_course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(**agg_dict).reset_index()

        if use_composition_features and 'LOWER_DIVISION_COURSES' in course_agg.columns:
            course_agg['LOWER_DIV_RATIO'] = course_agg['LOWER_DIVISION_COURSES'] / course_agg['NUM_COURSES']
            course_agg.drop(columns=['LOWER_DIVISION_COURSES'], inplace=True)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]
    if not course_agg.empty:
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not faculty_summary.empty:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_FACULTY=('FACULTY_ID', 'nunique')).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # --- Ablation Target 2: Seasonal Lag Features ---
    if use_lag_features:
        term_str = data['TERM_CODE'].astype(str)
        prev_year_term_code = (term_str.str[:4].astype(int) - 1).astype(str) + term_str.str[4:]
        data['PREV_YEAR_TERM_CODE'] = pd.to_numeric(prev_year_term_code, errors='coerce').fillna(0).astype(int)
        
        metrics_to_lag = ['TOTAL_SEATS', 'NUM_COURSES', 'TOTAL_ENROLLMENT']
        lag_cols = [col for col in metrics_to_lag if col in data.columns]
        
        if lag_cols:
            lag_features_source = data[['SUBJECT_ID_SORT', 'TERM_CODE'] + lag_cols].copy()
            lag_features_source.rename(columns={col: f'{col}_lag1y' for col in lag_cols}, inplace=True)
            data = pd.merge(data, lag_features_source, left_on=['SUBJECT_ID_SORT', 'PREV_YEAR_TERM_CODE'], right_on=['SUBJECT_ID_SORT', 'TERM_CODE'], how='left', suffixes=('', '_r'))
            data.drop(columns=['TERM_CODE_r'], inplace=True, errors='ignore')
        data.drop(columns=['PREV_YEAR_TERM_CODE'], inplace=True, errors='ignore')

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_pipeline(use_composition_features=True, use_lag_features=True, use_lgbm=True):
    """Main function to execute the training and validation pipeline, returns F1 score."""
    BASE_INPUT_DIR = './input'
    GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')
    
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

    dataframes = load_data(BASE_INPUT_DIR, GOLD_LABELS_PATH)
    data = feature_engineering(dataframes, use_composition_features, use_lag_features)

    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    validation_term = sorted_terms[-1]
    train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
    val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2: return 0.0

    y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

    pipeline_cat_features = [col for col in ['DEPARTMENT'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    # Model 1: RandomForest
    pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced'), pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)
    proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]

    # Model 2: LightGBM (Ablation Target 3)
    if use_lgbm:
        pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced'), pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
    else:
        proba_lgbm = 0

    # Model 3: TabPFN
    tabpfn_features = [f for f in data.columns if f in train_df.columns and (data[f].dtype in [np.number, 'object']) and f not in ['HIGH_ENROLLMENT']]
    X_train_tabpfn = train_df[tabpfn_features].apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x).fillna(0).iloc[:,:100]
    X_val_tabpfn = val_df[tabpfn_features].apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x).fillna(0).iloc[:,:100]
    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
    clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
    proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

    # Ensemble
    if use_lgbm:
        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
    else:
        proba_ensemble = (proba_rf + proba_tabpfn) / 2
        
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)

def run_ablation_study():
    """Orchestrates the ablation study by running different pipeline configurations."""
    create_dummy_data()
    
    results = {}
    print("--- Running Ablation Study ---")

    # Baseline
    baseline_score = run_pipeline(use_composition_features=True, use_lag_features=True, use_lgbm=True)
    results['Baseline (All Components)'] = (baseline_score, 0.0)

    # Ablation 1: No Course Composition Features
    score_no_comp = run_pipeline(use_composition_features=False, use_lag_features=True, use_lgbm=True)
    results['Ablation: No Course Composition Features'] = (score_no_comp, baseline_score - score_no_comp)
    
    # Ablation 2: No Seasonal Lag Features
    score_no_lag = run_pipeline(use_composition_features=True, use_lag_features=False, use_lgbm=True)
    results['Ablation: No Seasonal Lag Features'] = (score_no_lag, baseline_score - score_no_lag)

    # Ablation 3: No LGBM Model in Ensemble
    score_no_lgbm = run_pipeline(use_composition_features=True, use_lag_features=True, use_lgbm=False)
    results['Ablation: No LGBM Model'] = (score_no_lgbm, baseline_score - score_no_lgbm)

    print("\n--- Ablation Study Results ---")
    for name, (score, drop) in results.items():
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")

    # Determine the most impactful component
    if not results or baseline_score == 0:
        conclusion = "Study was inconclusive as baseline performance was zero."
    else:
        max_drop_component = max(results.items(), key=lambda item: item[1][1])
        if max_drop_component[1][1] > 0.001:
            component_name = max_drop_component[0].replace('Ablation: No ', '')
            conclusion = f"The component that contributes the most to the overall performance is: '{component_name}'"
        else:
            conclusion = "No single component removal resulted in a significant performance drop."

    print(f"\n--- Conclusion ---\n{conclusion}")


if __name__ == '__main__':
    try:
        run_ablation_study()
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
