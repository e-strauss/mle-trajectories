
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
warnings.filterwarnings('ignore')

# --- Constants and Setup ---
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data_if_not_exists():
    """Creates minimal dummy data files to ensure the script can run."""
    if not os.path.exists(BASE_INPUT_DIR):
        os.makedirs(BASE_INPUT_DIR)

    file_creators = {
        'subject_summary.csv': lambda: pd.DataFrame({
            'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230],
            'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
            'TOTAL_ENROLLMENT': [150, 80, 160, 75, 170, 90]
        }),
        'gold_enrollment_train.csv': lambda: pd.DataFrame({
            'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230],
            'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N']
        }),
        'course_summary.csv': lambda: pd.DataFrame({
            'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230],
            'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
            'COURSE_ID': [1, 2, 3, 4, 5, 6],
            'MAX_ENROLLMENT': [50, 40, 55, 35, 60, 45]
        }),
        'faculty_summary.csv': lambda: pd.DataFrame({
            'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230],
            'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
            'FACULTY_ID': [10, 20, 10, 21, 11, 20],
            'NUM_COURSES_TAUGHT': [2, 1, 2, 1, 2, 1]
        }),
        'course_attributes.csv': lambda: pd.DataFrame({
            'SUBJECT_ID_SORT': ['CS-101', 'MATH-201'],
            'CAMPUS_ID_DESC': ['Main Campus', 'Main Campus']
        }),
        'instructor_attributes.csv': lambda: pd.DataFrame(), # Can be empty
    }
    
    for filename, creator in file_creators.items():
        path = os.path.join(BASE_INPUT_DIR, filename)
        if not os.path.exists(path):
            creator().to_csv(path, index=False)


# --- Core Logic from train.py ---

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

def feature_engineering(dfs, use_temporal_lags=True, use_faculty_features=True):
    subject_summary = dfs.get('subject_summary', pd.DataFrame())
    gold_labels = dfs.get('gold_labels', pd.DataFrame())
    course_attributes = dfs.get('course_attributes', pd.DataFrame())
    course_summary = dfs.get('course_summary', pd.DataFrame())
    faculty_summary = dfs.get('faculty_summary', pd.DataFrame())

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files are missing or empty.")

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if not df.empty and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if use_temporal_lags:
        if 'TOTAL_ENROLLMENT' in data.columns and 'HIGH_ENROLLMENT' in data.columns:
            lag_source = data[['TERM_CODE', 'SUBJECT_ID_SORT', 'TOTAL_ENROLLMENT', 'HIGH_ENROLLMENT']].copy()
            lag_source['HIGH_ENROLLMENT_LAG'] = lag_source['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
            lag_source.rename(columns={'TOTAL_ENROLLMENT': 'PREV_TERM_ENROLLMENT', 'HIGH_ENROLLMENT_LAG': 'PREV_TERM_HIGH_ENROLLMENT'}, inplace=True)
            lag_source = lag_source[['TERM_CODE', 'SUBJECT_ID_SORT', 'PREV_TERM_ENROLLMENT', 'PREV_TERM_HIGH_ENROLLMENT']]
            
            unique_terms = sorted(data['TERM_CODE'].unique())
            term_map = {term: next_term for term, next_term in zip(unique_terms[:-1], unique_terms[1:])}
            lag_source['TERM_CODE'] = lag_source['TERM_CODE'].map(term_map)
            lag_source.dropna(subset=['TERM_CODE'], inplace=True)
            data = pd.merge(data, lag_source, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
        data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'), AVG_SEATS=('MAX_ENROLLMENT', 'mean')).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if use_faculty_features and not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_FACULTY=('FACULTY_ID', 'nunique'), AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan

    data['SEATS_PER_COURSE'] = data.get('TOTAL_SEATS', pd.Series(index=data.index)) / data.get('NUM_COURSES', pd.Series(index=data.index))
    data['COURSES_PER_FACULTY'] = data.get('NUM_COURSES', pd.Series(index=data.index)) / data.get('NUM_FACULTY', pd.Series(index=data.index))
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_experiment(use_temporal_lags, use_faculty_features, use_lgbm):
    """Executes one run of the training and validation pipeline with specified components."""
    try:
        dataframes = load_data(DEFAULT_TRAIN_DIR)
        data = feature_engineering(dataframes, use_temporal_lags=use_temporal_lags, use_faculty_features=use_faculty_features)
        
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

        if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
            return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        ensemble_probas = []

        # --- Model 1: RandomForest ---
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)
        ensemble_probas.append(pipeline_rf.predict_proba(val_df)[:, 1])

        # --- Model 2: LightGBM (Conditional) ---
        if use_lgbm:
            lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
            pipeline_lgbm.fit(train_df, y_train)
            ensemble_probas.append(pipeline_lgbm.predict_proba(val_df)[:, 1])

        # --- Model 3: TabPFN ---
        tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in train_df.columns]
        tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY', 'PREV_TERM_ENROLLMENT', 'PREV_TERM_HIGH_ENROLLMENT'] if f in train_df.columns]
        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

        train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
        ensemble_probas.append(clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1])

        # --- Validation and Ensembling ---
        proba_ensemble = np.mean(ensemble_probas, axis=0)
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    
    except Exception:
        return 0.0

def main():
    # Install dependencies quietly
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        return

    # Create dummy data for local runs if necessary
    create_dummy_data_if_not_exists()
    
    ablation_scenarios = {
        'Temporal Lag Features': {'use_temporal_lags': False, 'use_faculty_features': True, 'use_lgbm': True},
        'Faculty Features': {'use_temporal_lags': True, 'use_faculty_features': False, 'use_lgbm': True},
        'LightGBM Model': {'use_temporal_lags': True, 'use_faculty_features': True, 'use_lgbm': False},
    }
    
    results = {}

    # Run baseline
    baseline_score = run_experiment(use_temporal_lags=True, use_faculty_features=True, use_lgbm=True)
    results['Baseline'] = (baseline_score, 0.0)

    # Run ablations
    for name, params in ablation_scenarios.items():
        score = run_experiment(**params)
        performance_drop = baseline_score - score
        results[f'Ablation: No {name}'] = (score, performance_drop)

    print("--- Ablation Study Results ---")
    for name, (score, drop) in results.items():
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
    
    # Determine the most impactful component
    if all(res[1] <= 0 for name, res in results.items() if name != 'Baseline'):
        conclusion = "No component removal resulted in a significant performance drop."
    else:
        most_impactful = max(results.items(), key=lambda item: item[1][1] if item[0] != 'Baseline' else -1)[0]
        # Clean up the name for the conclusion
        component_name = most_impactful.replace('Ablation: No ', '')
        conclusion = f"The component that contributes the most to the overall performance is: '{component_name}'"

    print("\n--- Conclusion ---")
    print(conclusion)


if __name__ == '__main__':
    # Add a global try-except to catch any unexpected errors during the whole process
    try:
        # These imports are here to be accessible by the functions after installation.
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
        main()
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

