
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
import copy

# Define constants
BASE_INPUT_DIR = './abl_input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data():
    """Creates a small, self-contained dataset for the ablation study."""
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)

    subject_summary = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
        'TOTAL_ENROLLMENT': [50, 80, 60, 75, 70, 90]
    })
    subject_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'subject_summary.csv'), index=False)

    gold_labels = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
        'HIGH_ENROLLMENT': ['N', 'Y', 'Y', 'N', 'Y', 'N']
    })
    gold_labels.to_csv(GOLD_LABELS_PATH, index=False)

    course_summary = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
        'COURSE_ID': ['C1', 'M1', 'C1', 'M1', 'C1', 'M1'],
        'MAX_ENROLLMENT': [60, 100, 70, 90, 80, 110]
    })
    course_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'course_summary.csv'), index=False)

    faculty_summary = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202202, 202202, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'MATH-201'],
        'FACULTY_ID': ['F1', 'F2', 'F1', 'F2', 'F1', 'F2'],
        'NUM_COURSES_TAUGHT': [1, 1, 1, 1, 1, 1]
    })
    faculty_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv'), index=False)

def load_data(train_dir):
    """Loads all necessary CSV files."""
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
        except Exception as e:
            dataframes[name] = pd.DataFrame()
    return dataframes

def feature_engineering(dfs, use_trend_features=True, use_relational_features=True):
    """Merges dataframes and creates features, with flags to ablate specific parts."""
    subject_summary, gold_labels, course_summary, faculty_summary = \
        dfs.get('subject_summary'), dfs.get('gold_labels'), dfs.get('course_summary'), dfs.get('faculty_summary')

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not faculty_summary.empty:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_FACULTY=('FACULTY_ID', 'nunique')).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = np.nan

    # Ablation Target 1: Trend-Based Features
    if use_trend_features:
        data.sort_values(by=['SUBJECT_ID_SORT', 'TERM_CODE'], inplace=True)
        growth_features = ['TOTAL_SEATS', 'NUM_COURSES', 'TOTAL_ENROLLMENT', 'NUM_FACULTY']
        for feature in growth_features:
            if feature in data.columns:
                data[f'{feature}_growth'] = data.groupby('SUBJECT_ID_SORT')[feature].pct_change()

    # Ablation Target 2: Relational Features
    if use_relational_features and 'DEPARTMENT' in data.columns:
        dept_agg_cols = {'TOTAL_SEATS': 'sum', 'NUM_COURSES': 'sum', 'TOTAL_ENROLLMENT': 'sum'}
        valid_dept_agg_cols = {k: v for k, v in dept_agg_cols.items() if k in data.columns}
        if valid_dept_agg_cols:
            dept_summary = data.groupby(['TERM_CODE', 'DEPARTMENT']).agg(valid_dept_agg_cols).reset_index()
            dept_summary.rename(columns={k: f'DEPT_{k}' for k in valid_dept_agg_cols}, inplace=True)
            data = pd.merge(data, dept_summary, on=['TERM_CODE', 'DEPARTMENT'], how='left')
            if 'DEPT_TOTAL_SEATS' in data.columns:
                data['SEATS_PCT_OF_DEPT'] = data['TOTAL_SEATS'] / data['DEPT_TOTAL_SEATS']

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_pipeline(use_trend_features, use_relational_features, use_lgbm):
    """Executes the full pipeline with flags for ablation."""
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

        dataframes = load_data(BASE_INPUT_DIR)
        data = feature_engineering(dataframes, use_trend_features, use_relational_features)
        
        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        validation_term = data['TERM_CODE_INT'].unique()[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

        if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2: return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        # Define feature sets
        all_cols = data.select_dtypes(include=np.number).columns.tolist() + data.select_dtypes(include='object').columns.tolist()
        pipeline_cat_features = [col for col in ['DEPARTMENT'] if col in all_cols]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE_INT', 'TERM_CODE']]
        
        # Models
        pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced'), pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)

        probas = {'rf': pipeline_rf.predict_proba(val_df)[:, 1]}

        # Ablation Target 3: LGBM Model
        if use_lgbm:
            pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1), pipeline_num_features, pipeline_cat_features)
            pipeline_lgbm.fit(train_df, y_train)
            probas['lgbm'] = pipeline_lgbm.predict_proba(val_df)[:, 1]

        # TabPFN Model (always included as part of the core ensemble)
        tabpfn_features = [col for col in pipeline_num_features if col in train_df.columns]
        X_train_tabpfn = train_df[tabpfn_features].fillna(0)
        X_val_tabpfn = val_df[tabpfn_features].fillna(0)
        
        if X_train_tabpfn.shape[1] > 100:
            X_train_tabpfn = X_train_tabpfn.iloc[:, :100]
            X_val_tabpfn = X_val_tabpfn.iloc[:, :100]
            
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
        probas['tabpfn'] = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        # Ensemble
        proba_ensemble = np.mean(list(probas.values()), axis=0)
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    except Exception as e:
        # print(f"Pipeline failed with error: {e}", file=sys.stderr)
        # print(traceback.format_exc(), file=sys.stderr)
        return 0.0

def main():
    """Main function to run the ablation study."""
    # Ensure dependencies are installed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "scikit-learn", "--quiet"])
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: Failed to install dependencies. {e}", file=sys.stderr)
        return

    create_dummy_data()

    results = {}
    
    # Run experiments
    results['Baseline (All Features & Models)'] = run_pipeline(use_trend_features=True, use_relational_features=True, use_lgbm=True)
    results['Ablation: No Trend Features'] = run_pipeline(use_trend_features=False, use_relational_features=True, use_lgbm=True)
    results['Ablation: No Relational Features'] = run_pipeline(use_trend_features=True, use_relational_features=False, use_lgbm=True)
    results['Ablation: No LGBM Model'] = run_pipeline(use_trend_features=True, use_relational_features=True, use_lgbm=False)

    baseline_score = results.get('Baseline (All Features & Models)', 0.0)
    
    # Calculate performance drops
    performance_drops = {
        'Trend-Based Features': baseline_score - results.get('Ablation: No Trend Features', 0.0),
        'Relational Features': baseline_score - results.get('Ablation: No Relational Features', 0.0),
        'LGBM Model': baseline_score - results.get('Ablation: No LGBM Model', 0.0),
    }

    # Print results
    print("--- Ablation Study Results ---")
    print(f"{'Configuration':<35} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 80)
    for name, score in results.items():
        drop = baseline_score - score if 'Ablation' in name else 0.0
        print(f"{name:<35} | {score:<20.4f} | {drop:<20.4f}")

    print("\n--- Conclusion ---")
    if baseline_score == 0.0:
        print("Study was inconclusive as baseline performance was zero.")
    else:
        # Find component with the largest drop
        if all(v <= 0 for v in performance_drops.values()):
            print("No single component removal resulted in a significant performance drop.")
        else:
            most_impactful_component = max(performance_drops, key=performance_drops.get)
            print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

    # Cleanup
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)

if __name__ == '__main__':
    main()
