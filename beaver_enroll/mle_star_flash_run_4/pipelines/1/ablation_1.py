
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
import copy

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
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
            NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'), AVG_SEATS=('MAX_ENROLLMENT', 'mean')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
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
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)], remainder='drop'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def main():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print(f"Ablation study failed due to dependency issues: {e}", file=sys.stderr)
        return

    # --- Data Loading and Feature Engineering (run once) ---
    dataframes = load_data(DEFAULT_TRAIN_DIR)
    try:
        data = feature_engineering(dataframes)
    except ValueError as e:
        print(f"Ablation study failed during data prep: {e}", file=sys.stderr)
        return

    # --- Shared Feature Definitions ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in data.columns]
    tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in data.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

    scores = {}

    # --- Experiment setup ---
    # Time-based split
    data_sorted = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data_sorted['TERM_CODE_INT'].unique())
    validation_term = sorted_terms[-1]
    train_df_time = data_sorted[data_sorted['TERM_CODE_INT'] < validation_term].copy()
    val_df_time = data_sorted[data_sorted['TERM_CODE_INT'] == validation_term].copy()
    y_train_time = train_df_time['HIGH_ENROLLMENT']
    y_val_time = val_df_time['HIGH_ENROLLMENT']

    # --- Baseline: Full Model ---
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    lgbm_clf = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    tabpfn_clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    
    # Train pipeline models
    pipeline_rf = build_classifier_pipeline(rf_clf, pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm = build_classifier_pipeline(lgbm_clf, pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df_time, y_train_time)
    pipeline_lgbm.fit(train_df_time, y_train_time)

    # Train TabPFN
    train_df_tabpfn, val_df_tabpfn = train_df_time.copy(), val_df_time.copy()
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df_tabpfn[col])
        train_df_tabpfn[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)
    X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
    X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
    tabpfn_clf.fit(X_train_tabpfn, y_train_time, overwrite_warning=True)
    
    # Predict and ensemble
    proba_rf = pipeline_rf.predict_proba(val_df_time)[:, 1]
    proba_lgbm = pipeline_lgbm.predict_proba(val_df_time)[:, 1]
    proba_tabpfn = tabpfn_clf.predict_proba(X_val_tabpfn)[:, 1]
    proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
    preds = (proba_ensemble >= 0.5).astype(int)
    scores['Baseline (Full Model)'] = f1_score(y_val_time, preds, average='macro', zero_division=0)

    # --- Ablation 1: No LightGBM Model ---
    proba_ensemble_no_lgbm = (proba_rf + proba_tabpfn) / 2
    preds_no_lgbm = (proba_ensemble_no_lgbm >= 0.5).astype(int)
    scores['Ablation (No LightGBM)'] = f1_score(y_val_time, preds_no_lgbm, average='macro', zero_division=0)
    
    # --- Ablation 2: No `class_weight='balanced'` ---
    rf_clf_no_weight = RandomForestClassifier(random_state=42, n_jobs=-1)
    lgbm_clf_no_weight = LGBMClassifier(random_state=42, n_jobs=-1)
    pipeline_rf_no_weight = build_classifier_pipeline(rf_clf_no_weight, pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm_no_weight = build_classifier_pipeline(lgbm_clf_no_weight, pipeline_num_features, pipeline_cat_features)
    pipeline_rf_no_weight.fit(train_df_time, y_train_time)
    pipeline_lgbm_no_weight.fit(train_df_time, y_train_time)
    proba_rf_no_weight = pipeline_rf_no_weight.predict_proba(val_df_time)[:, 1]
    proba_lgbm_no_weight = pipeline_lgbm_no_weight.predict_proba(val_df_time)[:, 1]
    proba_ensemble_no_weight = (proba_rf_no_weight + proba_lgbm_no_weight + proba_tabpfn) / 3
    preds_no_weight = (proba_ensemble_no_weight >= 0.5).astype(int)
    scores['Ablation (No Class Weighting)'] = f1_score(y_val_time, preds_no_weight, average='macro', zero_division=0)

    # --- Ablation 3: Random Split instead of Time-based Split ---
    train_df_rand, val_df_rand = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    y_train_rand, y_val_rand = train_df_rand['HIGH_ENROLLMENT'], val_df_rand['HIGH_ENROLLMENT']
    
    pipeline_rf.fit(train_df_rand, y_train_rand)
    pipeline_lgbm.fit(train_df_rand, y_train_rand)
    
    train_df_tabpfn_rand, val_df_tabpfn_rand = train_df_rand.copy(), val_df_rand.copy()
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df_tabpfn_rand[col])
        train_df_tabpfn_rand[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn_rand[col] = val_df_tabpfn_rand[col].map(mapping).fillna(-1).astype(int)
    X_train_tabpfn_rand = train_df_tabpfn_rand[all_tabpfn_features].fillna(0).iloc[:,:100]
    X_val_tabpfn_rand = val_df_tabpfn_rand[all_tabpfn_features].fillna(0).iloc[:,:100]
    tabpfn_clf.fit(X_train_tabpfn_rand, y_train_rand, overwrite_warning=True)

    proba_rf_rand = pipeline_rf.predict_proba(val_df_rand)[:, 1]
    proba_lgbm_rand = pipeline_lgbm.predict_proba(val_df_rand)[:, 1]
    proba_tabpfn_rand = tabpfn_clf.predict_proba(X_val_tabpfn_rand)[:, 1]
    proba_ensemble_rand = (proba_rf_rand + proba_lgbm_rand + proba_tabpfn_rand) / 3
    preds_rand = (proba_ensemble_rand >= 0.5).astype(int)
    scores['Comparison (Random Split)'] = f1_score(y_val_rand, preds_rand, average='macro', zero_division=0)

    # --- Print Results ---
    print("--- Ablation Study Results ---")
    for name, score in scores.items():
        print(f'{name}: {score:.4f}')

    # --- Conclusion ---
    baseline_score = scores['Baseline (Full Model)']
    drop_lgbm = baseline_score - scores['Ablation (No LightGBM)']
    drop_weighting = baseline_score - scores['Ablation (No Class Weighting)']
    
    # We compare the time-based score to the random-split score. A higher random score suggests data leakage or less temporal drift.
    # The 'impact' is the magnitude of the difference.
    impact_split = abs(baseline_score - scores['Comparison (Random Split)'])

    performance_drops = {
        'LightGBM Model': drop_lgbm,
        'Class Weighting': drop_weighting,
        'Time-based Split Strategy': impact_split
    }

    most_impactful_component = max(performance_drops, key=performance_drops.get)
    
    print("\n--- Conclusion ---")
    print(f"The component that contributes most to the overall performance is the '{most_impactful_component}'.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        traceback.print_exc()

