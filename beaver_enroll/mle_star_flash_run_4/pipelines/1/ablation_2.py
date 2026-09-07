
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

# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
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

def feature_engineering_ablation(dfs, use_hierarchical_features=True):
    """
    Merges dataframes and creates features. Includes a flag for ablating hierarchical features.
    """
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
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not faculty_summary.empty:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Hierarchical Department Features (can be ablated)
    if use_hierarchical_features and 'TOTAL_SEATS' in data.columns and 'NUM_COURSES' in data.columns:
        dept_agg = data.groupby(['TERM_CODE', 'DEPARTMENT']).agg(
            DEPT_TOTAL_SEATS=('TOTAL_SEATS', 'sum'),
            DEPT_TOTAL_COURSES=('NUM_COURSES', 'sum')
        ).reset_index()
        data = pd.merge(data, dept_agg, on=['TERM_CODE', 'DEPARTMENT'], how='left')
        data['SEAT_PROPORTION_IN_DEPT'] = data['TOTAL_SEATS'] / data['DEPT_TOTAL_SEATS']
        data['COURSE_PROPORTION_IN_DEPT'] = data['NUM_COURSES'] / data['DEPT_TOTAL_COURSES']

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
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
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_ablation_study():
    """
    Performs an ablation study by training models with different components disabled.
    """
    print("--- Starting Ablation Study ---")
    results = {}

    try:
        # --- 0. Dependency Installation ---
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print(f"Error installing dependencies: {e}", file=sys.stderr)
        return

    # --- 1. Data Loading and Baseline Preprocessing ---
    dataframes = load_data(DEFAULT_TRAIN_DIR)
    
    # --- Experiment 1: Baseline (All components active) ---
    print("\nRunning Experiment: Baseline (All Features & Models)")
    try:
        data_baseline = feature_engineering_ablation(dataframes, use_hierarchical_features=True)

        # Time-based split
        data_baseline = data_baseline.sort_values('TERM_CODE_INT').reset_index(drop=True)
        validation_term = sorted(data_baseline['TERM_CODE_INT'].unique())[-1]
        train_df = data_baseline[data_baseline['TERM_CODE_INT'] < validation_term].copy()
        val_df = data_baseline[data_baseline['TERM_CODE_INT'] == validation_term].copy()
        y_train = train_df['HIGH_ENROLLMENT']
        y_val = val_df['HIGH_ENROLLMENT']

        # Define features
        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data_baseline.columns if c.endswith('_attr') and data_baseline[c].dtype == 'object'] if col in data_baseline.columns]
        pipeline_num_features = [col for col in data_baseline.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

        # Train models
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)

        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        
        # TabPFN setup and training
        tabpfn_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'NUM_FACULTY', 'SEAT_PROPORTION_IN_DEPT', 'COURSE_PROPORTION_IN_DEPT'] if f in train_df.columns]
        X_train_tabpfn = train_df[tabpfn_features].fillna(0)
        X_val_tabpfn = val_df[tabpfn_features].fillna(0)
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

        # Get probabilities
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        # Ensemble and score
        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        baseline_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        results['Baseline'] = baseline_score
        print(f"Performance: {baseline_score:.4f}")

    except Exception as e:
        print(f"Baseline failed: {e}", file=sys.stderr)
        results['Baseline'] = 0.0

    # --- Experiment 2: Ablation - No Hierarchical Features ---
    print("\nRunning Experiment: Ablation - No Hierarchical Features")
    try:
        # Regenerate data without the features
        data_no_hier = feature_engineering_ablation(dataframes, use_hierarchical_features=False)
        data_no_hier = data_no_hier.sort_values('TERM_CODE_INT').reset_index(drop=True)
        train_df_nh = data_no_hier[data_no_hier['TERM_CODE_INT'] < validation_term].copy()
        val_df_nh = data_no_hier[data_no_hier['TERM_CODE_INT'] == validation_term].copy()

        # Retrain models on ablated data
        pipeline_rf.fit(train_df_nh, train_df_nh['HIGH_ENROLLMENT'])
        pipeline_lgbm.fit(train_df_nh, train_df_nh['HIGH_ENROLLMENT'])
        
        tabpfn_features_nh = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'NUM_FACULTY'] if f in train_df_nh.columns]
        X_train_tabpfn_nh = train_df_nh[tabpfn_features_nh].fillna(0)
        X_val_tabpfn_nh = val_df_nh[tabpfn_features_nh].fillna(0)
        clf_tabpfn.fit(X_train_tabpfn_nh, train_df_nh['HIGH_ENROLLMENT'], overwrite_warning=True)

        # Ensemble and score
        proba_rf_nh = pipeline_rf.predict_proba(val_df_nh)[:, 1]
        proba_lgbm_nh = pipeline_lgbm.predict_proba(val_df_nh)[:, 1]
        proba_tabpfn_nh = clf_tabpfn.predict_proba(X_val_tabpfn_nh)[:, 1]
        proba_ensemble_nh = (proba_rf_nh + proba_lgbm_nh + proba_tabpfn_nh) / 3
        pred_ensemble_nh = (proba_ensemble_nh >= 0.5).astype(int)
        score = f1_score(val_df_nh['HIGH_ENROLLMENT'], pred_ensemble_nh, average='macro', zero_division=0)
        results['No Hierarchical Features'] = score
        print(f"Performance: {score:.4f}")
    except Exception as e:
        print(f"Ablation failed: {e}", file=sys.stderr)
        results['No Hierarchical Features'] = 0.0


    # --- Experiment 3: Ablation - No RandomForest in Ensemble ---
    print("\nRunning Experiment: Ablation - No RandomForest in Ensemble")
    try:
        # Use baseline models, just change ensemble logic
        proba_ensemble_norf = (proba_lgbm + proba_tabpfn) / 2
        pred_ensemble_norf = (proba_ensemble_norf >= 0.5).astype(int)
        score = f1_score(y_val, pred_ensemble_norf, average='macro', zero_division=0)
        results['No RandomForest'] = score
        print(f"Performance: {score:.4f}")
    except Exception as e:
        print(f"Ablation failed: {e}", file=sys.stderr)
        results['No RandomForest'] = 0.0
    
    # --- Experiment 4: Ablation - No Ensemble (LGBM Only) ---
    print("\nRunning Experiment: Ablation - No Ensemble (LGBM Only)")
    try:
        # Use only the LGBM model's predictions
        pred_lgbm_only = (proba_lgbm >= 0.5).astype(int)
        score = f1_score(y_val, pred_lgbm_only, average='macro', zero_division=0)
        results['LGBM Only'] = score
        print(f"Performance: {score:.4f}")
    except Exception as e:
        print(f"Ablation failed: {e}", file=sys.stderr)
        results['LGBM Only'] = 0.0

    # --- Conclusion ---
    print("\n--- Ablation Study Summary ---")
    baseline_score = results.get('Baseline', 0.0)
    print(f"Baseline F1 Score: {baseline_score:.4f}")
    
    max_drop = 0
    most_impactful = "None"
    
    for name, score in results.items():
        if name != 'Baseline':
            drop = baseline_score - score
            if drop > max_drop:
                max_drop = drop
                most_impactful = name
            print(f"Ablation '{name}': F1 Score = {score:.4f} (Drop = {drop:.4f})")
            
    if most_impactful == "No Hierarchical Features":
        conclusion = "The Hierarchical Department Features contribute the most to the overall performance."
    elif most_impactful == "No RandomForest":
        conclusion = "The RandomForest model contributes the most to the ensemble's performance."
    elif most_impactful == "LGBM Only":
        conclusion = "The ensembling strategy contributes the most to the overall performance."
    else:
        conclusion = "No single ablated component showed a significant impact on performance."
        
    print(f"\nConclusion: {conclusion}")

if __name__ == '__main__':
    try:
        run_ablation_study()
    except Exception as e:
        print(f"\nA critical error occurred in the ablation study: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

