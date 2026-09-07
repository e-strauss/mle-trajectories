
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

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
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
    """Merges dataframes and creates a combined set of features."""
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

def build_classifier_pipeline(classifier, numeric_features, categorical_features, imputation_strategy='median'):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy=imputation_strategy)
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
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def run_single_experiment(config):
    """
    Runs a single training and validation experiment based on a configuration dictionary.
    """
    try:
        # --- 1. Load and Prepare Data ---
        dataframes = load_data(DEFAULT_TRAIN_DIR)
        data = feature_engineering(dataframes)

        # --- 2. Data Splitting (Time-based) ---
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

        if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2 or val_df['HIGH_ENROLLMENT'].nunique() < 2:
            return 0.0

        y_train = train_df['HIGH_ENROLLMENT']
        y_val = val_df['HIGH_ENROLLMENT']

        # --- 3. Model Training ---
        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        # --- Model 1: RandomForest ---
        rf_class_weight = 'balanced' if config['use_class_weight'] else None
        rf_classifier = RandomForestClassifier(random_state=42, class_weight=rf_class_weight, n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features, config['imputation_strategy'])
        pipeline_rf.fit(train_df, y_train)

        # --- Model 2: LightGBM ---
        lgbm_class_weight = 'balanced' if config['use_class_weight'] else None
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight=lgbm_class_weight, n_jobs=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features, config['imputation_strategy'])
        pipeline_lgbm.fit(train_df, y_train)

        # --- Model 3: TabPFN ---
        tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
        tabpfn_num_features = ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY']
        tabpfn_cat_features = [f for f in tabpfn_cat_features if f in train_df.columns]
        tabpfn_num_features = [f for f in tabpfn_num_features if f in train_df.columns]
        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features
        train_df_tabpfn = train_df.copy()
        val_df_tabpfn = val_df.copy()

        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:, :100]
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:, :100]

        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

        # --- 4. Validation and Ensembling ---
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1] if config['use_rf_in_ensemble'] else 0
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        # Ensemble probabilities
        if config['use_rf_in_ensemble']:
            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        else:
            proba_ensemble = (proba_lgbm + proba_tabpfn) / 2
            
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)

    except Exception:
        # print(f"Error during experiment '{config['name']}': {e}", file=sys.stderr)
        # print(traceback.format_exc(), file=sys.stderr)
        return 0.0

def run_ablation_study():
    """
    Runs an ablation study by executing multiple experiment configurations.
    """
    # --- Install Dependencies ---
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
        # Make these available globally for run_single_experiment
        globals()['TabPFNClassifier'] = TabPFNClassifier
        globals()['LGBMClassifier'] = LGBMClassifier
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        return

    # --- Define Experiments ---
    experiments = [
        {
            'name': 'Baseline',
            'use_class_weight': True,
            'imputation_strategy': 'median',
            'use_rf_in_ensemble': True,
        },
        {
            'name': 'Ablation: No Class Weighting',
            'use_class_weight': False,
            'imputation_strategy': 'median',
            'use_rf_in_ensemble': True,
        },
        {
            'name': 'Ablation: Use Mean Imputation',
            'use_class_weight': True,
            'imputation_strategy': 'mean',
            'use_rf_in_ensemble': True,
        },
        {
            'name': 'Ablation: No RandomForest in Ensemble',
            'use_class_weight': True,
            'imputation_strategy': 'median',
            'use_rf_in_ensemble': False,
        }
    ]

    results = []
    baseline_score = 0

    print("--- Ablation Study Results ---")
    for i, config in enumerate(experiments):
        score = run_single_experiment(config)
        if i == 0:  # Baseline
            baseline_score = score
        
        perf_drop = baseline_score - score
        results.append({
            'name': config['name'],
            'score': score,
            'drop': perf_drop
        })
        print(f"{config['name']}: {score:.4f} (Performance Drop: {perf_drop:.4f})")

    # --- Determine Most Important Component ---
    # Filter out baseline for finding the max drop
    ablation_results = [res for res in results if 'Ablation' in res['name']]
    if not ablation_results:
        print("\nNo ablation experiments were run.")
        return
        
    most_impactful = max(ablation_results, key=lambda x: x['drop'])

    component_map = {
        'Ablation: No Class Weighting': 'Class Weighting',
        'Ablation: Use Mean Imputation': 'Median Imputation Strategy',
        'Ablation: No RandomForest in Ensemble': 'RandomForest in Ensemble'
    }
    
    conclusion = "The component that contributes the most to the overall performance is: '{}'".format(
        component_map.get(most_impactful['name'], 'Unknown')
    )
    
    print("\n--- Conclusion ---")
    print(conclusion)


if __name__ == '__main__':
    run_ablation_study()
