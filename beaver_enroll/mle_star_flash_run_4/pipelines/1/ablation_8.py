
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
warnings.filterwarnings("ignore", category=UserWarning)

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

# --- Original Helper Functions (with modifications for ablation) ---

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
        except Exception:
            dataframes[name] = pd.DataFrame()
    return dataframes

def feature_engineering(dfs, use_course_attributes=True):
    """
    Merges dataframes and creates features. The use of course_attributes can be toggled.
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

    if use_course_attributes and not course_attributes.empty:
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
        data['NUM_FACULTY'], data['AVG_FACULTY_LOAD'] = np.nan, np.nan

    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']
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
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def run_experiment(ablation_name="Baseline", use_multi_split=True, use_weighted_average=True, use_course_attributes=True):
    """
    Runs a single experiment with a specific configuration.
    """
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

    dataframes = load_data(DEFAULT_TRAIN_DIR)
    
    try:
        data = feature_engineering(dataframes, use_course_attributes=use_course_attributes)
    except ValueError:
        return 0.0

    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    if len(sorted_terms) < 2:
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data.get('HIGH_ENROLLMENT'))
    else:
        validation_term = sorted_terms[-1]
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

    if val_df.empty or val_df['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    train_dfs = []
    if len(sorted_terms) >= 2:
        if use_multi_split:
            num_splits = min(3, len(sorted_terms) - 1)
            for i in range(1, num_splits + 1):
                cutoff_term = sorted_terms[-i]
                train_df_split = data[data['TERM_CODE_INT'] < cutoff_term].copy()
                if not train_df_split.empty and train_df_split['HIGH_ENROLLMENT'].nunique() > 1:
                    train_dfs.append(train_df_split)
        else: # Single split (all data before validation term)
            cutoff_term = sorted_terms[-1]
            train_df_single = data[data['TERM_CODE_INT'] < cutoff_term].copy()
            if not train_df_single.empty and train_df_single['HIGH_ENROLLMENT'].nunique() > 1:
                train_dfs.append(train_df_single)
    else: # Random split case
        if not train_df.empty and train_df['HIGH_ENROLLMENT'].nunique() > 1:
            train_dfs.append(train_df)
            
    if not train_dfs:
        return 0.0
    
    y_val = val_df['HIGH_ENROLLMENT']
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    tabpfn_num_features = ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY']
    tabpfn_cat_features = [f for f in tabpfn_cat_features if f in data.columns]
    tabpfn_num_features = [f for f in tabpfn_num_features if f in data.columns]
    all_tabpfn_features = (tabpfn_num_features + tabpfn_cat_features)[:100]
    
    split_probas_rf, split_probas_lgbm, split_probas_tabpfn = [], [], []

    for train_df in train_dfs:
        y_train = train_df['HIGH_ENROLLMENT']
        
        pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)
        split_probas_rf.append(pipeline_rf.predict_proba(val_df)[:, 1])

        pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        split_probas_lgbm.append(pipeline_lgbm.predict_proba(val_df)[:, 1])

        train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)
        
        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0)
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0)
        
        if 0 < X_train_tabpfn.shape[0] < 1024 and X_train_tabpfn.shape[1] > 0:
            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
            split_probas_tabpfn.append(clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1])
        else:
            split_probas_tabpfn.append(np.full(len(val_df), 0.5))

    weights = None
    if use_weighted_average and len(train_dfs) > 1:
        num_splits = len(train_dfs)
        weights = np.arange(num_splits, 0, -1) / np.sum(np.arange(num_splits, 0, -1))
        
    avg_proba_rf = np.average(np.array(split_probas_rf), axis=0, weights=weights)
    avg_proba_lgbm = np.average(np.array(split_probas_lgbm), axis=0, weights=weights)
    avg_proba_tabpfn = np.average(np.array(split_probas_tabpfn), axis=0, weights=weights)

    proba_ensemble = (avg_proba_rf + avg_proba_lgbm + avg_proba_tabpfn) / 3
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    
    score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    return score

def main_ablation():
    """
    Main function to run the ablation study.
    """
    print("--- Running Ablation Study ---")
    
    try:
        baseline_score = run_experiment(
            ablation_name="Baseline",
            use_multi_split=True,
            use_weighted_average=True,
            use_course_attributes=True
        )

        ablation_1_score = run_experiment(
            ablation_name="No Multi-Split Training",
            use_multi_split=False,
            use_weighted_average=False, # Not applicable with single split
            use_course_attributes=True
        )

        ablation_2_score = run_experiment(
            ablation_name="No Weighted Averaging",
            use_multi_split=True,
            use_weighted_average=False,
            use_course_attributes=True
        )

        ablation_3_score = run_experiment(
            ablation_name="No Course Attributes Data",
            use_multi_split=True,
            use_weighted_average=True,
            use_course_attributes=False
        )

        results = [
            {"name": "Multi-Split Training", "score": ablation_1_score, "drop": baseline_score - ablation_1_score},
            {"name": "Weighted Averaging of Splits", "score": ablation_2_score, "drop": baseline_score - ablation_2_score},
            {"name": "Course Attributes Data", "score": ablation_3_score, "drop": baseline_score - ablation_3_score},
        ]

        print("\n--- Ablation Study Results ---")
        print(f"Baseline (All Components): {baseline_score:.4f} (Performance Drop: 0.0000)")
        print(f"Ablation: No Multi-Split Training: {results[0]['score']:.4f} (Performance Drop: {results[0]['drop']:.4f})")
        print(f"Ablation: No Weighted Averaging of Splits: {results[1]['score']:.4f} (Performance Drop: {results[1]['drop']:.4f})")
        print(f"Ablation: No Course Attributes Data: {results[2]['score']:.4f} (Performance Drop: {results[2]['drop']:.4f})")
        
        if baseline_score == 0 and all(r['drop'] == 0 for r in results):
             print("\n--- Conclusion ---")
             print("The ablation study was inconclusive as all models scored 0.0. A fundamental issue may exist in the data or pipeline.")
        else:
            most_impactful = max(results, key=lambda x: x['drop'])
            print("\n--- Conclusion ---")
            print(f"The component that contributes the most to the overall performance is: '{most_impactful['name']}'")

    except Exception as e:
        print(f"A critical error occurred during the ablation study: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

if __name__ == '__main__':
    main_ablation()
