
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

# Mock TabPFN and LGBM for self-contained execution
class MockClassifier:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(random_state=42, n_estimators=5)
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

TabPFNClassifier = MockClassifier
LGBMClassifier = MockClassifier

# Define constants
ABLATION_INPUT_DIR = './input_ablation'
GOLD_LABELS_PATH = os.path.join(ABLATION_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data():
    """Creates a small, self-contained dataset for the ablation study."""
    if os.path.exists(ABLATION_INPUT_DIR):
        shutil.rmtree(ABLATION_INPUT_DIR)
    os.makedirs(ABLATION_INPUT_DIR, exist_ok=True)

    # subject_summary.csv
    subject_summary_data = {
        'TERM_CODE': [202301]*4 + [202302]*4,
        'SUBJECT_ID_SORT': ['MATH-101', 'CS-101', 'PHYS-101', 'ART-101']*2,
        'SOME_METRIC': [10, 20, 5, 2] * 2
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(ABLATION_INPUT_DIR, 'subject_summary.csv'), index=False)

    # gold_enrollment_train.csv
    gold_labels_data = {
        'TERM_CODE': [202301]*4 + [202302]*4,
        'SUBJECT_ID_SORT': ['MATH-101', 'CS-101', 'PHYS-101', 'ART-101']*2,
        'HIGH_ENROLLMENT': ['Y', 'Y', 'N', 'N'] * 2
    }
    pd.DataFrame(gold_labels_data).to_csv(GOLD_LABELS_PATH, index=False)

    # course_attributes.csv
    course_attributes_data = {
        'SUBJECT_ID_SORT': ['MATH-101', 'CS-101', 'PHYS-101', 'ART-101'],
        'COURSE_LEVEL': [100, 100, 100, 100]
    }
    pd.DataFrame(course_attributes_data).to_csv(os.path.join(ABLATION_INPUT_DIR, 'course_attributes.csv'), index=False)
    
    # instructor_attributes.csv (This feature is predictive)
    instructor_attributes_data = {
        'SUBJECT_ID_SORT': ['MATH-101', 'CS-101', 'PHYS-101', 'ART-101'],
        'INSTRUCTOR_RATING': [5, 5, 2, 2] 
    }
    pd.DataFrame(instructor_attributes_data).to_csv(os.path.join(ABLATION_INPUT_DIR, 'instructor_attributes.csv'), index=False)

    # Other required files (can be minimal)
    pd.DataFrame({'TERM_CODE': [], 'SUBJECT_ID_SORT': []}).to_csv(os.path.join(ABLATION_INPUT_DIR, 'course_summary.csv'), index=False)
    pd.DataFrame({'TERM_CODE': [], 'SUBJECT_ID_SORT': []}).to_csv(os.path.join(ABLATION_INPUT_DIR, 'faculty_summary.csv'), index=False)


def run_pipeline(use_instructor_data=True, force_random_split=False, rf_estimators=100):
    """
    Executes a single run of the training and validation pipeline with specified configurations.
    """
    # --- 1. Data Loading ---
    paths = {
        'subject_summary': os.path.join(ABLATION_INPUT_DIR, 'subject_summary.csv'),
        'course_attributes': os.path.join(ABLATION_INPUT_DIR, 'course_attributes.csv'),
        'instructor_attributes': os.path.join(ABLATION_INPUT_DIR, 'instructor_attributes.csv'),
        'course_summary': os.path.join(ABLATION_INPUT_DIR, 'course_summary.csv'),
        'faculty_summary': os.path.join(ABLATION_INPUT_DIR, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }
    
    # Ablation point for instructor data
    if not use_instructor_data:
        paths.pop('instructor_attributes')

    dataframes = {}
    for name, path in paths.items():
        try:
            if os.path.exists(path):
                dataframes[name] = pd.read_csv(path)
            else:
                dataframes[name] = pd.DataFrame()
        except Exception:
            dataframes[name] = pd.DataFrame()

    # --- 2. Feature Engineering ---
    subject_summary = dataframes.get('subject_summary')
    gold_labels = dataframes.get('gold_labels')
    course_attributes = dataframes.get('course_attributes')
    instructor_attributes = dataframes.get('instructor_attributes', pd.DataFrame()) # Handle missing df
    
    for df in [subject_summary, gold_labels]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    if not instructor_attributes.empty:
        data = pd.merge(data, instructor_attributes.add_suffix('_inst'), on='SUBJECT_ID_SORT', how='left')

    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # --- 3. Data Splitting ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    # Ablation point for splitting strategy
    if force_random_split or len(sorted_terms) < 2:
        train_df, val_df = train_test_split(data, test_size=0.5, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- 4. Model Training & Validation ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    # Ablation point for RandomForest complexity
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, n_estimators=rf_estimators)
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', SimpleImputer(strategy='median'), pipeline_num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), pipeline_cat_features)
        ], remainder='passthrough')),
        ('classifier', rf_classifier)
    ])
    pipeline_rf.fit(train_df, y_train)
    proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]

    # Dummy models for ensemble
    pred_ensemble = (proba_rf >= 0.5).astype(int)
    final_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    
    return final_score


def run_ablation_study():
    """
    Runs the ablation study by executing the pipeline with different configurations.
    """
    create_dummy_data()

    scenarios = {
        "Baseline (All Components)": {
            "use_instructor_data": True, 
            "force_random_split": False, 
            "rf_estimators": 100
        },
        "Ablation: No Instructor Data": {
            "use_instructor_data": False,
            "force_random_split": False,
            "rf_estimators": 100
        },
        "Ablation: Use Random Split": {
            "use_instructor_data": True,
            "force_random_split": True,
            "rf_estimators": 100
        },
        "Ablation: Simplified RandomForest": {
            "use_instructor_data": True,
            "force_random_split": False,
            "rf_estimators": 10
        }
    }

    results = {}
    for name, config in scenarios.items():
        try:
            score = run_pipeline(**config)
            results[name] = score
        except Exception as e:
            # print(f"Error in scenario '{name}': {e}", file=sys.stderr)
            results[name] = 0.0

    # --- Reporting ---
    print("--- Ablation Study Results ---")
    print(f"{'Configuration':<35} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 80)

    baseline_score = results.get("Baseline (All Components)", 0.0)
    performance_drops = {}

    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name:<35} | {score:<20.4f} | {drop:<20.4f}")
        if "Ablation" in name:
            performance_drops[name] = drop

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    if not performance_drops:
        print("Could not run ablation scenarios.")
    elif all(v == 0 for v in performance_drops.values()) and baseline_score > 0:
        print("No single component removal resulted in a significant performance drop.")
    elif baseline_score == 0:
        print("Study was inconclusive as baseline performance was zero.")
    else:
        most_impactful = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful.split(': ')[1]}'")
    
    # Cleanup
    shutil.rmtree(ABLATION_INPUT_DIR)

if __name__ == '__main__':
    run_ablation_study()
