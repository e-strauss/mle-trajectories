
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import traceback
import sys
import subprocess

def create_dummy_data():
    """Creates a small, self-contained dummy dataset for the ablation study."""
    dfs = {}
    dfs['subject_summary'] = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'ART-303']
    })
    dfs['gold_labels'] = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202301, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202', 'ART-303'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'Y', 'N']
    })
    dfs['course_attributes'] = pd.DataFrame({
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'ART-303'],
        'ATTRIBUTE_1': ['Core', 'Core', 'Elective'],
        'ATTRIBUTE_2': ['Beginner', 'Intermediate', 'Advanced']
    })
    dfs['course_summary'] = pd.DataFrame({
        'TERM_CODE': [202201, 202201, 202301, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202'],
        'COURSE_ID': [1, 2, 3, 4],
        'MAX_ENROLLMENT': [100, 50, 120, 40]
    })
    dfs['faculty_summary'] = pd.DataFrame({
        'TERM_CODE': [202201, 202301],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101'],
        'FACULTY_ID': [10, 11],
        'NUM_COURSES_TAUGHT': [2, 3]
    })
    return dfs

def feature_engineering(dfs):
    """Merges dataframes and creates features."""
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    data = pd.merge(data, course_attributes.add_suffix('_attr'), left_on='SUBJECT_ID_SORT', right_on='SUBJECT_ID_SORT_attr', how='left')
    
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
        NUM_COURSES=('COURSE_ID', 'nunique'),
        TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
    ).reset_index()
    data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
        NUM_FACULTY=('FACULTY_ID', 'nunique')
    ).reset_index()
    data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features, use_scaler=True, use_svd=True):
    """Builds a scikit-learn pipeline with optional scaling and SVD."""
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaler:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_steps = [
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ]
    if use_svd:
        # Use a small n_components for dummy data
        categorical_steps.append(('svd', TruncatedSVD(n_components=2, random_state=42)))
    categorical_transformer = Pipeline(steps=categorical_steps)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        n_jobs=-1
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def run_ablation_study():
    """Main function to execute the ablation study."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print(f"Dependency installation failed: {e}", file=sys.stderr)
        return

    dataframes = create_dummy_data()
    data = feature_engineering(dataframes)

    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    validation_term = data['TERM_CODE_INT'].max()
    train_df = data[data['TERM_CODE_INT'] < validation_term]
    val_df = data[data['TERM_CODE_INT'] == validation_term]

    if train_df.empty or val_df.empty:
        print("Dummy data split resulted in empty train/val set. Cannot proceed.", file=sys.stderr)
        return

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- Define Features ---
    pipeline_cat_features = [c for c in data.columns if data[c].dtype == 'object' and c != 'HIGH_ENROLLMENT']
    pipeline_num_features = [c for c in data.select_dtypes(include=np.number).columns if c not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    # --- Define Ablation Scenarios ---
    scenarios = [
        {'name': 'Baseline (Scaler + SVD)', 'use_scaler': True, 'use_svd': True},
        {'name': 'Ablation: No StandardScaler', 'use_scaler': False, 'use_svd': True},
        {'name': 'Ablation: No TruncatedSVD', 'use_scaler': True, 'use_svd': False},
    ]
    
    results = {}

    for scenario in scenarios:
        try:
            # --- Model Training ---
            rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
            pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features, use_scaler=scenario['use_scaler'], use_svd=scenario['use_svd'])
            pipeline_rf.fit(train_df, y_train)

            lgbm_classifier = LGBMClassifier(random_state=42, n_jobs=-1)
            pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features, use_scaler=scenario['use_scaler'], use_svd=scenario['use_svd'])
            pipeline_lgbm.fit(train_df, y_train)

            # --- Validation ---
            proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
            proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
            
            # Simple 2-model ensemble for this study
            proba_ensemble = (proba_rf + proba_lgbm) / 2
            pred_ensemble = (proba_ensemble >= 0.5).astype(int)

            score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
            results[scenario['name']] = score
        except Exception as e:
            print(f"Error during scenario '{scenario['name']}': {e}", file=sys.stderr)
            results[scenario['name']] = 0.0

    # --- Print Results and Conclusion ---
    print("\n--- Ablation Study Results ---\n")
    baseline_score = results.get('Baseline (Scaler + SVD)', 0.0)
    performance_drops = {}

    print(f"{'Configuration':<30} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 75)

    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name:<30} | {score:<20.4f} | {drop:<20.4f}")
        if name != 'Baseline (Scaler + SVD)':
            performance_drops[name.replace('Ablation: ', '')] = drop
            
    if not performance_drops:
        most_impactful = "N/A"
    else:
        most_impactful = max(performance_drops, key=performance_drops.get)

    print("\n--- Conclusion ---\n")
    if all(v <= 0 for v in performance_drops.values()) and baseline_score > 0:
        print("The baseline configuration performed best. Adding StandardScaler and TruncatedSVD is beneficial.")
    elif baseline_score == 0:
        print("Study was inconclusive as baseline performance was zero.")
    else:
        print(f"The component that contributes the most to the overall performance is: '{most_impactful}'")

if __name__ == '__main__':
    try:
        run_ablation_study()
    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
