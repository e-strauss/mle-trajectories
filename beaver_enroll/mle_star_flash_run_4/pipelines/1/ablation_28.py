
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import traceback
import sys
import subprocess
from lightgbm import LGBMClassifier


def create_dummy_dataframes():
    """Creates a set of in-memory pandas DataFrames with dummy data."""
    dfs = {}
    
    # Consistent subjects across terms
    subjects = [f'SUBJ-{i:02d}' for i in range(1, 11)]
    
    # subject_summary & gold_labels (training terms)
    train_data = []
    for term in [202201, 202202]:
        for i, subj in enumerate(subjects):
            train_data.append([term, subj, 'Y' if i % 2 == 0 else 'N'])
    
    # subject_summary & gold_labels (validation term)
    val_data = []
    for i, subj in enumerate(subjects):
        # Slightly different pattern for validation
        val_data.append([202301, subj, 'Y' if i % 3 == 0 else 'N'])
        
    all_summary_data = train_data + val_data
    df_summary = pd.DataFrame(all_summary_data, columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT_DUMMY'])

    dfs['subject_summary'] = df_summary[['TERM_CODE', 'SUBJECT_ID_SORT']]
    dfs['gold_labels'] = df_summary.rename(columns={'HIGH_ENROLLMENT_DUMMY': 'HIGH_ENROLLMENT'})
    
    # course_attributes
    attr_data = []
    for i, subj in enumerate(subjects):
        attr_data.append([subj, f'Campus {(i%2)+1}', f'Dept {(i%3)+1}'])
    dfs['course_attributes'] = pd.DataFrame(attr_data, columns=['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'])
    
    # course_summary
    course_data = []
    for term in [202201, 202202, 202301]:
        for i, subj in enumerate(subjects):
            num_courses = (i % 4) + 1
            for c in range(num_courses):
                course_data.append([term, subj, f'{subj}-C{c}', 50 + i*5 - c*2])
    dfs['course_summary'] = pd.DataFrame(course_data, columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'COURSE_ID', 'MAX_ENROLLMENT'])

    # faculty_summary
    faculty_data = []
    for term in [202201, 202202, 202301]:
        for i, subj in enumerate(subjects):
             faculty_data.append([term, subj, f'FACULTY_{i}', (i % 3) + 1])
    dfs['faculty_summary'] = pd.DataFrame(faculty_data, columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'FACULTY_ID', 'NUM_COURSES_TAUGHT'])
    
    return dfs

def feature_engineering(dfs):
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), left_on='SUBJECT_ID_SORT', right_on='SUBJECT_ID_SORT_attr', how='left').drop(columns=['SUBJECT_ID_SORT_attr'])
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'),
            AVG_SEATS=('MAX_ENROLLMENT', 'mean')).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'), AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_custom_pipeline(classifier, numeric_features, categorical_features, use_scaler=True, svd_components=100):
    """Builds a customizable scikit-learn pipeline."""
    if use_scaler:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
    else:
        numeric_transformer = SimpleImputer(strategy='median')
        
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('svd', TruncatedSVD(n_components=svd_components, random_state=42))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop')
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return pipeline

def run_scenario(use_scaler=True, svd_components=100, use_rf=True):
    """Executes a single training and validation scenario."""
    try:
        dataframes = create_dummy_dataframes()
        data = feature_engineering(dataframes)

        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        validation_term = sorted(data['TERM_CODE_INT'].unique())[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term]
        val_df = data[data['TERM_CODE_INT'] == validation_term]

        y_train = train_df['HIGH_ENROLLMENT']
        y_val = val_df['HIGH_ENROLLMENT']

        cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr')] if col in data.columns]
        num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

        # --- Model Training ---
        probas = []

        # LGBM Model (always present)
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=1, verbosity=-1)
        pipeline_lgbm = build_custom_pipeline(lgbm_classifier, num_features, cat_features, use_scaler, svd_components)
        pipeline_lgbm.fit(train_df, y_train)
        probas.append(pipeline_lgbm.predict_proba(val_df)[:, 1])

        # RandomForest Model (optional)
        if use_rf:
            rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
            pipeline_rf = build_custom_pipeline(rf_classifier, num_features, cat_features, use_scaler, svd_components)
            pipeline_rf.fit(train_df, y_train)
            probas.append(pipeline_rf.predict_proba(val_df)[:, 1])

        # --- Ensembling ---
        proba_ensemble = np.mean(probas, axis=0)
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    except Exception as e:
        print(f"Scenario failed: {e}", file=sys.stderr)
        return 0.0

def main():
    """Main function to run the ablation study."""
    results = {}

    # Baseline: All components enabled
    print("Running: Baseline (RobustScaler, SVD n=100, RF in ensemble)...")
    results['Baseline'] = run_scenario(use_scaler=True, svd_components=100, use_rf=True)

    # Ablation 1: No RobustScaler
    print("Running: Ablation (No RobustScaler)...")
    results['No RobustScaler'] = run_scenario(use_scaler=False, svd_components=100, use_rf=True)
    
    # Ablation 2: Fewer SVD Components
    print("Running: Ablation (SVD n=5)...")
    results['Fewer SVD Components'] = run_scenario(use_scaler=True, svd_components=5, use_rf=True)

    # Ablation 3: No RandomForest in Ensemble
    print("Running: Ablation (No RandomForest)...")
    results['No RandomForest'] = run_scenario(use_scaler=True, svd_components=100, use_rf=False)

    # --- Print Results ---
    baseline_score = results.get('Baseline', 0.0)
    performance_drops = {
        'RobustScaler': baseline_score - results.get('No RobustScaler', 0.0),
        'High-dim SVD': baseline_score - results.get('Fewer SVD Components', 0.0),
        'RandomForest Model': baseline_score - results.get('No RandomForest', 0.0),
    }

    print("\n--- Ablation Study Results ---")
    print(f"{'Configuration':<30} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 75)
    print(f"{'Baseline':<30} | {results.get('Baseline', 0.0):<20.4f} | {0.0:<20.4f}")
    print(f"{'Ablation: No RobustScaler':<30} | {results.get('No RobustScaler', 0.0):<20.4f} | {performance_drops['RobustScaler']:<20.4f}")
    print(f"{'Ablation: Fewer SVD Components':<30} | {results.get('Fewer SVD Components', 0.0):<20.4f} | {performance_drops['High-dim SVD']:<20.4f}")
    print(f"{'Ablation: No RandomForest':<30} | {results.get('No RandomForest', 0.0):<20.4f} | {performance_drops['RandomForest Model']:<20.4f}")
    
    # --- Conclusion ---
    print("\n--- Conclusion ---")
    if all(drop <= 0 for drop in performance_drops.values()):
        if baseline_score == 0.0:
            print("Study was inconclusive as baseline performance was zero.")
        else:
            print("No component removal resulted in a significant performance drop.")
    else:
        most_impactful_component = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

if __name__ == '__main__':
    main()
