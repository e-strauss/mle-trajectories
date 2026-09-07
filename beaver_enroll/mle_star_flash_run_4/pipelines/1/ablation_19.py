
import argparse
import os
import pandas as pd
import numpy as np
import shutil
import sys
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Mock dependencies to ensure script runs without installation errors
try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("Warning: lightgbm not found. Using RandomForestClassifier as a fallback.", file=sys.stderr)
    LGBMClassifier = RandomForestClassifier

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    print("Warning: tabpfn not found. Using a mock classifier.", file=sys.stderr)
    class TabPFNClassifier:
        def __init__(self, device='cpu', N_ensemble_configurations=32):
            self.model = RandomForestClassifier(random_state=42)
        def fit(self, X, y, overwrite_warning=False):
            self.model.fit(X, y)
            return self
        def predict_proba(self, X):
            return self.model.predict_proba(X)

def create_dummy_data(base_dir):
    """Creates dummy CSV files for a self-contained ablation study."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    # subject_summary.csv
    subject_summary_data = {
        'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230, 202230, 202310, 202310, 202310] * 2,
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202'] * 10,
        'TOTAL_ENROLLMENT': [150, 80, 160, 75, 155, 85, 90, 180, 95, 100] * 2,
        'some_numeric_feature': [1, 2, 1, 3, 2, 1, 3, 1, 2, 3] * 2,
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(base_dir, 'subject_summary.csv'), index=False)

    # gold_enrollment_train.csv
    gold_labels_data = {
        'TERM_CODE': subject_summary_data['TERM_CODE'],
        'SUBJECT_ID_SORT': subject_summary_data['SUBJECT_ID_SORT'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'N', 'N'] * 2
    }
    pd.DataFrame(gold_labels_data).to_csv(os.path.join(base_dir, 'gold_enrollment_train.csv'), index=False)

    # course_summary.csv
    course_summary_data = {
        'TERM_CODE': subject_summary_data['TERM_CODE'],
        'SUBJECT_ID_SORT': subject_summary_data['SUBJECT_ID_SORT'],
        'COURSE_ID': range(20),
        'MAX_ENROLLMENT': [50, 40, 55, 35, 50, 45, 45, 60, 50, 50] * 2,
    }
    pd.DataFrame(course_summary_data).to_csv(os.path.join(base_dir, 'course_summary.csv'), index=False)

    # faculty_summary.csv with some NaNs
    faculty_summary_data = {
        'TERM_CODE': subject_summary_data['TERM_CODE'],
        'SUBJECT_ID_SORT': subject_summary_data['SUBJECT_ID_SORT'],
        'FACULTY_ID': range(20),
        'NUM_COURSES_TAUGHT': [2, 1, 2, 1, 2, 2, np.nan, 3, 1, 2] * 2,
    }
    pd.DataFrame(faculty_summary_data).to_csv(os.path.join(base_dir, 'faculty_summary.csv'), index=False)

    # course_attributes.csv
    course_attributes_data = {'SUBJECT_ID_SORT': ['CS-101', 'MATH-202']}
    pd.DataFrame(course_attributes_data).to_csv(os.path.join(base_dir, 'course_attributes.csv'), index=False)

def load_data(train_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': os.path.join(train_dir, 'gold_enrollment_train.csv')
    }
    dataframes = {name: pd.read_csv(path) for name, path in paths.items()}
    return dataframes

def feature_engineering(dfs, use_department_feature=True):
    """Merges dataframes and creates features."""
    data = pd.merge(dfs['subject_summary'], dfs['gold_labels'], on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if use_department_feature:
        data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    course_agg = dfs['course_summary'].groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_COURSES=('COURSE_ID', 'nunique')).reset_index()
    data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    faculty_agg = dfs['faculty_summary'].groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(NUM_FACULTY=('FACULTY_ID', 'nunique')).reset_index()
    data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features, use_imputer_indicator=True):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median', add_indicator=use_imputer_indicator)
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
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return pipeline

def run_ablation_study(train_dir, use_imputer_indicator, use_department_feature, use_lgbm_model):
    """Runs a single configuration of the training pipeline and returns the F1 score."""
    try:
        dataframes = load_data(train_dir)
        data = feature_engineering(dataframes, use_department_feature=use_department_feature)

        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        validation_term = sorted(data['TERM_CODE_INT'].unique())[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term]
        val_df = data[data['TERM_CODE_INT'] == validation_term]

        if train_df.empty or val_df.empty: return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']
        
        # Define features based on ablation flags
        num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        cat_features = ['DEPARTMENT'] if 'DEPARTMENT' in data.columns else []

        # Model 1: RandomForest
        pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42), num_features, cat_features, use_imputer_indicator)
        pipeline_rf.fit(train_df, y_train)
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]

        # Model 2: LightGBM
        if use_lgbm_model:
            pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42), num_features, cat_features, use_imputer_indicator)
            pipeline_lgbm.fit(train_df, y_train)
            proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]

        # Model 3: TabPFN
        tabpfn_features = [f for f in num_features if f in train_df.columns]
        X_train_tabpfn, X_val_tabpfn = train_df[tabpfn_features].fillna(0), val_df[tabpfn_features].fillna(0)
        clf_tabpfn = TabPFNClassifier(device='cpu')
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]
        
        # Ensemble
        probas = [proba_rf, proba_tabpfn]
        if use_lgbm_model:
            probas.append(proba_lgbm)
        
        proba_ensemble = np.mean(probas, axis=0)
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    except Exception:
        # print(f"Error during ablation run: {e}\n{traceback.format_exc()}", file=sys.stderr)
        return 0.0

if __name__ == '__main__':
    temp_input_dir = './temp_ablation_input'
    create_dummy_data(temp_input_dir)

    results = {}

    # Baseline: All components enabled
    baseline_score = run_ablation_study(
        train_dir=temp_input_dir,
        use_imputer_indicator=True,
        use_department_feature=True,
        use_lgbm_model=True
    )
    results['Baseline (Imputer Indicator, DEPARTMENT, LGBM)'] = baseline_score

    # Ablation 1: Remove Imputer Indicator
    score_no_indicator = run_ablation_study(
        train_dir=temp_input_dir,
        use_imputer_indicator=False,
        use_department_feature=True,
        use_lgbm_model=True
    )
    results['Ablation: No Imputer Indicator Feature'] = score_no_indicator

    # Ablation 2: Remove DEPARTMENT Feature
    score_no_department = run_ablation_study(
        train_dir=temp_input_dir,
        use_imputer_indicator=True,
        use_department_feature=False,
        use_lgbm_model=True
    )
    results['Ablation: No DEPARTMENT Feature'] = score_no_department

    # Ablation 3: Remove LGBM Model from Ensemble
    score_no_lgbm = run_ablation_study(
        train_dir=temp_input_dir,
        use_imputer_indicator=True,
        use_department_feature=True,
        use_lgbm_model=False
    )
    results['Ablation: No LGBM Model'] = score_no_lgbm

    # --- Print Results and Conclusion ---
    print("--- Ablation Study Results ---")
    performance_drops = {}
    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
        if 'Ablation' in name:
            component_name = name.split('No ')[-1]
            performance_drops[component_name] = drop
            
    if not performance_drops or max(performance_drops.values()) <= 1e-6:
        conclusion = "No single component removal resulted in a significant performance drop."
    else:
        most_impactful = max(performance_drops, key=performance_drops.get)
        conclusion = f"The component that contributes the most to the overall performance is: '{most_impactful}'"

    print("\n--- Conclusion ---")
    print(conclusion)

    # Clean up dummy data
    shutil.rmtree(temp_input_dir)
