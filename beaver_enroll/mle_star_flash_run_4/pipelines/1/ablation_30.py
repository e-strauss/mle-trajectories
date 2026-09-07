
import os
import subprocess
import sys
import pandas as pd
import shutil
import re

def create_dummy_data():
    """Creates a small, self-contained dummy dataset for the ablation study."""
    os.makedirs('./input', exist_ok=True)

    subject_summary_data = {
        'TERM_CODE': [202201, 202202, 202301] * 3,
        'SUBJECT_ID_SORT': ['CS-101'] * 3 + ['MATH-202'] * 3 + ['BIO-101'] * 3,
        'TOTAL_ENROLLMENT': [80, 85, 90, 50, 45, 60, 120, 110, 100],
    }
    pd.DataFrame(subject_summary_data).to_csv('./input/subject_summary.csv', index=False)

    gold_labels_data = {
        'TERM_CODE': [202201, 202202, 202301] * 3,
        'SUBJECT_ID_SORT': ['CS-101'] * 3 + ['MATH-202'] * 3 + ['BIO-101'] * 3,
        'HIGH_ENROLLMENT': ['N', 'N', 'Y', 'N', 'N', 'N', 'Y', 'Y', 'N'],
    }
    pd.DataFrame(gold_labels_data).to_csv('./input/gold_enrollment_train.csv', index=False)

    course_summary_data = {
        'TERM_CODE': [202201, 202202, 202301] * 3,
        'SUBJECT_ID_SORT': ['CS-101'] * 3 + ['MATH-202'] * 3 + ['BIO-101'] * 3,
        'COURSE_ID': [f'C{i}' for i in range(9)],
        'MAX_ENROLLMENT': [90, 90, 100, 60, 60, 70, 130, 130, 110],
    }
    pd.DataFrame(course_summary_data).to_csv('./input/course_summary.csv', index=False)

    faculty_summary_data = {
        'TERM_CODE': [202201, 202202, 202301] * 3,
        'SUBJECT_ID_SORT': ['CS-101'] * 3 + ['MATH-202'] * 3 + ['BIO-101'] * 3,
        'FACULTY_ID': [f'F{i}' for i in range(9)],
        'NUM_COURSES_TAUGHT': [2, 2, 3, 1, 1, 2, 4, 3, 3],
    }
    pd.DataFrame(faculty_summary_data).to_csv('./input/faculty_summary.csv', index=False)

    course_attributes_data = {
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'BIO-101'],
        'CAMPUS_ID_DESC': ['Main', 'Main', 'Science'],
    }
    pd.DataFrame(course_attributes_data).to_csv('./input/course_attributes.csv', index=False)


def get_main_script_content():
    """Returns the content of the main training script as a string."""
    return """
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

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

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
        except Exception as e:
            dataframes[name] = pd.DataFrame()
    return dataframes

def feature_engineering(dfs):
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files (subject_summary or gold_enrollment_train) are missing or empty.")

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
    
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data = data.sort_values(['SUBJECT_ID_SORT', 'TERM_CODE_INT'])

    # ABLATION_TARGET_1: Historical Features using Expanding Window
    historical_features = ['TOTAL_ENROLLMENT', 'TOTAL_SEATS', 'NUM_COURSES']
    grouped_by_subject = data.groupby('SUBJECT_ID_SORT')
    for feature in historical_features:
        if feature in data.columns:
            expanding_mean = grouped_by_subject[feature].expanding().mean()
            data[f'TEMP_AVG'] = expanding_mean.reset_index(level=0, drop=True)
            data[f'HIST_AVG_{feature}'] = data.groupby('SUBJECT_ID_SORT')[f'TEMP_AVG'].shift(1)
            data.drop(columns=[f'TEMP_AVG'], inplace=True)

    # ABLATION_TARGET_2: Departmental Context Features
    if 'DEPARTMENT' in data.columns and 'TOTAL_ENROLLMENT' in data.columns and 'TOTAL_SEATS' in data.columns:
        dept_term_summary = data.groupby(['TERM_CODE_INT', 'DEPARTMENT']).agg(
            DEPT_TOTAL_ENROLLMENT=('TOTAL_ENROLLMENT', 'sum'),
            DEPT_TOTAL_SEATS=('TOTAL_SEATS', 'sum')
        ).reset_index()
        data = pd.merge(data, dept_term_summary, on=['TERM_CODE_INT', 'DEPARTMENT'], how='left')
        data['ENROLLMENT_SHARE_IN_DEPT'] = data['TOTAL_ENROLLMENT'] / data['DEPT_TOTAL_ENROLLMENT']
        data['SEATS_SHARE_IN_DEPT'] = data['TOTAL_SEATS'] / data['DEPT_TOTAL_SEATS']
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
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

def main():
    parser = argparse.ArgumentParser(description='Predict high enrollment for courses.')
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR)
    args = parser.parse_args()

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except Exception as e:
        print('Final Validation Performance: 0.0')
        return

    dataframes = load_data(args.train_data_dir)
    try:
        data = feature_engineering(dataframes)
    except ValueError as e:
        print('Final Validation Performance: 0.0')
        return

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

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        print('Final Validation Performance: 0.0')
        return

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    # ABLATION_TARGET_3: LGBM Model
    lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm.fit(train_df, y_train)

    tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT']
    tabpfn_num_features = [c for c in pipeline_num_features if 'HIST_' not in c and 'SHARE' not in c]
    tabpfn_cat_features = [f for f in tabpfn_cat_features if f in train_df.columns]
    tabpfn_num_features = [f for f in tabpfn_num_features if f in train_df.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

    train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df_tabpfn[col])
        train_df_tabpfn[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

    X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0)
    X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0)

    if X_train_tabpfn.shape[1] > 100:
        X_train_tabpfn = X_train_tabpfn.iloc[:, :100]
        X_val_tabpfn = X_val_tabpfn.iloc[:, :100]

    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

    if not y_val.empty:
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        final_validation_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        print(f'Final Validation Performance: {final_validation_score}')
    else:
        print('Final Validation Performance: 0.0')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print('Final Validation Performance: 0.0')

"""

def run_scenario(scenario_name, modifications):
    """Runs a modified version of the training script and returns the F1 score."""
    script_content = get_main_script_content()
    for original, replacement in modifications.items():
        script_content = script_content.replace(original, replacement)

    temp_script_path = 'temp_train.py'
    with open(temp_script_path, 'w') as f:
        f.write(script_content)

    result = subprocess.run([sys.executable, temp_script_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"--- Error in scenario: {scenario_name} ---")
        print(result.stderr)
        return 0.0

    output = result.stdout
    score = 0.0
    for line in output.splitlines():
        if 'Final Validation Performance:' in line:
            try:
                score = float(line.split(':')[1].strip())
            except (ValueError, IndexError):
                score = 0.0
    return score

def main():
    """Main function to run the ablation study."""
    create_dummy_data()

    # Define ablation scenarios
    mod_no_hist_features = {
        """    # ABLATION_TARGET_1: Historical Features using Expanding Window
    historical_features = ['TOTAL_ENROLLMENT', 'TOTAL_SEATS', 'NUM_COURSES']
    grouped_by_subject = data.groupby('SUBJECT_ID_SORT')
    for feature in historical_features:
        if feature in data.columns:
            expanding_mean = grouped_by_subject[feature].expanding().mean()
            data[f'TEMP_AVG'] = expanding_mean.reset_index(level=0, drop=True)
            data[f'HIST_AVG_{feature}'] = data.groupby('SUBJECT_ID_SORT')[f'TEMP_AVG'].shift(1)
            data.drop(columns=[f'TEMP_AVG'], inplace=True)""":
        "    # Historical features removed"
    }

    mod_no_dept_context = {
        """    # ABLATION_TARGET_2: Departmental Context Features
    if 'DEPARTMENT' in data.columns and 'TOTAL_ENROLLMENT' in data.columns and 'TOTAL_SEATS' in data.columns:
        dept_term_summary = data.groupby(['TERM_CODE_INT', 'DEPARTMENT']).agg(
            DEPT_TOTAL_ENROLLMENT=('TOTAL_ENROLLMENT', 'sum'),
            DEPT_TOTAL_SEATS=('TOTAL_SEATS', 'sum')
        ).reset_index()
        data = pd.merge(data, dept_term_summary, on=['TERM_CODE_INT', 'DEPARTMENT'], how='left')
        data['ENROLLMENT_SHARE_IN_DEPT'] = data['TOTAL_ENROLLMENT'] / data['DEPT_TOTAL_ENROLLMENT']
        data['SEATS_SHARE_IN_DEPT'] = data['TOTAL_SEATS'] / data['DEPT_TOTAL_SEATS']""":
        "    # Departmental context features removed"
    }

    mod_no_lgbm = {
        "    # ABLATION_TARGET_3: LGBM Model\n    lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)\n    pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)\n    pipeline_lgbm.fit(train_df, y_train)": 
        "# LGBM Model Removed",
        "        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]": "        proba_lgbm = 0",
        "        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3": "        proba_ensemble = (proba_rf + proba_tabpfn) / 2"
    }

    scenarios = {
        "Baseline": {},
        "No Historical Features": mod_no_hist_features,
        "No Departmental Context": mod_no_dept_context,
        "No LGBM Model": mod_no_lgbm,
    }

    results = {}
    print("--- Running Ablation Study ---")
    for name, mods in scenarios.items():
        score = run_scenario(name, mods)
        results[name] = score
        print(f"Scenario: {name:<25} | Macro F1 Score: {score:.4f}")

    baseline_score = results.get("Baseline", 0.0)
    
    if baseline_score == 0.0:
        print("\n--- Conclusion ---")
        print("Study was inconclusive as baseline performance was zero.")
        return

    performance_drops = {
        name: baseline_score - score for name, score in results.items() if name != "Baseline"
    }

    # Find the component with the largest performance drop
    if not performance_drops:
        most_impactful_component = "N/A"
    else:
        most_impactful_component = max(performance_drops, key=performance_drops.get)

    print("\n--- Ablation Study Results Summary ---")
    header = f"{'Configuration':<25} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}"
    print(header)
    print("-" * len(header))
    print(f"{'Baseline':<25} | {baseline_score:<20.4f} | {0.0:<20.4f}")
    for name, drop in performance_drops.items():
        score = results[name]
        print(f"{name:<25} | {score:<20.4f} | {drop:<20.4f}")
    
    print("\n--- Conclusion ---")
    if all(drop <= 0 for drop in performance_drops.values()):
        print("No single component removal resulted in a significant performance drop.")
    else:
        print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up created files and directories
        if os.path.exists('./input'):
            shutil.rmtree('./input')
        if os.path.exists('temp_train.py'):
            os.remove('temp_train.py')

