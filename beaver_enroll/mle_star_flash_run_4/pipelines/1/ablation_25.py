
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

# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def create_dummy_data():
    """Creates dummy data files for a self-contained run."""
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)
    os.makedirs(BASE_INPUT_DIR, exist_ok=True)

    # Note: instructor_attributes.csv is intentionally omitted to test strict loading
    
    subject_summary = pd.DataFrame({
        'TERM_CODE': [2022, 2022, 2023, 2023, 2024, 2024, 2024],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-303', 'CS-101', 'MATH-202', 'PHYS-404']
    })
    subject_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'subject_summary.csv'), index=False)

    gold_labels = pd.DataFrame({
        'TERM_CODE': [2022, 2022, 2023, 2023, 2024, 2024],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'ART-303', 'CS-101', 'MATH-202'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'Y', 'Y', 'N']
    })
    gold_labels.to_csv(GOLD_LABELS_PATH, index=False)

    course_attributes = pd.DataFrame({
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'ART-303', 'PHYS-404'],
        'CAMPUS_ID_DESC': ['Main', 'Main', 'Online', 'Main'],
        'DEPARTMENT_ID_DESC': ['CompSci', 'Mathematics', 'Fine Arts', 'Physics']
    })
    course_attributes.to_csv(os.path.join(BASE_INPUT_DIR, 'course_attributes.csv'), index=False)

    course_summary = pd.DataFrame({
        'TERM_CODE': [2022, 2023, 2024],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101'],
        'COURSE_ID': [1, 2, 3],
        'MAX_ENROLLMENT': [100, 150, 200]
    })
    course_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'course_summary.csv'), index=False)
    
    faculty_summary = pd.DataFrame({
        'TERM_CODE': [2022, 2023, 2024],
        'SUBJECT_ID_SORT': ['CS-101', 'CS-101', 'CS-101'],
        'FACULTY_ID': [10, 11, 12],
        'NUM_COURSES_TAUGHT': [1, 2, 1]
    })
    faculty_summary.to_csv(os.path.join(BASE_INPUT_DIR, 'faculty_summary.csv'), index=False)


def load_data(train_dir, strict_loading=False):
    """
    Loads all necessary CSV files.
    If strict_loading is True, it will raise an error for missing files.
    Otherwise, it will create empty DataFrames and show a warning.
    """
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
                if strict_loading:
                    raise FileNotFoundError(f"Strict loading enabled: {name}.csv not found at {path}.")
                print(f"Warning: {name}.csv not found at {path}. Proceeding without this data.", file=sys.stderr)
                dataframes[name] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading {name} data: {e}", file=sys.stderr)
            dataframes[name] = pd.DataFrame()

    return dataframes

def feature_engineering(dfs, core_join_type='left'):
    """
    Merges dataframes and creates features.
    `core_join_type` controls the merge between subject_summary and gold_labels.
    """
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

    # The join type here is an ablation target
    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how=core_join_type)

    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'),
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    data['TERM_CODE_INT'] = data['TERM_CODE']
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

def run_ablation_scenario(strict_loading=False, core_join_type='left', disable_split_fallback=False):
    """
    Runs a single scenario of the training and validation pipeline.
    Returns the final validation score.
    """
    try:
        dataframes = load_data(DEFAULT_TRAIN_DIR, strict_loading=strict_loading)
        data = feature_engineering(dataframes, core_join_type=core_join_type)

        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        sorted_terms = sorted(data['TERM_CODE_INT'].unique())
        
        train_df, val_df = pd.DataFrame(), pd.DataFrame()

        if len(sorted_terms) < 2:
            if not disable_split_fallback:
                train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
        else:
            validation_term = sorted_terms[-1]
            train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
            val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
            
            if (train_df.empty or val_df.empty) and not disable_split_fallback:
                train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

        if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2 or val_df['HIGH_ENROLLMENT'].nunique() < 2:
            return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        cat_features = ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object']
        num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf, num_features, cat_features)
        pipeline_rf.fit(train_df, y_train)

        preds = pipeline_rf.predict(val_df)
        score = f1_score(y_val, preds, average='macro', zero_division=0)
        return score

    except Exception as e:
        print(f"  Scenario failed with error: {e}", file=sys.stderr)
        return 0.0


def main():
    """Main function to setup and run the ablation study."""
    # Suppress dependency installation output for cleaner logs
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass # Ignore errors if already installed or fails

    create_dummy_data()

    scenarios = {
        "Baseline": {
            "strict_loading": False, "core_join_type": 'left', "disable_split_fallback": False
        },
        "Ablation: Strict Data Loading": {
            "strict_loading": True, "core_join_type": 'left', "disable_split_fallback": False
        },
        "Ablation: Inner Join for Core Data": {
            "strict_loading": False, "core_join_type": 'inner', "disable_split_fallback": False
        },
        "Ablation: No Split Fallback": {
            "strict_loading": False, "core_join_type": 'left', "disable_split_fallback": True
        }
    }

    results = {}
    print("--- Running Ablation Study ---")
    for name, params in scenarios.items():
        print(f"Running: {name}...")
        score = run_ablation_scenario(**params)
        results[name] = score
    print("--- Ablation Study Complete ---")

    baseline_score = results.get("Baseline", 0.0)
    ablation_results = []
    
    for name, score in results.items():
        if name != "Baseline":
            performance_drop = baseline_score - score
            ablation_results.append({"component": name.replace("Ablation: ", ""), "score": score, "drop": performance_drop})

    # Print summary table
    print("\n--- Ablation Study Results ---")
    print(f"{'Configuration':<40} {'F1 Score (Macro)':<20} {'Performance Drop':<20}")
    print("-" * 80)
    print(f"{'Baseline':<40} {baseline_score:<20.4f} {0.0:<20.4f}")
    for res in ablation_results:
        print(f"{'Ablation: ' + res['component']:<40} {res['score']:<20.4f} {res['drop']:<20.4f}")
    print("-" * 80)

    # Determine and print conclusion
    print("\n--- Conclusion ---")
    if baseline_score == 0.0:
        print("Study was inconclusive as baseline performance was zero.")
    elif not ablation_results or all(res['drop'] <= 0 for res in ablation_results):
        print("No single component change resulted in a significant performance drop.")
    else:
        most_impactful = max(ablation_results, key=lambda x: x['drop'])
        print(f"The component that contributes the most to the overall performance is: '{most_impactful['component']}'")
        print(f"Its alteration caused a performance drop of {most_impactful['drop']:.4f}.")
    
    # Cleanup
    if os.path.exists(BASE_INPUT_DIR):
        shutil.rmtree(BASE_INPUT_DIR)


if __name__ == '__main__':
    main()

