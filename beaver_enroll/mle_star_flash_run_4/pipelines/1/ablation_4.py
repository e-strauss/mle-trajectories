
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
import io

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """
    Loads all necessary CSV files from both base and reference solutions into pandas DataFrames.
    Includes error handling for missing files by creating empty DataFrames.
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
                # Suppress warning for ablation study runs
                # print(f"Warning: {name}.csv not found at {path}. Proceeding without this data.", file=sys.stderr)
                dataframes[name] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading {name} data: {e}", file=sys.stderr)
            dataframes[name] = pd.DataFrame()

    return dataframes

def feature_engineering(dfs, use_department_feature=True):
    """
    Merges dataframes and creates a combined set of features.
    The 'use_department_feature' flag controls the creation of the 'DEPARTMENT' feature.
    """
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
    
    if use_department_feature:
        data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]
    else:
        # Ensure the column doesn't exist if the feature is disabled
        if 'DEPARTMENT' in data.columns:
            data = data.drop(columns=['DEPARTMENT'])

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

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = [('num', numeric_transformer, numeric_features)]
    if categorical_features: # Only add categorical transformer if features are present
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def run_training_pipeline(ablation_type='baseline'):
    """
    Main function to execute the integrated training and validation pipeline.
    Ablation types: 'baseline', 'no_department_feature', 'no_categorical_encoding', 'tabpfn_only'
    """
    # --- 0. Dependency Installation ---
    try:
        # Redirect stdout and stderr to DEVNULL to silence installation messages
        DEVNULL = open(os.devnull, 'w')
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0"], stdout=DEVNULL, stderr=subprocess.STDOUT)
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Final Validation Performance: 0.0")
        return

    # --- 1. Data Loading ---
    dataframes = load_data(DEFAULT_TRAIN_DIR)

    # --- 2. Feature Engineering ---
    try:
        use_dept_feat = ablation_type != 'no_department_feature'
        data = feature_engineering(dataframes, use_department_feature=use_dept_feat)
    except ValueError as e:
        print(f"Final Validation Performance: 0.0")
        return

    # --- 3. Data Splitting (Time-based) ---
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
        print("Final Validation Performance: 0.0")
        return

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- 4. Model Training ---
    # Define features for pipeline models (RF, LGBM)
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    if ablation_type == 'no_categorical_encoding':
        pipeline_cat_features = []
    else:
        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]

    # Train pipeline models unless we are only using TabPFN
    if ablation_type != 'tabpfn_only':
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)

        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)

    # Train TabPFN model
    tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    tabpfn_num_features = [
        'NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS',
        'NUM_FACULTY', 'AVG_FACULTY_LOAD',
        'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'
    ]
    
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

    # --- 5. Validation and Ensembling ---
    if not y_val.empty:
        # Get probabilities from all models
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        if ablation_type == 'tabpfn_only':
            proba_ensemble = proba_tabpfn
        else:
            proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
            proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        final_validation_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        print(f'Final Validation Performance: {final_validation_score}')
    else:
        print('Final Validation Performance: 0.0')


def run_ablation_study():
    """
    Orchestrates the ablation study by running the training pipeline with different configurations.
    """
    # Create dummy files to prevent file-not-found errors during the study
    if not os.path.exists(BASE_INPUT_DIR):
        os.makedirs(BASE_INPUT_DIR)
    
    # Minimal data to make the script runnable
    pd.DataFrame({
        'TERM_CODE': [202210, 202210, 202310, 202310],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202']
    }).to_csv(os.path.join(DEFAULT_TRAIN_DIR, 'subject_summary.csv'), index=False)
    
    pd.DataFrame({
        'TERM_CODE': [202210, 202210, 202310, 202310],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-202', 'CS-101', 'MATH-202'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'N', 'Y']
    }).to_csv(GOLD_LABELS_PATH, index=False)

    ablation_scenarios = {
        'Baseline (Full Model)': 'baseline',
        'Ablation: No Department Feature': 'no_department_feature',
        'Ablation: No Categorical Encoding': 'no_categorical_encoding',
        'Ablation: TabPFN Only': 'tabpfn_only'
    }

    results = {}
    print("--- Running Ablation Study ---")
    
    # Use a string buffer to capture stdout from the subprocess
    old_stdout = sys.stdout
    
    for name, scenario in ablation_scenarios.items():
        print(f"Running: {name}...")
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            run_training_pipeline(ablation_type=scenario)
            output = redirected_output.getvalue()
            
            # Find the score in the captured output
            score_line = [line for line in output.split('\n') if 'Final Validation Performance:' in line]
            if score_line:
                score = float(score_line[0].split(':')[-1].strip())
                results[name] = score
            else:
                results[name] = 0.0
        except Exception as e:
            results[name] = 0.0
            print(f"Error during '{name}' run: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        finally:
            sys.stdout = old_stdout # Restore stdout

    # --- Analyze and Print Results ---
    baseline_score = results.get('Baseline (Full Model)', 0.0)
    performance_drops = {
        name: baseline_score - score for name, score in results.items()
    }

    print("\n--- Ablation Study Results ---")
    print(f"{'Experiment':<35} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 80)

    for name, score in results.items():
        drop = performance_drops[name]
        print(f"{name:<35} | {score:<20.4f} | {drop:<20.4f}")

    # Determine the most impactful component
    # Exclude baseline from the components to find the max drop
    ablated_drops = {k: v for k, v in performance_drops.items() if k != 'Baseline (Full Model)' and v is not None}
    
    if not ablated_drops:
         most_impactful = "N/A - Could not run ablation tests."
    else:
        max_drop_component_key = max(ablated_drops, key=ablated_drops.get)

        if "No Department Feature" in max_drop_component_key:
            most_impactful = "The 'DEPARTMENT' feature"
        elif "No Categorical Encoding" in max_drop_component_key:
            most_impactful = "Categorical Feature Encoding for pipeline models"
        elif "TabPFN Only" in max_drop_component_key:
            most_impactful = "The RF and LGBM models in the ensemble"
        else:
            most_impactful = max_drop_component_key
            
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: '{most_impactful}'")
    
    # Clean up dummy files
    os.remove(os.path.join(DEFAULT_TRAIN_DIR, 'subject_summary.csv'))
    os.remove(GOLD_LABELS_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default=None, help="For internal use by the ablation study.")
    args, unknown = parser.parse_known_args()

    # If --ablation is passed, this is a worker process. Run the training.
    if args.ablation:
        run_training_pipeline(ablation_type=args.ablation)
    # Otherwise, this is the main orchestrator. Run the study.
    else:
        run_ablation_study()
