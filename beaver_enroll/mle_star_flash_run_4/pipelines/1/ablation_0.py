
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
import copy

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def install_dependencies():
    """Install required packages quietly."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (ImportError, subprocess.CalledProcessError):
        print("Error: Failed to install dependencies. Ablation study cannot proceed.", file=sys.stderr)
        return False

# --- Core Functions from the original script ---

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
        if os.path.exists(path):
            dataframes[name] = pd.read_csv(path)
        else:
            dataframes[name] = pd.DataFrame()
    return dataframes

def feature_engineering(dfs, use_advanced_features=True):
    subject_summary = dfs.get('subject_summary', pd.DataFrame()).copy()
    gold_labels = dfs.get('gold_labels', pd.DataFrame()).copy()
    course_attributes = dfs.get('course_attributes', pd.DataFrame()).copy()
    course_summary = dfs.get('course_summary', pd.DataFrame()).copy()
    faculty_summary = dfs.get('faculty_summary', pd.DataFrame()).copy()

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files are missing.")

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if not df.empty and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
        data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if use_advanced_features:
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
        
        # Fill missing advanced feature columns if data was missing
        for col in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD']:
             if col not in data.columns:
                 data[col] = np.nan

        data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
        data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
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
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_experiment(description, data, use_rf=True, use_lgbm=True, use_tabpfn=True):
    """Runs a single experiment configuration and returns the F1 score."""
    from tabpfn import TabPFNClassifier
    from lightgbm import LGBMClassifier

    # --- Data Splitting ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    if len(sorted_terms) < 2:
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term]
        val_df = data[data['TERM_CODE_INT'] == validation_term]
        if train_df.empty or val_df.empty:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

    if train_df.empty or val_df.empty or val_df['HIGH_ENROLLMENT'].nunique() < 2:
        print(f"Skipping '{description}': Not enough data for validation.", file=sys.stderr)
        return 0.0

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']
    
    probas = []
    
    # --- RF/LGBM Training ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

    if use_rf:
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)
        probas.append(pipeline_rf.predict_proba(val_df)[:, 1])

    if use_lgbm:
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        probas.append(pipeline_lgbm.predict_proba(val_df)[:, 1])
        
    # --- TabPFN Training ---
    if use_tabpfn:
        tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in train_df.columns]
        tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in train_df.columns]
        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

        if not all_tabpfn_features:
            # Add a default feature if all specific ones are missing (e.g., in no-advanced-features run)
            all_tabpfn_features = [f for f in pipeline_num_features if f in train_df.columns][:5]

        train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0)
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0)

        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        if X_train_tabpfn.shape[1] > 0 and not X_train_tabpfn.empty:
            clf_tabpfn.fit(X_train_tabpfn.iloc[:, :100], y_train, overwrite_warning=True)
            probas.append(clf_tabpfn.predict_proba(X_val_tabpfn.iloc[:, :100])[:, 1])

    # --- Validation ---
    if not probas:
        print(f"Skipping '{description}': No models were trained.", file=sys.stderr)
        return 0.0

    proba_ensemble = np.mean(probas, axis=0)
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    
    print(f"Performance for '{description}': {score:.4f}")
    return score

def main():
    """Main function to run the ablation study."""
    if not install_dependencies():
        return

    parser = argparse.ArgumentParser(description='Ablation study for high enrollment prediction.')
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR)
    args = parser.parse_args()

    try:
        dataframes = load_data(args.train_data_dir)
        
        # Prepare data for different feature sets
        data_full_features = feature_engineering(dataframes, use_advanced_features=True)
        data_basic_features = feature_engineering(dataframes, use_advanced_features=False)

        print("\n--- Starting Ablation Study ---\n")
        
        scores = {}
        
        # Baseline: Full model with all features
        scores['Baseline (All Models & Features)'] = run_experiment(
            "Baseline (All Models & Features)", 
            data=data_full_features, 
            use_rf=True, use_lgbm=True, use_tabpfn=True
        )

        # Ablation 1: Remove TabPFN model
        scores['Ablation: No TabPFN Model'] = run_experiment(
            "Ablation: No TabPFN Model",
            data=data_full_features, 
            use_rf=True, use_lgbm=True, use_tabpfn=False
        )

        # Ablation 2: Remove advanced features
        scores['Ablation: No Advanced Features'] = run_experiment(
            "Ablation: No Advanced Features",
            data=data_basic_features, 
            use_rf=True, use_lgbm=True, use_tabpfn=True
        )

        print("\n--- Ablation Study Results ---")
        
        baseline_score = scores.get('Baseline (All Models & Features)', 0)
        performance_drops = {}
        for name, score in scores.items():
            if name != 'Baseline (All Models & Features)':
                drop = baseline_score - score
                performance_drops[name] = drop
        
        if not performance_drops:
             print("\nCould not run any ablations to compare.")
             return
             
        # Find the component that caused the biggest drop
        most_impactful_component = max(performance_drops, key=performance_drops.get)
        
        if 'No TabPFN' in most_impactful_component:
            conclusion = "the TabPFN model"
        elif 'No Advanced Features' in most_impactful_component:
            conclusion = "the advanced features from course and faculty summaries"
        else:
            conclusion = "an unknown component"

        print(f"\nConclusion: Based on the performance drop, {conclusion} contributes the most to the overall performance.")

    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == '__main__':
    main()
