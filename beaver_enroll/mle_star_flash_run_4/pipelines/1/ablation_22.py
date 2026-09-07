
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

warnings.filterwarnings("ignore", category=UserWarning)

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def setup_environment():
    """Installs required packages quietly."""
    try:
        # First, upgrade pip and setuptools to ensure a modern build environment and availability of pkg_resources. [2, 8]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"])
        # Then, install the specific packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        return True
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: Failed to install dependencies. {e}", file=sys.stderr)
        return False

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

def feature_engineering(dfs, use_ratio_features=True):
    """Merges dataframes and creates features, with an option to ablate ratio features."""
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

    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan

    # Ablation point for ratio features
    if use_ratio_features:
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
        ], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_ablation_scenario(use_ratio_features=True, use_rf_in_ensemble=True, force_random_split=False):
    """
    Executes one full training and validation pipeline based on ablation settings.
    Returns the macro F1 score.
    """
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
        
        # 1. Data Loading
        dataframes = load_data(DEFAULT_TRAIN_DIR)

        # 2. Feature Engineering
        data = feature_engineering(dataframes, use_ratio_features=use_ratio_features)

        # 3. Data Splitting
        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        
        # Ablation point for splitting strategy
        if force_random_split:
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
        else:
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
            return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        # 4. Model Training
        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

        # Train shared models (LGBM, TabPFN)
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        
        tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'DEPARTMENT'] if f in train_df.columns]
        tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD'] if f in train_df.columns]
        if use_ratio_features:
            tabpfn_num_features.extend([f for f in ['SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in train_df.columns])
        
        X_train_tabpfn = train_df.copy()
        X_val_tabpfn = val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(X_train_tabpfn[col])
            X_train_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            X_val_tabpfn[col] = X_val_tabpfn[col].map(mapping).fillna(-1).astype(int)

        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features
        X_train_tabpfn = X_train_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        X_val_tabpfn = X_val_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

        # 5. Validation and Ensembling
        probas = []
        probas.append(pipeline_lgbm.predict_proba(val_df)[:, 1])
        probas.append(clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1])

        # Ablation point for RandomForest in ensemble
        if use_rf_in_ensemble:
            rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
            pipeline_rf.fit(train_df, y_train)
            probas.append(pipeline_rf.predict_proba(val_df)[:, 1])
            
        proba_ensemble = np.mean(probas, axis=0)
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    
    except Exception:
        # traceback.print_exc()
        return 0.0

if __name__ == '__main__':
    baseline_score = 0.0
    if not setup_environment():
        print("Environment setup failed. Exiting.", file=sys.stderr)
    else:
        ablation_results = {}
        
        print("--- Running Ablation Study ---")

        print("\nRunning: Baseline (All Components)")
        baseline_score = run_ablation_scenario(
            use_ratio_features=True, 
            use_rf_in_ensemble=True, 
            force_random_split=False
        )
        ablation_results['Baseline'] = (baseline_score, 0.0)

        print("Running: Ablation - No Ratio Features")
        score_no_ratio = run_ablation_scenario(
            use_ratio_features=False, 
            use_rf_in_ensemble=True, 
            force_random_split=False
        )
        ablation_results['No Ratio Features'] = (score_no_ratio, baseline_score - score_no_ratio)

        print("Running: Ablation - No RandomForest in Ensemble")
        score_no_rf = run_ablation_scenario(
            use_ratio_features=True, 
            use_rf_in_ensemble=False, 
            force_random_split=False
        )
        ablation_results['No RandomForest in Ensemble'] = (score_no_rf, baseline_score - score_no_rf)

        print("Running: Ablation - Forced Random Split")
        score_random_split = run_ablation_scenario(
            use_ratio_features=True, 
            use_rf_in_ensemble=True, 
            force_random_split=True
        )
        ablation_results['Forced Random Split'] = (score_random_split, baseline_score - score_random_split)
        
        print("\n--- Ablation Study Results ---")
        print(f"{'Configuration':<35} | {'Macro F1-Score':<18} | {'Performance Drop':<20}")
        print("-" * 78)
        for config, (score, drop) in ablation_results.items():
            print(f"{config:<35} | {score:<18.4f} | {drop:<20.4f}")

        print("\n--- Conclusion ---")
        impact_scores = {k: v[1] for k, v in ablation_results.items() if k not in ['Baseline', 'Forced Random Split'] and v[1] > 0}

        if not impact_scores:
            if baseline_score == 0:
                print("Study was inconclusive as baseline performance was zero.")
            else:
                print("No single component removal resulted in a significant performance drop.")
        else:
            most_impactful_component = max(impact_scores, key=impact_scores.get)
            print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

    print(f"Final Validation Performance: {baseline_score}")
