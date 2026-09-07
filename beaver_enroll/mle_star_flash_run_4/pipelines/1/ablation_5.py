
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
warnings.filterwarnings('ignore')

# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """Loads all necessary CSV files into pandas DataFrames."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
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

def feature_engineering(dfs, use_department_feature=True, use_course_attributes=True):
    """Merges dataframes and creates features, with options to ablate specific parts."""
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

    # Ablation point: Include course_attributes data
    if use_course_attributes and not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), left_on='SUBJECT_ID_SORT', right_on='SUBJECT_ID_SORT_attr', how='left')
         if 'SUBJECT_ID_SORT_attr' in data.columns:
             data.drop('SUBJECT_ID_SORT_attr', axis=1, inplace=True)


    # Ablation point: Create DEPARTMENT feature
    if use_department_feature:
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
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return pipeline

def run_experiment(name, use_optimized_weights=True, use_department_feature=True, use_course_attributes=True):
    """
    Runs a single experiment with a specific configuration.
    """
    print(f"--- Running: {name} ---")
    
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
        from scipy.optimize import minimize

        dataframes = load_data(DEFAULT_TRAIN_DIR)
        data = feature_engineering(dataframes, use_department_feature, use_course_attributes)

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
            return 0.0

        y_train = train_df['HIGH_ENROLLMENT']
        y_val = val_df['HIGH_ENROLLMENT']

        # --- Model Training ---
        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)

        pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)

        tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in train_df.columns]
        tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'] if f in train_df.columns]
        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

        train_df_tabpfn = train_df.copy()
        val_df_tabpfn = val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0).iloc[:,:100]
        
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train)

        # --- Validation and Ensembling ---
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        # Ablation point: Optimized Weights vs. Simple Average
        if use_optimized_weights:
            val_probas = np.vstack([proba_rf, proba_lgbm, proba_tabpfn])
            def f1_objective(weights):
                weighted_probas = np.average(val_probas, axis=0, weights=weights)
                preds = (weighted_probas >= 0.5).astype(int)
                return -f1_score(y_val, preds, average='macro', zero_division=0)
            
            opt_result = minimize(f1_objective, np.array([1/3]*3), method='SLSQP', bounds=[(0,1)]*3, constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}))
            proba_ensemble = np.average(val_probas, axis=0, weights=opt_result.x)
        else:
            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        return score

    except Exception as e:
        print(f"Error during experiment '{name}': {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 0.0

if __name__ == '__main__':
    # Install dependencies quietly
    try:
        # First, ensure setuptools and pip are up-to-date to avoid pkg_resources errors.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "--quiet"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "scipy", "--quiet"])
    except Exception as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)
        # Do not exit, but the rest of the script will likely fail.
    
    # Run all experiments
    results = {}
    
    # Baseline
    results['Baseline (Optimized Weights & All Features)'] = run_experiment(
        name='Baseline',
        use_optimized_weights=True,
        use_department_feature=True,
        use_course_attributes=True
    )
    
    # Ablation 1: Simple Averaging instead of Optimized Weights
    results['Ablation: Simple Average Ensemble'] = run_experiment(
        name='No Optimized Weights',
        use_optimized_weights=False,
        use_department_feature=True,
        use_course_attributes=True
    )

    # Ablation 2: Remove the 'DEPARTMENT' feature
    results['Ablation: No DEPARTMENT Feature'] = run_experiment(
        name='No DEPARTMENT Feature',
        use_optimized_weights=True,
        use_department_feature=False,
        use_course_attributes=True
    )

    # Ablation 3: Remove features from 'course_attributes.csv'
    results['Ablation: No Course Attributes Data'] = run_experiment(
        name='No Course Attributes',
        use_optimized_weights=True,
        use_department_feature=True,
        use_course_attributes=False
    )
    
    # --- Print Results ---
    print("\n--- Ablation Study Results ---")
    baseline_score = results.get('Baseline (Optimized Weights & All Features)', 0.0)
    performance_drops = {}

    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
        if 'Ablation' in name:
            # Map the friendly name to a component name
            if 'Simple Average' in name:
                component = 'Optimized Ensemble Weighting'
            elif 'DEPARTMENT' in name:
                component = 'DEPARTMENT Feature'
            elif 'Course Attributes' in name:
                component = 'Course Attributes Data'
            performance_drops[component] = drop
            
    # --- Conclusion ---
    if not performance_drops:
        most_impactful_component = "N/A (No valid ablations completed)"
    else:
        most_impactful_component = max(performance_drops, key=performance_drops.get)

    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

    # Print the final validation score as required
    final_validation_score = results.get('Baseline (Optimized Weights & All Features)', 0.0)
    print(f"\nFinal Validation Performance: {final_validation_score}")
