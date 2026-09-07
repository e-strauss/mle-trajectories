
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

# Suppress verbose output from subprocess calls
DEVNULL = open(os.devnull, 'w')

# Define constants from the original problem
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def run_pipeline(use_weighted_ensemble: bool, use_threshold_optimization: bool):
    """
    Executes the full training and validation pipeline with specific components enabled/disabled for the ablation study.
    
    Args:
        use_weighted_ensemble (bool): If True, uses F1 scores to weight the model ensemble. Otherwise, uses simple averaging.
        use_threshold_optimization (bool): If True, finds the best prediction threshold on the validation set. Otherwise, uses a fixed 0.5.
        
    Returns:
        float: The final macro F1 validation score for the given configuration.
    """
    original_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    score = 0.0

    try:
        # --- 0. Dependency Installation ---
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0"], stdout=DEVNULL, stderr=DEVNULL)
            from tabpfn import TabPFNClassifier
            from lightgbm import LGBMClassifier
        except (ImportError, subprocess.CalledProcessError):
            return 0.0

        # --- 1. Data Loading ---
        def load_data(train_dir):
            paths = {
                'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
                'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
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

        dataframes = load_data(DEFAULT_TRAIN_DIR)

        # --- 2. Feature Engineering ---
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
            
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data['TERM_CODE_INT'] = data['TERM_CODE']
            
            data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
            data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
            return data

        try:
            data = feature_engineering(dataframes)
        except ValueError:
            return 0.0

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
            return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        # --- 4. Model Training ---
        def build_classifier_pipeline(classifier, numeric_features, categorical_features):
            numeric_transformer = SimpleImputer(strategy='median')
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
            return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)
        
        pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1), pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)

        tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
        tabpfn_num_features = ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY']
        tabpfn_cat_features = [f for f in tabpfn_cat_features if f in train_df.columns]
        tabpfn_num_features = [f for f in tabpfn_num_features if f in train_df.columns]
        all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

        train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            val_df_tabpfn[col] = val_df_tabpfn[col].map({label: i for i, label in enumerate(uniques)}).fillna(-1).astype(int)

        X_train_tabpfn, X_val_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0), val_df_tabpfn[all_tabpfn_features].fillna(0)
        
        if X_train_tabpfn.shape[1] > 100:
            X_train_tabpfn, X_val_tabpfn = X_train_tabpfn.iloc[:, :100], X_val_tabpfn.iloc[:, :100]

        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

        # --- 5. Validation and Ensembling (Ablation Logic) ---
        if not y_val.empty:
            proba_rf, proba_lgbm, proba_tabpfn = pipeline_rf.predict_proba(val_df)[:, 1], pipeline_lgbm.predict_proba(val_df)[:, 1], clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

            if use_weighted_ensemble:
                f1_rf, f1_lgbm, f1_tabpfn = f1_score(y_val, (proba_rf >= 0.5).astype(int), average='macro', zero_division=0), f1_score(y_val, (proba_lgbm >= 0.5).astype(int), average='macro', zero_division=0), f1_score(y_val, (proba_tabpfn >= 0.5).astype(int), average='macro', zero_division=0)
                total_weight = f1_rf + f1_lgbm + f1_tabpfn
                proba_ensemble = ((proba_rf * f1_rf + proba_lgbm * f1_lgbm + proba_tabpfn * f1_tabpfn) / total_weight) if total_weight > 0 else ((proba_rf + proba_lgbm + proba_tabpfn) / 3)
            else: # Simple averaging
                proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3

            if use_threshold_optimization:
                best_f1 = 0.0
                for threshold in np.arange(0.0, 1.01, 0.01):
                    preds = (proba_ensemble >= threshold).astype(int)
                    current_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
                    if current_f1 > best_f1: best_f1 = current_f1
                final_validation_score = best_f1
            else: # Fixed 0.5 threshold
                final_validation_score = f1_score(y_val, (proba_ensemble >= 0.5).astype(int), average='macro', zero_division=0)
            
            print(f'Final Validation Performance: {final_validation_score}')

    except Exception:
        return 0.0
    finally:
        sys.stdout = original_stdout

    output_text = captured_output.getvalue()
    for line in reversed(output_text.splitlines()):
        if 'Final Validation Performance:' in line:
            try:
                score = float(line.split(':')[1].strip())
                return score
            except (ValueError, IndexError):
                return 0.0
    return 0.0

def main():
    """Main function to orchestrate the ablation study."""
    scenarios = {
        'Baseline (Weighted Ensemble + Threshold Opt)': {'use_weighted_ensemble': True, 'use_threshold_optimization': True},
        'Ablation: No Threshold Optimization': {'use_weighted_ensemble': True, 'use_threshold_optimization': False},
        'Ablation: No Weighted Ensemble': {'use_weighted_ensemble': False, 'use_threshold_optimization': True},
    }

    results = {name: run_pipeline(**config) for name, config in scenarios.items()}
    baseline_score = results.get('Baseline (Weighted Ensemble + Threshold Opt)', 0.0)
    performance_drops = {name: baseline_score - score for name, score in results.items() if 'Ablation' in name}

    print("--- Ablation Study Results ---")
    print(f"{'Configuration':<50} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 95)
    print(f"{'Baseline (Weighted Ensemble + Threshold Opt)':<50} | {baseline_score:<20.4f} | {0.0:<20.4f}")
    
    for name, drop in performance_drops.items():
        print(f"{name:<50} | {results[name]:<20.4f} | {drop:<20.4f}")

    print("\n--- Conclusion ---")
    if not performance_drops:
        print("No ablation studies were performed.")
    elif all(d <= 0 for d in performance_drops.values()):
        print("Study was inconclusive as baseline performance was zero." if baseline_score == 0 else "No component removal resulted in a significant performance drop.")
    else:
        most_impactful_component = max(performance_drops, key=performance_drops.get)
        conclusion_name = 'Threshold Optimization' if 'No Threshold Optimization' in most_impactful_component else 'F1-Weighted Ensemble'
        print(f"The component that contributes the most to the overall performance is: '{conclusion_name}'")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"The ablation study script failed: {e}", file=sys.stderr)
