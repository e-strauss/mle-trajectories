
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

# --- Helper function to create a clean environment ---
def setup_environment():
    """Creates dummy data and directories for a self-contained run."""
    if os.path.exists('./input'):
        shutil.rmtree('./input')
    os.makedirs('./input', exist_ok=True)
    
    # Gold Labels
    gold_labels_data = {
        'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230, 202230],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'CS-102', 'MATH-201'],
        'HIGH_ENROLLMENT': ['N', 'N', 'Y', 'N', 'Y', 'Y', 'N']
    }
    pd.DataFrame(gold_labels_data).to_csv('./input/gold_enrollment_train.csv', index=False)
    
    # Subject Summary
    subject_summary_data = {
        'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230, 202230],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'CS-102', 'MATH-201'],
        'TOTAL_ENROLLMENT': [50, 20, 150, 25, 160, 200, 30]
    }
    pd.DataFrame(subject_summary_data).to_csv('./input/subject_summary.csv', index=False)
    
    # Course Summary
    course_summary_data = {
        'TERM_CODE': [202210, 202210, 202220, 202220, 202230, 202230, 202230, 202230],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-201', 'CS-101', 'MATH-201', 'CS-101', 'CS-102', 'CS-102', 'MATH-201'],
        'COURSE_ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'MAX_ENROLLMENT': [60, 30, 160, 30, 170, 100, 110, 40]
    }
    pd.DataFrame(course_summary_data).to_csv('./input/course_summary.csv', index=False)


def run_ablation_study():
    """
    Runs an ablation study by toggling different components of the ML pipeline.
    """
    # --- Install Dependencies ---
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: Failed to install or import dependencies. {e}", file=sys.stderr)
        return

    # --- Define Ablation Components ---
    BASE_INPUT_DIR = './input'
    GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

    def load_data(train_dir):
        paths = {
            'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
            'course_summary': os.path.join(train_dir, 'course_summary.csv'),
            'gold_labels': GOLD_LABELS_PATH
        }
        dataframes = {}
        for name, path in paths.items():
            dataframes[name] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        return dataframes

    def feature_engineering(dfs, use_dept_context_features=True):
        subject_summary = dfs.get('subject_summary')
        gold_labels = dfs.get('gold_labels')
        course_summary = dfs.get('course_summary')

        for df in [subject_summary, gold_labels, course_summary]:
            if not df.empty:
                df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

        data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

        if not course_summary.empty:
            course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                NUM_COURSES=('COURSE_ID', 'nunique'),
                TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
            ).reset_index()
            data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        
        # --- Ablation Point: Departmental Contextual Features ---
        if use_dept_context_features:
            features_to_normalize = ['TOTAL_SEATS', 'TOTAL_ENROLLMENT', 'NUM_COURSES']
            for col in features_to_normalize:
                if col in data.columns:
                    dept_avg_col = f'DEPT_AVG_{col}'
                    data[dept_avg_col] = data.groupby(['DEPARTMENT', 'TERM_CODE'])[col].transform('mean')
                    ratio_col = f'{col.replace("TOTAL_", "")}_vs_DEPT_AVG_RATIO'
                    data[ratio_col] = data[col] / data[dept_avg_col]
        
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
            transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)],
            remainder='drop'
        )
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

    def run_pipeline(use_dept_context_features, use_rf_model, use_ensemble):
        dataframes = load_data(BASE_INPUT_DIR)
        data = feature_engineering(dataframes, use_dept_context_features=use_dept_context_features)
        
        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        validation_term = sorted(data['TERM_CODE_INT'].unique())[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

        if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
            return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        pipeline_cat_features = ['DEPARTMENT']
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
        
        # LGBM (always present unless ensemble is off)
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]

        # --- Ablation Point: RandomForest in Ensemble ---
        proba_rf = None
        if use_rf_model:
            rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
            pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
            pipeline_rf.fit(train_df, y_train)
            proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
            
        # --- Ablation Point: Ensemble vs Single Model ---
        if use_ensemble:
            probas = [p for p in [proba_lgbm, proba_rf] if p is not None]
            if not probas: return 0.0
            proba_ensemble = np.mean(probas, axis=0)
        else: # Use LGBM Only
            proba_ensemble = proba_lgbm

        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)

    # --- Execute Ablation Runs ---
    setup_environment()
    results = {}
    
    # Baseline
    results["Baseline (Context Feats + RF + Ensemble)"] = run_pipeline(use_dept_context_features=True, use_rf_model=True, use_ensemble=True)
    
    # Ablation 1: No Departmental Context Features
    results["Ablation: No Departmental Context Features"] = run_pipeline(use_dept_context_features=False, use_rf_model=True, use_ensemble=True)

    # Ablation 2: No RandomForest in Ensemble
    results["Ablation: No RandomForest in Ensemble"] = run_pipeline(use_dept_context_features=True, use_rf_model=False, use_ensemble=True)
    
    # Ablation 3: LGBM Only (No Ensemble)
    results["Ablation: LGBM Only (No Ensemble)"] = run_pipeline(use_dept_context_features=True, use_rf_model=False, use_ensemble=False)

    # --- Print Results and Conclusion ---
    baseline_score = results["Baseline (Context Feats + RF + Ensemble)"]
    performance_drops = {
        'Departmental Context Features': baseline_score - results["Ablation: No Departmental Context Features"],
        'RandomForest in Ensemble': baseline_score - results["Ablation: No RandomForest in Ensemble"],
        'Ensemble Method (vs. LGBM Only)': baseline_score - results["Ablation: LGBM Only (No Ensemble)"]
    }

    print("--- Ablation Study Results ---")
    print(f"{'Configuration':<45} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
    print("-" * 90)
    for name, score in results.items():
        drop = baseline_score - score if name != "Baseline (Context Feats + RF + Ensemble)" else 0.0
        print(f"{name:<45} | {score:<20.4f} | {drop:<20.4f}")

    print("\n--- Conclusion ---")
    if baseline_score == 0.0:
        print("Study was inconclusive as baseline performance was zero.")
    elif not performance_drops or all(v <= 0 for v in performance_drops.values()):
        print("No component removal resulted in a significant performance drop.")
    else:
        most_impactful = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful}'")
    
    # Clean up
    shutil.rmtree('./input')

if __name__ == '__main__':
    run_ablation_study()
