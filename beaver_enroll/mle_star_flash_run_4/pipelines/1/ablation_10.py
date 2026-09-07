
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback
import sys
import subprocess
from collections import OrderedDict

# --- Dependency Installation ---
try:
    # Install necessary packages, including imblearn for SMOTE
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "scikit-learn==1.3.2", "imbalanced-learn==0.11.0", "--quiet"])
    from tabpfn import TabPFNClassifier
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except (ImportError, subprocess.CalledProcessError) as e:
    print(f"Error: Failed to install or import dependencies. {e}", file=sys.stderr)
    # Define dummy classifiers to avoid crashing the script if imports fail
    class DummyClassifier:
        def fit(self, *args, **kwargs): pass
        def predict_proba(self, X): return np.zeros((len(X), 2))
    TabPFNClassifier = LGBMClassifier = RandomForestClassifier = DummyClassifier
    print("Warning: Using dummy classifiers due to import failure.", file=sys.stderr)


# Define constants
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """Loads all necessary CSV files."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'instructor_attributes': os.path.join(train_dir, 'instructor_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }
    dataframes = {name: pd.read_csv(path) if os.path.exists(path) else pd.DataFrame() for name, path in paths.items()}
    return dataframes

def feature_engineering(dfs):
    """Merges dataframes and creates features."""
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files are missing.")
    
    for df in [df for df in dfs.values() if not df.empty and 'TERM_CODE' in df.columns]:
        df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    # Feature Engineering from various sources
    if not dfs.get('course_attributes').empty:
        data = pd.merge(data, dfs['course_attributes'].add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not dfs.get('course_summary').empty:
        course_agg = dfs['course_summary'].groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not dfs.get('faculty_summary').empty:
        faculty_agg = dfs['faculty_summary'].groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique')).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def run_ablation_study():
    """
    Runs an ablation study on the modeling pipeline, focusing on SMOTE and StandardScaler.
    """
    results = OrderedDict()

    for scenario in ["Baseline (SMOTE + StandardScaler)", "Ablation: No SMOTE", "Ablation: No StandardScaler"]:
        print(f"--- Running: {scenario} ---")
        score = 0.0
        try:
            # --- 1. Data Loading and Preprocessing ---
            dataframes = load_data(DEFAULT_TRAIN_DIR)
            data = feature_engineering(dataframes)

            # --- 2. Data Splitting ---
            data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
            sorted_terms = sorted(data['TERM_CODE_INT'].unique())
            if len(sorted_terms) < 2:
                train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
            else:
                validation_term = sorted_terms[-1]
                train_df = data[data['TERM_CODE_INT'] < validation_term]
                val_df = data[data['TERM_CODE_INT'] == validation_term]

            if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
                raise ValueError("Insufficient data for training/validation.")

            y_train = train_df['HIGH_ENROLLMENT']
            y_val = val_df['HIGH_ENROLLMENT']

            # --- 3. Feature and Pipeline Definition (Ablation Logic) ---
            use_smote = "No SMOTE" not in scenario
            use_scaler = "No StandardScaler" not in scenario
            
            pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
            pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
            
            numeric_transformer = StandardScaler() if use_scaler else SimpleImputer(strategy='median')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, pipeline_num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), pipeline_cat_features)
                ], remainder='passthrough')

            # Build Pipelines for RF and LGBM
            def build_model_pipeline(classifier):
                if use_smote:
                    return ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)), ('classifier', classifier)])
                else:
                    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

            pipeline_rf = build_model_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
            pipeline_lgbm = build_model_pipeline(LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1))

            # --- 4. Model Training ---
            pipeline_rf.fit(train_df, y_train)
            pipeline_lgbm.fit(train_df, y_train)

            # TabPFN training (remains consistent across ablations)
            tabpfn_features = [f for f in ['SUBJECT_ID_SORT'] + pipeline_num_features if f in train_df.columns]
            X_train_tabpfn = train_df[tabpfn_features].copy()
            X_val_tabpfn = val_df[tabpfn_features].copy()
            for col in X_train_tabpfn.select_dtypes(include=['object']).columns:
                codes, uniques = pd.factorize(X_train_tabpfn[col])
                X_train_tabpfn[col] = codes
                mapping = {label: i for i, label in enumerate(uniques)}
                X_val_tabpfn[col] = X_val_tabpfn[col].map(mapping).fillna(-1)
            X_train_tabpfn = X_train_tabpfn.fillna(0).iloc[:, :100]
            X_val_tabpfn = X_val_tabpfn.fillna(0).iloc[:, :100]
            
            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

            # --- 5. Validation and Ensembling ---
            proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
            proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
            proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
            pred_ensemble = (proba_ensemble >= 0.5).astype(int)
            score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        
        except Exception as e:
            print(f"Error during '{scenario}': {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            score = 0.0

        results[scenario] = score
        print(f"--- Performance for {scenario}: {score:.4f} ---\n")

    # --- 6. Conclusion ---
    print("--- Ablation Study Results ---")
    baseline_score = results.get("Baseline (SMOTE + StandardScaler)", 0.0)
    performance_drops = {}
    
    for scenario, score in results.items():
        drop = baseline_score - score
        print(f"{scenario}: {score:.4f} (Performance Drop: {drop:.4f})")
        if "Ablation" in scenario:
            component_name = scenario.split("Ablation: No ")[1]
            performance_drops[component_name] = drop

    print("\n--- Conclusion ---")
    if not performance_drops or max(performance_drops.values()) <= 0:
        print("No component removal resulted in a significant performance drop.")
    else:
        most_impactful = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful}'")

if __name__ == '__main__':
    run_ablation_study()
