
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

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """
    Loads all necessary CSV files into pandas DataFrames.
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
                dataframes[name] = pd.DataFrame()
        except Exception:
            dataframes[name] = pd.DataFrame()
    return dataframes

def feature_engineering(dfs, create_historical_features=True):
    """
    Merges dataframes and creates features. Includes an option to ablate historical features.
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

    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

    if create_historical_features:
        data = data.sort_values(['SUBJECT_ID_SORT', 'TERM_CODE']).reset_index(drop=True)
        historical_metrics = ['TOTAL_ENROLLMENT', 'TOTAL_SEATS', 'NUM_COURSES']
        for metric in historical_metrics:
            if metric in data.columns:
                grouped_by_subject = data.groupby('SUBJECT_ID_SORT')
                expanding_mean = grouped_by_subject[metric].expanding().mean().reset_index(level=0, drop=True)
                data[f'temp_mean_{metric}'] = expanding_mean
                data[f'hist_mean_{metric}'] = grouped_by_subject[f'temp_mean_{metric}'].shift(1)
                data.drop(columns=[f'temp_mean_{metric}'], inplace=True)

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
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_ablation_scenario(use_historical_features, use_tabpfn, use_lgbm):
    """
    Runs a single training and validation scenario based on ablation flags.
    """
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

    dataframes = load_data(DEFAULT_TRAIN_DIR)
    
    try:
        data = feature_engineering(dataframes, create_historical_features=use_historical_features)
    except ValueError as e:
        return 0.0

    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    if len(sorted_terms) < 2:
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data.get('HIGH_ENROLLMENT'))
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- Model Training ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

    # Model 1: RandomForest (always trained)
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    # Model 2: LightGBM (conditional)
    if use_lgbm:
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)

    # Model 3: TabPFN (conditional)
    if use_tabpfn:
        tabpfn_features = [c for c in train_df.columns if c in val_df.columns and c not in ['HIGH_ENROLLMENT', 'TERM_CODE_INT']]
        X_train_tabpfn = train_df[tabpfn_features].apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x).fillna(0)
        X_val_tabpfn = val_df[tabpfn_features].apply(lambda x: pd.factorize(x)[0] if x.dtype == 'object' else x).fillna(0)
        if X_train_tabpfn.shape[1] > 100:
            X_train_tabpfn = X_train_tabpfn.iloc[:, :100]
            X_val_tabpfn = X_val_tabpfn.iloc[:, :100]
        clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
        clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

    # --- Validation and Ensembling ---
    if y_val.empty:
        return 0.0

    probas = []
    probas.append(pipeline_rf.predict_proba(val_df)[:, 1])
    if use_lgbm:
        probas.append(pipeline_lgbm.predict_proba(val_df)[:, 1])
    if use_tabpfn:
        probas.append(clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1])
    
    proba_ensemble = np.mean(probas, axis=0)
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    
    return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)


if __name__ == '__main__':
    results = {}
    
    try:
        # --- Baseline ---
        baseline_score = run_ablation_scenario(
            use_historical_features=True, 
            use_tabpfn=True, 
            use_lgbm=True
        )
        results['Baseline (All Components)'] = baseline_score

        # --- Ablation 1: No Robust Historical Features ---
        no_hist_score = run_ablation_scenario(
            use_historical_features=False, 
            use_tabpfn=True, 
            use_lgbm=True
        )
        results['Ablation: No Robust Historical Features'] = no_hist_score
        
        # --- Ablation 2: No TabPFN Model ---
        no_tabpfn_score = run_ablation_scenario(
            use_historical_features=True, 
            use_tabpfn=False, 
            use_lgbm=True
        )
        results['Ablation: No TabPFN Model'] = no_tabpfn_score

        # --- Ablation 3: RandomForest Only (No Ensemble) ---
        rf_only_score = run_ablation_scenario(
            use_historical_features=True, 
            use_tabpfn=False, 
            use_lgbm=False
        )
        results['Ablation: RandomForest Only (No Ensemble)'] = rf_only_score

        print("--- Ablation Study Results ---")
        for name, score in results.items():
            performance_drop = baseline_score - score
            print(f"{name}: {score:.4f} (Performance Drop: {performance_drop:.4f})")

        # --- Conclusion ---
        performance_drops = {
            'Robust Historical Features': baseline_score - no_hist_score,
            'TabPFN Model': baseline_score - no_tabpfn_score,
            'Ensemble (RF+LGBM+TabPFN vs RF only)': baseline_score - rf_only_score
        }

        if not performance_drops or max(performance_drops.values()) <= 0.0001:
            most_impactful = "No single component removal resulted in a significant performance drop."
        else:
            most_impactful_component = max(performance_drops, key=performance_drops.get)
            most_impactful = f"The '{most_impactful_component}' contribute(s) the most to the overall performance."

        print("\n--- Conclusion ---")
        print(most_impactful)

    except Exception as e:
        print(f"An error occurred during the ablation study: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
