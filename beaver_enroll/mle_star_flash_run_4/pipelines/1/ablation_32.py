
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
import io

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def create_dummy_data():
    """Creates a dictionary of in-memory pandas DataFrames for the ablation study."""
    dfs = {}

    # subject_summary.csv
    subject_summary_data = """TERM_CODE,SUBJECT_ID_SORT,TOTAL_ENROLLMENT
202201,CS-101,80
202201,MATH-201,120
202301,CS-101,95
202301,MATH-201,110
202401,CS-101,105
202401,MATH-201,130
"""
    dfs['subject_summary'] = pd.read_csv(io.StringIO(subject_summary_data))

    # gold_enrollment_train.csv
    gold_labels_data = """TERM_CODE,SUBJECT_ID_SORT,HIGH_ENROLLMENT
202201,CS-101,N
202201,MATH-201,Y
202301,CS-101,Y
202301,MATH-201,Y
202401,CS-101,Y
202401,MATH-201,Y
"""
    dfs['gold_labels'] = pd.read_csv(io.StringIO(gold_labels_data))

    # course_attributes.csv
    course_attributes_data = """SUBJECT_ID_SORT,CAMPUS_ID_DESC,DEPARTMENT_ID_DESC
CS-101,Main Campus,Computer Science
MATH-201,Main Campus,Mathematics
"""
    dfs['course_attributes'] = pd.read_csv(io.StringIO(course_attributes_data))

    # course_summary.csv
    course_summary_data = """TERM_CODE,SUBJECT_ID_SORT,COURSE_ID,MAX_ENROLLMENT
202201,CS-101,CS-101-001,50
202201,CS-101,CS-101-002,50
202201,MATH-201,MATH-201-001,150
202301,CS-101,CS-101-001,100
202301,MATH-201,MATH-201-001,120
202401,CS-101,CS-101-001,110
202401,MATH-201,MATH-201-001,70
202401,MATH-201,MATH-201-002,70
"""
    dfs['course_summary'] = pd.read_csv(io.StringIO(course_summary_data))

    # faculty_summary.csv
    faculty_summary_data = """TERM_CODE,SUBJECT_ID_SORT,FACULTY_ID,NUM_COURSES_TAUGHT
202201,CS-101,F10,2
202201,MATH-201,F20,1
202301,CS-101,F10,1
202301,MATH-201,F21,1
202401,CS-101,F11,1
202401,MATH-201,F20,2
"""
    dfs['faculty_summary'] = pd.read_csv(io.StringIO(faculty_summary_data))

    # instructor_attributes is not used by default feature engineering, create empty
    dfs['instructor_attributes'] = pd.DataFrame()
    return dfs

def feature_engineering(dfs, use_lag_features=True):
    """
    Merges dataframes and creates features. Includes a flag for ablating lag features.
    """
    subject_summary = dfs.get('subject_summary').copy()
    gold_labels = dfs.get('gold_labels').copy()
    course_attributes = dfs.get('course_attributes').copy()
    course_summary = dfs.get('course_summary').copy()
    faculty_summary = dfs.get('faculty_summary').copy()

    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if not course_attributes.empty:
        # Fix potential key error by not modifying the merge key in place
        ca_renamed = course_attributes.add_suffix('_attr')
        ca_renamed.rename(columns={'SUBJECT_ID_SORT_attr': 'SUBJECT_ID_SORT'}, inplace=True)
        data = pd.merge(data, ca_renamed, on='SUBJECT_ID_SORT', how='left')

    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'), TOTAL_SEATS=('MAX_ENROLLMENT', 'sum')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if use_lag_features:
        data = data.sort_values(['SUBJECT_ID_SORT', 'TERM_CODE'])
        for feature in ['TOTAL_ENROLLMENT', 'NUM_COURSES', 'TOTAL_SEATS', 'NUM_FACULTY']:
            if feature in data.columns:
                data[f'LAG1_{feature}'] = data.groupby('SUBJECT_ID_SORT')[feature].shift(1)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)], remainder='drop')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def run_pipeline(dataframes, use_lag_features=True, no_lgbm_model=False, tabpfn_only=False):
    """
    Main function to execute the training and validation pipeline.
    Accepts flags to control ablation scenarios.
    """
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except ImportError as e:
        # Silently fail if dependencies not installed, as per original code logic
        return 0.0

    # --- 1. Feature Engineering ---
    try:
        data = feature_engineering(dataframes, use_lag_features=use_lag_features)
    except Exception:
        return 0.0

    # --- 2. Data Splitting (Time-based) ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    validation_term = data['TERM_CODE_INT'].max()
    train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
    val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        return 0.0

    y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

    # --- 3. Model Training ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    # Model 1: RandomForest
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
    pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    # Model 2: LightGBM
    lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=1)
    pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm.fit(train_df, y_train)

    # Model 3: TabPFN
    tabpfn_features = [f for f in ['SUBJECT_ID_SORT', 'NUM_COURSES', 'TOTAL_SEATS', 'NUM_FACULTY'] if f in train_df.columns]
    X_train_tabpfn = train_df[tabpfn_features].copy()
    for col in X_train_tabpfn.select_dtypes(include='object').columns:
        X_train_tabpfn[col], _ = pd.factorize(X_train_tabpfn[col])
    X_val_tabpfn = val_df[tabpfn_features].copy()
    for col in X_val_tabpfn.select_dtypes(include='object').columns:
        X_val_tabpfn[col], _ = pd.factorize(X_val_tabpfn[col])

    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
    clf_tabpfn.fit(X_train_tabpfn.fillna(0), y_train, overwrite_warning=True)

    # --- 4. Validation and Ensembling ---
    proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
    proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
    proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn.fillna(0))[:, 1]
    
    # Ablation logic for ensembling
    if tabpfn_only:
        proba_final = proba_tabpfn
    elif no_lgbm_model:
        proba_final = (proba_rf + proba_tabpfn) / 2
    else: # Baseline
        proba_final = (proba_rf + proba_lgbm + proba_tabpfn) / 3

    pred_final = (proba_final >= 0.5).astype(int)
    final_score = f1_score(y_val, pred_final, average='macro', zero_division=0)
    
    return final_score

if __name__ == '__main__':
    try:
        # --- 1. Setup ---
        dummy_dfs = create_dummy_data()
        results = {}

        # --- 2. Run Experiments ---
        print("Running ablation study...")

        # Baseline: Full model with all features and models
        results['Baseline (Lag Features, Full Ensemble)'] = run_pipeline(dummy_dfs, use_lag_features=True, no_lgbm_model=False, tabpfn_only=False)

        # Ablation 1: Remove Lag Features
        results['Ablation: No Lag Features'] = run_pipeline(dummy_dfs, use_lag_features=False, no_lgbm_model=False, tabpfn_only=False)

        # Ablation 2: Remove LGBM from the ensemble
        results['Ablation: No LGBM Model'] = run_pipeline(dummy_dfs, use_lag_features=True, no_lgbm_model=True, tabpfn_only=False)
        
        # Ablation 3: Use only the TabPFN model (no ensembling)
        results['Ablation: TabPFN Only'] = run_pipeline(dummy_dfs, use_lag_features=True, no_lgbm_model=False, tabpfn_only=True)

        # --- 3. Report Results ---
        baseline_score = results['Baseline (Lag Features, Full Ensemble)']
        performance_drops = {}
        for name, score in results.items():
            if name != 'Baseline (Lag Features, Full Ensemble)':
                performance_drops[name.replace('Ablation: ', '')] = baseline_score - score

        print("\n--- Ablation Study Results ---")
        print(f"{'Configuration':<40} | {'F1 Score (Macro)':<20} | {'Performance Drop':<20}")
        print("-" * 85)
        for name, score in results.items():
            drop = baseline_score - score
            print(f"{name:<40} | {score:<20.4f} | {drop:<20.4f}")

        print("\n--- Conclusion ---")
        if not performance_drops or all(v <= 0 for v in performance_drops.values()):
             if baseline_score == 0:
                print("Study was inconclusive as baseline performance was zero.")
             else:
                print("No component removal resulted in a significant performance drop.")
        else:
            most_impactful_component = max(performance_drops, key=performance_drops.get)
            print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

    except Exception as e:
        print(f"An error occurred during the ablation study: {e}")
        print(traceback.format_exc())

