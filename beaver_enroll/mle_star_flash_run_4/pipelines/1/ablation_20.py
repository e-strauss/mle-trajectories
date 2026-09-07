
import pandas as pd
import numpy as np
import os
import shutil
import sys
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback

# --- Dummy Data Creation ---
# To ensure the script is self-contained and runnable
def create_dummy_data(base_dir='./tmp_input'):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # subject_summary.csv
    subject_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202203, 202203],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'ART-101', 'ART-201']
    }
    pd.DataFrame(subject_summary_data).to_csv(os.path.join(base_dir, 'subject_summary.csv'), index=False)

    # gold_enrollment_train.csv
    gold_labels_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202203, 202203],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'ART-101', 'ART-201'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y']
    }
    pd.DataFrame(gold_labels_data).to_csv(os.path.join(base_dir, 'gold_enrollment_train.csv'), index=False)

    # course_summary.csv
    course_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202203, 202203],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'ART-101', 'ART-201'],
        'COURSE_ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'MAX_ENROLLMENT': [100, 50, 120, 60, 80, 70, 40, 90]
    }
    pd.DataFrame(course_summary_data).to_csv(os.path.join(base_dir, 'course_summary.csv'), index=False)

    # faculty_summary.csv
    faculty_summary_data = {
        'TERM_CODE': [202201, 202201, 202202, 202202, 202203, 202203, 202203, 202203],
        'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'CS-101', 'MATH-101', 'ART-101', 'ART-201'],
        'FACULTY_ID': [10, 20, 10, 21, 11, 22, 30, 31],
        'NUM_COURSES_TAUGHT': [2, 1, 2, 1, 1, 2, 1, 1]
    }
    pd.DataFrame(faculty_summary_data).to_csv(os.path.join(base_dir, 'faculty_summary.csv'), index=False)
    
    # course_attributes.csv
    course_attributes_data = {
       'SUBJECT_ID_SORT': ['CS-101', 'MATH-101', 'ART-101', 'ART-201'],
       'CAMPUS_ID_DESC': ['Main', 'Main', 'South', 'Main']
    }
    pd.DataFrame(course_attributes_data).to_csv(os.path.join(base_dir, 'course_attributes.csv'), index=False)

    return base_dir

# --- Core Logic from train.py (modified for ablation) ---

def load_data(train_dir):
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'instructor_attributes': os.path.join(train_dir, 'instructor_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': os.path.join(train_dir, 'gold_enrollment_train.csv')
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

def feature_engineering(dfs, use_contextual_features=True):
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
    
    if not faculty_summary.empty:
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
    
    # --- Ablated Component: Contextual Interaction Features ---
    if use_contextual_features:
        metrics_to_compare = ['TOTAL_SEATS', 'NUM_COURSES', 'NUM_FACULTY']
        for metric in metrics_to_compare:
            if metric in data.columns:
                dept_avg_col = f'DEPT_AVG_{metric}'
                data[dept_avg_col] = data.groupby(['TERM_CODE', 'DEPARTMENT'])[metric].transform('mean')
                ratio_col = f'{metric}_VS_DEPT_AVG_RATIO'
                data[ratio_col] = data[metric] / data[dept_avg_col]
                diff_col = f'{metric}_VS_DEPT_AVG_DIFF'
                data[diff_col] = data[metric] - data[dept_avg_col]
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features, imputer_strategy='median'):
    # --- Ablated Component: Imputer Strategy ---
    numeric_transformer = SimpleImputer(strategy=imputer_strategy)
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

def run_ablation_study(train_dir, use_contextual_features=True, use_tabpfn=True, imputer_strategy='median'):
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier

        dataframes = load_data(train_dir)
        data = feature_engineering(dataframes, use_contextual_features=use_contextual_features)

        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        sorted_terms = sorted(data['TERM_CODE_INT'].unique())
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

        if train_df.empty or val_df.empty: return 0.0

        y_train, y_val = train_df['HIGH_ENROLLMENT'], val_df['HIGH_ENROLLMENT']

        pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
        pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

        # Train RF and LGBM
        pipeline_rf = build_classifier_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced'), pipeline_num_features, pipeline_cat_features, imputer_strategy)
        pipeline_rf.fit(train_df, y_train)
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]

        pipeline_lgbm = build_classifier_pipeline(LGBMClassifier(random_state=42, class_weight='balanced'), pipeline_num_features, pipeline_cat_features, imputer_strategy)
        pipeline_lgbm.fit(train_df, y_train)
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]

        # --- Ablated Component: TabPFN Model ---
        if use_tabpfn:
            tabpfn_cat_features = [f for f in ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if f in train_df.columns]
            tabpfn_num_features = [f for f in ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY'] if f in train_df.columns]
            all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

            train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
            for col in tabpfn_cat_features:
                codes, uniques = pd.factorize(train_df_tabpfn[col])
                train_df_tabpfn[col] = codes
                mapping = {label: i for i, label in enumerate(uniques)}
                val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

            X_train_tabpfn, X_val_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0), val_df_tabpfn[all_tabpfn_features].fillna(0)
            
            if X_train_tabpfn.shape[1] > 100:
                X_train_tabpfn = X_train_tabpfn.iloc[:, :100]
                X_val_tabpfn = X_val_tabpfn.iloc[:, :100]
            
            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=8) # Reduced for speed
            clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
            proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]
            
            proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        else:
            proba_ensemble = (proba_rf + proba_lgbm) / 2

        pred_ensemble = (proba_ensemble >= 0.5).astype(int)
        return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    except Exception:
        # traceback.print_exc() # Uncomment for debugging
        return 0.0

def main():
    # Install dependencies quietly
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
    except Exception:
        print("Failed to install dependencies. Cannot run study.", file=sys.stderr)
        return

    dummy_data_dir = create_dummy_data()
    results = {}

    print("--- Running Ablation Study ---")
    
    # Baseline
    results['Baseline (All Features + TabPFN + Median Impute)'] = run_ablation_study(dummy_data_dir, use_contextual_features=True, use_tabpfn=True, imputer_strategy='median')
    
    # Ablation 1: No Contextual Features
    results['Ablation: No Contextual Features'] = run_ablation_study(dummy_data_dir, use_contextual_features=False, use_tabpfn=True, imputer_strategy='median')
    
    # Ablation 2: No TabPFN Model
    results['Ablation: No TabPFN Model'] = run_ablation_study(dummy_data_dir, use_contextual_features=True, use_tabpfn=False, imputer_strategy='median')
    
    # Ablation 3: Change Imputer to Mean
    results['Ablation: Use Mean Imputation'] = run_ablation_study(dummy_data_dir, use_contextual_features=True, use_tabpfn=True, imputer_strategy='mean')
    
    # Clean up dummy data
    shutil.rmtree(dummy_data_dir)

    # --- Print and Analyze Results ---
    baseline_score = results.get('Baseline (All Features + TabPFN + Median Impute)', 0.0)
    performance_drops = {}

    print("\n--- Ablation Study Results ---")
    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
        if 'Ablation' in name:
            component_name = name.replace('Ablation: ', '').strip()
            performance_drops[component_name] = drop

    print("\n--- Conclusion ---")
    if not performance_drops or max(performance_drops.values()) <= 0.0:
        print("No single component removal resulted in a significant performance drop.")
    else:
        most_impactful_component = max(performance_drops, key=performance_drops.get)
        print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

if __name__ == '__main__':
    main()
