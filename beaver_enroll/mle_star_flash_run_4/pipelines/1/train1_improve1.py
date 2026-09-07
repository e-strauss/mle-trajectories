
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
                print(f"Warning: {name}.csv not found at {path}. Proceeding without this data.", file=sys.stderr)
                dataframes[name] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading {name} data: {e}", file=sys.stderr)
            dataframes[name] = pd.DataFrame()

    return dataframes

def feature_engineering(dfs):
    """
    Merges dataframes and creates a combined set of features from both base and reference solutions,
    including new hierarchical features.
    """
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files (subject_summary or gold_enrollment_train) are missing or empty.")

    # --- Start with core data ---
    # Ensure TERM_CODE is consistent for merging
    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    # Merge summary with gold labels
    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # --- Base Solution Feature Engineering ---
    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    # --- Reference Solution Feature Engineering ---
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
        # Create empty columns if faculty data is missing
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan

    # Original ratio features
    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

    # --- Improvement: Hierarchical & Contextual Department Features ---
    # This section contextualizes a subject's metrics relative to its department for a given term.
    if 'TOTAL_SEATS' in data.columns and 'NUM_COURSES' in data.columns:
        # 1. Aggregate key metrics at the department level for each term
        dept_agg = data.groupby(['TERM_CODE', 'DEPARTMENT']).agg(
            DEPT_TOTAL_SEATS=('TOTAL_SEATS', 'sum'),
            DEPT_TOTAL_COURSES=('NUM_COURSES', 'sum')
        ).reset_index()

        # 2. Merge department-level aggregates back to the main dataframe
        data = pd.merge(data, dept_agg, on=['TERM_CODE', 'DEPARTMENT'], how='left')

        # 3. Create powerful ratio features to normalize subject metrics by department totals
        # This captures the proportion of department resources/offerings a subject represents.
        data['SEAT_PROPORTION_IN_DEPT'] = data['TOTAL_SEATS'] / data['DEPT_TOTAL_SEATS']
        data['COURSE_PROPORTION_IN_DEPT'] = data['NUM_COURSES'] / data['DEPT_TOTAL_COURSES']

    # --- Final Processing ---
    # Handle infinities from division by zero and NaNs created
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create sortable integer term code
    data['TERM_CODE_INT'] = data['TERM_CODE']
    
    # Convert target variable 'Y'/'N' to binary 1/0, drop rows where target is missing
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
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def main():
    """
    Main function to execute the integrated training and validation pipeline.
    """
    parser = argparse.ArgumentParser(description='Predict high enrollment for courses.')
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR,
                        help='Directory containing training data tables.')
    args = parser.parse_args()

    # --- 0. Dependency Installation ---
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: Failed to install or import dependencies. {e}", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    # --- 1. Data Loading ---
    dataframes = load_data(args.train_data_dir)

    # --- 2. Feature Engineering ---
    try:
        data = feature_engineering(dataframes)
    except ValueError as e:
        print(f"Feature engineering failed: {e}", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    # --- 3. Data Splitting (Time-based) ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    if len(sorted_terms) < 2:
        print("Warning: Only one term available. Using random split for validation.", file=sys.stderr)
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
        
        if train_df.empty or val_df.empty:
            print("Warning: Time-based split resulted in an empty train or validation set. Using random split.", file=sys.stderr)
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        print("Final dataset is too small to split or train. Cannot train or validate.", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- 4. Model Training ---

    # --- Model 1: RandomForest (Base Solution) ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_rf.fit(train_df, y_train)

    # --- Model 2: LightGBM (New Addition) ---
    lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
    pipeline_lgbm.fit(train_df, y_train)

    # --- Model 3: TabPFN (Reference Solution) ---
    tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    tabpfn_num_features = [
        'NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS',
        'NUM_FACULTY', 'AVG_FACULTY_LOAD',
        'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'
    ]
    
    tabpfn_cat_features = [f for f in tabpfn_cat_features if f in train_df.columns]
    tabpfn_num_features = [f for f in tabpfn_num_features if f in train_df.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features

    train_df_tabpfn = train_df.copy()
    val_df_tabpfn = val_df.copy()

    # Factorize categorical features correctly (fit on train, transform val)
    for col in tabpfn_cat_features:
        codes, uniques = pd.factorize(train_df_tabpfn[col])
        train_df_tabpfn[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

    X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0)
    X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0)

    if X_train_tabpfn.shape[1] > 100:
        print(f"Warning: Number of features ({X_train_tabpfn.shape[1]}) for TabPFN exceeds 100. Truncating.", file=sys.stderr)
        X_train_tabpfn = X_train_tabpfn.iloc[:, :100]
        X_val_tabpfn = X_val_tabpfn.iloc[:, :100]

    clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)

    # --- 5. Validation and Ensembling ---
    if not y_val.empty:
        # Get probabilities from all models
        proba_rf = pipeline_rf.predict_proba(val_df)[:, 1]
        proba_lgbm = pipeline_lgbm.predict_proba(val_df)[:, 1]
        proba_tabpfn = clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1]

        # Simple averaging ensemble
        proba_ensemble = (proba_rf + proba_lgbm + proba_tabpfn) / 3
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        final_validation_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        print(f'Final Validation Performance: {final_validation_score}')
    else:
        print("Validation set is empty. Cannot compute performance.", file=sys.stderr)
        print('Final Validation Performance: 0.0')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred in the main execution block: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # To ensure the evaluation system receives a score even on failure
        print('Final Validation Performance: 0.0')
