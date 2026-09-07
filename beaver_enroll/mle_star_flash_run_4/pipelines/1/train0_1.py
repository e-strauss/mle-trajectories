
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

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input' # Adjusted to match reference solution's assumption
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
    Merges dataframes and creates a combined set of features from both base and reference solutions.
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
        if 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = df['TERM_CODE'].astype(float).astype(int)

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
    
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = 0
        data['AVG_FACULTY_LOAD'] = 0

    # New features from reference solution
    data['SEATS_PER_COURSE'] = data['NUM_COURSES'].where(data['NUM_COURSES'] > 0, np.nan)
    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['SEATS_PER_COURSE']
    data['COURSES_PER_FACULTY'] = data['NUM_FACULTY'].where(data['NUM_FACULTY'] > 0, np.nan)
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['COURSES_PER_FACULTY']
    
    # --- Final Processing ---
    # Handle infinities from division by zero and NaNs created
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create sortable integer term code
    data['TERM_CODE_INT'] = data['TERM_CODE']
    
    # Convert target variable 'Y'/'N' to binary 1/0, drop rows where target is missing
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def build_pipeline(numeric_features, categorical_features):
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
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
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

    # --- 1. Data Loading ---
    dataframes = load_data(args.train_data_dir)

    # --- 2. Feature Engineering ---
    data = feature_engineering(dataframes)

    # --- 3. Data Splitting (Time-based) ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    if len(sorted_terms) < 2:
        print("Warning: Only one term available. Using random split for validation.")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
        
        if train_df.empty or val_df.empty:
            print("Warning: Time-based split resulted in an empty train or validation set. Using random split.")
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

    if train_df.empty or val_df.empty:
        print("Final dataset is too small to split. Cannot train or validate.")
        print('Final Validation Performance: 0.0')
        return

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # --- 4. Model Pipeline Setup & Training ---

    # --- Model 1: Base Solution ---
    base_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    base_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT'] and '_attr' in col]
    
    pipeline_base = build_pipeline(base_num_features, base_cat_features)
    pipeline_base.fit(train_df, y_train)

    # --- Model 2: Reference Solution ---
    ref_cat_features = [col for col in ['CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC'] if col in data.columns]
    ref_num_features = [col for col in [
        'NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD',
        'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'
    ] if col in data.columns]

    pipeline_ref = build_pipeline(ref_num_features, ref_cat_features)
    pipeline_ref.fit(train_df, y_train)
    
    # --- 5. Validation and Ensembling ---
    if not y_val.empty:
        # Get probabilities from both models
        proba_base = pipeline_base.predict_proba(val_df)[:, 1]
        proba_ref = pipeline_ref.predict_proba(val_df)[:, 1]

        # Simple averaging ensemble
        proba_ensemble = (proba_base + proba_ref) / 2
        pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        final_validation_score = f1_score(y_val, pred_ensemble, average='macro')
        print(f'Final Validation Performance: {final_validation_score}')
    else:
        print("Validation set is empty. Cannot compute performance.")
        print('Final Validation Performance: 0.0')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred in the main execution block: {e}")
        print(traceback.format_exc())
        # To ensure the evaluation system receives a score even on failure
        print('Final Validation Performance: 0.0')

