
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
import sys
import subprocess

def main():
    """
    Main function to run the training and evaluation pipeline for predicting high enrollment.
    """
    # --- Installation of dependencies ---
    # The 'tabpfn' library is a required dependency. This block ensures it's installed.
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn"])
        from tabpfn import TabPFNClassifier
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: Failed to install or import 'tabpfn'. {e}", file=sys.stderr)
        print(f'Final Validation Performance: {0.0}')
        return

    # --- Path definitions ---
    # Data is expected to be in the './input' directory.
    TRAIN_DATA_DIR = "./input"

    # Construct full paths to the data files.
    COURSE_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'course_summary.csv')
    FACULTY_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'faculty_summary.csv')
    SUBJECT_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'subject_summary.csv')
    GOLD_ENROLLMENT_PATH = os.path.join(TRAIN_DATA_DIR, 'gold_enrollment_train.csv')

    # --- Loading data ---
    try:
        course_summary = pd.read_csv(COURSE_SUMMARY_PATH)
        subject_summary = pd.read_csv(SUBJECT_SUMMARY_PATH)
        gold_enrollment = pd.read_csv(GOLD_ENROLLMENT_PATH)
        # Faculty summary might be optional.
        if os.path.exists(FACULTY_SUMMARY_PATH):
            faculty_summary = pd.read_csv(FACULTY_SUMMARY_PATH)
        else:
            faculty_summary = pd.DataFrame()
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Please ensure data is in the './input' directory.", file=sys.stderr)
        # As per requirements, do not exit, but return and print a default score.
        print(f'Final Validation Performance: {0.0}')
        return

    # --- Data Preprocessing and Feature Engineering ---
    # Convert TERM_CODE to a sortable integer for time-based splitting.
    for df in [course_summary, faculty_summary, subject_summary, gold_enrollment]:
        if 'TERM_CODE' in df.columns:
            # Handle potential float or object types for TERM_CODE.
            df['TERM_CODE'] = df['TERM_CODE'].astype(str).str.split('.').str[0]
            df['TERM_CODE_INT'] = df['TERM_CODE'].astype(int)

    # Start with subject_summary as the base table.
    data = subject_summary.copy()

    # Merge gold standard labels.
    data = pd.merge(data, gold_enrollment, on=['TERM_CODE', 'SUBJECT_ID_SORT', 'TERM_CODE_INT'], how='left')

    # Aggregate course features by subject and term.
    course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
        NUM_COURSES=('COURSE_ID', 'nunique'),
        TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'),
        AVG_SEATS=('MAX_ENROLLMENT', 'mean')
    ).reset_index()
    data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Aggregate faculty features by subject and term.
    if not faculty_summary.empty and all(col in faculty_summary.columns for col in ['FACULTY_ID', 'NUM_COURSES_TAUGHT']):
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        data['NUM_FACULTY'] = 0
        data['AVG_FACULTY_LOAD'] = 0

    # Create new interaction features.
    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

    # Handle missing values and infinities from division by zero.
    data.fillna(0, inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)

    # Define target and features.
    target = 'HIGH_ENROLLMENT'
    data.dropna(subset=[target], inplace=True)
    data[target] = data[target].apply(lambda x: 1 if x == 'Y' else 0)

    # --- Prepare Features for TabPFN ---
    # TabPFN requires numerical inputs. We will use label encoding for categoricals.
    categorical_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    numerical_features = [
        'NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS',
        'NUM_FACULTY', 'AVG_FACULTY_LOAD',
        'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'
    ]
    
    for col in categorical_features:
        if col in data.columns:
            data[col] = pd.factorize(data[col])[0]
            
    features = numerical_features + categorical_features
    # Ensure all selected feature columns exist in the dataframe.
    features = [f for f in features if f in data.columns]
    
    # Check feature dimension constraint for TabPFN.
    if len(features) > 100:
        print(f"Warning: Number of features ({len(features)}) exceeds TabPFN's recommended limit of 100. Truncating feature set.", file=sys.stderr)
        features = features[:100]

    # --- Time-based Validation Split ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    all_terms = sorted(data['TERM_CODE_INT'].unique())

    if len(all_terms) > 1:
        validation_term = all_terms[-1] # Use the latest term for validation.
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
    else: # Fallback if only one term exists.
        # This case is unlikely given the task description but is good practice.
        # We need to stratify to ensure both classes are present if possible.
        try:
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])
        except ValueError: # Stratify fails if one class is missing.
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

    # Check for empty splits, which would prevent training or evaluation.
    if train_df.empty or val_df.empty or train_df[target].nunique() < 2:
       print("Training or validation set is empty or contains only one class. Cannot train model.", file=sys.stderr)
       print(f'Final Validation Performance: {0.0}')
       return

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]
    
    # --- Model Training ---
    # TabPFN is designed for small datasets (N_samples <= 1024).
    # The library handles larger datasets by subsampling a random 1024-sample subset internally during fit.
    clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    clf.fit(X_train, y_train)

    # --- Evaluation ---
    # Make predictions on the validation set.
    y_pred, _ = clf.predict(X_val, return_winning_probability=True)

    # Calculate the macro F1 score.
    # Check if there are true labels in the validation set to score against.
    if len(y_val) > 0:
        final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)
    else:
        final_validation_score = 0.0

    print(f'Final Validation Performance: {final_validation_score}')

if __name__ == "__main__":
    main()
