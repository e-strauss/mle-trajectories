
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
import sys

def main():
    """
    Main function to run the training and evaluation pipeline for predicting high enrollment.
    """
    # --- Path definitions ---
    # The original FileNotFoundError indicated an incorrect path to the data.
    # The error message "./input/table_splits/train/course_summary.csv not found"
    # suggests the subdirectory structure was wrong.
    # Based on the instruction "All the provided input data is stored in './input' directory",
    # the paths are corrected to point directly into the './input' directory.
    TRAIN_DATA_DIR = "./input"

    COURSE_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'course_summary.csv')
    FACULTY_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'faculty_summary.csv')
    SUBJECT_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'subject_summary.csv')
    # The gold standard file is also expected to be in the main input directory for training.
    GOLD_ENROLLMENT_PATH = os.path.join(TRAIN_DATA_DIR, 'gold_enrollment_train.csv')

    # --- Loading data ---
    # Load data from the corrected paths.
    # If a file is missing, print an error and exit gracefully by returning.
    try:
        course_summary = pd.read_csv(COURSE_SUMMARY_PATH)
        subject_summary = pd.read_csv(SUBJECT_SUMMARY_PATH)
        gold_enrollment = pd.read_csv(GOLD_ENROLLMENT_PATH)
        # Faculty summary might be optional or not exist in all datasets.
        if os.path.exists(FACULTY_SUMMARY_PATH):
            faculty_summary = pd.read_csv(FACULTY_SUMMARY_PATH)
        else:
            faculty_summary = pd.DataFrame()
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Please check the input directory structure.", file=sys.stderr)
        print(f'Final Validation Performance: {0.0}')
        return

    # --- Data Preprocessing and Feature Engineering ---
    # Convert TERM_CODE to a sortable integer for time-based splitting.
    # Using a loop to process all relevant dataframes.
    for df in [course_summary, faculty_summary, subject_summary, gold_enrollment]:
        if 'TERM_CODE' in df.columns:
            # TERM_CODE can be float, convert to int then string to handle both cases.
            df['TERM_CODE'] = df['TERM_CODE'].astype(float).astype(int).astype(str)
            df['TERM_CODE_INT'] = df['TERM_CODE'].astype(int)

    # Start with subject_summary as it contains the keys for prediction (TERM_CODE, SUBJECT_ID_SORT).
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
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else: # If faculty summary is empty or missing key columns, create placeholder columns.
        data['NUM_FACULTY'] = 0
        data['AVG_FACULTY_LOAD'] = 0

    # Feature Engineering.
    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

    # Handle missing values that may have resulted from merges or divisions.
    data.fillna(0, inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True) # Replace infinities from division by zero.

    # Define target variable, dropping rows where target is missing (if any).
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # --- Feature Selection ---
    # Define categorical and numerical features for the model.
    # Exclude TERM_CODE as TERM_CODE_INT is used for time-based splitting.
    categorical_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    numerical_features = [
        'NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS',
        'NUM_FACULTY', 'AVG_FACULTY_LOAD',
        'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'
    ]
    features = numerical_features + categorical_features
    target = 'HIGH_ENROLLMENT'
    
    # Ensure all selected feature columns exist in the dataframe.
    features = [f for f in features if f in data.columns]
    
    # --- Time-based Validation Split ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    all_terms = sorted(data['TERM_CODE_INT'].unique())
    
    train_df = None
    val_df = None

    if len(all_terms) > 1:
        validation_term = all_terms[-1] # Use the last term for validation.
        train_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
    else: # Fallback for a single term in data.
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target])
    
    # Check for empty splits.
    if train_df.empty or val_df.empty:
       # If splits are empty, it's not possible to train/evaluate.
       # Print a score of 0 as a failure case.
       print(f'Final Validation Performance: {0.0}')
       return

    # --- Preprocessing for RandomForest ---
    # One-Hot Encode categorical features.
    X_train = pd.get_dummies(train_df[features], columns=categorical_features, dummy_na=True)
    y_train = train_df[target]
    
    X_val = pd.get_dummies(val_df[features], columns=categorical_features, dummy_na=True)
    y_val = val_df[target]

    # Align columns between training and validation sets.
    # This ensures both dataframes have the same dummy variables.
    train_cols, val_cols = X_train.align(X_val, join='left', axis=1, fill_value=0)
    
    # Reassign aligned dataframes.
    X_train = train_cols
    X_val = val_cols
    
    # --- Model Training ---
    # Initialize and train the Random Forest Classifier.
    # Use class_weight='balanced' to handle potential class imbalance.
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', oob_score=False)
    rf_clf.fit(X_train, y_train)

    # --- Evaluation ---
    # Make predictions on the validation set.
    y_pred = rf_clf.predict(X_val)

    # Calculate the macro F1 score.
    final_validation_score = f1_score(y_val, y_pred, average='macro')

    print(f'Final Validation Performance: {final_validation_score}')

if __name__ == "__main__":
    main()
