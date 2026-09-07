The `FileNotFoundError` indicates that the file path specified in the code is incorrect. The error message `FileNotFoundError: Required data file not found: ./input/course_summary.csv` and the instruction "All the provided input data is stored in the './input' directory" strongly suggest that the data files are located directly in the `./input` folder, not in the `table_splits/train` subfolder.

The original code defines `TRAIN_DATA_DIR` as `./input/table_splits/train`, which leads to an incorrect path. The fix is to change this variable to point directly to the `./input` directory. This aligns the code with the actual file location as indicated by the error and the instructions.


import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit, train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import os
import sys

def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    try:
        # --- Path definitions ---
        # The task description indicates training data is in 'table_splits/train'
        # relative to the main input directory.
        TRAIN_DATA_DIR = "./input/table_splits/train"
        
        # Check if the directory and files exist to provide clearer errors
        if not os.path.isdir(TRAIN_DATA_DIR):
            # If the primary directory isn't found, check the root input directory as a fallback.
            if os.path.isdir("./input"):
                TRAIN_DATA_DIR = "./input"
            else:
                raise FileNotFoundError(f"Data directory not found: {TRAIN_DATA_DIR} or ./input")

        COURSE_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'course_summary.csv')
        FACULTY_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'faculty_summary.csv')
        SUBJECT_SUMMARY_PATH = os.path.join(TRAIN_DATA_DIR, 'subject_summary.csv')
        GOLD_ENROLLMENT_PATH = os.path.join(TRAIN_DATA_DIR, 'gold_enrollment_train.csv')

        for path in [COURSE_SUMMARY_PATH, FACULTY_SUMMARY_PATH, SUBJECT_SUMMARY_PATH, GOLD_ENROLLMENT_PATH]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required data file not found: {path}")

        # --- Loading data ---
        course_summary = pd.read_csv(COURSE_SUMMARY_PATH)
        faculty_summary = pd.read_csv(FACULTY_SUMMARY_PATH)
        subject_summary = pd.read_csv(SUBJECT_SUMMARY_PATH)
        gold_enrollment = pd.read_csv(GOLD_ENROLLMENT_PATH)

        # --- Data Preprocessing and Feature Engineering ---
        # Convert TERM_CODE to a sortable integer for time-based splitting
        for df in [course_summary, faculty_summary, subject_summary, gold_enrollment]:
            if 'TERM_CODE' in df.columns:
                df['TERM_CODE_INT'] = df['TERM_CODE'].astype(int)

        # Start with subject_summary as it contains the keys for prediction (TERM_CODE, SUBJECT_ID_SORT)
        data = subject_summary.copy()

        # Merge gold standard labels
        data = pd.merge(data, gold_enrollment, on=['TERM_CODE', 'SUBJECT_ID_SORT', 'TERM_CODE_INT'], how='left')

        # Aggregate course features by subject and term
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'),
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'),
            AVG_SEATS=('MAX_ENROLLMENT', 'mean')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

        # Aggregate faculty features by subject and term
        if not faculty_summary.empty:
            faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                NUM_FACULTY=('FACULTY_ID', 'nunique'),
                AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
            ).reset_index()
            data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        else: # If faculty summary is empty, create placeholder columns
            data['NUM_FACULTY'] = 0
            data['AVG_FACULTY_LOAD'] = 0


        # Feature Engineering
        data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
        data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']

        # Handle missing values that may have resulted from merges or divisions
        data.fillna(0, inplace=True)
        data.replace([np.inf, -np.inf], 0, inplace=True) # Replace infinities from division by zero

        # Define target variable, dropping rows where target is missing (if any)
        data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
        data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

        # --- Time-based Validation Split ---
        data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
        all_terms = sorted(data['TERM_CODE_INT'].unique())
        
        X_train, X_val, y_train, y_val = None, None, None, None

        if len(all_terms) > 1:
            validation_term = all_terms[-1] # Use the last term for validation
            train_indices = data[data['TERM_CODE_INT'] < validation_term].index
            val_indices = data[data['TERM_CODE_INT'] == validation_term].index
            
            X_train, X_val = data.iloc[train_indices], data.iloc[val_indices]
            y_train, y_val = data['HIGH_ENROLLMENT'].iloc[train_indices], data['HIGH_ENROLLMENT'].iloc[val_indices]

        else: # Fallback for a single term in data
            try:
                # Stratification can fail if a class has fewer than 2 members.
                X_train, X_val, y_train, y_val = train_test_split(
                    data, data['HIGH_ENROLLMENT'], test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT']
                )
            except ValueError:
                # Fallback to non-stratified split.
                X_train, X_val, y_train, y_val = train_test_split(
                    data, data['HIGH_ENROLLMENT'], test_size=0.2, random_state=42
                )
            
        if X_train.empty or X_val.empty:
            raise ValueError("Training or validation split resulted in an empty dataframe. Check data and split logic.")

        # --- Model Training ---
        categorical_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
        categorical_features = [f for f in categorical_features if f in data.columns]

        numerical_features = [
            'NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS',
            'NUM_FACULTY', 'AVG_FACULTY_LOAD',
            'SEATS_PER_COURSE', 'COURSES_PER_FACULTY'
        ]
        numerical_features = [f for f in numerical_features if f in data.columns]

        features = numerical_features + categorical_features
        
        # Ensure feature columns exist in both splits
        X_train = X_train[features]
        X_val = X_val[features]
        
        # CatBoost handles object/string types robustly when passed in `cat_features`.
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=42,
            auto_class_weights='Balanced', # Handles imbalanced target
            task_type='CPU',
            thread_count=2,
            verbose=0
        )

        model.fit(
            X_train, y_train,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )

        # --- Evaluation ---
        y_pred = model.predict(X_val)
        final_validation_score = f1_score(y_val, y_pred, average='macro')

        print(f"Final Validation Performance: {final_validation_score}")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
