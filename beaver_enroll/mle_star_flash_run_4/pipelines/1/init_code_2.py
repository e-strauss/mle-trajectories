
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

# Define constants based on the problem description
# All input data is stored in the "./input" directory.
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = os.path.join(BASE_INPUT_DIR, 'table_splits/train')
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir, gold_labels_path):
    """
    Loads all necessary CSV files into pandas DataFrames.
    Includes error handling for missing files.
    """
    try:
        subject_summary_path = os.path.join(train_dir, 'subject_summary.csv')
        course_attributes_path = os.path.join(train_dir, 'course_attributes.csv')
        
        if not os.path.exists(subject_summary_path):
            raise FileNotFoundError(f"subject_summary.csv not found at {subject_summary_path}")
        if not os.path.exists(course_attributes_path):
            raise FileNotFoundError(f"course_attributes.csv not found at {course_attributes_path}")
        if not os.path.exists(gold_labels_path):
            raise FileNotFoundError(f"Gold labels file not found at {gold_labels_path}")

        subject_summary = pd.read_csv(subject_summary_path)
        course_attributes = pd.read_csv(course_attributes_path)
        gold_labels = pd.read_csv(gold_labels_path)
        
        # The 'instructor_attributes.csv' file might be optional.
        instructor_path = os.path.join(train_dir, 'instructor_attributes.csv')
        instructor_attributes = pd.read_csv(instructor_path) if os.path.exists(instructor_path) else None

        return subject_summary, course_attributes, instructor_attributes, gold_labels
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def feature_engineering(subject_summary, course_attributes, instructor_attributes, gold_labels):
    """
    Merges dataframes and creates features for the model.
    """
    # Merge summary with gold labels, which contains the target variable.
    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'])

    # Merge with course attributes. Assuming a common key 'SUBJECT_ID_SORT'.
    # A left merge is used to keep all rows from the primary data.
    if 'SUBJECT_ID_SORT' in course_attributes.columns:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')

    # Engineer features
    # Extract department from the subject ID, e.g., 'CSCI' from 'CSCI-101-A'
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]
    
    # Ensure TERM_CODE is numeric for sorting and time-based splitting.
    data['TERM_CODE'] = pd.to_numeric(data['TERM_CODE'])

    # Convert target variable 'Y'/'N' to binary 1/0.
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def main():
    """
    Main function to execute the training and validation pipeline.
    """
    parser = argparse.ArgumentParser(description='Predict high enrollment for courses.')
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR,
                        help='Directory containing training data tables.')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='Directory for test data (placeholder, not used in training).')
    
    args = parser.parse_args()

    # --- 1. Data Loading ---
    subject_summary, course_attributes, instructor_attributes, gold_labels = load_data(
        args.train_data_dir, GOLD_LABELS_PATH
    )

    # --- 2. Feature Engineering ---
    data = feature_engineering(subject_summary, course_attributes, instructor_attributes, gold_labels)

    # --- 3. Data Splitting (Time-based) ---
    # Sort by term and use the latest term for validation, as per instructions.
    sorted_terms = sorted(data['TERM_CODE'].unique())
    
    if len(sorted_terms) < 2:
        print("Warning: Only one term available. Using random split for validation.")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])
    else:
        validation_term = sorted_terms[-1]
        train_df = data[data['TERM_CODE'] < validation_term].copy()
        val_df = data[data['TERM_CODE'] == validation_term].copy()
        
        if train_df.empty or val_df.empty:
            print("Warning: Time-based split resulted in an empty train or validation set. Using random split.")
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['HIGH_ENROLLMENT'])

    y_train = train_df['HIGH_ENROLLMENT']
    X_train = train_df.drop(columns=['HIGH_ENROLLMENT'])
    y_val = val_df['HIGH_ENROLLMENT']
    X_val = val_df.drop(columns=['HIGH_ENROLLMENT'])

    # --- 4. Model Pipeline Setup ---
    # Identify feature types
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Remove identifiers and non-feature columns from feature lists
    id_cols = ['TERM_CODE', 'SUBJECT_ID_SORT']
    categorical_features = [col for col in categorical_features if col not in id_cols]
    numeric_features = [col for col in numeric_features if col not in id_cols]

    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a preprocessor to apply transformations to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly handled
    )

    # Define the classifier
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # --- 5. Training ---
    pipeline.fit(X_train, y_train)

    # --- 6. Validation and Evaluation ---
    if not y_val.empty:
        y_pred = pipeline.predict(X_val)
        final_validation_score = f1_score(y_val, y_pred, average='macro')
        print(f'Final Validation Performance: {final_validation_score}')
    else:
        print("Validation set is empty. Cannot compute performance.")
        print('Final Validation Performance: 0.0')


if __name__ == '__main__':
    # Wrap the main execution in a try-except block to catch any unhandled exceptions
    # and provide a detailed traceback, preventing silent crashes.
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred in the main execution block: {e}")
        print(traceback.format_exc())
