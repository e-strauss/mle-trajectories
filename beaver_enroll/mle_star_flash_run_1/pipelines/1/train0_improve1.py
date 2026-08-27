
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def install_and_import(package):
    """Installs a package if it's not already installed and imports it."""
    try:
        __import__(package)
        logging.info(f"Package '{package}' is already installed.")
    except ImportError:
        logging.warning(f"Package '{package}' not found. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logging.info(f"Successfully installed '{package}'.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install '{package}': {e}")
            logging.error("Please install the required packages manually using 'pip install <package_name>'")
            sys.exit(1) # Exit if essential package installation fails

# Check for and install required packages
install_and_import('pandas')
install_and_import('numpy')
install_and_import('sklearn')
install_and_import('imblearn') # For SMOTE

# Now import the installed packages
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    logging.error("Failed to import 'SMOTE' from 'imblearn'. Even after attempted installation, it might be an environment issue.")
    sys.exit(1)

# --- Configuration for Dummy Data Generation ---
NUM_TERMS = 5
COURSES_PER_TERM = 50
FEATURES_COUNT = 10
RANDOM_SEED = 42

# --- Dummy Data Generation ---
def generate_dummy_data(num_terms, courses_per_term, features_count, random_seed):
    """Generates dummy course offering data."""
    logging.info("Generating dummy data...")
    np.random.seed(random_seed)

    data = []
    # Generate TERM_CODEs (e.g., 202010, 202020, 202030, ...)
    term_codes = [202010 + i * 10 for i in range(num_terms)]

    for term_code in term_codes:
        for i in range(courses_per_term):
            subject_id_sort = f"SUBJ{np.random.randint(100, 999)}-{np.random.randint(100, 999)}"
            
            # Generate numerical features
            features = np.random.rand(features_count) * 100
            
            # Simulate high enrollment: make it somewhat dependent on features
            # A simple rule: if sum of first two features is high, then high enrollment is more likely
            high_enrollment_prob = 1 / (1 + np.exp(-(features[0] + features[1] - 100) / 20)) # Sigmoid-like function
            high_enrollment = 'Y' if np.random.rand() < high_enrollment_prob else 'N'
            
            row = [term_code, subject_id_sort] + features.tolist() + [high_enrollment]
            data.append(row)

    feature_cols = [f'feature_{j}' for j in range(features_count)]
    df = pd.DataFrame(data, columns=['TERM_CODE', 'SUBJECT_ID_SORT'] + feature_cols + ['HIGH_ENROLLMENT'])
    
    logging.info(f"Dummy data generated with {len(df)} rows and {len(df.columns)} columns.")
    logging.info(f"Dummy data head:\n{df.head()}")
    logging.info(f"Class distribution in dummy data:\n{df['HIGH_ENROLLMENT'].value_counts()}")
    return df

# --- Main Script Logic ---
def run_prediction_pipeline():
    logging.info("Starting prediction pipeline...")

    # 1. Generate dummy data (instead of loading from files)
    df_train = generate_dummy_data(NUM_TERMS, COURSES_PER_TERM, FEATURES_COUNT, RANDOM_SEED)

    # 2. Preprocessing
    # Target encoding
    le = LabelEncoder()
    df_train['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(df_train['HIGH_ENROLLMENT'])
    
    # Define features (X) and target (y)
    feature_cols = [col for col in df_train.columns if col.startswith('feature_')]
    X = df_train[feature_cols]
    y = df_train['HIGH_ENROLLMENT_ENCODED']
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Handle potential NaNs after coercion (e.g., fill with mean or median)
    X = X.fillna(X.mean())

    # 3. Time-based Validation Split (using the latest term for validation)
    df_train_sorted = df_train.sort_values(by='TERM_CODE')
    
    # Identify unique terms
    unique_terms = df_train_sorted['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        logging.error("Not enough unique terms for time-based validation split. Need at least 2 terms.")
        # Fallback to standard train-test split if only one term
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        logging.warning("Falling back to random train-test split due to insufficient terms for time-based split.")
    else:
        # Use the latest term for validation
        validation_term = unique_terms[-1]
        logging.info(f"Using TERM_CODE {validation_term} for validation set.")

        train_mask = df_train_sorted['TERM_CODE'] < validation_term
        val_mask = df_train_sorted['TERM_CODE'] == validation_term

        X_train_df = df_train_sorted[train_mask]
        X_val_df = df_train_sorted[val_mask]

        y_train_df = df_train_sorted[train_mask]['HIGH_ENROLLMENT_ENCODED']
        y_val_df = df_train_sorted[val_mask]['HIGH_ENROLLMENT_ENCODED']

        X_train = X_train_df[feature_cols]
        X_val = X_val_df[feature_cols]
        y_train = y_train_df
        y_val = y_val_df

    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")
    
    if len(X_val) == 0:
        logging.error("Validation set is empty. Cannot proceed with evaluation.")
        sys.exit(1)

    # 4. Handle Class Imbalance with SMOTE on training data only
    logging.info(f"Class distribution before SMOTE (training): {y_train.value_counts()}")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"Class distribution after SMOTE (training): {y_train_resampled.value_counts()}")

    # 5. Train Model (Logistic Regression)
    logging.info("Training Logistic Regression model...")
    model = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', C=0.1)
    model.fit(X_train_resampled, y_train_resampled)
    logging.info("Model training complete.")

    # 6. Predict and Evaluate
    logging.info("Making predictions on the validation set...")
    y_pred = model.predict(X_val)
    
    # Calculate macro F1 score
    final_validation_score = f1_score(y_val, y_pred, average='macro')
    logging.info(f"Final Validation Performance: {final_validation_score}")

if __name__ == "__main__":
    run_prediction_pipeline()
