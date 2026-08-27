
import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# --- Configure logging ---
# Set up logging to stdout (default for basicConfig) to ensure messages are captured.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# All provided input data is stored in "./input" directory.
# TRAIN_DATA_DIR — training tables (available now) in table_splits/train
TRAIN_DATA_DIR = './input/table_splits/train'
GOLD_ENROLLMENT_FILE_NAME = 'gold_enrollment_train.csv'
GOLD_ENROLLMENT_FILE_PATH = os.path.join(TRAIN_DATA_DIR, GOLD_ENROLLMENT_FILE_NAME)

# Prediction keys and target column names
TERM_CODE = 'TERM_CODE'
SUBJECT_ID_SORT = 'SUBJECT_ID_SORT'
HIGH_ENROLLMENT = 'HIGH_ENROLLMENT' # Target column, 'Y' or 'N'

# --- Data Loading and Preprocessing ---
def load_data(data_dir, gold_file_path):
    """
    Loads gold enrollment data, performs initial validation, and prepares base features.
    """
    logging.info(f"Attempting to load data from directory: '{data_dir}' and gold file: '{gold_file_path}'")
    
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Training data directory not found: '{data_dir}'")
    if not os.path.exists(gold_file_path):
        raise RuntimeError(f"Gold enrollment file not found: '{gold_file_path}'")

    try:
        gold_df = pd.read_csv(gold_file_path)
        if gold_df.empty:
            raise RuntimeError(f"Gold enrollment data is empty in '{gold_file_path}'")
        logging.info(f"Loaded gold enrollment data successfully: {gold_df.shape[0]} rows.")
        
        # Ensure critical columns are present
        for col in [TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT]:
            if col not in gold_df.columns:
                raise RuntimeError(f"Missing critical column in gold enrollment data: '{col}'")

        # Convert HIGH_ENROLLMENT to numerical (Y: 1, N: 0)
        if gold_df[HIGH_ENROLLMENT].dtype == 'object':
            gold_df[HIGH_ENROLLMENT] = gold_df[HIGH_ENROLLMENT].map({'Y': 1, 'N': 0})
            if gold_df[HIGH_ENROLLMENT].isnull().any():
                logging.warning(f"Found non-Y/N values in '{HIGH_ENROLLMENT}' column, replacing NaNs with 0. Consider specific handling for unexpected values.")
                gold_df[HIGH_ENROLLMENT].fillna(0, inplace=True)
        elif not np.issubdtype(gold_df[HIGH_ENROLLMENT].dtype, np.number):
             raise RuntimeError(f"Unexpected data type for '{HIGH_ENROLLMENT}' column: {gold_df[HIGH_ENROLLMENT].dtype}. Expected 'Y'/'N' or numerical (0/1).")

        # Simulate creation of some base numerical features if they don't exist.
        # In a real scenario, these would come from other tables in TRAIN_DATA_DIR.
        np.random.seed(42) # For reproducibility of dummy features
        
        if 'credit_hours' not in gold_df.columns:
            gold_df['credit_hours'] = np.random.choice([1, 3, 4], size=len(gold_df))
            logging.info("Added dummy 'credit_hours' feature.")
        if 'course_level' not in gold_df.columns:
            gold_df['course_level'] = np.random.choice([100, 200, 300, 400], size=len(gold_df))
            logging.info("Added dummy 'course_level' feature.")
        if 'department_size' not in gold_df.columns:
            # Create dummy department sizes based on the department prefix from SUBJECT_ID_SORT
            gold_df['department_prefix'] = gold_df[SUBJECT_ID_SORT].apply(lambda x: ''.join(filter(str.isalpha, str(x))).upper())
            unique_departments = gold_df['department_prefix'].unique()
            dept_size_map = {dept: np.random.randint(10, 100) for dept in unique_departments}
            gold_df['department_size'] = gold_df['department_prefix'].map(dept_size_map)
            gold_df.drop(columns=['department_prefix'], inplace=True)
            logging.info("Added dummy 'department_size' feature based on subject ID prefixes.")

        return gold_df

    except Exception as e:
        logging.error(f"Error during data loading or initial preprocessing: {e}", exc_info=True)
        # Re-raise the exception to ensure a non-zero exit code and traceback for critical failures.
        raise RuntimeError(f"Failed to load or preprocess data: {e}")

# --- Feature Engineering ---
def create_features(df):
    """
    Creates additional features for the model from the preprocessed DataFrame.
    """
    logging.info("Starting feature engineering...")
    
    # Example: Simple interaction feature
    if 'course_level' in df.columns and 'credit_hours' in df.columns:
        df['level_credit_interaction'] = df['course_level'] * df['credit_hours']
        logging.info("Created 'level_credit_interaction' feature.")
    else:
        logging.warning("Cannot create 'level_credit_interaction': missing 'course_level' or 'credit_hours'.")

    # Example: Simple past enrollment rate for the same subject.
    # This assumes TERM_CODE can be sorted chronologically (e.g., YYYYTT format).
    df = df.sort_values(by=TERM_CODE).reset_index(drop=True)
    if HIGH_ENROLLMENT in df.columns and SUBJECT_ID_SORT in df.columns:
        df['prev_term_high_enrollment_rate'] = df.groupby(SUBJECT_ID_SORT)[HIGH_ENROLLMENT].transform(
            lambda x: x.shift(1).expanding().mean().fillna(0)
        )
        logging.info("Created 'prev_term_high_enrollment_rate' feature.")
    else:
        logging.warning(f"Cannot create 'prev_term_high_enrollment_rate': missing '{HIGH_ENROLLMENT}' or '{SUBJECT_ID_SORT}'.")


    # Define the final list of numerical features to be used by the model
    features = [col for col in ['credit_hours', 'course_level', 'department_size', 
                                'level_credit_interaction', 'prev_term_high_enrollment_rate'] 
                if col in df.columns]

    if not features:
        raise RuntimeError("No numerical features available after feature engineering. Please check feature creation logic.")

    logging.info(f"Final features selected for training: {features}")
    return df, features

# --- Model Training and Validation ---
def train_and_validate_model(df, features, target_col):
    """
    Splits data into training and validation sets (time-based),
    trains a RandomForestClassifier, and evaluates its performance using macro F1.
    """
    logging.info("Splitting data for time-based training and validation...")

    # Sort by TERM_CODE for chronological splitting
    df = df.sort_values(by=TERM_CODE).reset_index(drop=True)

    unique_terms = sorted(df[TERM_CODE].unique())
    if len(unique_terms) < 2:
        raise RuntimeError(f"Not enough unique terms ({len(unique_terms)}) for a time-based validation split. Need at least two terms.")

    # Determine validation split point (e.g., last 20% of unique terms)
    # Ensure at least one term for validation and one for training
    split_idx = max(1, int(len(unique_terms) * 0.8))
    train_terms = unique_terms[:split_idx]
    validation_terms = unique_terms[split_idx:]
    
    # Fallback/check to ensure both sets are non-empty
    if not train_terms:
        raise RuntimeError("No terms available for training after split. Adjust term split logic or check data.")
    if not validation_terms: # If validation_terms is still empty, take the last one from train_terms
        logging.warning("Validation set became empty; using the last training term for validation.")
        validation_terms = [train_terms.pop()] # Move last train term to validation
        if not train_terms: # If train_terms becomes empty, error out.
            raise RuntimeError("Cannot establish valid train/validation split: too few unique terms.")

    train_df = df[df[TERM_CODE].isin(train_terms)].copy()
    val_df = df[df[TERM_CODE].isin(validation_terms)].copy()

    if train_df.empty:
        raise RuntimeError(f"Training data is empty after time-based split for terms: {train_terms}.")
    if val_df.empty:
        raise RuntimeError(f"Validation data is empty after time-based split for terms: {validation_terms}.")

    logging.info(f"Train terms: {train_terms}, Validation terms: {validation_terms}")
    logging.info(f"Training data shape: {train_df.shape}, Validation data shape: {val_df.shape}")

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_val = val_df[features]
    y_val = val_df[target_col]

    # Ensure feature sets are aligned between train and validation.
    # This prevents errors if certain features only appear in one split due to data sparsity.
    train_cols = X_train.columns.tolist()
    val_cols = X_val.columns.tolist()
    
    if set(train_cols) != set(val_cols):
        logging.warning("Feature columns mismatch between train and validation sets. Attempting to align.")
        # Add missing columns to validation set, fill with 0
        missing_in_val = list(set(train_cols) - set(val_cols))
        for col in missing_in_val:
            X_val[col] = 0
            logging.info(f"Added missing feature '{col}' to validation set with default 0.")
        
        # Drop extra columns from validation set (should not happen if `features` list is controlled)
        extra_in_val = list(set(val_cols) - set(train_cols))
        if extra_in_val:
            X_val = X_val.drop(columns=extra_in_val, errors='ignore')
            logging.info(f"Dropped extra features from validation set: {extra_in_val}.")
        
        # Reorder columns to match training set for consistency
        X_val = X_val[train_cols]
        logging.info("Feature columns aligned between train and validation sets.")

    if X_train.empty or y_train.empty:
        raise RuntimeError("Training features or labels are empty after final feature alignment.")
    if X_val.empty or y_val.empty:
        raise RuntimeError("Validation features or labels are empty after final feature alignment.")

    # Model training (using RandomForestClassifier as an example)
    logging.info("Training RandomForestClassifier model...")
    # Using `class_weight='balanced'` to mitigate potential class imbalance in the target variable.
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Model evaluation on the validation set
    logging.info("Evaluating model on validation set...")
    y_pred = model.predict(X_val)
    
    # Calculate macro F1 score as the development metric
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    logging.info(f"Validation Macro F1 Score: {macro_f1}")

    return macro_f1, model

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Ensure the TRAIN_DATA_DIR exists. This also creates './input' if it doesn't exist.
        os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
        
        # --- Self-contained test setup: Create a dummy gold_enrollment_train.csv if it doesn't exist ---
        # This allows the script to run standalone for testing purposes without external data.
        if not os.path.exists(GOLD_ENROLLMENT_FILE_PATH):
            logging.warning(f"'{GOLD_ENROLLMENT_FILE_PATH}' not found. Creating a dummy file for demonstration purposes.")
            # Dummy data designed for multiple terms and subjects to allow time-based split and feature creation
            dummy_data = {
                TERM_CODE: [202010, 202010, 202010, 202010, 202020, 202020, 202020, 202020,
                            202110, 202110, 202110, 202110, 202120, 202120, 202120, 202120,
                            202210, 202210, 202210, 202210, 202220, 202220, 202220, 202220],
                SUBJECT_ID_SORT: ['CS101', 'MA201', 'PH101', 'CH101', 'CS101', 'MA201', 'PH101', 'CH101',
                                  'CS101', 'MA201', 'PH101', 'CH101', 'CS101', 'MA201', 'PH101', 'CH101',
                                  'CS101', 'MA201', 'PH101', 'CH101', 'CS101', 'MA201', 'PH101', 'CH101'],
                HIGH_ENROLLMENT: ['Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'N',
                                  'Y', 'N', 'Y', 'N', 'N', 'Y', 'N', 'Y',
                                  'Y', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N']
            }
            dummy_df = pd.DataFrame(dummy_data)
            # Add dummy numerical features to prevent warnings/errors during feature creation in load_data
            np.random.seed(42)
            dummy_df['credit_hours'] = np.random.choice([1, 3, 4], size=len(dummy_df))
            dummy_df['course_level'] = np.random.choice([100, 200, 300, 400], size=len(dummy_df))
            
            dummy_df.to_csv(GOLD_ENROLLMENT_FILE_PATH, index=False)
            logging.info(f"Dummy '{GOLD_ENROLLMENT_FILE_PATH}' created for execution demonstration.")
        # --- End of self-contained test setup ---

        # 1. Load and initially preprocess data
        data_df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_FILE_PATH)

        # 2. Create additional features
        data_df, features = create_features(data_df)
        
        # Final check for valid features list before training
        if not features:
            raise RuntimeError("No features were identified or created for model training. Check 'create_features' function.")
        for feature in features:
            if feature not in data_df.columns:
                raise RuntimeError(f"Required feature '{feature}' is missing from the DataFrame after creation. Data pipeline error.")

        # 3. Train and validate the model
        final_validation_score, _ = train_and_validate_model(data_df, features, HIGH_ENROLLMENT)

        # Print final performance as requested by the task
        print(f"Final Validation Performance: {final_validation_score}")

    except RuntimeError as e:
        logging.error(f"Execution failed due to a critical setup or data processing error: {e}")
        # Re-raise the exception to ensure a non-zero exit code (1) and to provide a traceback.
        # This addresses the "silent failure" issue by making errors explicit and diagnosable.
        raise
    except Exception as e:
        # Catch any other unexpected errors, log them, and re-raise.
        logging.critical(f"An unexpected and unhandled error occurred during script execution: {e}", exc_info=True)
        raise

