
# Required libraries:
# pip install pandas scikit-learn numpy

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging to capture informational messages and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use the specified input directory for data
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")

# TEST_DATA_DIR is a placeholder; it's not available yet and not used during training.
TEST_DATA_DIR = None

# --- Data Loading ---
def load_data(data_dir, gold_enrollment_path=None):
    """Loads all relevant tables and the gold enrollment file from the specified directory."""
    dfs = {}
    try:
        # Load gold enrollment data first if provided
        if gold_enrollment_path and os.path.exists(gold_enrollment_path):
            dfs['gold_enrollment'] = pd.read_csv(gold_enrollment_path)
            logging.info(f"Loaded gold enrollment data from {gold_enrollment_path}. Shape: {dfs['gold_enrollment'].shape}")
        else:
            if gold_enrollment_path:
                logging.error(f"Gold enrollment file not found: {gold_enrollment_path}. This file is critical.")
                raise FileNotFoundError(f"Gold enrollment file not found at {gold_enrollment_path}")
            else:
                logging.warning("No gold enrollment path provided. This might lead to errors later.")

        # Load other tables from the data directory
        if data_dir and os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                # Ensure we only load CSV files and not the gold enrollment file again
                if filename.endswith(".csv") and filename != os.path.basename(gold_enrollment_path):
                    filepath = os.path.join(data_dir, filename)
                    table_name = os.path.splitext(filename)[0]
                    try:
                        dfs[table_name] = pd.read_csv(filepath)
                        logging.info(f"Loaded {table_name} from {filepath}. Shape: {dfs[table_name].shape}")
                    except Exception as e:
                        logging.warning(f"Could not load {filename}: {e}")
        else:
            logging.warning(f"Data directory not found or empty: {data_dir}. No additional tables loaded.")

    except FileNotFoundError: # Re-raise for specific handling in main
        raise
    except Exception as e:
        logging.error(f"An error occurred during data loading: {e}", exc_info=True)
        raise # Re-raise to stop execution if data loading fails critically

    if not dfs:
        raise ValueError("No dataframes were loaded. Check paths and file presence.")

    return dfs

# --- Data Preprocessing ---
def preprocess_data(dfs):
    """
    Combines and preprocesses the loaded data for model training.
    This includes feature engineering, handling categorical variables, and target encoding.
    """
    if 'gold_enrollment' not in dfs or dfs['gold_enrollment'].empty:
        logging.error("Gold enrollment data is missing or empty. Cannot preprocess.")
        raise ValueError("Missing or empty gold enrollment data.")

    main_df = dfs['gold_enrollment'].copy()
    logging.info(f"Initial main_df shape: {main_df.shape}")

    # Feature Engineering
    # 1. Temporal features from TERM_CODE
    main_df['TERM_YEAR'] = main_df['TERM_CODE'].astype(str).str[:4].astype(int)
    main_df['TERM_QUARTER'] = main_df['TERM_CODE'].astype(str).str[4:].astype(int)
    logging.info("Generated TERM_YEAR and TERM_QUARTER features.")

    # 2. Incorporate SUBJECT_ID_SORT as a categorical feature
    # It's a key, but also potentially a strong predictor of enrollment patterns
    main_df['SUBJECT_ID_SORT'] = main_df['SUBJECT_ID_SORT'].astype(str) # Ensure it's treated as categorical
    logging.info("Converted SUBJECT_ID_SORT to categorical type.")

    # 3. Merge with other tables if available and relevant
    merged_with_additional_data = False
    for key, df_other in dfs.items():
        if key != 'gold_enrollment' and isinstance(df_other, pd.DataFrame) and not df_other.empty:
            # Keys to try for merging. Common identifiers between enrollment and other course data.
            merge_on_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']
            
            # Check if all merge keys exist in both dataframes
            if all(col in main_df.columns for col in merge_on_keys) and \
               all(col in df_other.columns for col in merge_on_keys):
                
                df_other_cleaned = df_other.drop(columns=['HIGH_ENROLLMENT'], errors='ignore') # Avoid duplicate target col
                initial_shape = main_df.shape
                main_df = pd.merge(main_df, df_other_cleaned, on=merge_on_keys, how='left', suffixes=('', f'_{key}_info'))
                logging.info(f"Merged with '{key}' table on {merge_on_keys}. Shape changed from {initial_shape} to {main_df.shape}")
                merged_with_additional_data = True
            else:
                logging.warning(f"Could not merge with '{key}' table. Missing common keys {merge_on_keys} in one of the dataframes.")
    
    if not merged_with_additional_data:
        logging.info("No additional tables were successfully merged into the main dataframe (or none were available).")


    # Encode the target variable
    le = LabelEncoder()
    main_df['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(main_df['HIGH_ENROLLMENT'])
    logging.info(f"Target variable 'HIGH_ENROLLMENT' encoded to numerical values: {le.classes_}")

    # Select features (columns for X)
    # Exclude original TERM_CODE, original target, and the encoded target from the feature set.
    # 'SUBJECT_ID_SORT' is kept for one-hot encoding.
    feature_cols_candidate = [col for col in main_df.columns if col not in ['TERM_CODE', 'HIGH_ENROLLMENT', 'HIGH_ENROLLMENT_ENCODED']]

    # Handle categorical features (One-hot encoding)
    categorical_cols = main_df[feature_cols_candidate].select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        logging.info(f"One-hot encoding categorical columns: {list(categorical_cols)}")
        main_df = pd.get_dummies(main_df, columns=categorical_cols, prefix=categorical_cols.tolist(), dummy_na=False) # Keep NaNs as separate category if present
        # Update feature_cols_candidate after one-hot encoding to include new dummy variables
        feature_cols_candidate = [col for col in main_df.columns if col not in ['TERM_CODE', 'HIGH_ENROLLMENT', 'HIGH_ENROLLMENT_ENCODED']]
    else:
        logging.info("No categorical columns found for one-hot encoding.")


    # Final feature selection and NaN handling for numerical features
    final_feature_cols = []
    for col in feature_cols_candidate:
        if pd.api.types.is_numeric_dtype(main_df[col]):
            if main_df[col].isnull().any():
                mean_val = main_df[col].mean()
                main_df[col] = main_df[col].fillna(mean_val) # Simple mean imputation for numerical features
                logging.warning(f"Imputed missing values in numeric column '{col}' with mean: {mean_val:.2f}")
            final_feature_cols.append(col)
        else:
            logging.warning(f"Dropping non-numeric feature column '{col}' from feature set after all processing stages.")
            # This handles cases where columns might remain object type unexpectedly.

    if not final_feature_cols:
        logging.error("No numeric features available after preprocessing. Check data and feature engineering steps.")
        raise ValueError("No features available for training. Ensure data can form numerical features.")

    X = main_df[final_feature_cols]
    y = main_df['HIGH_ENROLLMENT_ENCODED']

    logging.info(f"Final features (X) shape: {X.shape}, target (y) shape: {y.shape}")
    logging.info(f"Features used: {X.columns.tolist()}")

    # Return features, target, and metadata needed for time-based split
    return X, y, main_df[['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]

# --- Model Training and Validation ---
def train_and_validate(X, y, df_metadata):
    """
    Trains a model and validates it using a time-based split,
    calculating macro F1 score as the development metric.
    """
    # Combine features and metadata for time-based splitting
    # Ensure X and df_metadata align by index or key if needed, here assuming they align.
    df_for_split = X.copy()
    df_for_split['TERM_CODE'] = df_metadata['TERM_CODE']
    df_for_split['HIGH_ENROLLMENT_ENCODED'] = y

    # Sort data by TERM_CODE to ensure a time-based split
    df_for_split = df_for_split.sort_values(by='TERM_CODE').reset_index(drop=True)

    # Determine validation split point (e.g., latest 20% of unique terms for validation)
    unique_terms = df_for_split['TERM_CODE'].unique()
    if len(unique_terms) < 2:
        logging.error("Not enough unique terms for a time-based split. At least two terms are required.")
        raise ValueError("Insufficient unique terms for time-based validation split.")

    split_point_index = int(len(unique_terms) * 0.8)
    train_terms = unique_terms[:split_point_index]
    val_terms = unique_terms[split_point_index:]

    X_train = df_for_split[df_for_split['TERM_CODE'].isin(train_terms)].drop(columns=['TERM_CODE', 'HIGH_ENROLLMENT_ENCODED'])
    y_train = df_for_split[df_for_split['TERM_CODE'].isin(train_terms)]['HIGH_ENROLLMENT_ENCODED']
    X_val = df_for_split[df_for_split['TERM_CODE'].isin(val_terms)].drop(columns=['TERM_CODE', 'HIGH_ENROLLMENT_ENCODED'])
    y_val = df_for_split[df_for_split['TERM_CODE'].isin(val_terms)]['HIGH_ENROLLMENT_ENCODED']

    logging.info(f"Training on {len(train_terms)} terms (up to {train_terms[-1] if train_terms.size > 0 else 'N/A'}), validating on {len(val_terms)} terms (from {val_terms[0] if val_terms.size > 0 else 'N/A'}).")
    logging.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Fallback to random split if time-based split results in empty sets (e.g., very few terms)
    if X_train.empty or X_val.empty:
        logging.warning("Time-based split resulted in empty training or validation sets. Falling back to random train_test_split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logging.info(f"Random split: Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
    if X_train.empty or X_val.empty:
        logging.error("Training or validation set is still empty after splitting. Cannot train or evaluate model.")
        raise ValueError("Training or validation sets are empty. Check data size and split logic.")

    # Initialize and train a RandomForestClassifier
    # Using 'balanced' class_weight to handle potential target imbalance
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    # Predict and evaluate on the validation set
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    logging.info(f"Validation Macro F1 Score: {f1:.4f}")

    return model, f1

# --- Main Execution Flow ---
if __name__ == "__main__":
    try:
        # 1. Load Data
        logging.info("Step 1: Starting data loading...")
        loaded_dfs = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_PATH)

        # 2. Preprocess Data
        logging.info("Step 2: Starting data preprocessing...")
        X, y, metadata_df = preprocess_data(loaded_dfs)

        # 3. Train and Validate Model
        logging.info("Step 3: Starting model training and validation...")
        model, final_validation_score = train_and_validate(X, y, metadata_df)

        # 4. Print final performance as required by the task
        print(f'Final Validation Performance: {final_validation_score}')

        logging.info("Script finished successfully.")

    except FileNotFoundError as e:
        logging.critical(f"Critical error: Required file not found: {e}. Please ensure input directory and files are correctly placed.", exc_info=True)
        print(f"Error: A critical data file was not found. Please check the 'input' directory structure. Details: {e}")
    except ValueError as e:
        logging.critical(f"Critical error during data processing or model setup: {e}", exc_info=True)
        print(f"Error during data processing or model setup: {e}. Check logs for more details.")
    except Exception as e:
        logging.critical(f"An unexpected and critical error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}. Check logs for more details.")

