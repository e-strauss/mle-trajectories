
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from pathlib import Path
import logging

# Configure logging to capture output even if stdout/stderr are redirected
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_load_csv(filepath, **kwargs):
    """Safely loads a CSV, printing an error if it fails."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        logging.info(f"Successfully loaded {filepath} with {len(df)} rows.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None

def main():
    try:
        logging.info("Starting script execution.")

        # --- Configuration ---
        INPUT_DIR = Path("./input")
        TRAIN_DATA_DIR = INPUT_DIR / "table_splits" / "train"
        TEST_DATA_DIR = INPUT_DIR / "test" # As per the prompt's instruction to replace __TEST_DATA_DIR__ with 'test/'
        GOLD_TRAIN_FILE = INPUT_DIR / "eval" / "gold_enrollment_train.csv"

        logging.info(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
        logging.info(f"TEST_DATA_DIR: {TEST_DATA_DIR}")
        logging.info(f"GOLD_TRAIN_FILE: {GOLD_TRAIN_FILE}")

        # Ensure directories exist
        if not TRAIN_DATA_DIR.exists():
            logging.error(f"Error: Training data directory not found: {TRAIN_DATA_DIR}")
            return
        if not TEST_DATA_DIR.exists():
            logging.error(f"Error: Test data directory not found: {TEST_DATA_DIR}")
            return
        if not GOLD_TRAIN_FILE.exists():
            logging.error(f"Error: Gold training file not found: {GOLD_TRAIN_FILE}")
            return

        # --- Load Data ---
        # Load gold labels for training
        gold_enrollment_train_df = safe_load_csv(GOLD_TRAIN_FILE)
        if gold_enrollment_train_df is None: return

        # Load subject summary data for train and test
        def load_term_data(data_dir):
            all_files = list(data_dir.glob("*.csv"))
            if not all_files:
                logging.error(f"No CSV files found in {data_dir}")
                return None
            
            df_list = []
            for f in all_files:
                logging.info(f"Loading {f.name}...")
                term_df = safe_load_csv(f)
                if term_df is None: continue
                # Extract TERM_CODE from filename (e.g., term_2208_subject_summary.csv -> 2208)
                term_code = int(f.stem.split('_')[1])
                term_df['TERM_CODE'] = term_code
                df_list.append(term_df)
            
            if not df_list:
                logging.error(f"Failed to load any data from {data_dir}")
                return None
            
            combined_df = pd.concat(df_list, ignore_index=True)
            logging.info(f"Combined data from {data_dir} has {len(combined_df)} rows.")
            return combined_df

        train_summary_df = load_term_data(TRAIN_DATA_DIR)
        if train_summary_df is None: return
        
        test_summary_df = load_term_data(TEST_DATA_DIR)
        if test_summary_df is None: return

        # --- Feature Engineering and Label Alignment ---
        logging.info("Starting feature engineering and label alignment.")

        def prepare_data(summary_df, gold_df=None, is_training=True):
            df = summary_df.copy()
            
            # Basic Features (can be expanded)
            df['ENROLLMENT_CAP_RATIO'] = df['ENROLLMENT'] / df['CAPACITY']
            df['FILLING_RATE'] = df['ENROLLMENT'] / df['CAPACITY']
            df['ACTIVE_SECTIONS_RATIO'] = df['ACTIVE_SECTIONS'] / df['MAX_SECTIONS']
            df['AVG_CREDIT_HOURS'] = df['TOTAL_CREDIT_HOURS'] / df['ACTIVE_SECTIONS']
            
            # Handle potential division by zero
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True) # Or a more sophisticated imputation
            
            # Lagged features (requires sorting and grouping, example: previous term's enrollment)
            # This is a simplified example; real lagged features require careful handling of missing previous terms.
            df.sort_values(by=['SUBJECT_ID_SORT', 'TERM_CODE'], inplace=True)
            df['PREV_TERM_ENROLLMENT'] = df.groupby('SUBJECT_ID_SORT')['ENROLLMENT'].shift(1)
            df['PREV_TERM_ENROLLMENT'].fillna(0, inplace=True) # Fill NaN for first term offerings
            
            # Merge with gold labels if training
            if is_training and gold_df is not None:
                logging.info(f"Merging with gold labels. Gold shape: {gold_df.shape}")
                df = pd.merge(df, gold_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
                logging.info(f"After merging with gold labels, df shape: {df.shape}")
                if df.empty:
                    logging.error("Merged training data is empty. Check TERM_CODE and SUBJECT_ID_SORT alignment.")
                    return None
            
            return df

        train_data = prepare_data(train_summary_df, gold_enrollment_train_df, is_training=True)
        if train_data is None: return
        test_data = prepare_data(test_summary_df, is_training=False)
        if test_data is None: return

        logging.info(f"Prepared training data shape: {train_data.shape}")
        logging.info(f"Prepared test data shape: {test_data.shape}")

        # --- Define Features and Target ---
        # Features to use for the model. Select numerical features.
        # Exclude original 'ENROLLMENT' if it's considered leakage for the 'HIGH_ENROLLMENT' target.
        # However, the task is "predict whether it will have high enrollment (yes or no) relative to other courses"
        # and "top quartile among offerings with positive enrollment". This suggests ENROLLMENT itself
        # might be a feature *if* the relative calculation is done within the target definition.
        # For predicting "high enrollment" directly, `ENROLLMENT` itself would be leakage.
        # Assuming `HIGH_ENROLLMENT` is pre-calculated in the gold file based on the quartile logic.
        
        # Identify common features between train and test
        feature_cols = [col for col in train_data.columns if col not in [
            'TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT', 'ENROLLMENT', 'CAPACITY', # Exclude actual enrollment as it's the target's source
            'SUBJECT_ID', 'CAMPUS_ID', 'COURSE_ID', 'COURSE_NUMBER' # Identifiers
        ] and col in test_data.columns]
        
        # Ensure all feature columns are numeric, convert if necessary
        for col in feature_cols:
            if train_data[col].dtype == 'object':
                try:
                    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
                    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
                except Exception as e:
                    logging.warning(f"Could not convert column {col} to numeric: {e}. Removing from features.")
                    feature_cols.remove(col)
            
            train_data[col].fillna(0, inplace=True)
            test_data[col].fillna(0, inplace=True)

        logging.info(f"Selected {len(feature_cols)} features: {feature_cols}")

        X = train_data[feature_cols]
        y = train_data['HIGH_ENROLLMENT']

        # --- Time-based Validation Split ---
        # Get unique term codes from training data
        unique_terms = sorted(train_data['TERM_CODE'].unique())
        if len(unique_terms) < 2:
            logging.error("Not enough unique terms in training data for time-based split.")
            return

        # Use the latest term(s) for validation
        validation_terms = unique_terms[-1:] # Example: last term for validation
        train_terms = unique_terms[:-1] # All but the last for training

        X_train_val = train_data[train_data['TERM_CODE'].isin(train_terms)][feature_cols]
        y_train_val = train_data[train_data['TERM_CODE'].isin(train_terms)]['HIGH_ENROLLMENT']
        X_val = train_data[train_data['TERM_CODE'].isin(validation_terms)][feature_cols]
        y_val = train_data[train_data['TERM_CODE'].isin(validation_terms)]['HIGH_ENROLLMENT']
        
        logging.info(f"Training data terms: {train_terms}")
        logging.info(f"Validation data terms: {validation_terms}")
        logging.info(f"X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape}")
        logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        if X_train_val.empty or X_val.empty:
            logging.error("Time-based split resulted in empty training or validation set. Adjust split logic.")
            return
        
        # --- Model Training ---
        logging.info("Training the model.")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train_val, y_train_val)
        logging.info("Model training complete.")

        # --- Validation Performance ---
        y_pred_val = model.predict(X_val)
        validation_f1_macro = f1_score(y_val, y_pred_val, average='macro')
        print(f'Final Validation Performance: {validation_f1_macro}')
        logging.info(f'Final Validation Performance: {validation_f1_macro}')

        # --- Prediction on Test Data ---
        logging.info("Generating predictions for the test set.")
        X_test = test_data[feature_cols]
        # Align columns of X_test with X_train_val
        missing_cols_in_test = set(X_train_val.columns) - set(X_test.columns)
        for c in missing_cols_in_test:
            X_test[c] = 0 # Fill missing columns with 0, or appropriate default
            logging.warning(f"Feature '{c}' missing in test data, added with default value 0.")
        X_test = X_test[X_train_val.columns] # Reorder columns to match training data

        test_predictions = model.predict(X_test)
        
        # --- Save Predictions ---
        output_df = pd.DataFrame({
            'TERM_CODE': test_data['TERM_CODE'],
            'SUBJECT_ID_SORT': test_data['SUBJECT_ID_SORT'],
            'HIGH_ENROLLMENT': test_predictions
        })
        
        output_path = "predictions.csv"
        output_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logging.exception("An unexpected error occurred during script execution.")
        # This will print the full traceback to the log, which should be captured.

if __name__ == "__main__":
    main()
