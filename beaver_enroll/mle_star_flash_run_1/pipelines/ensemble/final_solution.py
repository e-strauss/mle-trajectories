

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    import subprocess
    import sys
    print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# Define paths
# Assuming the script is run from the root directory where 'input' is present.
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

# Test data paths (now available for submission phase)
TEST_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "test")
GOLD_ENROLLMENT_TEST_PATH = os.path.join(INPUT_DIR, "gold_enrollment_test.csv") # This file provides the keys for prediction

print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")
print(f"TEST_DATA_DIR: {TEST_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TEST_PATH: {GOLD_ENROLLMENT_TEST_PATH}")


# --- Helper function to load a table if it exists ---
def load_table_if_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        print(f"Loading {filename} from {directory}")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        print(f"Warning: {filename} not found at {filepath}. Skipping.")
        return pd.DataFrame() # Return empty DataFrame if file not found


# --- TRAINING DATA PREPARATION ---
# --- 1. Load Gold Labels (Training) ---
try:
    gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train)} rows.")
    if gold_enrollment_train.empty:
        raise ValueError("gold_enrollment_train.csv is empty.")
    if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
        raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N']
    })
    print("Using dummy gold_enrollment_train data.")

# --- 2. Load Features from TRAIN_DATA_DIR ---
terms_train_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_train_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

# Create a base dataframe for merging features, starting with gold labels
data = gold_enrollment_train.copy()

# Add features from offerings_train_df if available and has required columns
if not offerings_train_df.empty:
    if all(col in offerings_train_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
        offerings_train_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_train_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_train_df['CAPACITY'] = pd.to_numeric(offerings_train_df['CAPACITY'], errors='coerce')

        agg_features_train = offerings_train_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            max_capacity=('CAPACITY', 'max'),
            num_offerings=('TERM_CODE', 'count'),
            sum_capacity=('CAPACITY', 'sum')
        ).reset_index()
        data = pd.merge(data, agg_features_train, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings training data. Data shape: {data.shape}")
    else:
        print("Warning: offerings_train_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_train_df is empty. Proceeding with limited features for training.")

# Add features from terms_train_df if available and has required columns
if not terms_train_df.empty:
    if 'TERM_CODE' in terms_train_df.columns and 'YEAR' in terms_train_df.columns:
        data = pd.merge(data, terms_train_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms training data. Data shape: {data.shape}")
    else:
        print("Warning: terms_train_df missing expected columns (YEAR). Skipping merge.")

# --- 3. Feature Engineering (Training) ---
data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

le_subject = LabelEncoder()
data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

# Define features and target (for training and test)
features = ['TERM_YEAR', 'TERM_SEMESTER', 'SUBJECT_ID_SORT_encoded']

if 'avg_enrollment' in data.columns:
    features.append('avg_enrollment')
if 'max_capacity' in data.columns:
    features.append('max_capacity')
if 'num_offerings' in data.columns:
    features.append('num_offerings')
if 'sum_capacity' in data.columns:
    features.append('sum_capacity')
if 'YEAR' in data.columns:
    features.append('YEAR')

target = 'HIGH_ENROLLMENT_TARGET'

# Drop rows with NaN in features or target for training
initial_rows = data.shape[0]
data.dropna(subset=features + [target], inplace=True)
if data.shape[0] < initial_rows:
    print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target for training.")

# Check if there's enough data after dropping NaNs
if data.empty:
    print("Error: No training data remaining after feature engineering and NaN removal. Cannot train model or generate submission.")
    # If training data is empty, create an empty submission file
    submission_df = pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'])
    os.makedirs("./final", exist_ok=True)
    submission_df.to_csv("./final/submission.csv", index=False)
    print("Created an empty submission file due to no training data.")
    final_validation_score = 0.0 # Default score if no training
else:
    X_full = data[features]
    y_full = data[target]

    # Store the mean of training features for imputation in test data
    feature_means = {col: X_full[col].mean() for col in features if X_full[col].dtype in ['int64', 'float64', 'float32']}
    
    # Fill NaNs in X_full itself (if any remain from previous merges that were not part of dropna subset)
    for col, mean_val in feature_means.items():
        if col in X_full.columns:
            X_full[col].fillna(mean_val, inplace=True)
    X_full.fillna(0, inplace=True) # Fallback for any other NaNs

    print(f"Features used: {features}")
    print(f"Shape of X_full (for training): {X_full.shape}, Shape of y_full: {y_full.shape}")

    # --- 5. Model Training on full training data (for final submission) ---
    print("Training RandomForestClassifier on full training data...")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model_rf.fit(X_full, y_full)

    print("Training GradientBoostingClassifier on full training data...")
    model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model_gb.fit(X_full, y_full)

    # --- Test Data Preparation and Prediction ---
    print("\n--- Preparing Test Data for Prediction ---")

    # 1. Load Test Prediction Keys
    test_prediction_keys = load_table_if_exists(INPUT_DIR, 'gold_enrollment_test.csv')
    if test_prediction_keys.empty:
        print("Error: gold_enrollment_test.csv not found or empty. Cannot create submission.")
        submission_df = pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'])
    else:
        print(f"Loaded gold_enrollment_test.csv with {len(test_prediction_keys)} rows.")
        if not all(col in test_prediction_keys.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT']):
            print("Warning: gold_enrollment_test.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT. Submission might be incomplete.")
        
        # 2. Load Features from TEST_DATA_DIR
        terms_test_df = load_table_if_exists(TEST_DATA_DIR, 'terms.csv')
        offerings_test_df = load_table_if_exists(TEST_DATA_DIR, 'offerings.csv')

        # Create a base dataframe for merging features, starting with test keys
        test_data = test_prediction_keys.copy()

        # Add features from offerings_test_df
        if not offerings_test_df.empty:
            if all(col in offerings_test_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
                offerings_test_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_test_df['ACTUAL_ENROLLMENT'], errors='coerce')
                offerings_test_df['CAPACITY'] = pd.to_numeric(offerings_test_df['CAPACITY'], errors='coerce')

                agg_features_test = offerings_test_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                    avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
                    max_capacity=('CAPACITY', 'max'),
                    num_offerings=('TERM_CODE', 'count'),
                    sum_capacity=('CAPACITY', 'sum')
                ).reset_index()
                test_data = pd.merge(test_data, agg_features_test, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
                print(f"Merged with aggregated offerings test data. Test data shape: {test_data.shape}")
            else:
                print("Warning: offerings_test_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
        else:
            print("Warning: offerings_test_df is empty. Proceeding with limited features for test.")

        # Add features from terms_test_df
        if not terms_test_df.empty:
            if 'TERM_CODE' in terms_test_df.columns and 'YEAR' in terms_test_df.columns:
                test_data = pd.merge(test_data, terms_test_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
                print(f"Merged with terms test data. Test data shape: {test_data.shape}")
            else:
                print("Warning: terms_test_df missing expected columns (YEAR). Skipping merge.")

        # 3. Feature Engineering (Test)
        test_data['TERM_CODE_str'] = test_data['TERM_CODE'].astype(str)
        test_data['TERM_YEAR'] = pd.to_numeric(test_data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        test_data['TERM_SEMESTER'] = pd.to_numeric(test_data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

        # Handle SUBJECT_ID_SORT_encoded for test data using the fitted LabelEncoder
        subject_to_encoded = {cls: i for i, cls in enumerate(le_subject.classes_)}
        default_encoded_value = len(le_subject.classes_) # New category ID for unseen subjects

        test_data['SUBJECT_ID_SORT_encoded'] = test_data['SUBJECT_ID_SORT'].map(
            lambda x: subject_to_encoded.get(x, default_encoded_value)
        )

        # Prepare X_test
        X_test = test_data[features]

        # Fill NaNs in X_test using the imputation strategy (e.g., means from training data or 0)
        # "You should not drop any test samples" -> so fill NaNs, don't drop.
        for col in X_test.columns:
            if X_test[col].isnull().any():
                if col in feature_means:
                    X_test[col].fillna(feature_means[col], inplace=True)
                else:
                    if X_test[col].dtype in ['int64', 'float64', 'float32']:
                        X_test[col].fillna(0, inplace=True) # Simple imputation for numeric features not in feature_means
                    # For categorical/other types, they should ideally be handled earlier or not included as numeric features
        X_test.fillna(0, inplace=True) # Final catch-all for any remaining NaNs

        print(f"Shape of X_test: {X_test.shape}")

        # 4. Predict on Test Data
        print("Generating predictions on test data...")
        rf_test_probabilities = model_rf.predict_proba(X_test)[:, 1]
        gb_test_probabilities = model_gb.predict_proba(X_test)[:, 1]

        final_blended_probabilities_test = (rf_test_probabilities + gb_test_probabilities) / 2
        test_predictions_binary = (final_blended_probabilities_test > 0.5).astype(int)
        
        # Convert binary predictions to 'Y'/'N'
        test_predictions_labels = np.where(test_predictions_binary == 1, 'Y', 'N')

        # 5. Create Submission File
        submission_df = pd.DataFrame({
            'TERM_CODE': test_prediction_keys['TERM_CODE'],
            'SUBJECT_ID_SORT': test_prediction_keys['SUBJECT_ID_SORT'],
            'HIGH_ENROLLMENT': test_predictions_labels
        })

    # Ensure the final directory exists
    os.makedirs("./final", exist_ok=True)
    submission_path = "./final/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at {submission_path}")

    # --- Validation Phase (for performance tracking, using time-based split of original training data) ---
    print("\n--- Validation Phase (for performance tracking) ---")
    if not data.empty:
        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_train_year = sorted_years[-1]

            train_df_val = data[data['TERM_YEAR'] < latest_train_year]
            val_df_val = data[data['TERM_YEAR'] == latest_train_year]

            if val_df_val.empty and len(sorted_years) > 1:
                second_latest_train_year = sorted_years[-2]
                print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
                train_df_val = data[data['TERM_YEAR'] < second_latest_train_year]
                val_df_val = data[data['TERM_YEAR'] == second_latest_train_year]
            elif val_df_val.empty:
                 print("Warning: Only one or two years of data available, and time-based split created empty validation. Using random split.")
                 train_df_val, val_df_val = train_test_split(data, test_size=0.2, random_state=42, stratify=y_full)
            else:
                print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
        else:
            print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
            train_df_val, val_df_val = train_test_split(data, test_size=0.2, random_state=42, stratify=y_full)

        X_train_val, y_train_val = train_df_val[features], train_df_val[target]
        X_val_val, y_val_val = val_df_val[features], val_df_val[target]

        # Fill NaNs in validation splits using means from the main training data
        for col, mean_val in feature_means.items():
            if col in X_train_val.columns:
                X_train_val[col].fillna(mean_val, inplace=True)
            if col in X_val_val.columns:
                X_val_val[col].fillna(mean_val, inplace=True)
        X_train_val.fillna(0, inplace=True) # Catch-all for any remaining
        X_val_val.fillna(0, inplace=True)


        print(f"Train set shape for validation: {X_train_val.shape}, Val set shape: {X_val_val.shape}")

        if X_train_val.empty or X_val_val.empty or len(np.unique(y_train_val)) < 2 or len(np.unique(y_val_val)) < 2:
            print("Error: Training or validation set for validation score is empty, or target has only one class. Cannot proceed with model training for validation score.")
            final_validation_score = 0.0
        else:
            print("Training models for validation score calculation...")
            model_rf_val = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model_rf_val.fit(X_train_val, y_train_val)
            rf_val_probabilities = model_rf_val.predict_proba(X_val_val)[:, 1]

            model_gb_val = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            model_gb_val.fit(X_train_val, y_train_val) # Trained on validation-split train data
            gb_val_probabilities = model_gb_val.predict_proba(X_val_val)[:, 1]

            final_blended_probabilities_val = (rf_val_probabilities + gb_val_probabilities) / 2
            blended_val_predictions = (final_blended_probabilities_val > 0.5).astype(int)
            final_validation_score = f1_score(y_val_val, blended_val_predictions, average='macro')
    else:
        final_validation_score = 0.0 # If training data itself was empty

    print(f"Final Validation Performance: {final_validation_score}")

