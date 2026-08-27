

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# MODIFICATION: Define TEST_DATA_DIR and SAMPLE_SUBMISSION_PATH for ensemble part
TEST_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "test")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")


print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")
print(f"TEST_DATA_DIR: {TEST_DATA_DIR}")
print(f"SAMPLE_SUBMISSION_PATH: {SAMPLE_SUBMISSION_PATH}")


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

# --- Helper function to safely transform with LabelEncoder ---
def safe_label_transform(encoder, series, fill_value=0):
    # Map unseen labels to a default fill_value (e.g., 0)
    # This avoids ValueError when `transform` encounters labels not seen during fit.
    known_classes_map = {cls: encoder.transform([cls])[0] for cls in encoder.classes_}
    transformed_series = series.map(known_classes_map).fillna(fill_value).astype(int)
    return transformed_series


# --- 1. Load Gold Labels (Train) ---
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
terms_df_train = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_df_train = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

# Create a base dataframe for merging features, starting with gold labels
data = gold_enrollment_train.copy()

# Add features from offerings_df_train if available and has required columns
if not offerings_df_train.empty:
    if all(col in offerings_df_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
        offerings_df_train['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df_train['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_df_train['CAPACITY'] = pd.to_numeric(offerings_df_train['CAPACITY'], errors='coerce')

        agg_features_train = offerings_df_train.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            max_capacity=('CAPACITY', 'max'),
            num_offerings=('TERM_CODE', 'count'),
            sum_capacity=('CAPACITY', 'sum')
        ).reset_index()
        data = pd.merge(data, agg_features_train, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated train offerings data. Data shape: {data.shape}")
    else:
        print("Warning: offerings_df_train missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_df_train is empty. Proceeding with limited features for training.")

# Add features from terms_df_train if available and has required columns
if not terms_df_train.empty:
    if 'TERM_CODE' in terms_df_train.columns and 'YEAR' in terms_df_train.columns:
        data = pd.merge(data, terms_df_train[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with train terms data. Data shape: {data.shape}")
    else:
        print("Warning: terms_df_train missing expected columns (YEAR). Skipping merge.")

# --- 3. Feature Engineering (Train Data) ---
data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

le_subject = LabelEncoder()
data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

# Define features and target (dynamic based on available merged features)
features = ['TERM_YEAR', 'TERM_SEMESTER', 'SUBJECT_ID_SORT_encoded']
if 'avg_enrollment' in data.columns: features.append('avg_enrollment')
if 'max_capacity' in data.columns: features.append('max_capacity')
if 'num_offerings' in data.columns: features.append('num_offerings')
if 'sum_capacity' in data.columns: features.append('sum_capacity')
if 'YEAR' in data.columns: features.append('YEAR')

target = 'HIGH_ENROLLMENT_TARGET'

# Drop rows with NaN in features or target for training data
initial_rows = data.shape[0]
data.dropna(subset=features + [target], inplace=True)
if data.shape[0] < initial_rows:
    print(f"Dropped {initial_rows - data.shape[0]} rows from training data due to NaN in features or target.")

# Check if there's enough data after dropping NaNs
final_validation_score = 0.0 # Default score if training is not possible
if data.empty:
    print("Error: No training data remaining after feature engineering and NaN removal. Cannot train model.")
    print(f"Final Validation Performance: {final_validation_score}")
else:
    # --- 4. Data Splitting (Time-based validation - for initial performance metric) ---
    X = data[features]
    y = data[target]

    print(f"Features used: {features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        train_df = data[data['TERM_YEAR'] < latest_train_year]
        val_df = data[data['TERM_YEAR'] == latest_train_year]

        if val_df.empty and len(sorted_years) > 1:
            second_latest_train_year = sorted_years[-2]
            print(f"Validation set from latest year ({latest_train_year}) was empty. Using second latest year ({second_latest_train_year}) for validation and prior years for training.")
            train_df = data[data['TERM_YEAR'] < second_latest_train_year]
            val_df = data[data['TERM_YEAR'] == second_latest_train_year]
        elif val_df.empty:
             print("Warning: Only one or two years of data available, and time-based split created empty validation. Using random split.")
             train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        else:
            print(f"Using latest year ({latest_train_year}) for validation. Training on years prior to {latest_train_year}.")
    else:
        print("Warning: 'TERM_YEAR' not available or only one year of data. Using random split for validation.")
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)


    X_train_split, y_train_split = train_df[features], train_df[target]
    X_val_split, y_val_split = val_df[features], val_df[target]

    print(f"Train set shape for validation: {X_train_split.shape}, Val set shape: {X_val_split.shape}")

    if X_train_split.empty or X_val_split.empty or len(np.unique(y_train_split)) < 2 or len(np.unique(y_val_split)) < 2:
        print("Error: Training or validation set is empty, or target has only one class for validation. Cannot proceed with model training.")
    else:
        # --- 5. Model Training (for validation metric) ---
        model_for_validation = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model_for_validation.fit(X_train_split, y_train_split)

        # --- 6. Evaluation (for initial performance metric) ---
        val_predictions = model_for_validation.predict(X_val_split)
        final_validation_score = f1_score(y_val_split, val_predictions, average='macro')

    # --- 7. Print Final Validation Performance (as required by prompt) ---
    print(f"Final Validation Performance: {final_validation_score}")


    # =========================================================================
    # --- ENSEMBLE PLAN IMPLEMENTATION STARTS HERE ---
    # =========================================================================

    print("\n--- Starting Ensemble Model Training and Test Prediction ---")

    # --- Prepare Full Training Data for Ensemble ---
    X_full = data[features]
    y_full = data[target]
    print(f"Full training data for ensemble (X_full): {X_full.shape}, (y_full): {y_full.shape}")

    # --- Load Test Data for Prediction ---
    try:
        sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        print(f"Loaded sample_submission.csv with {len(sample_submission)} rows.")
        test_data_for_pred = sample_submission[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()
    except FileNotFoundError:
        print(f"Error: {SAMPLE_SUBMISSION_PATH} not found. Cannot prepare test data for prediction.")
        test_data_for_pred = pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT']) # Create empty to avoid further errors

    if not test_data_for_pred.empty:
        terms_df_test = load_table_if_exists(TEST_DATA_DIR, 'terms.csv')
        offerings_df_test = load_table_if_exists(TEST_DATA_DIR, 'offerings.csv')

        # Add features from offerings_df_test
        if not offerings_df_test.empty:
            if all(col in offerings_df_test.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
                offerings_df_test['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df_test['ACTUAL_ENROLLMENT'], errors='coerce')
                offerings_df_test['CAPACITY'] = pd.to_numeric(offerings_df_test['CAPACITY'], errors='coerce')

                agg_features_test = offerings_df_test.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                    avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
                    max_capacity=('CAPACITY', 'max'),
                    num_offerings=('TERM_CODE', 'count'),
                    sum_capacity=('CAPACITY', 'sum')
                ).reset_index()
                test_data_for_pred = pd.merge(test_data_for_pred, agg_features_test, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
                print(f"Merged with aggregated test offerings data. Test data shape: {test_data_for_pred.shape}")
            else:
                print("Warning: offerings_df_test missing expected columns for aggregation. Skipping merge.")
        else:
            print("Warning: offerings_df_test is empty for prediction.")

        # Add features from terms_df_test
        if not terms_df_test.empty:
            if 'TERM_CODE' in terms_df_test.columns and 'YEAR' in terms_df_test.columns:
                test_data_for_pred = pd.merge(test_data_for_pred, terms_df_test[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
                print(f"Merged with test terms data. Test data shape: {test_data_for_pred.shape}")
            else:
                print("Warning: terms_df_test missing expected columns (YEAR). Skipping merge.")

        # --- Feature Engineering (Test Data) ---
        test_data_for_pred['TERM_CODE_str'] = test_data_for_pred['TERM_CODE'].astype(str)
        test_data_for_pred['TERM_YEAR'] = pd.to_numeric(test_data_for_pred['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
        test_data_for_pred['TERM_SEMESTER'] = pd.to_numeric(test_data_for_pred['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

        # Apply LabelEncoder fitted on training data to test data
        test_data_for_pred['SUBJECT_ID_SORT_encoded'] = safe_label_transform(le_subject, test_data_for_pred['SUBJECT_ID_SORT'])

        # Ensure all features from training are present in test data and fill NaNs
        X_test = pd.DataFrame(index=test_data_for_pred.index) # Create an empty df to populate
        for f in features:
            if f in test_data_for_pred.columns:
                X_test[f] = test_data_for_pred[f]
            else:
                X_test[f] = 0 # Fill missing features with a default value (e.g., 0)

        # Fill any remaining NaNs in numeric features for X_test
        X_test = X_test[features].fillna(0) # Fill with 0 for numeric features
        print(f"Prepared X_test for prediction. Shape: {X_test.shape}")

        # --- Self-Ensembling Loop ---
        N_ENSEMBLE = 10 # Number of models in the ensemble
        all_test_proba_preds = []

        if not X_full.empty and not X_test.empty and all(col in X_test.columns for col in features):
            print(f"\nStarting {N_ENSEMBLE} model ensemble for test prediction...")
            for i in range(N_ENSEMBLE):
                # print(f"Training ensemble model {i+1}/{N_ENSEMBLE}...")
                ensemble_model = RandomForestClassifier(n_estimators=100, random_state=i, class_weight='balanced', n_jobs=-1)
                ensemble_model.fit(X_full, y_full) # Train on full training data
                test_proba = ensemble_model.predict_proba(X_test)[:, 1] # Probability of the positive class ('Y')
                all_test_proba_preds.append(test_proba)

            # --- Aggregate predictions ---
            avg_test_proba = np.mean(all_test_proba_preds, axis=0)

            # --- Convert averaged probabilities to final labels ---
            final_test_labels_numeric = (avg_test_proba > 0.5).astype(int)
            final_test_labels_str = np.where(final_test_labels_numeric == 1, 'Y', 'N')

            # At this point, final_test_labels_str contains the predictions for submission.
            # The prompt requests not to modify the submission part due to formatting issues
            # and only print the validation performance. So, we stop here for the output.
            print("Ensemble predictions generated for test set internally.")
        else:
            print("Error: Either full training data or test data is empty, or features mismatch. Skipping ensemble prediction.")
    else:
        print("Error: Test data for prediction is empty. Skipping ensemble prediction.")

