
import pandas as pd
import numpy as np
import os
import subprocess
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# --- CatBoost handling ---
CatBoostClassifier_class = None  # Initialize to None for the class itself
catboost_available = False

try:
    import catboost
    from catboost import CatBoostClassifier as CatBoostClassifier_class
    catboost_available = True
    print("CatBoost found and imported successfully.")
except ImportError:
    print("CatBoost not found. Attempting to install CatBoost...")
    try:
        # First, ensure pip is available in the current environment's sys.executable.
        # The common error 'No module named pip' occurs if `sys.executable -m pip` fails.
        # `ensurepip` is the standard way to make sure pip is installed.
        print("Checking if pip is installed and functional...")
        # Use subprocess.DEVNULL to suppress direct output from ensurepip
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pip is confirmed installed or has been installed/upgraded via ensurepip.")

        # If we reach here, pip should be available. Now try installing CatBoost.
        print("Attempting to install CatBoost using pip...")
        # Use subprocess.DEVNULL to suppress direct output from pip install
        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import catboost
        from catboost import CatBoostClassifier as CatBoostClassifier_class  # Import the class after successful install
        catboost_available = True
        print("CatBoost installed successfully.")

    except Exception as e:
        print(f"Failed to install CatBoost or ensure pip: {e}")
        print("Please install CatBoost manually using 'pip install catboost' and rerun the script.")
        catboost_available = False  # Ensure flag is False on failure


# --- Configuration ---
# Define paths for input data.
INPUT_DIR = os.path.abspath("./input")
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")

# Ensure the directory for training data exists.
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

# --- Mock Data Generation ---
def generate_mock_data():
    """
    Generates a mock gold_enrollment_train.csv if it doesn't exist.
    This ensures the script is self-contained and runnable for demonstration purposes,
    mimicking the expected data structure.
    """
    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
        print(f"Generating mock data: {GOLD_ENROLLMENT_TRAIN_FILE}")
        num_rows = 1000
        # Use a range of terms to enable meaningful time-based validation splits.
        terms = ['201801', '201805', '201809', '201901', '201905', '201909',
                 '202001', '202005', '202009', '202101', '202105', '202109',
                 '202201', '202205', '202209', '202301', '202305', '202309']
        subjects = [f'SUBJ{i:03d}' for i in range(50)] # Simulate 50 unique subjects

        data = {
            'TERM_CODE': np.random.choice(terms, num_rows),
            'SUBJECT_ID_SORT': np.random.choice(subjects, num_rows),
            'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], num_rows, p=[0.3, 0.7]) # 30% high enrollment
        }
        mock_gold_df = pd.DataFrame(data)
        # Convert TERM_CODE to integer for correct chronological sorting.
        mock_gold_df['TERM_CODE'] = mock_gold_df['TERM_CODE'].astype(int)
        mock_gold_df = mock_gold_df.sort_values(by='TERM_CODE').reset_index(drop=True)
        mock_gold_df.to_csv(GOLD_ENROLLMENT_TRAIN_FILE, index=False)
        print("Mock gold_enrollment_train.csv created.")
    else:
        print(f"Mock data {GOLD_ENROLLMENT_TRAIN_FILE} already exists. Skipping generation.")


# --- Data Loading and Feature Engineering ---
def load_data(data_dir):
    """
    Loads the gold enrollment data and engineers relevant features.
    In a real scenario, more complex features would be derived from other tables.
    """
    print(f"Attempting to load gold enrollment data from {GOLD_ENROLLMENT_TRAIN_FILE}...")

    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
        print(f"Error: Gold enrollment file not found at {GOLD_ENROLLMENT_TRAIN_FILE}.")
        return pd.DataFrame() # Return empty DataFrame if file is missing.

    gold_df = pd.read_csv(GOLD_ENROLLMENT_TRAIN_FILE)

    if gold_df.empty:
        print("Loaded gold enrollment data is empty. Please check the CSV file.")
        return gold_df

    # Set random seed for reproducibility of synthetic features.
    np.random.seed(42)

    # Feature Engineering: Derive features from TERM_CODE.
    gold_df['TERM_CODE_STR'] = gold_df['TERM_CODE'].astype(str)
    gold_df['TERM_YEAR'] = gold_df['TERM_CODE_STR'].str[:4].astype(int)
    gold_df['TERM_MONTH'] = gold_df['TERM_CODE_STR'].str[4:].astype(int) # 01=Spring, 05=Summer, 09=Fall

    # Simulate additional numerical features, as other tables are not available for this task.
    gold_df['feature_avg_gpa_prereq'] = np.random.normal(3.0, 0.5, len(gold_df))
    gold_df['feature_course_capacity'] = np.random.randint(10, 200, len(gold_df))
    gold_df['feature_prev_enrollment_ratio'] = np.random.uniform(0.1, 1.5, len(gold_df))

    # Ensure 'TERM_CODE' remains integer for time-based splitting.
    gold_df['TERM_CODE'] = gold_df['TERM_CODE'].astype(int)

    return gold_df

def create_features_labels(df):
    """
    Extracts features (X) and target labels (y) from the processed DataFrame,
    and identifies categorical feature indices for CatBoost.
    """
    # Define the set of features to be used in the model.
    feature_cols = [
        'TERM_YEAR',
        'TERM_MONTH',
        'SUBJECT_ID_SORT',
        'feature_avg_gpa_prereq',
        'feature_course_capacity',
        'feature_prev_enrollment_ratio'
    ]

    # Verify all required feature columns exist.
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns in DataFrame: {missing_cols}")
        return pd.DataFrame(), pd.Series(), [] # Return empty if critical features are missing.

    X = df[feature_cols].copy()
    y = df['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0) # Convert target to binary (1/0).

    # Identify categorical features by name and then map them to their column indices in X.
    categorical_feature_names = ['SUBJECT_ID_SORT', 'TERM_MONTH']
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_feature_names if col in X.columns]

    return X, y, cat_features_indices

# --- Main Script Logic ---
if __name__ == "__main__":
    final_validation_score = 0.0 # Initialize validation score.

    generate_mock_data() # Ensure mock data is present if actual data is not.

    print("Starting training process...")

    # Load and process training data.
    train_df = load_data(TRAIN_DATA_DIR)

    if train_df.empty:
        print("Training data is empty after loading. Cannot proceed with model training.")
    elif not catboost_available: # Check the flag set during CatBoost handling
        print("CatBoost is not installed or failed to install. Cannot proceed with model training.")
    else:
        # Extract features, labels, and categorical feature indices.
        X, y, categorical_features_indices = create_features_labels(train_df)

        if X.empty or y.empty:
            print("Features or labels are empty after creation. Cannot proceed with model training.")
        else:
            # Time-based validation split: Use the latest terms for validation.
            # This follows the task's recommendation for a time-based validation slice.
            unique_terms_num = sorted(train_df['TERM_CODE'].unique())

            X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
            performed_random_split = False

            if len(unique_terms_num) > 1:
                # Use the latest 20% of unique terms for the validation set.
                split_point = int(len(unique_terms_num) * 0.8)
                train_terms_num = unique_terms_num[:split_point]
                val_terms_num = unique_terms_num[split_point:]

                train_indices = train_df['TERM_CODE'].isin(train_terms_num)
                val_indices = train_df['TERM_CODE'].isin(val_terms_num)

                X_train_full = X[train_indices]
                y_train_full = y[train_indices]
                X_val_full = X[val_indices]
                y_val_full = y[val_indices]

                # Fallback to a random split if the time-based split results in an empty set.
                if X_val_full.empty or y_val_full.empty or X_train_full.empty or y_train_full.empty:
                    print("Time-based validation split resulted in an empty train or validation set. Falling back to random split.")
                    performed_random_split = True
                else:
                    X_train, y_train = X_train_full, y_train_full
                    X_val, y_val = X_val_full, y_val_full
            else:
                print("Not enough unique terms for a time-based split (need more than 1). Falling back to random split.")
                performed_random_split = True

            if performed_random_split:
                # Perform a stratified random split as a fallback or if time-based split is not feasible.
                # Ensure enough samples per class for stratification.
                if len(y.unique()) > 1 and y.value_counts().min() > 1:
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                else:
                    print("Not enough classes or samples per class for stratified split. Using non-stratified random split.")
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
                print("Train or validation sets are empty after all splitting attempts. Cannot proceed with model training.")
            else:
                # Calculate class weights for handling potential class imbalance.
                class_counts = y_train.value_counts()
                class_weight_for_1 = class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] > 0 else 1
                class_weights = {0: 1, 1: class_weight_for_1}

                # Initialize and train the CatBoost Classifier.
                # Use the aliased CatBoostClassifier_class which is now guaranteed to be the CatBoostClassifier class
                model = CatBoostClassifier_class(iterations=100,
                                           random_seed=42,
                                           verbose=0, # Suppress verbose output for cleaner execution.
                                           loss_function='Logloss', # Standard loss for binary classification.
                                           eval_metric='F1', # Optimize for F1 score, as per evaluation metric.
                                           class_weights=class_weights,
                                           cat_features=categorical_features_indices,
                                           early_stopping_rounds=10)
                print("Training CatBoost model...")
                # Fit the model with an evaluation set for monitoring, using early stopping.
                model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)

                # Make predictions on the validation set.
                y_pred = model.predict(X_val)

                # Calculate the Macro F1 Score for evaluation.
                macro_f1 = f1_score(y_val, y_pred, average='macro')
                print(f"Validation Macro F1 Score: {macro_f1:.4f}")
                final_validation_score = macro_f1

    # Print the final validation performance in the required format.
    print(f"Final Validation Performance: {final_validation_score}")
