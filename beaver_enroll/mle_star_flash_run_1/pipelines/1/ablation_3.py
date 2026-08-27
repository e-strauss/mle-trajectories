
import pandas as pd
import numpy as np
import os
import logging
import subprocess
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(TRAIN_DATA_DIR, 'gold_enrollment_train.csv')

TERM_CODE = 'TERM_CODE'
SUBJECT_ID_SORT = 'SUBJECT_ID_SORT'
HIGH_ENROLLMENT = 'HIGH_ENROLLMENT' # Target column, 'Y' or 'N'

# --- Data Loading and Preprocessing (adapted from original) ---
def load_data_ablation(data_dir, gold_file_path, include_department_size_feature=True):
    """
    Loads gold enrollment data, performs initial validation, and prepares base features.
    Adapted to allow skipping 'department_size'.
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

        for col in [TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT]:
            if col not in gold_df.columns:
                raise RuntimeError(f"Missing critical column in gold enrollment data: '{col}'")

        if gold_df[HIGH_ENROLLMENT].dtype == 'object':
            gold_df[HIGH_ENROLLMENT] = gold_df[HIGH_ENROLLMENT].map({'Y': 1, 'N': 0})
            if gold_df[HIGH_ENROLLMENT].isnull().any():
                logging.warning(f"Found non-Y/N values in '{HIGH_ENROLLMENT}' column, replacing NaNs with 0. Consider specific handling for unexpected values.")
                gold_df[HIGH_ENROLLMENT].fillna(0, inplace=True)
        elif not np.issubdtype(gold_df[HIGH_ENROLLMENT].dtype, np.number):
             raise RuntimeError(f"Unexpected data type for '{HIGH_ENROLLMENT}' column: {gold_df[HIGH_ENROLLMENT].dtype}. Expected 'Y'/'N' or numerical (0/1).")

        np.random.seed(42) # For reproducibility of dummy features
        if 'credit_hours' not in gold_df.columns:
            gold_df['credit_hours'] = np.random.choice([1, 3, 4], size=len(gold_df))
            logging.info("Added dummy 'credit_hours' feature.")
        if 'course_level' not in gold_df.columns:
            gold_df['course_level'] = np.random.choice([100, 200, 300, 400], size=len(gold_df))
            logging.info("Added dummy 'course_level' feature.")

        if include_department_size_feature:
            if 'department_size' not in gold_df.columns:
                gold_df['department_prefix'] = gold_df[SUBJECT_ID_SORT].apply(lambda x: ''.join(filter(str.isalpha, str(x))).upper())
                unique_departments = gold_df['department_prefix'].unique()
                dept_size_map = {dept: np.random.randint(10, 100) for dept in unique_departments}
                gold_df['department_size'] = gold_df['department_prefix'].map(dept_size_map)
                gold_df.drop(columns=['department_prefix'], inplace=True)
                logging.info("Added dummy 'department_size' feature based on subject ID prefixes.")
        else:
            logging.info("Skipping 'department_size' feature as per ablation study.")
            # If the feature already exists (e.g., from a pre-created dummy file), remove it.
            if 'department_size' in gold_df.columns:
                gold_df = gold_df.drop(columns=['department_size'])

        return gold_df

    except Exception as e:
        logging.error(f"Error during data loading or initial preprocessing: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load or preprocess data: {e}")

# --- Feature Engineering (adapted from original) ---
def create_features_ablation(df, include_engineered_features=True):
    """
    Creates additional features for the model from the preprocessed DataFrame.
    Adapted to allow skipping engineered features.
    """
    logging.info("Starting feature engineering...")

    if include_engineered_features:
        if 'course_level' in df.columns and 'credit_hours' in df.columns:
            df['level_credit_interaction'] = df['course_level'] * df['credit_hours']
            logging.info("Created 'level_credit_interaction' feature.")
        else:
            logging.warning("Cannot create 'level_credit_interaction': missing 'course_level' or 'credit_hours'.")

        df = df.sort_values(by=TERM_CODE).reset_index(drop=True) # Ensure sorted for prev_term_high_enrollment_rate
        if HIGH_ENROLLMENT in df.columns and SUBJECT_ID_SORT in df.columns:
            df['prev_term_high_enrollment_rate'] = df.groupby(SUBJECT_ID_SORT)[HIGH_ENROLLMENT].transform(
                lambda x: x.shift(1).expanding().mean().fillna(0)
            )
            logging.info("Created 'prev_term_high_enrollment_rate' feature.")
        else:
            logging.warning(f"Cannot create 'prev_term_high_enrollment_rate': missing '{HIGH_ENROLLMENT}' or '{SUBJECT_ID_SORT}'.")
    else:
        logging.info("Skipping engineered features as per ablation study.")

    # Define the final list of numerical features to be used by the model
    features_list = [col for col in ['credit_hours', 'course_level', 'department_size'] if col in df.columns]

    if include_engineered_features:
        if 'level_credit_interaction' in df.columns:
            features_list.append('level_credit_interaction')
        if 'prev_term_high_enrollment_rate' in df.columns:
            features_list.append('prev_term_high_enrollment_rate')

    if not features_list:
        raise RuntimeError("No numerical features available after feature engineering. Please check feature creation logic.")

    logging.info(f"Final features selected for training: {features_list}")
    return df, features_list

# --- Model Training and Validation (adapted from original) ---
def train_and_validate_model_ablation(df, features, target_col, rf_n_estimators=100):
    """
    Splits data into training and validation sets (time-based),
    trains a RandomForestClassifier, and evaluates its performance using macro F1.
    Adapted to allow changing rf_n_estimators.
    """
    logging.info(f"Splitting data for time-based training and validation with n_estimators={rf_n_estimators}...")

    df = df.sort_values(by=TERM_CODE).reset_index(drop=True)

    unique_terms = sorted(df[TERM_CODE].unique())
    if len(unique_terms) < 2:
        raise RuntimeError(f"Not enough unique terms ({len(unique_terms)}) for a time-based validation split. Need at least two terms.")

    split_idx = max(1, int(len(unique_terms) * 0.8)) # Use the first 80% of terms for training
    train_terms = unique_terms[:split_idx]
    validation_terms = unique_terms[split_idx:]
    
    # Fallback to ensure validation_terms is not empty, by moving a term from train if necessary.
    if not validation_terms:
        logging.warning("Validation set became empty; attempting to use the last training term for validation.")
        if len(train_terms) > 0:
            validation_terms = [train_terms.pop()]
        else:
            raise RuntimeError("Cannot establish valid train/validation split: too few unique terms.")
        if not train_terms:
            raise RuntimeError("Cannot establish valid train/validation split: training set became empty after moving term to validation.")

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

    train_cols = X_train.columns.tolist()
    val_cols = X_val.columns.tolist()

    if set(train_cols) != set(val_cols):
        logging.warning("Feature columns mismatch between train and validation sets. Attempting to align.")
        missing_in_val = list(set(train_cols) - set(val_cols))
        for col in missing_in_val:
            X_val[col] = 0
            logging.info(f"Added missing feature '{col}' to validation set with default 0.")
        extra_in_val = list(set(val_cols) - set(train_cols))
        if extra_in_val:
            X_val = X_val.drop(columns=extra_in_val, errors='ignore')
            logging.info(f"Dropped extra features from validation set: {extra_in_val}.")
        X_val = X_val[train_cols]
        logging.info("Feature columns aligned between train and validation sets.")

    if X_train.empty or y_train.empty:
        raise RuntimeError("Training features or labels are empty after final feature alignment.")
    if X_val.empty or y_val.empty:
        raise RuntimeError("Validation features or labels are empty after final feature alignment.")

    logging.info("Training RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    logging.info("Model training complete.")

    logging.info("Evaluating model on validation set...")
    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    logging.info(f"Validation Macro F1 Score: {macro_f1}")

    return macro_f1

# --- Ablation Study Orchestration ---
def run_ablation_experiment(experiment_name, include_engineered_features, rf_n_estimators, include_department_size):
    """
    Runs a single experiment with specified ablation settings.
    """
    logging.info(f"\n--- Running Experiment: {experiment_name} ---")
    try:
        data_df = load_data_ablation(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_PATH, include_department_size_feature=include_department_size)
        data_df, features = create_features_ablation(data_df, include_engineered_features=include_engineered_features)
        
        # Ensure 'department_size' is not in features if it was explicitly excluded
        if not include_department_size and 'department_size' in features:
            features.remove('department_size')
            logging.info(f"Explicitly removed 'department_size' from features list for {experiment_name}.")

        if not features:
            raise RuntimeError("No features were identified or created for model training. Check 'create_features' function.")
        for feature in features:
            if feature not in data_df.columns:
                raise RuntimeError(f"Required feature '{feature}' is missing from the DataFrame after creation for {experiment_name}. Data pipeline error.")

        score = train_and_validate_model_ablation(data_df, features, HIGH_ENROLLMENT, rf_n_estimators=rf_n_estimators)
        return score
    except Exception as e:
        logging.error(f"Error during experiment '{experiment_name}': {e}")
        return 0.0 # Return 0.0 for failed experiments for comparison

if __name__ == "__main__":
    # Ensure the TRAIN_DATA_DIR exists.
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

    # --- Self-contained test setup: Create a dummy gold_enrollment_train.csv if it doesn't exist ---
    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_PATH):
        logging.warning(f"'{GOLD_ENROLLMENT_TRAIN_PATH}' not found. Creating a dummy file for demonstration purposes.")
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
        np.random.seed(42)
        dummy_df['credit_hours'] = np.random.choice([1, 3, 4], size=len(dummy_df))
        dummy_df['course_level'] = np.random.choice([100, 200, 300, 400], size=len(dummy_df))
        dummy_df.to_csv(GOLD_ENROLLMENT_TRAIN_PATH, index=False)
        logging.info(f"Dummy '{GOLD_ENROLLMENT_TRAIN_PATH}' created for execution demonstration.")
    # --- End of self-contained test setup ---

    results = {}

    # 1. Baseline Experiment (Original Configuration)
    print("\n--- Running Baseline Experiment ---")
    baseline_score = run_ablation_experiment(
        "Baseline",
        include_engineered_features=True,
        rf_n_estimators=100,
        include_department_size=True
    )
    results["Baseline"] = baseline_score
    print(f"Baseline Performance (Macro F1): {baseline_score:.4f}")

    # 2. Ablation: Remove Engineered Features (level_credit_interaction, prev_term_high_enrollment_rate)
    print("\n--- Running Ablation: No Engineered Features ---")
    no_engineered_features_score = run_ablation_experiment(
        "No Engineered Features",
        include_engineered_features=False,
        rf_n_estimators=100,
        include_department_size=True
    )
    results["No Engineered Features"] = no_engineered_features_score
    print(f"Ablation 'No Engineered Features' Performance (Macro F1): {no_engineered_features_score:.4f}")

    # 3. Ablation: Reduced n_estimators in RandomForestClassifier (e.g., 10 instead of 100)
    print("\n--- Running Ablation: Reduced n_estimators (10) ---")
    reduced_estimators_score = run_ablation_experiment(
        "Reduced n_estimators (10)",
        include_engineered_features=True,
        rf_n_estimators=10, # Reduced
        include_department_size=True
    )
    results["Reduced n_estimators (10)"] = reduced_estimators_score
    print(f"Ablation 'Reduced n_estimators (10)' Performance (Macro F1): {reduced_estimators_score:.4f}")

    # 4. Ablation: Remove 'department_size' base feature
    print("\n--- Running Ablation: No 'department_size' Feature ---")
    no_department_size_score = run_ablation_experiment(
        "No 'department_size' Feature",
        include_engineered_features=True,
        rf_n_estimators=100,
        include_department_size=False # Removed
    )
    results["No 'department_size' Feature"] = no_department_size_score
    print(f"Ablation 'No 'department_size' Feature' Performance (Macro F1): {no_department_size_score:.4f}")

    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    # Determine the most contributing part (the one whose removal/modification causes the biggest drop)
    if baseline_score > 0:
        impacts = {
            "Engineered Features": baseline_score - no_engineered_features_score,
            "RF n_estimators (from 100 to 10)": baseline_score - reduced_estimators_score,
            "Base Feature 'department_size'": baseline_score - no_department_size_score
        }

        most_impactful_change = max(impacts, key=impacts.get)
        highest_impact_value = impacts[most_impactful_change]

        if highest_impact_value > 0:
            print(f"\nConclusion: The part that contributes the most to the overall performance (largest drop when removed/modified) is: '{most_impactful_change}' with a performance drop of {highest_impact_value:.4f}.")
        else:
            print("\nConclusion: No single ablation significantly degraded performance, or some even improved it (indicating potential issues with the baseline or features).")
    else:
        print("\nConclusion: Baseline performance is 0.0, so it's hard to determine positive contributions. Investigate the baseline first.")

