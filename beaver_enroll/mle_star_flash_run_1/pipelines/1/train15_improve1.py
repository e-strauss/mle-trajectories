
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
TEST_DATA_DIR = None # Not available for training phase

print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
print(f"GOLD_ENROLLMENT_TRAIN_PATH: {GOLD_ENROLLMENT_TRAIN_PATH}")

# --- 1. Load Gold Labels ---
try:
    gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    print(f"Loaded gold_enrollment_train.csv with {len(gold_enrollment_train)} rows.")
    if gold_enrollment_train.empty:
        raise ValueError("gold_enrollment_train.csv is empty.")
    if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
        raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading gold_enrollment_train.csv: {e}. Creating dummy data for execution.")
    # Create a dummy dataframe for development purposes if file is missing or invalid.
    # In a real scenario, this would typically be a fatal error if data is critical.
    gold_enrollment_train = pd.DataFrame({
        'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101'],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N']
    })
    print("Using dummy gold_enrollment_train data.")

# --- 2. Load Features from TRAIN_DATA_DIR ---
# Helper function to load a table if it exists
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

# Load potential feature tables
# Assuming common academic data tables
terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')
# You could load more tables like 'courses.csv', 'subjects.csv' here and merge as needed.

# Create a base dataframe for merging features, starting with gold labels
data = gold_enrollment_train.copy()

# Add features from offerings_df if available and has required columns
if not offerings_df.empty:
    if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
        # Ensure 'ACTUAL_ENROLLMENT' and 'CAPACITY' are numeric
        offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

        # Aggregate offerings data per (TERM_CODE, SUBJECT_ID_SORT)
        agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            max_capacity=('CAPACITY', 'max'),
            num_offerings=('TERM_CODE', 'count'),
            sum_capacity=('CAPACITY', 'sum')
        ).reset_index()
        data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings data. Data shape: {data.shape}")
    else:
        print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
else:
    print("Warning: offerings_df is empty. Proceeding with limited features.")

# Add features from terms_df if available and has required columns
if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms data. Data shape: {data.shape}")
    else:
        print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

# --- 3. Feature Engineering ---
# Convert target to numeric
data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

# Extract features from TERM_CODE
data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

# Label Encode SUBJECT_ID_SORT
le_subject = LabelEncoder()
data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

# Define features and target
features = ['TERM_YEAR', 'TERM_SEMESTER', 'SUBJECT_ID_SORT_encoded']

# Define TARGET_COLUMN before its usage
TARGET_COLUMN = 'HIGH_ENROLLMENT_TARGET'

# Dynamically add aggregated features if they exist after merging, based on feature importances
candidate_agg_features_list = [
    'avg_enrollment',
    'max_capacity',
    'num_offerings',
    'sum_capacity',
    'YEAR' # If 'YEAR' was merged from terms_df
]

# Identify which of these candidate aggregated features are actually present in the data
available_agg_features = [col for col in candidate_agg_features_list if col in data.columns]

# Proceed with feature importance-based selection only if there are available aggregated features
if available_agg_features:
    # Construct the feature set for the temporary Random Forest model.
    # This set includes the 'core' features (already in 'features') and ALL available aggregated features.
    # Using a set ensures no duplicates if some aggregated features were somehow already in 'features'.
    temp_model_features_for_rf = list(set(features + available_agg_features))

    # Ensure the target column is not used as a feature for the temporary model.
    if TARGET_COLUMN in temp_model_features_for_rf:
        temp_model_features_for_rf.remove(TARGET_COLUMN)

    # Check if we have valid data and features to train the temporary model.
    # Also, ensure TARGET_COLUMN exists and has more than one unique value and is not entirely null.
    if (not data.empty and TARGET_COLUMN in data.columns and 
        not data[TARGET_COLUMN].isnull().all() and 
        data[TARGET_COLUMN].nunique() > 1 and 
        len(temp_model_features_for_rf) > 0):

        # Prepare data for the temporary model.
        # Create copies to avoid SettingWithCopyWarning and allow modifications for fitting.
        X_temp = data[temp_model_features_for_rf].copy()
        y_temp = data[TARGET_COLUMN].copy()

        # Create a temporary DataFrame for training to handle NaNs robustly without affecting original 'data'.
        temp_df_for_fit = pd.concat([X_temp, y_temp], axis=1).dropna()

        if not temp_df_for_fit.empty and temp_df_for_fit[TARGET_COLUMN].nunique() > 1:
            X_temp_fit = temp_df_for_fit[temp_model_features_for_rf]
            y_temp_fit = temp_df_for_fit[TARGET_COLUMN]

            # Train a lightweight Random Forest model.
            # Using RandomForestClassifier as F1 score optimization typically implies a classification task.
            # Parameters are chosen to make it lightweight and prevent overfitting for feature selection.
            # `class_weight='balanced'` is good practice for F1 score, which is sensitive to class imbalance.
            try:
                temp_rf_model = RandomForestClassifier(
                    n_estimators=75,      # A moderate number of trees
                    max_depth=8,          # Limited depth
                    random_state=42,
                    n_jobs=-1,            # Use all available CPU cores for faster training
                    class_weight='balanced'
                )
                temp_rf_model.fit(X_temp_fit, y_temp_fit)

                # Get feature importances
                importances = temp_rf_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': temp_model_features_for_rf,
                    'importance': importances
                })

                # Filter for importances of only the *aggregated* features
                agg_feature_importances = feature_importance_df[
                    feature_importance_df['feature'].isin(available_agg_features)
                ].sort_values(by='importance', ascending=False)

                selected_agg_features = []
                if not agg_feature_importances.empty:
                    # Determine a threshold for selecting significant aggregated features.
                    # This approach uses a dynamic threshold based on the maximum importance
                    # among the aggregated features, along with an absolute minimum.
                    max_agg_importance = agg_feature_importances['importance'].max()
                    
                    # Threshold: at least 15% of the max importance among aggregated features,
                    # and an absolute minimum importance to filter out very low noise features.
                    importance_threshold_relative = max_agg_importance * 0.15
                    min_absolute_importance_threshold = 0.0005 # A very small threshold

                    final_importance_threshold = max(importance_threshold_relative, min_absolute_importance_threshold)

                    selected_agg_features = agg_feature_importances[
                        agg_feature_importances['importance'] >= final_importance_threshold
                    ]['feature'].tolist()

                    # Fallback: If no features meet the dynamic threshold but available_agg_features is not empty,
                    # select the top N (e.g., top 2) most important aggregated features to ensure some are included.
                    if not selected_agg_features and len(available_agg_features) > 0:
                        num_to_select_fallback = min(2, len(available_agg_features))
                        selected_agg_features = agg_feature_importances['feature'].head(num_to_select_fallback).tolist()

                # Append the selected, important aggregated features to the 'features' list.
                # Ensure no duplicates are added.
                for agg_f in selected_agg_features:
                    if agg_f not in features:
                        features.append(agg_f)

            except ValueError:
                # Catch potential errors during model fitting (e.g., target has only one class after dropna)
                # If fitting fails, skip adding importance-based features via this method.
                pass
    # Else (if data/target/features not valid for training), no aggregated features are added.
# Else (if no available_agg_features), no aggregated features are added.


target = TARGET_COLUMN

# Drop rows with NaN in features or target
initial_rows = data.shape[0]
data.dropna(subset=features + [target], inplace=True)
if data.shape[0] < initial_rows:
    print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

# Check if there's enough data after dropping NaNs
if data.empty:
    print("Error: No data remaining after feature engineering and NaN removal. Cannot train model.")
    print("Final Validation Performance: 0.0")
else:
    X = data[features]
    y = data[target]

    print(f"Features used: {features}")
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    # --- 4. Data Splitting (Time-based validation) ---
    # Use the latest year in the training data for validation
    if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
        sorted_years = sorted(data['TERM_YEAR'].unique())
        latest_train_year = sorted_years[-1]

        train_df = data[data['TERM_YEAR'] < latest_train_year]
        val_df = data[data['TERM_YEAR'] == latest_train_year]

        if val_df.empty and len(sorted_years) > 1:
            # Fallback if the latest year created an empty validation set
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


    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    print(f"Train set shape: {X_train.shape}, Val set shape: {X_val.shape}")

    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("Error: Training or validation set is empty, or target has only one class. Cannot proceed with model training.")
        final_validation_score = 0.0 # Default score if training is not possible
    else:
        # --- 5. Model Training ---
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # --- 6. Evaluation ---
        val_predictions = model.predict(X_val)
        final_validation_score = f1_score(y_val, val_predictions, average='macro')

    # --- 7. Print Final Validation Performance ---
    print(f"Final Validation Performance: {final_validation_score}")
