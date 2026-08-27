
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

import pandas as pd
import numpy as np

# Add features from offerings_df if available and has required columns
if not offerings_df.empty:
    required_cols_offerings = ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']
    if all(col in offerings_df.columns for col in required_cols_offerings):
        # Ensure 'ACTUAL_ENROLLMENT' and 'CAPACITY' are numeric
        offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

        # Calculate enrollment rate, handling division by zero and potential NaNs
        # Replace 0 capacity with NaN to avoid division by zero, then replace inf/-inf with NaN.
        # Finally, fill any remaining NaNs in enrollment_rate with 0.0.
        offerings_df['enrollment_rate'] = offerings_df['ACTUAL_ENROLLMENT'] / offerings_df['CAPACITY'].replace(0, np.nan)
        offerings_df['enrollment_rate'] = offerings_df['enrollment_rate'].replace([np.inf, -np.inf], np.nan)
        offerings_df['enrollment_rate'] = offerings_df['enrollment_rate'].fillna(0.0)

        # --- Improvement Plan Part 1: Historical Median and Standard Deviation of enrollment_rate for each SUBJECT_ID_SORT ---
        # Calculate these across all available historical data in offerings_df for a stable baseline.
        subject_history_features = offerings_df.groupby('SUBJECT_ID_SORT').agg(
            hist_median_enrollment_rate=('enrollment_rate', 'median'),
            hist_std_enrollment_rate=('enrollment_rate', 'std')
        ).reset_index()
        # Fill NaN std (e.g., if only one offering for a subject)
        subject_history_features['hist_std_enrollment_rate'] = subject_history_features['hist_std_enrollment_rate'].fillna(0)

        data = pd.merge(data, subject_history_features, on='SUBJECT_ID_SORT', how='left')
        print(f"Merged with historical subject enrollment rate features. Data shape: {data.shape}")

        # --- Improvement Plan Part 2 & 3: Term-Subject level aggregations ---

        # Part 3 setup: Discretize enrollment_rate into categorical bins at the offering level
        bins = [-np.inf, 0.7, 1.0, np.inf]
        labels = ['underfilled', 'optimal', 'overfilled']
        offerings_df['enrollment_bin'] = pd.cut(offerings_df['enrollment_rate'], bins=bins, labels=labels, right=True)

        # Basic aggregations (similar to original code, but adding mean_enrollment_rate)
        # and new categorical bin proportions
        agg_features_list = {
            'avg_enrollment': ('ACTUAL_ENROLLMENT', 'mean'),
            'max_capacity': ('CAPACITY', 'max'),
            'num_offerings': ('TERM_CODE', 'count'),
            'sum_capacity': ('CAPACITY', 'sum'),
            'mean_enrollment_rate': ('enrollment_rate', 'mean') # Required for percentile rank and general info
        }
        agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(**agg_features_list).reset_index()

        # Part 3 continued: Calculate proportions of enrollment bins per (TERM_CODE, SUBJECT_ID_SORT)
        # This creates a DataFrame where columns are the bin labels ('prop_enroll_underfilled', etc.)
        # and values are the proportions for each group.
        bin_proportions = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT'])['enrollment_bin'] \
                                     .value_counts(normalize=True) \
                                     .unstack(fill_value=0) \
                                     .add_prefix('prop_enroll_') \
                                     .reset_index()
        agg_features = pd.merge(agg_features, bin_proportions, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')


        # Part 2: Within each TERM_CODE, compute the percentile rank of a subject's mean_enrollment_rate
        agg_features['term_enroll_rate_percentile'] = agg_features.groupby('TERM_CODE')['mean_enrollment_rate'].rank(pct=True)

        # Merge the aggregated features (including mean_enrollment_rate, percentile, and bin proportions)
        data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with aggregated offerings data (including new features). Data shape: {data.shape}")
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

# Dynamically add aggregated features if they exist after merging
if 'avg_enrollment' in data.columns:
    features.append('avg_enrollment')
if 'max_capacity' in data.columns:
    features.append('max_capacity')
if 'num_offerings' in data.columns:
    features.append('num_offerings')
if 'sum_capacity' in data.columns:
    features.append('sum_capacity')
if 'YEAR' in data.columns: # If 'YEAR' was merged from terms_df
    features.append('YEAR')


target = 'HIGH_ENROLLMENT_TARGET'

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
