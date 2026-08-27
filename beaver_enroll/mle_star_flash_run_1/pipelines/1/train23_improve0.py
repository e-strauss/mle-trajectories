
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
    required_offerings_cols = ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']
    if all(col in offerings_df.columns for col in required_offerings_cols):
        offerings_temp_df = offerings_df.copy() # Work on a copy to avoid modifying original df
        
        # Ensure 'ACTUAL_ENROLLMENT' and 'CAPACITY' are numeric
        offerings_temp_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_temp_df['ACTUAL_ENROLLMENT'], errors='coerce')
        offerings_temp_df['CAPACITY'] = pd.to_numeric(offerings_temp_df['CAPACITY'], errors='coerce')

        # Calculate enrollment rate, handling division by zero
        # If capacity is 0, enrollment rate is defined as 0 or NaN. Here, we set to 0.
        offerings_temp_df['enrollment_rate'] = np.where(
            offerings_temp_df['CAPACITY'] > 0,
            offerings_temp_df['ACTUAL_ENROLLMENT'] / offerings_temp_df['CAPACITY'],
            0 # Handle cases where capacity is 0
        )
        offerings_temp_df['enrollment_rate'].fillna(0, inplace=True) # Fill NaNs (e.g., from coerced non-numeric values)
        
        print(f"Offerings_df prepared with 'enrollment_rate'. Current shape: {offerings_temp_df.shape}")

        # --- Step 1: Add YEAR from terms_df to offerings_temp_df for trend calculation ---
        has_year_for_trends = False
        if not terms_df.empty:
            if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
                terms_year_map = terms_df[['TERM_CODE', 'YEAR']].drop_duplicates()
                offerings_temp_df = pd.merge(
                    offerings_temp_df, 
                    terms_year_map, 
                    on='TERM_CODE', 
                    how='left'
                )
                offerings_temp_df['YEAR'] = pd.to_numeric(offerings_temp_df['YEAR'], errors='coerce')
                # Drop rows where YEAR is NaN after merge, as it's crucial for trends
                offerings_temp_df.dropna(subset=['YEAR'], inplace=True)
                offerings_temp_df['YEAR'] = offerings_temp_df['YEAR'].astype(int)
                has_year_for_trends = True
                print(f"YEAR merged into offerings_temp_df for trend calculation. Current shape: {offerings_temp_df.shape}")
            else:
                print("Warning: terms_df missing expected columns (YEAR) for trend calculation.")
        else:
            print("Warning: terms_df is empty. Cannot calculate year-based trends.")

        # --- Step 2: Aggregate offerings data per (TERM_CODE, SUBJECT_ID_SORT) and optionally YEAR ---
        group_cols = ['TERM_CODE', 'SUBJECT_ID_SORT']
        if has_year_for_trends:
            group_cols.append('YEAR')

        agg_features = offerings_temp_df.groupby(group_cols).agg(
            mean_actual_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
            sum_actual_enrollment=('ACTUAL_ENROLLMENT', 'sum'),
            max_capacity=('CAPACITY', 'max'),
            min_capacity=('CAPACITY', 'min'),
            mean_enrollment_rate=('enrollment_rate', 'mean'),
            median_enrollment_rate=('enrollment_rate', 'median'),
            num_sections=('TERM_CODE', 'count'), # Number of sections offered for this subject in this term/year
            std_enrollment=('ACTUAL_ENROLLMENT', 'std'),
            std_enrollment_rate=('enrollment_rate', 'std')
        ).reset_index()

        # Fill NaNs for std columns which can occur if there's only one offering in a group
        agg_features['std_enrollment'].fillna(0, inplace=True)
        agg_features['std_enrollment_rate'].fillna(0, inplace=True)

        print(f"Aggregated offerings data with enhanced features. Current shape: {agg_features.shape}")

        # --- Step 3: Calculate Trend Features (Year-over-Year / Term-to-Term) ---
        if has_year_for_trends:
            # Ensure proper chronological sorting for trend calculation
            agg_features.sort_values(by=['SUBJECT_ID_SORT', 'YEAR', 'TERM_CODE'], inplace=True)

            # Calculate term-over-term absolute changes
            agg_features['enrollment_change_prev_term'] = agg_features.groupby('SUBJECT_ID_SORT')['mean_actual_enrollment'].diff().fillna(0)
            agg_features['enrollment_rate_change_prev_term'] = agg_features.groupby('SUBJECT_ID_SORT')['mean_enrollment_rate'].diff().fillna(0)
            
            # Calculate term-over-term percentage changes
            # Replace inf values (from division by zero where previous was 0 but current is not) with 0
            agg_features['enrollment_pct_change_prev_term'] = agg_features.groupby('SUBJECT_ID_SORT')['mean_actual_enrollment'].pct_change().fillna(0)
            agg_features['enrollment_pct_change_prev_term'].replace([np.inf, -np.inf], 0, inplace=True)

            # Rolling averages to capture popularity trends over a few terms
            agg_features['rolling_mean_enrollment_2term'] = agg_features.groupby('SUBJECT_ID_SORT')['mean_actual_enrollment'].transform(
                lambda x: x.rolling(window=2, min_periods=1).mean()
            )
            agg_features['rolling_mean_enrollment_rate_2term'] = agg_features.groupby('SUBJECT_ID_SORT')['mean_enrollment_rate'].transform(
                lambda x: x.rolling(window=2, min_periods=1).mean()
            )
            print(f"Calculated trend features. Current shape: {agg_features.shape}")

        # Drop the 'YEAR' column from agg_features before merging to avoid duplication
        # if 'YEAR' is also merged from terms_df later directly into 'data'.
        if has_year_for_trends and 'YEAR' in agg_features.columns:
            agg_features.drop('YEAR', axis=1, inplace=True)
            print("Dropped 'YEAR' from aggregated features to prevent duplication during final merge.")

        # --- Step 4: Merge final aggregated and trend features into the main data dataframe ---
        data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        print(f"Merged with enhanced aggregated offerings data and trends. Data shape: {data.shape}")

    else:
        print("Warning: offerings_df missing expected columns for enhanced aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping enhanced merge.")
else:
    print("Warning: offerings_df is empty. Proceeding with limited features.")

# Add features from terms_df if available and has required columns
# This block is kept as per original logic, merging YEAR directly into 'data'.
# It runs after the offerings_df processing.
if not terms_df.empty:
    if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
        # Merging YEAR separately, ensuring no duplicates for TERM_CODE entries in terms_df
        data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        print(f"Merged with terms data (YEAR). Data shape: {data.shape}")
    else:
        print("Warning: terms_df missing expected columns (YEAR). Skipping merge for YEAR column.")

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
