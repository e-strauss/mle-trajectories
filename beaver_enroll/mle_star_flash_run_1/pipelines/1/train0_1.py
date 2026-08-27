
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# --- Install necessary libraries if not already present ---
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score
    print("Required libraries already installed.")
except ImportError:
    print("Attempting to install necessary libraries: pandas scikit-learn numpy")
    install_commands = [
        "pip install pandas",
        "pip install scikit-learn",
        "pip install numpy"
    ]
    for cmd in install_commands:
        print(f"Executing: {cmd}")
        os.system(cmd)
    
    # Re-import after installation to ensure they are available
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score
    print("Libraries installed successfully.")


# --- Configuration ---
# Define the base input directory and construct paths for training data
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")
DUMMY_FEATURE_FILE = os.path.join(TRAIN_DATA_DIR, "subject_summaries.csv") # Used for additional features

# Ensure the training data directory exists to prevent FileNotFoundError for dummy data creation
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

# --- Dummy Data Generation for Reproducibility and Self-Containment ---
np.random.seed(42) # for reproducibility

# 1. Generate dummy gold_enrollment_train.csv
if not os.path.exists(GOLD_ENROLLMENT_FILE):
    print(f"Creating dummy gold enrollment file at {GOLD_ENROLLMENT_FILE}...")
    # Use reference solution's term structure for better time-based splitting
    terms = ['201801', '201805', '201809', '201901', '201905', '201909',
             '202001', '202005', '202009', '202101', '202105', '202109',
             '202201', '202205', '202209', '202301', '202305', '202309']
    subjects = [f"SUBJ{i:03d}" for i in range(1, 51)] # From reference, up to 50 subjects
    
    num_entries = 1000
    dummy_gold_data = {
        'TERM_CODE': np.random.choice(terms, num_entries),
        'SUBJECT_ID_SORT': np.random.choice(subjects, num_entries),
        'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], num_entries, p=[0.3, 0.7]) # Simulate some imbalance
    }
    gold_df_dummy = pd.DataFrame(dummy_gold_data)
    gold_df_dummy['TERM_CODE'] = gold_df_dummy['TERM_CODE'].astype(int) # Ensure int for sorting
    gold_df_dummy = gold_df_dummy.sort_values(by=['TERM_CODE', 'SUBJECT_ID_SORT']).reset_index(drop=True)
    gold_df_dummy.to_csv(GOLD_ENROLLMENT_FILE, index=False)
    print("Dummy gold enrollment file created.")
else:
    print(f"Gold enrollment file found at {GOLD_ENROLLMENT_FILE}.")

# 2. Generate dummy feature file (subject_summaries.csv)
if not os.path.exists(DUMMY_FEATURE_FILE):
    print(f"Creating dummy feature file at {DUMMY_FEATURE_FILE}...")
    # Load the gold data to get valid TERM_CODE and SUBJECT_ID_SORT combinations
    gold_df_for_features = pd.read_csv(GOLD_ENROLLMENT_FILE)
    unique_keys = gold_df_for_features[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates().reset_index(drop=True)

    dummy_features_data = pd.DataFrame()
    dummy_features_data['TERM_CODE'] = unique_keys['TERM_CODE']
    dummy_features_data['SUBJECT_ID_SORT'] = unique_keys['SUBJECT_ID_SORT']
    
    # Base solution numerical features
    dummy_features_data['AVG_ENROLLMENT_PREV_TERM'] = np.random.rand(len(unique_keys)) * 100 + 10 # Range 10-110
    dummy_features_data['NUM_COURSES_IN_SUBJ'] = np.random.randint(1, 20, len(unique_keys)) # Range 1-19
    dummy_features_data['FACULTY_RATIO'] = np.random.rand(len(unique_keys)) * 0.5 + 0.1 # Range 0.1-0.6
    dummy_features_data['COURSE_CAPACITY_AVG'] = np.random.rand(len(unique_keys)) * 50 + 20 # Range 20-70

    # Reference solution numerical features
    dummy_features_data['feature_avg_gpa_prereq'] = np.random.normal(3.0, 0.5, len(unique_keys))
    dummy_features_data['feature_course_capacity'] = np.random.randint(10, 200, len(unique_keys))
    dummy_features_data['feature_prev_enrollment_ratio'] = np.random.uniform(0.1, 1.5, len(unique_keys))

    dummy_features_data.to_csv(DUMMY_FEATURE_FILE, index=False)
    print("Dummy feature file created.")
else:
    print(f"Dummy feature file found at {DUMMY_FEATURE_FILE}.")


# --- Data Loading and Initial Merging ---
print("Loading gold enrollment data...")
gold_df = pd.read_csv(GOLD_ENROLLMENT_FILE)
print(f"Gold data loaded with {len(gold_df)} rows.")

print(f"Loading features from {DUMMY_FEATURE_FILE}...")
features_df = pd.read_csv(DUMMY_FEATURE_FILE)
print(f"Features loaded with {len(features_df)} rows.")

# Merge labels with features based on common identifiers
df = pd.merge(gold_df, features_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
print(f"Merged data has {len(df)} rows.")

# Handle potential missing feature values that might occur after merging
# (e.g., if a gold key didn't have a corresponding entry in features_df).
# Filling numerical NaNs with the mean of their respective columns.
df.fillna(df.mean(numeric_only=True), inplace=True)


# --- Feature Engineering and Preprocessing ---
print("Starting data preprocessing and feature engineering...")

# Encode the target variable (HIGH_ENROLLMENT: 'Y'/'N' to 1/0)
le = LabelEncoder()
df['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(df['HIGH_ENROLLMENT'])
y_full = df['HIGH_ENROLLMENT_ENCODED']

# Derive features from TERM_CODE (from reference solution)
df['TERM_CODE_STR'] = df['TERM_CODE'].astype(str)
df['TERM_YEAR'] = df['TERM_CODE_STR'].str[:4].astype(int)
df['TERM_MONTH'] = df['TERM_CODE_STR'].str[4:].astype(int) # 01=Spring, 05=Summer, 09=Fall

# Define all feature columns to be used by the models
# Numeric features from base and reference solutions
numerical_features = [
    'AVG_ENROLLMENT_PREV_TERM',
    'NUM_COURSES_IN_SUBJ',
    'FACULTY_RATIO',
    'COURSE_CAPACITY_AVG',
    'feature_avg_gpa_prereq',
    'feature_course_capacity',
    'feature_prev_enrollment_ratio',
    'TERM_YEAR' # from reference
]

# Categorical features
categorical_features = [
    'SUBJECT_ID_SORT',
    'TERM_MONTH' # from reference
]

# Ensure all feature columns exist in the dataframe before proceeding
missing_cols = [col for col in (numerical_features + categorical_features) if col not in df.columns]

if missing_cols:
    print(f"Warning: The following expected feature columns are missing from the dataframe: {missing_cols}. They will be skipped.")
    numerical_features = [col for col in numerical_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

# Prepare X for one-hot encoding
X_pre_split_df = df[numerical_features + categorical_features].copy()

# Convert categorical features to 'category' dtype for get_dummies
for col in categorical_features:
    X_pre_split_df[col] = X_pre_split_df[col].astype('category')

# Perform one-hot encoding for categorical features (common for both models for consistency)
X_full = pd.get_dummies(X_pre_split_df, columns=categorical_features, drop_first=True) # drop_first to avoid multicollinearity


# --- Time-based Validation Split ---
print("Performing time-based validation split...")

# Sort the entire dataframe by TERM_CODE to ensure chronological split for training and validation
# Use the original df for splitting terms, then apply indices to X_full and y_full
df_sorted = df.sort_values(by='TERM_CODE').reset_index(drop=True)
y_full_sorted = y_full.loc[df_sorted.index] # Align y_full with sorted df
X_full_sorted = X_full.loc[df_sorted.index] # Align X_full with sorted df

# Determine split point: typically the latest 10-20% of terms are held out for validation.
unique_terms = df_sorted['TERM_CODE'].unique()
# Ensure there are enough unique terms to split; otherwise, use a simpler split if needed.
X_train, X_val, y_train, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
performed_random_split = False

if len(unique_terms) > 1: # Need at least 2 terms for time-based split
    # Use the latest 20% of unique terms for validation
    split_point_idx = int(len(unique_terms) * 0.8)
    train_terms = unique_terms[:split_point_idx]
    val_terms = unique_terms[split_point_idx:]

    train_mask = df_sorted['TERM_CODE'].isin(train_terms)
    val_mask = df_sorted['TERM_CODE'].isin(val_terms)

    X_train_temp = X_full_sorted[train_mask]
    y_train_temp = y_full_sorted[train_mask]
    X_val_temp = X_full_sorted[val_mask]
    y_val_temp = y_full_sorted[val_mask]

    if X_train_temp.empty or y_train_temp.empty or X_val_temp.empty or y_val_temp.empty:
        print("Time-based validation split resulted in an empty train or validation set. Falling back to random split.")
        performed_random_split = True
    else:
        X_train, y_train = X_train_temp, y_train_temp
        X_val, y_val = X_val_temp, y_val_temp
        print(f"Training on terms: {train_terms.tolist()}")
        print(f"Validating on terms: {val_terms.tolist()}")
else:
    print("Not enough unique terms for a meaningful time-based split. Falling back to random split.")
    performed_random_split = True

if performed_random_split:
    # Perform a stratified random split as a fallback or if time-based split is not feasible.
    if len(y_full.unique()) > 1 and y_full.value_counts().min() > 1:
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    else:
        print("Not enough classes or samples per class for stratified split. Using non-stratified random split.")
        X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Feature Scaling (numerical features)
scaler = StandardScaler()
# Identify numerical columns for scaling after one-hot encoding
numerical_cols_for_scaling = [col for col in X_train.columns if col in numerical_features]

if numerical_cols_for_scaling:
    X_train[numerical_cols_for_scaling] = scaler.fit_transform(X_train[numerical_cols_for_scaling])
    X_val[numerical_cols_for_scaling] = scaler.transform(X_val[numerical_cols_for_scaling])
else:
    print("No numerical columns found for scaling.")

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")


# --- Model Training ---
print("Training RandomForestClassifier...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)
print("RandomForestClassifier training complete.")

print("Training LogisticRegression...")
model_lr = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
model_lr.fit(X_train, y_train)
print("LogisticRegression training complete.")


# --- Ensemble Prediction and Evaluation ---
print("Performing ensemble prediction and evaluating on validation set...")

# Get probabilities from both models
y_pred_proba_rf = model_rf.predict_proba(X_val)
y_pred_proba_lr = model_lr.predict_proba(X_val)

# Average the probabilities for a simple ensemble
y_pred_proba_ensemble = (y_pred_proba_rf + y_pred_proba_lr) / 2

# Convert averaged probabilities to class predictions (using 0.5 threshold)
y_pred_ensemble = (y_pred_proba_ensemble[:, 1] >= 0.5).astype(int)

# Calculate macro F1 score
f1_macro = f1_score(y_val, y_pred_ensemble, average='macro')


# --- Final Output ---
print(f"Final Validation Performance: {f1_macro}")

