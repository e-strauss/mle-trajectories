
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import xgboost as xgb
import subprocess # For installing xgboost
import sys # For sys.executable

# Try to install xgboost if not found
try:
    import xgboost as xgb
except ImportError:
    print("xgboost not found. Attempting to install...")
    try:
        # Use sys.executable to ensure pip associated with the current Python environment is used
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])
        import xgboost as xgb
        print("xgboost installed successfully.")
    except Exception as e:
        print(f"Failed to install xgboost: {e}")
        # If installation fails, the script will likely terminate later if xgboost is used.
        # It's better to let it raise an ImportError if installation is critical and fails.
        raise ImportError(f"xgboost is required but could not be installed: {e}")

# Define data directories
# The problem statement specifies: "All the provided input data is stored in './input' directory."
# So we prepend './input/' to the TRAIN_DATA_DIR to correctly locate the files.
# Using os.path.join for robustness across different operating systems.
TRAIN_DATA_DIR = os.path.join('./input', os.getenv('TRAIN_DATA_DIR', 'table_splits/train'))
# TEST_DATA_DIR is a placeholder and not used in this training phase
TEST_DATA_DIR = os.getenv('TEST_DATA_DIR', None)

# --- Data Loading ---
print("Loading data...")
try:
    df_gold = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'gold_enrollment_train.csv'))
    df_catalog = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'course_catalog.csv'))
    df_sections = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'course_sections.csv'))
    print("Data loaded successfully.")
except FileNotFoundError as e:
    # Providing a more informative error message if files are not found.
    print(f"Error loading data: {e}. Please ensure the input directory structure is correct. Looked in: {TRAIN_DATA_DIR}")
    raise # Re-raise the error to stop execution if essential files are missing

# --- Feature Engineering ---
print("Starting feature engineering...")

# Merge gold enrollment data with course catalog for course details
df = pd.merge(df_gold, df_catalog, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

# Extract course level and subject prefix from SUBJECT_ID_SORT
# Using n=1 to split only on the first hyphen, and expand=True to create new columns.
# Fill any resulting None/NaN values robustly.
split_id = df['SUBJECT_ID_SORT'].str.split('-', n=1, expand=True)
df['SUBJECT_PREFIX'] = split_id[0].fillna('UNKNOWN_PREFIX').astype(str)
# If no hyphen exists, split_id[1] will be None; fill with a default '000' or similar placeholder.
df['COURSE_LEVEL'] = split_id[1].fillna('000').astype(str)

# Aggregate course sections data to create features like number of sections, avg enrollment, and total capacity
section_summary = df_sections.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
    num_sections=('CRN', 'nunique'),
    avg_enrollment=('CURRENT_ENROLLMENT', 'mean'),
    max_capacity=('MAX_CAPACITY', 'sum') # Sum capacity across all sections for a given course offering
).reset_index()

# Calculate enrollment_capacity_ratio, explicitly handling potential division by zero.
# If max_capacity is 0, the ratio is set to 0 to prevent NaNs or Infs.
section_summary['enrollment_capacity_ratio'] = np.where(
    section_summary['max_capacity'] > 0,
    section_summary['avg_enrollment'] / section_summary['max_capacity'],
    0
)

# Merge the aggregated section data back into the main DataFrame
df = pd.merge(df, section_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

# Fill NaNs for newly merged numerical features (e.g., if a course had no sections data)
# Assuming 0 as a reasonable default for counts, averages, and ratios if no data was found.
for col in ['num_sections', 'avg_enrollment', 'max_capacity', 'enrollment_capacity_ratio']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Derive TERM_YEAR and TERM_SEMESTER from TERM_CODE for temporal features
df['TERM_YEAR'] = df['TERM_CODE'].astype(str).str[:4].astype(int)
df['TERM_SEMESTER'] = df['TERM_CODE'].astype(str).str[4:].astype(int)

print("Feature engineering completed.")

# --- Preprocessing ---
print("Starting preprocessing...")

# Convert the target variable 'HIGH_ENROLLMENT' into a numerical format (1 for 'Y', 0 for 'N')
df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

# Define columns that act as identifiers or the target, and should not be used as features directly.
identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']

# Identify categorical features: columns with object or category dtype, excluding identifiers.
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features = [col for col in categorical_features if col not in identifier_cols]

# Identify numerical features: columns with numeric dtype, excluding identifiers.
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
numerical_features = [col for col in numerical_features if col not in identifier_cols]

# Handle missing values for numerical features using median imputation.
for col in numerical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Handle missing values for categorical features by filling with a 'Missing' category.
for col in categorical_features:
    if df[col].isnull().any():
        df[col] = df[col].fillna('Missing')

# One-hot encode categorical features *before* the time-based train-validation split.
# This critical step ensures that the training and validation sets have an identical set of columns,
# preventing `KeyError` or dimension mismatch issues during model prediction.
df_encoded = pd.get_dummies(df, columns=categorical_features, dummy_na=False)

# Define the final list of feature columns after one-hot encoding.
# This list includes all processed numerical features and the newly created one-hot encoded columns,
# while excluding original identifiers and the target variable.
feature_columns_final = [col for col in df_encoded.columns if col not in identifier_cols]

# Perform a final check to ensure all selected feature columns are indeed numeric.
# This guards against unexpected non-numeric columns that might have slipped through.
final_numeric_features = []
for col in feature_columns_final:
    if pd.api.types.is_numeric_dtype(df_encoded[col]):
        final_numeric_features.append(col)
    else:
        print(f"Warning: Non-numeric feature column '{col}' detected and will be dropped from features.")

print("Preprocessing completed.")

# --- Time-based Validation Split ---
print("Splitting data into training and validation sets...")
unique_terms = sorted(df_encoded['TERM_CODE'].unique())

# Use the latest 20% of unique terms for the validation set to simulate a realistic temporal split.
# This assumes that `TERM_CODE` is chronologically ordered.
# Ensure there are enough terms to make a meaningful split.
if len(unique_terms) < 2:
    # If there are too few unique terms, a time-based split is not practical.
    # This scenario would make evaluation difficult or impossible.
    print(f"Error: Only {len(unique_terms)} unique terms found. Cannot perform a meaningful time-based split.")
    raise ValueError("Insufficient unique terms for time-based validation split.")

val_terms = unique_terms[int(len(unique_terms) * 0.8):]

# Create training and validation DataFrames based on the selected terms.
train_df = df_encoded[~df_encoded['TERM_CODE'].isin(val_terms)]
val_df = df_encoded[df_encoded['TERM_CODE'].isin(val_terms)]

# Separate features (X) and target (y) for both training and validation sets.
X_train = train_df[final_numeric_features]
y_train = train_df['HIGH_ENROLLMENT']
X_val = val_df[final_numeric_features]
y_val = val_df['HIGH_ENROLLMENT']

print(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, Target shape: {y_val.shape}")
print(f"Validation terms: {val_terms}")

# Critical checks: ensure validation set is not empty and has both classes for F1 score calculation.
if X_val.empty or y_val.empty:
    print("Error: Validation set is empty after splitting. Cannot proceed with model evaluation.")
    raise ValueError("Validation set is empty, cannot evaluate the model.")

if y_train.nunique() < 2:
    print("Error: Training set has only one class for HIGH_ENROLLMENT. Cannot train a binary classifier.")
    raise ValueError("Training set has only one class for target variable.")

if y_val.nunique() < 2:
    print("Warning: Validation set has only one class for HIGH_ENROLLMENT. F1 score might be undefined or misleading.")
    # Proceed, but note that F1 score might be 0 or throw an error depending on `zero_division` parameter.

# --- Model Training ---
print("Training XGBoost model...")

# Calculate `scale_pos_weight` to address potential class imbalance in the training data.
# This parameter helps XGBoost to give more importance to the minority class.
positive_count = y_train.sum()
negative_count = len(y_train) - positive_count
# Handle the case where there might be no positive samples to avoid division by zero.
scale_pos_weight_value = negative_count / positive_count if positive_count > 0 else 1.0

print(f"Positive samples in train: {positive_count}, Negative samples in train: {negative_count}, Scale Pos Weight: {scale_pos_weight_value:.2f}")

# Initialize and train the XGBoost Classifier.
# `objective='binary:logistic'` for binary classification.
# `eval_metric='logloss'` is a common metric for binary classification.
# `use_label_encoder` is deprecated and removed in recent XGBoost versions; it's omitted here.
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight_value,
    random_state=42
)

model.fit(X_train, y_train)
print("Model training completed.")

# --- Prediction and Evaluation ---
print("Making predictions and evaluating...")
y_pred_val = model.predict(X_val)

# Calculate the macro F1 score as the primary evaluation metric.
# `zero_division=0` ensures that if a class has no true samples or no predicted samples,
# its F1 score contributes 0 to the macro average, preventing errors.
final_validation_score = f1_score(y_val, y_pred_val, average='macro', zero_division=0)
print(f"Final Validation Performance: {final_validation_score}")

