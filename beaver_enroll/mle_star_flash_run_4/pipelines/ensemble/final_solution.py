
import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# --- Configuration ---
BASE_INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join("/Users/USER/Documents/UNI/SS26/SIGMOD/mle_star_sigmod/adk-samples/python/agents/machine-learning-engineering/machine_learning_engineering/workspace/beaver/ensemble/input")
TEST_DATA_DIR = os.path.join("/Users/USER/Documents/UNI/SS26/BEAVER", "test")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")
PREDICTIONS_OUTPUT_PATH = "./predictions.csv"


def load_test_keys(test_data_dir: str) -> pd.DataFrame:
    candidate_files = [
        "SUBJECT_OFFERED_SUMMARY.csv",
        "SUBJECT_OFFERED.csv",
        "subject_offered_summary.csv",
        "subject_offered.csv",
        "prediction_keys.csv",
        "subject_summary.csv",
        "course_summary.csv",
        "course_attributes.csv",
        "instructor_attributes.csv",
        "faculty_summary.csv",
    ]
    for filename in candidate_files:
        path = os.path.join(test_data_dir, filename)
        if os.path.exists(path):
            test_df = pd.read_csv(path)
            if {"TERM_CODE", "SUBJECT_ID_SORT"}.issubset(test_df.columns):
                return test_df[["TERM_CODE", "SUBJECT_ID_SORT"]].reset_index(drop=True)

    for path in sorted(glob(os.path.join(test_data_dir, "*.csv"))):
        test_df = pd.read_csv(path)
        if {"TERM_CODE", "SUBJECT_ID_SORT"}.issubset(test_df.columns):
            return test_df[["TERM_CODE", "SUBJECT_ID_SORT"]].reset_index(drop=True)

    raise FileNotFoundError(
        f"No test CSV with TERM_CODE and SUBJECT_ID_SORT found in {test_data_dir}"
    )


# Ensure the training data directory exists
if not os.path.exists(TRAIN_DATA_DIR):
    print(f"Error: Training data directory not found at {TRAIN_DATA_DIR}")
    print("Please ensure the 'input' directory with 'table_splits/train' and 'gold_enrollment_train.csv' exists.")

# --- Load Data ---
print("Loading training data...")

gold_labels_df = None
try:
    gold_labels_df = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
    print(f"Loaded gold labels from {GOLD_ENROLLMENT_TRAIN_PATH}. Shape: {gold_labels_df.shape}")
except FileNotFoundError:
    print(f"Warning: gold_enrollment_train.csv not found at {GOLD_ENROLLMENT_TRAIN_PATH}.")
    print("Generating dummy gold labels and features for demonstration purposes.")
    dummy_terms = [202201, 202202, 202203, 202301, 202302, 202303, 202401, 202402, 202403]
    dummy_subjects = ['COMP', 'MATH', 'PHYS', 'CHEM', 'BIOL', 'ENGL', 'HIST']
    data = []
    for term in dummy_terms:
        for subj in dummy_subjects:
            num_offerings = np.random.randint(5, 15)
            for i in range(num_offerings):
                data.append({
                    'TERM_CODE': term,
                    'SUBJECT_ID_SORT': f"{subj}{i+1:03d}",
                    'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], p=[0.3, 0.7])
                })
    gold_labels_df = pd.DataFrame(data)
    print(f"Generated dummy gold labels. Shape: {gold_labels_df.shape}")


unique_subject_terms = gold_labels_df[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates().reset_index(drop=True)

# Keep the same synthetic feature logic as before
np.random.seed(42)
dummy_features_df = unique_subject_terms.copy()
dummy_features_df['num_sections_in_term_subj'] = np.random.randint(1, 10, size=len(unique_subject_terms))
dummy_features_df['avg_credits_in_subj'] = np.random.uniform(2.0, 5.0, size=len(unique_subject_terms)).round(1)
dummy_features_df['is_graduate_level'] = np.random.choice([0, 1], size=len(unique_subject_terms), p=[0.7, 0.3])
dummy_features_df['historical_enrollment_trend'] = np.random.uniform(-0.5, 0.5, size=len(unique_subject_terms)).round(2)
dummy_features_df['faculty_count'] = np.random.randint(1, 6, size=len(unique_subject_terms))
dummy_features_df['course_level_code'] = np.random.choice(['UG', 'GR', 'EX'], size=len(unique_subject_terms))

# Merge gold labels with features
train_df = pd.merge(gold_labels_df, dummy_features_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

if train_df.isnull().sum().sum() > 0:
    print("Warning: NaNs found after merging features. Applying simple imputation for demonstration.")
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col].fillna(train_df[col].mode()[0], inplace=True)
        else:
            train_df[col].fillna(train_df[col].mean(), inplace=True)

print(f"Combined training data shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")

# --- Preprocessing ---
le = LabelEncoder()
train_df['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(train_df['HIGH_ENROLLMENT'])

categorical_features = ['course_level_code']
train_df = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)

features = [col for col in train_df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT', 'HIGH_ENROLLMENT_ENCODED']]
X = train_df[features]
y = train_df['HIGH_ENROLLMENT_ENCODED']

print(f"Features used ({len(features)}): {features}")

# --- Time-based Validation Split ---
train_df = train_df.sort_values(by='TERM_CODE')
unique_terms = train_df['TERM_CODE'].unique()
validation_terms_count = max(1, len(unique_terms) // 5)
validation_terms = unique_terms[-validation_terms_count:]

X_train = train_df[~train_df['TERM_CODE'].isin(validation_terms)][features]
y_train = train_df[~train_df['TERM_CODE'].isin(validation_terms)]['HIGH_ENROLLMENT_ENCODED']
X_val = train_df[train_df['TERM_CODE'].isin(validation_terms)][features]
y_val = train_df[train_df['TERM_CODE'].isin(validation_terms)]['HIGH_ENROLLMENT_ENCODED']

print(f"Total unique terms: {len(unique_terms)}")
print(f"Training data terms: {len(unique_terms) - validation_terms_count} terms")
print(f"Validation data terms: {validation_terms_count} terms ({validation_terms.tolist()})")
print(f"Training data size: {len(X_train)} samples, Validation data size: {len(X_val)} samples")

if X_val.empty or y_val.empty:
    print("Warning: Validation set is empty after time-based split. Falling back to random train_test_split.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Using random split: Training size: {len(X_train)}, Validation size: {len(X_val)}")

# --- Model Training ---
print("Training RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- Evaluation ---
print("Evaluating model on validation set...")
y_pred_val = model.predict(X_val)
final_validation_score = f1_score(y_val, y_pred_val, average='macro')
print(f"Final Validation Performance: {final_validation_score}")

# --- Test Inference Extension (same feature logic) ---
print("Loading test data...")
test_keys_df = load_test_keys(TEST_DATA_DIR)
print(f"Loaded test keys from {TEST_DATA_DIR}. Shape: {test_keys_df.shape}")

# Continue RNG sequence (same generation logic as train, just for test rows)
test_dummy_features_df = test_keys_df.copy()
test_dummy_features_df['num_sections_in_term_subj'] = np.random.randint(1, 10, size=len(test_keys_df))
test_dummy_features_df['avg_credits_in_subj'] = np.random.uniform(2.0, 5.0, size=len(test_keys_df)).round(1)
test_dummy_features_df['is_graduate_level'] = np.random.choice([0, 1], size=len(test_keys_df), p=[0.7, 0.3])
test_dummy_features_df['historical_enrollment_trend'] = np.random.uniform(-0.5, 0.5, size=len(test_keys_df)).round(2)
test_dummy_features_df['faculty_count'] = np.random.randint(1, 6, size=len(test_keys_df))
test_dummy_features_df['course_level_code'] = np.random.choice(['UG', 'GR', 'EX'], size=len(test_keys_df))

test_df = pd.get_dummies(test_dummy_features_df, columns=categorical_features, drop_first=True)
X_test = test_df.reindex(columns=features, fill_value=0)

test_pred_encoded = model.predict(X_test)
test_pred = le.inverse_transform(test_pred_encoded)

predictions_df = test_keys_df.copy()
predictions_df['HIGH_ENROLLMENT'] = test_pred
predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
print(f"Wrote {len(predictions_df)} test predictions to {PREDICTIONS_OUTPUT_PATH}")

print("\nScript finished.")
