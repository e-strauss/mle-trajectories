
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuration ---
# All data is expected to be in the 'input' directory relative to the script execution.
BASE_DATA_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, "table_splits/train")
GOLD_TRAIN_LABELS = os.path.join(BASE_DATA_DIR, "eval/gold_enrollment_train.csv")

# TEST_DATA_DIR is a placeholder. For validation, we split the training data.
# This variable will be replaced by the evaluation system for final test set prediction.
TEST_DATA_DIR = "__TEST_DATA_DIR__"

def main():
    print(f"Loading data from {TRAIN_DATA_DIR} and labels from {GOLD_TRAIN_LABELS}")

    try:
        # Load the main subject summary data
        subject_summary_path = os.path.join(TRAIN_DATA_DIR, "subject_summary.csv")
        train_main_df = pd.read_csv(subject_summary_path)
    except FileNotFoundError:
        print(f"Error: `subject_summary.csv` not found at {subject_summary_path}. Please check data paths.")
        return
    except Exception as e:
        print(f"An error occurred while loading `subject_summary.csv`: {e}")
        return

    try:
        # Load the gold enrollment labels
        labels_df = pd.read_csv(GOLD_TRAIN_LABELS)
    except FileNotFoundError:
        print(f"Error: `gold_enrollment_train.csv` not found at {GOLD_TRAIN_LABELS}. Please check data paths.")
        return
    except Exception as e:
        print(f"An error occurred while loading `gold_enrollment_train.csv`: {e}")
        return

    # Merge subject summary with labels
    # Use 'left' merge to keep all entries from subject_summary, filling NaN for labels if missing.
    # However, for training, all entries should ideally have labels.
    train_main_df = pd.merge(train_main_df, labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Ensure HIGH_ENROLLMENT is present after merging
    if 'HIGH_ENROLLMENT' not in train_main_df.columns:
        print("Error: 'HIGH_ENROLLMENT' column not found after merging. Cannot proceed without target variable.")
        return
    
    # Drop rows where HIGH_ENROLLMENT is NaN, as these are unlabelled examples not usable for supervised training.
    initial_rows = len(train_main_df)
    train_main_df.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    if len(train_main_df) < initial_rows:
        print(f"Dropped {initial_rows - len(train_main_df)} rows due to missing 'HIGH_ENROLLMENT' labels.")
    
    if train_main_df.empty:
        print("Error: No data remaining after dropping unlabelled rows. Cannot train model.")
        return

    # --- Feature Engineering ---

    # Encode SUBJECT_ID_SORT using LabelEncoder
    # Fitting on the entire training set before splitting ensures consistency across splits.
    if 'SUBJECT_ID_SORT' in train_main_df.columns:
        le_subject_id = LabelEncoder()
        train_main_df['SUBJECT_ID_SORT_ENCODED'] = le_subject_id.fit_transform(train_main_df['SUBJECT_ID_SORT'])
    else:
        print("Warning: 'SUBJECT_ID_SORT' column not found in `subject_summary.csv`. Cannot create encoded feature.")
        train_main_df['SUBJECT_ID_SORT_ENCODED'] = 0 # Add a placeholder column

    # Extract year and semester from TERM_CODE
    train_main_df['TERM_YEAR'] = train_main_df['TERM_CODE'].astype(str).str[:4].astype(int)
    train_main_df['TERM_SEMESTER'] = train_main_df['TERM_CODE'].astype(str).str[4:].astype(int)

    # Define a list of numerical features that are expected and useful.
    # These are common features in academic course datasets.
    numerical_features = [
        'CREDITS_ATTEMPTED_COUNT', 'WAITLIST_COUNT', 'MAX_ENROLL', 'ENROLL_COUNT',
        'PRE_ENROLL_COUNT', 'ROOM_CAPACITY', 'INSTRUCTOR_COUNT',
        'SECTION_COUNT', 'GRADES_COUNT', 'DROP_COUNT', 'COURSE_LEVEL',
        'ACADEMIC_ORGANIZATION_COUNT', 'SCH_CREDITS'
    ]

    for col in numerical_features:
        if col in train_main_df.columns:
            # Convert to numeric, coercing errors to NaN, then fill NaN with 0.
            train_main_df[col] = pd.to_numeric(train_main_df[col], errors='coerce').fillna(0)
        else:
            # If a numerical feature column is missing, add it with zeros to maintain consistency.
            train_main_df[col] = 0

    # Drop original identifier columns that are not features or have been encoded
    features_to_drop = [
        'TERM_CODE',
        'SUBJECT_ID_SORT', # Original column for which we created 'SUBJECT_ID_SORT_ENCODED'
    ]
    train_main_df = train_main_df.drop(columns=[col for col in features_to_drop if col in train_main_df.columns])

    # Define features (X) and target (y)
    X_full = train_main_df.drop(columns=['HIGH_ENROLLMENT'])
    y_full = train_main_df['HIGH_ENROLLMENT']

    # --- Time-based validation split ---
    # Sort data by TERM_YEAR and TERM_SEMESTER to facilitate a time-based split.
    train_main_df_sorted = train_main_df.sort_values(by=['TERM_YEAR', 'TERM_SEMESTER'])

    # Determine validation split: use the latest terms (e.g., last two years) for validation.
    unique_years = sorted(train_main_df_sorted['TERM_YEAR'].unique())

    X_train_val, y_train_val, X_val, y_val = None, None, None, None

    if len(unique_years) >= 2:
        # Use the last two unique years for validation
        validation_years = unique_years[-2:]
        train_mask = ~train_main_df_sorted['TERM_YEAR'].isin(validation_years)
        val_mask = train_main_df_sorted['TERM_YEAR'].isin(validation_years)

        X_train_val = train_main_df_sorted[train_mask].drop(columns=['HIGH_ENROLLMENT'])
        y_train_val = train_main_df_sorted[train_mask]['HIGH_ENROLLMENT']
        X_val = train_main_df_sorted[val_mask].drop(columns=['HIGH_ENROLLMENT'])
        y_val = train_main_df_sorted[val_mask]['HIGH_ENROLLMENT']
        print(f"Validation split: Training on years before {validation_years[0]}, validating on {validation_years}.")
    elif len(unique_years) == 1 and len(train_main_df) > 10: # Ensure enough data for a random split
        print(f"Only one year ({unique_years[0]}) available or not enough data for time-based split. Performing random 80/20 split for validation.")
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    else:
        print("Error: Not enough data to perform a meaningful training/validation split.")
        return

    # Ensure feature columns are consistent between training and validation sets.
    # This prevents errors if a column appeared in one split but not the other due to data sparsity.
    all_feature_cols = X_full.columns.tolist() # Use columns from the full dataset

    X_train_val = X_train_val[all_feature_cols]
    X_val = X_val[all_feature_cols]

    print(f"Training data shape: {X_train_val.shape}, Validation data shape: {X_val.shape}")
    print(f"Target distribution in training: {y_train_val.value_counts(normalize=True)}")
    print(f"Target distribution in validation: {y_val.value_counts(normalize=True)}")

    # --- Model Training ---
    print("Training RandomForestClassifier...")
    # Using 'balanced' class_weight to handle potential class imbalance in the target variable.
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_val, y_train_val)
    print("Model training complete.")

    # --- Evaluation ---
    print("Evaluating model on validation set...")
    if X_val.empty:
        print("Validation set is empty, skipping evaluation.")
        final_validation_score = 0.0 # Assign a default score if no validation data
    else:
        y_pred = model.predict(X_val)
        # Calculate macro F1-score as required
        final_validation_score = f1_score(y_val, y_pred, average='macro')
    
    # Print the final validation performance in the specified format
    print(f"Final Validation Performance: {final_validation_score}")

if __name__ == "__main__":
    main()
