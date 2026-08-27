
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
import subprocess
import sys

# Install xgboost if not already installed
try:
    import xgboost
except ImportError:
    print("xgboost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost
from xgboost import XGBClassifier


# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        # Identifier columns used for merging and to be excluded from direct feature selection
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        # Merge gold_df (which now contains numerical 'HIGH_ENROLLMENT') with subject_summary_df to get features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        # Select numerical features for the model.
        # Exclude original identifiers and the target column.
        feature_cols_candidate = [
            col for col in df.columns 
            if col not in identifier_cols + ['HIGH_ENROLLMENT']
        ]
        
        # Filter for truly numeric columns from the candidate list.
        numeric_feature_cols = df[feature_cols_candidate].select_dtypes(include=np.number).columns.tolist()
        
        # Encode TERM_CODE and SUBJECT_ID_SORT to use them as numerical features
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        if 'TERM_CODE_ENCODED' not in numeric_feature_cols:
            numeric_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        if 'SUBJECT_ID_SORT_ENCODED' not in numeric_feature_cols:
            numeric_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(numeric_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found after merge, add a dummy feature.
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge (if some gold entries don't have summary data)
        # or from missing values in the subject_summary. Simple imputation with 0.
        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        
        # Store feature columns for later use
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        # Fallback to the minimal features if subject_summary.csv is not found
        
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        # Add a simple dummy numerical feature
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        # Fill potential NaNs in dummy features (not expected if created from existing data, but for robustness)
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script ---
def run_training_and_validation():
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        print("Final Validation Performance: 0.0")
        return

    # Use the last term for validation to simulate a future term, as per problem description.
    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    # Retrieve feature columns determined during data loading
    feature_cols = getattr(df, '_feature_cols', [])
    
    # Fallback to identify feature columns if not properly set (should not happen with robust load_data)
    if not feature_cols: 
        warnings.warn("Feature columns not correctly identified by load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print("No usable features available for training after all fallback attempts.")
        print("Final Validation Performance: 0.0")
        return

    # Align feature columns for train and validation sets to ensure consistency
    X_train_cols = [col for col in feature_cols if col in train_df.columns]
    X_val_cols = [col for col in feature_cols if col in val_df.columns]

    common_feature_cols = list(set(X_train_cols) & set(X_val_cols))

    if not common_feature_cols:
        warnings.warn("No common feature columns found between train and validation sets. This indicates a potential issue in feature generation or data consistency across splits. Falling back to minimal common features.")
        temp_common_features = []
        if 'TERM_CODE_ENCODED' in train_df.columns and 'TERM_CODE_ENCODED' in val_df.columns:
            temp_common_features.append('TERM_CODE_ENCODED')
        if 'SUBJECT_ID_SORT_ENCODED' in train_df.columns and 'SUBJECT_ID_SORT_ENCODED' in val_df.columns:
            temp_common_features.append('SUBJECT_ID_SORT_ENCODED')
        
        if not temp_common_features: # If still no features, add a guaranteed dummy feature
            if 'DUMMY_FEATURE' not in train_df.columns:
                train_df['DUMMY_FEATURE'] = 0
            if 'DUMMY_FEATURE' not in val_df.columns:
                val_df['DUMMY_FEATURE'] = 0
            temp_common_features.append('DUMMY_FEATURE')
        
        common_feature_cols = temp_common_features
        # Fill NaNs for the now explicit common_feature_cols in case they were missed or created dynamically
        train_df[common_feature_cols] = train_df[common_feature_cols].fillna(0)
        val_df[common_feature_cols] = val_df[common_feature_cols].fillna(0)
    
    if not common_feature_cols:
        print("No usable features available for training after aligning train and validation sets.")
        print("Final Validation Performance: 0.0")
        return

    X_train = train_df[common_feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[common_feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for empty training data
    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        print("Final Validation Performance: 0.0")
        return

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
        # Proceeding with training, as this is a valid (though not ideal) training scenario.
    
    # --- Model Training 1: RandomForestClassifier (Base Solution) ---
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_pred_proba = rf_model.predict_proba(X_val)[:, 1] # Get probabilities for the positive class (1)

    # --- Model Training 2: XGBoost Classifier (Reference Solution) ---
    print("Training XGBClassifier...")
    # use_label_encoder=False is deprecated in recent XGBoost versions but kept for direct integration
    # from the reference solution. It is safe to remove if it causes warnings with newer XGBoost.
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, 
                              n_estimators=500, learning_rate=0.05, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_val)[:, 1] # Get probabilities for the positive class (1)

    # --- Ensemble Predictions ---
    print("Ensembling predictions...")
    # Simple averaging of predicted probabilities from both models
    ensemble_pred_proba = (rf_pred_proba + xgb_pred_proba) / 2
    
    # Convert ensembled probabilities to binary predictions using a default threshold of 0.5
    ensemble_y_pred = (ensemble_pred_proba >= 0.5).astype(int)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        print("Validation set is empty. F1 score cannot be calculated.")
    else:
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(ensemble_y_pred)

        # 2. Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            print(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set
            # zero_division=0 ensures that if a class has no true instances or no predicted instances,
            # its F1 score contribution is 0, preventing division by zero warnings/errors.
            final_validation_score = f1_score(y_val, ensemble_y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}")

# Run the training and validation process
if __name__ == "__main__":
    run_training_and_validation()

