
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
import subprocess
import sys

# Install necessary libraries if not already installed
try:
    import xgboost
except ImportError:
    print("xgboost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost
from xgboost import XGBClassifier

try:
    import catboost
except ImportError:
    print("catboost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    import catboost
from catboost import CatBoostClassifier

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

def load_and_preprocess_data(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data, performs feature engineering,
    and prepares data for multiple models, distinguishing between numerical/encoded
    features for RandomForest/XGBoost and native categorical handling for CatBoost.
    """
    gold_df = pd.read_csv(gold_file)

    # Convert HIGH_ENROLLMENT to numerical (0 for 'N', 1 for 'Y') as it's the target.
    gold_df['HIGH_ENROLLMENT'] = gold_df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Initialize lists to store feature column names for different model types
    rf_xgb_numerical_features = [] # For RandomForest, XGBoost (expecting numerical/encoded categoricals)
    catboost_numerical_features = [] # For CatBoost (its numerical features)
    catboost_categorical_features = [] # For CatBoost (its native categorical features)
    
    # Identifier columns used for merging.
    identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT']

    # --- Pre-merge feature engineering for identifiers (always present in gold_df) ---
    
    # For CatBoost: use original identifier columns as categorical features.
    # Convert them to string type and fill NaNs to ensure robustness.
    catboost_categorical_features.extend(identifier_cols)
    for col in identifier_cols:
        df[col] = df[col].astype(str).fillna(f'Missing_{col}')

    # For RandomForest/XGBoost: use LabelEncoded versions of identifiers as numerical features.
    le_term = LabelEncoder()
    df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
    rf_xgb_numerical_features.append('TERM_CODE_ENCODED')

    le_subject = LabelEncoder()
    df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
    rf_xgb_numerical_features.append('SUBJECT_ID_SORT_ENCODED')

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        # Merge the main DataFrame with subject_summary_df to get additional features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        # Identify additional features from subject_summary_df (excluding identifiers, which are already handled).
        feature_candidates_from_summary = [
            col for col in subject_summary_df_full.columns 
            if col not in identifier_cols
        ]
        
        for col in feature_candidates_from_summary:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Add to both numerical feature sets
                rf_xgb_numerical_features.append(col)
                catboost_numerical_features.append(col)
            else: # Treat other types (object, string) as categorical for CatBoost.
                catboost_categorical_features.append(col)
                df[col] = df[col].astype(str).fillna('Missing_Category') # Fill NaNs for CatBoost's categorical features.
                # For RF/XGB, these additional categorical columns are not included unless explicitly one-hot encoded,
                # which aligns with the base solution's numerical-focused approach.

        # Fill any NaNs that might result from the left merge or from missing values in subject_summary.
        # Simple imputation with 0 for numerical features.
        df[rf_xgb_numerical_features] = df[rf_xgb_numerical_features].fillna(0)
        df[catboost_numerical_features] = df[catboost_numerical_features].fillna(0)

    except FileNotFoundError:
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        # If subject_summary.csv is not found, feature lists will only contain those derived from gold_df.
        # (i.e., TERM_CODE_ENCODED/SUBJECT_ID_SORT_ENCODED for RF/XGB and TERM_CODE/SUBJECT_ID_SORT for CatBoost).

    # --- Fallback for empty feature sets ---
    # Ensure all feature lists have at least one column for model training.
    if not rf_xgb_numerical_features and not catboost_numerical_features and not catboost_categorical_features:
        warnings.warn("No usable features identified after all steps. Adding a dummy feature for all models.")
        df['DUMMY_FEATURE'] = 0
        rf_xgb_numerical_features.append('DUMMY_FEATURE')
        catboost_numerical_features.append('DUMMY_FEATURE')
    
    # Store feature column names as attributes on the DataFrame for later use, ensuring uniqueness.
    df._rf_xgb_feature_cols = list(set(rf_xgb_numerical_features))
    df._catboost_numerical_features = list(set(catboost_numerical_features))
    df._catboost_categorical_features = list(set(catboost_categorical_features))
    
    return df

def run_training_and_validation():
    df = load_and_preprocess_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return

    # Ensure TERM_CODE is treated as string for consistent sorting behavior.
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    # Sort data by TERM_CODE to facilitate a time-based validation split.
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Retrieve feature lists determined during data loading.
    rf_xgb_feature_cols = getattr(df, '_rf_xgb_feature_cols', [])
    catboost_numerical_features = getattr(df, '_catboost_numerical_features', [])
    catboost_categorical_features = getattr(df, '_catboost_categorical_features', [])
    
    # Filter feature columns to ensure they are actually present in the DataFrame.
    rf_xgb_feature_cols = [col for col in rf_xgb_feature_cols if col in df.columns]
    catboost_numerical_features = [col for col in catboost_numerical_features if col in df.columns]
    catboost_categorical_features = [col for col in catboost_categorical_features if col in df.columns]

    # Combine all feature columns for CatBoost (numerical and categorical).
    catboost_all_features = catboost_numerical_features + catboost_categorical_features

    # Final check: if no features are available for any model type, exit.
    if not rf_xgb_feature_cols and not catboost_all_features:
        print("No usable features available for training after all preprocessing and fallback attempts.")
        print("Final Validation Performance: 0.0")
        return

    # Determine validation set: Use the latest TERM_CODE for validation, simulating a future term.
    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        print("Final Validation Performance: 0.0")
        return

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term].copy()
    val_df = df[df['TERM_CODE'] == validation_term].copy()

    if train_df.empty or val_df.empty:
        print("One of the training or validation sets is empty after time-based split.")
        print("Final Validation Performance: 0.0")
        return

    # Prepare X (features) and y (target) for training and validation for each model type.
    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # For RandomForest and XGBoost (using a common set of numerical/encoded features)
    X_train_rf_xgb = train_df[rf_xgb_feature_cols]
    X_val_rf_xgb = val_df[rf_xgb_feature_cols]

    # For CatBoost (using its specific combined feature set)
    X_train_catboost = train_df[catboost_all_features]
    X_val_catboost = val_df[catboost_all_features]

    if X_train_rf_xgb.empty or X_train_catboost.empty or y_train.empty:
        print("Training set is empty for at least one model's feature set. Cannot train models.")
        print("Final Validation Performance: 0.0")
        return

    if len(y_train.unique()) < 2:
        warnings.warn(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # --- Model Training 1: RandomForestClassifier (Base Solution Model) ---
    rf_pred_proba = np.array([])
    if not X_train_rf_xgb.empty and not X_val_rf_xgb.empty:
        print("Training RandomForestClassifier...")
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train_rf_xgb, y_train)
        rf_pred_proba = rf_model.predict_proba(X_val_rf_xgb)[:, 1] # Probabilities for the positive class (1)
    else:
        print("Skipping RandomForestClassifier training due to empty feature set.")

    # --- Model Training 2: XGBoost Classifier (Additional Model) ---
    xgb_pred_proba = np.array([])
    if not X_train_rf_xgb.empty and not X_val_rf_xgb.empty:
        print("Training XGBClassifier...")
        xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, 
                                  n_estimators=500, learning_rate=0.05, use_label_encoder=False)
        xgb_model.fit(X_train_rf_xgb, y_train)
        xgb_pred_proba = xgb_model.predict_proba(X_val_rf_xgb)[:, 1]
    else:
        print("Skipping XGBClassifier training due to empty feature set.")

    # --- Model Training 3: CatBoost Classifier (Reference Solution Model) ---
    cat_pred_proba = np.array([])
    if not X_train_catboost.empty and not X_val_catboost.empty:
        print("Training CatBoostClassifier...")
        # CatBoost needs to know which columns are categorical from X_train_catboost.
        cat_features_for_catboost_in_X = [col for col in catboost_categorical_features if col in X_train_catboost.columns]

        cat_model = CatBoostClassifier(
            iterations=500, learning_rate=0.05, random_seed=42, 
            loss_function='Logloss', eval_metric='F1', verbose=0, early_stopping_rounds=50
        )
        cat_model.fit(
            X_train_catboost, y_train,
            cat_features=cat_features_for_catboost_in_X,
            eval_set=(X_val_catboost, y_val),
        )
        cat_pred_proba = cat_model.predict_proba(X_val_catboost)[:, 1]
    else:
        print("Skipping CatBoostClassifier training due to empty feature set.")

    # --- Ensemble Predictions ---
    print("Ensembling predictions...")
    all_pred_probas = []
    if rf_pred_proba.size > 0:
        all_pred_probas.append(rf_pred_proba)
    if xgb_pred_proba.size > 0:
        all_pred_probas.append(xgb_pred_proba)
    if cat_pred_proba.size > 0:
        all_pred_probas.append(cat_pred_proba)

    if not all_pred_probas:
        print("No models were successfully trained, cannot ensemble.")
        print("Final Validation Performance: 0.0")
        return

    # Simple average of predicted probabilities from all successfully trained models.
    ensemble_pred_proba = np.mean(all_pred_probas, axis=0)
    
    # Convert ensembled probabilities to binary predictions using a default threshold of 0.5.
    ensemble_y_pred = (ensemble_pred_proba >= 0.5).astype(int)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    if y_val.empty:
        print("Validation set is empty. F1 score cannot be calculated.")
    else:
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(ensemble_y_pred)

        # Handle cases where the validation set contains only one class.
        if len(unique_y_val) < 2:
            warnings.warn(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0 # Perfect score if all predictions match the single class.
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set.
            # `zero_division=0` handles cases where a class has no true instances or no predicted instances,
            # assigning 0 to its F1 score contribution.
            final_validation_score = f1_score(y_val, ensemble_y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}")

# Run the training and validation process when the script is executed.
if __name__ == "__main__":
    run_training_and_validation()
