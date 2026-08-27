
import pandas as pd
import numpy as np
import os
import warnings
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# Define paths
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

def load_and_preprocess_data(data_dir, gold_file):
    """
    Loads gold labels and subject summary data, performs merges, feature engineering,
    and prepares data for CatBoost, identifying numerical and categorical features.
    """
    gold_df = pd.read_csv(gold_file)

    # Convert HIGH_ENROLLMENT to numerical (0 for 'N', 1 for 'Y') as CatBoost expects
    # numerical targets for binary classification.
    gold_df['HIGH_ENROLLMENT'] = gold_df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    # Start with gold_df for merging
    df = gold_df.copy()

    # Identifier columns used for merging. These will also be treated as categorical features.
    identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 

    # Lists to store feature column names
    numerical_features = []
    categorical_features = []
    
    # Always include TERM_CODE and SUBJECT_ID_SORT as categorical features from the start
    # as they are key identifiers and likely important.
    categorical_features.extend(identifier_cols)

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        # Merge gold_df with subject_summary_df to get additional features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        # Identify feature candidates (all columns except identifiers and the target)
        feature_candidates = [
            col for col in df.columns 
            if col not in identifier_cols + ['HIGH_ENROLLMENT']
        ]
        
        # Separate numerical and categorical features from the subject summary data.
        for col in feature_candidates:
            # Check for numerical types. Consider int64 as well.
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_features.append(col)
            else: # Treat other types (object, string) as categorical
                categorical_features.append(col)
        
        # Handle missing values:
        # For numerical features, a simple imputation with 0.
        # More sophisticated imputation (mean/median) could be used in a complex solution.
        df[numerical_features] = df[numerical_features].fillna(0)
        
        # For categorical features, convert to string type and fill NaNs with a placeholder.
        # CatBoost can handle string-represented NaNs as a distinct category.
        for col in categorical_features:
            df[col] = df[col].astype(str).fillna('Missing_Category')

    except FileNotFoundError:
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df (TERM_CODE, SUBJECT_ID_SORT).")
        # If subject_summary.csv is not found, we only have TERM_CODE and SUBJECT_ID_SORT as features.
        # Ensure they are string type for CatBoost categorical handling.
        for col in identifier_cols:
            df[col] = df[col].astype(str).fillna(f'Missing_{col}')
        
        # No additional numerical features if summary file is missing.
        numerical_features = [] 

    # Store feature column names for later use. Use set to ensure uniqueness then convert back to list.
    # Convert to sets first to handle duplicates that might arise from identifier_cols being in both.
    df._numerical_features = list(set(numerical_features))
    df._categorical_features = list(set(categorical_features))

    return df

def run_training_and_validation():
    df = load_and_preprocess_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return

    # Ensure TERM_CODE is treated as string for consistent sorting behavior and categorical encoding
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    # Sort data by TERM_CODE to facilitate time-based validation split
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Retrieve feature lists determined during data loading.
    numerical_features = getattr(df, '_numerical_features', [])
    categorical_features = getattr(df, '_categorical_features', [])
    
    # Combine all identified feature columns.
    all_feature_cols = numerical_features + categorical_features

    # Fallback if no features were identified (e.g., if data loading completely failed).
    # This block is moved BEFORE the train/validation split to ensure 'DUMMY_FEATURE'
    # is present in both train_df and val_df if it gets added.
    if not all_feature_cols:
        warnings.warn("No features identified after data loading. Adding a dummy numerical feature.")
        df['DUMMY_FEATURE'] = 0 
        all_feature_cols = ['DUMMY_FEATURE']
        numerical_features = ['DUMMY_FEATURE']
        categorical_features = [] # No categorical features in this specific fallback scenario.

    # Determine validation set: Use the latest TERM_CODE for validation as per problem description.
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough unique terms to create both training and validation sets.
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        print("Final Validation Performance: 0.0")
        return

    # The last term is chosen for validation, simulating a future term.
    validation_term = unique_terms[-1] 
    
    # Use .copy() to prevent SettingWithCopyWarning
    train_df = df[df['TERM_CODE'] != validation_term].copy()
    val_df = df[df['TERM_CODE'] == validation_term].copy()

    # Check for empty training or validation data after split.
    if train_df.empty or val_df.empty:
        print("One of the training or validation sets is empty after time-based split.")
        print("Final Validation Performance: 0.0")
        return

    # Prepare X (features) and y (target) for CatBoost.
    X_train = train_df[all_feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[all_feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Filter `categorical_features` to only include those that are actually present in `X_train`.
    # This is a robustness check, ensuring CatBoost doesn't receive non-existent column names.
    cat_features_in_X_train = [col for col in categorical_features if col in X_train.columns]
    
    # Initialize and train CatBoost Classifier.
    # Parameters chosen for a reasonable baseline without extensive hyperparameter tuning.
    model = CatBoostClassifier(
        iterations=500,                  # Number of boosting iterations (trees)
        learning_rate=0.05,              # Step size shrinkage to prevent overfitting
        random_seed=42,                  # For reproducibility
        loss_function='Logloss',         # Logloss is standard for binary classification
        eval_metric='F1',                # Metric to optimize for and display during training
        verbose=0,                       # Suppress detailed training output
        early_stopping_rounds=50,        # Stop if validation F1 doesn't improve for 50 rounds
        # CatBoost can automatically use GPU if available and configured.
        # task_type="GPU" # Uncomment this line to enable GPU training if available
    )

    # CatBoost model training.
    # The `eval_set` is used for monitoring performance during training and for early stopping.
    try:
        model.fit(
            X_train, y_train,
            cat_features=cat_features_in_X_train, # Crucial for CatBoost to identify categorical features
            eval_set=(X_val, y_val),               # Validation set for early stopping
        )
    except Exception as e:
        print(f"Error during CatBoost model training: {e}")
        print("Final Validation Performance: 0.0")
        return

    # Make predictions on the validation set.
    y_pred = model.predict(X_val)

    # --- Macro F1 Score Calculation ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    if y_val.empty:
        print("Validation set is empty. F1 score cannot be calculated.")
    else:
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # Handle edge case: validation set has only one class.
        if len(unique_y_val) < 2:
            warnings.warn(f"Validation set target 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted.")
            # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set.
            # `zero_division=0` handles cases where a class has no true or predicted instances,
            # assigning 0 to its F1 score contribution.
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}")

# Run the training and validation process when the script is executed.
if __name__ == "__main__":
    run_training_and_validation()
