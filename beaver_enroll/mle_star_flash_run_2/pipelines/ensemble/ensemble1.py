

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features (from Python Solution 1) ---
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
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        # Merge gold_df (which now contains numerical 'HIGH_ENROLLMENT') with subject_summary_df to get features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        # Initialize the list of feature columns
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
        current_feature_cols.extend(existing_numeric_cols)

        # 2. Encode TERM_CODE and SUBJECT_ID_SORT and add them as numerical features
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        current_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Identify candidate columns for advanced feature engineering from subject_summary
        # These are generally the continuous/count-like features from subject_summary, not identifiers.
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        # Select up to 3 features from `numeric_summary_cols` for squaring to avoid feature explosion.
        poly_features_to_square = numeric_summary_cols[:3]
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        # Create interaction terms between encoded identifiers and some numerical summary features,
        # and between a pair of numerical summary features.
        
        # Interaction between encoded TERM_CODE and a numeric summary feature
        if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
            col_to_interact = numeric_summary_cols[0] 
            new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
            current_feature_cols.append(new_col_name)

        # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature (different from above if possible)
        if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
            col_to_interact = numeric_summary_cols[1] 
            new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
            current_feature_cols.append(new_col_name)
        elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1: # Fallback to first if only one
             col_to_interact = numeric_summary_cols[0] 
             new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
             df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
             current_feature_cols.append(new_col_name)

        # Interaction between two distinct numeric summary features
        if len(numeric_summary_cols) >= 2:
            col1 = numeric_summary_cols[0]
            col2 = numeric_summary_cols[1]
            new_col_name = f"{col1}_x_{col2}"
            df[new_col_name] = df[col1] * df[col2]
            current_feature_cols.append(new_col_name)

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(current_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found after all engineering, add a dummy feature.
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge or missing values in summary data.
        # This applies to all original and newly engineered features.
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

# --- Refactored function for individual model training and validation (Step 1) ---
def train_and_validate_single_model(df_processed, model_initializer, model_params, random_state):
    """
    Trains and validates a single model instance.
    Returns validation probabilities for class 1, true labels, and individual F1 score.
    """
    df = df_processed.copy()

    if df.empty:
        warnings.warn("Loaded DataFrame is empty. Cannot proceed with training.")
        return None, None, 0.0

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        warnings.warn("Not enough unique terms for a time-based validation split (at least 2 required). Returning 0.0 F1.")
        return None, None, 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

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
        warnings.warn("No usable features available for training after all fallback attempts. Returning 0.0 F1.")
        return None, None, 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        warnings.warn("Training set is empty. Cannot train a model. Returning 0.0 F1.")
        return None, None, 0.0

    if len(y_train.unique()) < 2:
        warnings.warn(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    model = model_initializer(random_state=random_state, **model_params)
    model.fit(X_train, y_train)

    # F1 Score Calculation and Probability Prediction
    individual_f1_score = 0.0
    y_pred_proba = np.array([]) # Initialize as empty numpy array for robustness

    if y_val.empty:
        warnings.warn("Validation set is empty. F1 score and probabilities cannot be calculated.")
    else:
        # Predict probabilities for the positive class (class 1)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate F1 score for the individual model using a default threshold (e.g., 0.5)
        # This F1 score is used for weighting in the ensemble, not for the final evaluation
        y_pred_binary_for_f1 = (y_pred_proba >= 0.5).astype(int)

        unique_y_val = np.unique(y_val)
        unique_y_pred_f1 = np.unique(y_pred_binary_for_f1)

        if len(unique_y_val) < 2:
            # Handle cases where the validation set contains only one class
            warnings.warn(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred_f1) == 1 and unique_y_pred_f1[0] == unique_y_val[0]:
                individual_f1_score = 1.0
            else:
                individual_f1_score = 0.0
        else:
            individual_f1_score = f1_score(y_val, y_pred_binary_for_f1, average='macro', zero_division=0)
            
    return y_pred_proba, y_val, individual_f1_score

# --- Main script for ensemble ---
if __name__ == "__main__":
    # Load data once
    df_processed = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df_processed.empty:
        print("Initial data loading resulted in an empty DataFrame. Cannot proceed with ensemble.")
        print("Final Validation Performance: 0.0")
        exit()

    # Step 2: Define Diverse Model Configurations
    # Adhering to "ensemble 1 Python Solutions", we create configurations for RandomForestClassifier.
    # We include multiple sets of hyperparameters and random states for diversity within the ensemble.
    model_configurations = [
        (RandomForestClassifier, {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'}, [42, 100, 200]),
        (RandomForestClassifier, {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 5, 'max_features': 'log2'}, [42, 300]),
        (RandomForestClassifier, {'n_estimators': 80, 'max_depth': 8, 'min_samples_leaf': 10, 'max_features': None}, [42, 400]),
    ]

    all_validation_probabilities = []
    all_individual_f1_scores = []
    true_validation_labels = None

    # Step 3: Execute Multiple Model Runs and Collect Results
    for model_class, params, random_states_list in model_configurations:
        for rs in random_states_list:
            print(f"Training {model_class.__name__} with params={params}, random_state={rs}...")
            
            y_pred_proba, y_val, f1 = train_and_validate_single_model(df_processed, model_class, params, rs)

            if y_pred_proba is not None and y_val is not None and f1 is not None:
                all_validation_probabilities.append(y_pred_proba)
                all_individual_f1_scores.append(f1)
                
                # Store true labels only once as they should be identical across runs for the same validation split
                if true_validation_labels is None:
                    true_validation_labels = y_val
                # A robust check would assert consistency: assert np.array_equal(true_validation_labels, y_val)
            else:
                print(f"Skipping model run due to issues: {model_class.__name__}, random_state={rs}. F1: {f1}")

    if not all_validation_probabilities or true_validation_labels is None:
        print("No successful model runs to form an ensemble.")
        print("Final Validation Performance: 0.0")
        exit()

    # Step 4: Apply Weighted Probability Averaging
    total_f1_score_sum = sum(all_individual_f1_scores)
    
    if total_f1_score_sum == 0:
        warnings.warn("All individual model F1 scores are zero. Using simple average of probabilities.")
        # If all F1 scores are 0, weights become effectively equal
        weighted_avg_probabilities = np.mean(all_validation_probabilities, axis=0)
    else:
        weighted_avg_probabilities = np.average(all_validation_probabilities, axis=0, weights=all_individual_f1_scores)

    # Step 5: Optimize Classification Threshold
    best_threshold = 0.5 # Default threshold
    max_ensemble_f1 = 0.0
    
    # Check if there's diversity in true labels for threshold optimization
    if len(np.unique(true_validation_labels)) > 1:
        thresholds = np.linspace(0.01, 0.99, 99) # Iterate through a fine range of thresholds
        for threshold in thresholds:
            ensemble_predictions = (weighted_avg_probabilities >= threshold).astype(int)
            current_f1 = f1_score(true_validation_labels, ensemble_predictions, average='macro', zero_division=0)
            if current_f1 > max_ensemble_f1:
                max_ensemble_f1 = current_f1
                best_threshold = threshold
    else:
        warnings.warn("True validation labels have only one class. Cannot optimize threshold. Using default 0.5.")
        # For single-class validation set, F1 is 1.0 if all predictions match, else 0.0
        ensemble_predictions = (weighted_avg_probabilities >= best_threshold).astype(int)
        unique_y_val = np.unique(true_validation_labels)
        unique_ensemble_pred = np.unique(ensemble_predictions)
        if len(unique_y_val) == 1:
            if len(unique_ensemble_pred) == 1 and unique_ensemble_pred[0] == unique_y_val[0]:
                max_ensemble_f1 = 1.0
            else:
                max_ensemble_f1 = 0.0

    # Step 6: Final Ensemble Evaluation
    final_ensemble_predictions = (weighted_avg_probabilities >= best_threshold).astype(int)
    
    # Final F1 calculation with robustness for single-class validation set
    final_validation_score = 0.0
    unique_true_val = np.unique(true_validation_labels)
    unique_final_pred = np.unique(final_ensemble_predictions)

    if len(unique_true_val) < 2:
        if len(unique_final_pred) == 1 and unique_final_pred[0] == unique_true_val[0]:
            final_validation_score = 1.0
        else:
            final_validation_score = 0.0
    else:
        final_validation_score = f1_score(true_validation_labels, final_ensemble_predictions, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}")

