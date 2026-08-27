
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

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, use_freq_encoding_term=False, include_subject_id_encoded=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters:
    - use_freq_encoding_term: If True, uses Frequency Encoding for TERM_CODE instead of Label Encoding.
    - include_subject_id_encoded: If False, excludes SUBJECT_ID_SORT_ENCODED from features.
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
        if use_freq_encoding_term:
            term_code_frequencies = df['TERM_CODE'].value_counts(normalize=True)
            df['TERM_CODE_ENCODED'] = df['TERM_CODE'].map(term_code_frequencies)
            # Fill NaN for TERM_CODE_ENCODED in case a term in gold_df is not in subject_summary_df_full for freq encoding.
            # This should generally only happen if a TERM_CODE exists in `gold_df` but not in `subject_summary_df_full`,
            # or if it's introduced in a validation set not seen during training.
            df['TERM_CODE_ENCODED'] = df['TERM_CODE_ENCODED'].fillna(0)
        else:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        current_feature_cols.append('TERM_CODE_ENCODED')

        if include_subject_id_encoded:
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
        
        # Interaction between encoded TERM_CODE and a numeric summary feature
        if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
            col_to_interact = numeric_summary_cols[0] 
            new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
            current_feature_cols.append(new_col_name)

        # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature (different from above if possible)
        if include_subject_id_encoded and 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
            col_to_interact = numeric_summary_cols[1] 
            new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
            current_feature_cols.append(new_col_name)
        elif include_subject_id_encoded and 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1: # Fallback to first if only one
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
        warnings.warn(f"{subject_summary_path} not found. Using minimal features from gold_df.")
        # Fallback to the minimal features if subject_summary.csv is not found
        
        if use_freq_encoding_term:
            term_code_frequencies = df['TERM_CODE'].value_counts(normalize=True)
            df['TERM_CODE_ENCODED'] = df['TERM_CODE'].map(term_code_frequencies).fillna(0)
        else:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        df._feature_cols = ['TERM_CODE_ENCODED'] # Start with only TERM_CODE_ENCODED
        
        if include_subject_id_encoded:
            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            df._feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Add a simple dummy numerical feature if only one or no features exist after identifier encoding
        if not df._feature_cols or (len(df._feature_cols) == 1 and 'TERM_CODE_ENCODED' in df._feature_cols and not include_subject_id_encoded):
            dummy_val = df['TERM_CODE_ENCODED'] % 5 if 'TERM_CODE_ENCODED' in df.columns else 0
            if include_subject_id_encoded and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                 dummy_val += df['SUBJECT_ID_SORT_ENCODED'] % 7
            
            df['DUMMY_FEATURE'] = dummy_val
            if 'DUMMY_FEATURE' not in df._feature_cols:
                df._feature_cols.append('DUMMY_FEATURE')
        
        # Fill potential NaNs in dummy features (not expected if created from existing data, but for robustness)
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script ---
def run_training_and_validation(rf_max_depth=None, use_freq_encoding_term=False, include_subject_id_encoded=True):
    """
    Runs the training and validation process with specified ablation parameters.
    Returns the final validation F1 score.
    """
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, 
                   use_freq_encoding_term=use_freq_encoding_term, 
                   include_subject_id_encoded=include_subject_id_encoded)

    if df.empty:
        return 0.0

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        return 0.0

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
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for empty training data
    if X_train.empty or y_train.empty:
        return 0.0

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        warnings.warn(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
        # Proceeding with training, as this is a valid (though not ideal) training scenario.
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=rf_max_depth)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        pass # score remains 0.0
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # 2. Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            warnings.warn(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set
            # zero_division=0 ensures that if a class has no true instances or no predicted instances,
            # its F1 score contribution is 0, preventing division by zero warnings/errors.
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# --- Ablation Study Execution ---
if __name__ == "__main__":
    results = {}
    print_statements = []

    # Baseline: Original configuration
    print_statements.append("Running Baseline (Original configuration)...")
    results['Baseline'] = run_training_and_validation()
    print_statements.append(f"Baseline F1 Score: {results['Baseline']:.4f}")

    # Ablation 1: Frequency Encoding for TERM_CODE instead of Label Encoding
    print_statements.append("\nRunning Ablation 1: Frequency Encoding for TERM_CODE instead of Label Encoding...")
    results['Ablation 1 (Freq Encoding for TERM_CODE)'] = run_training_and_validation(use_freq_encoding_term=True)
    print_statements.append(f"Ablation 1 F1 Score: {results['Ablation 1 (Freq Encoding for TERM_CODE)']:.4f}")

    # Ablation 2: Exclude SUBJECT_ID_SORT_ENCODED feature
    print_statements.append("\nRunning Ablation 2: Exclude SUBJECT_ID_SORT_ENCODED feature...")
    results['Ablation 2 (No SUBJECT_ID_SORT_ENCODED feature)'] = run_training_and_validation(include_subject_id_encoded=False)
    print_statements.append(f"Ablation 2 F1 Score: {results['Ablation 2 (No SUBJECT_ID_SORT_ENCODED feature)']:.4f}")

    # Ablation 3: RandomForest max_depth=1
    print_statements.append("\nRunning Ablation 3: RandomForest max_depth=1...")
    results['Ablation 3 (RF max_depth=1)'] = run_training_and_validation(rf_max_depth=1)
    print_statements.append(f"Ablation 3 F1 Score: {results['Ablation 3 (RF max_depth=1)']:.4f}")

    # Determine the most impactful part
    baseline_score = results['Baseline']
    impacts = {name: baseline_score - score for name, score in results.items() if name != 'Baseline'}

    most_impactful_message = "No single component had a clearly detrimental impact, or the dataset is too simple to show meaningful differences."
    
    # Filter for actual performance drops
    performance_drops = {name: drop for name, drop in impacts.items() if drop > 0}

    if performance_drops:
        max_impact_name = max(performance_drops, key=performance_drops.get)
        max_impact_value = performance_drops[max_impact_name]
        most_impactful_message = f"The most impactful part (causing the largest performance drop) is '{max_impact_name}' with a drop of {max_impact_value:.4f}."

    print_statements.append("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print_statements.append(f"{name}: F1 Score = {score:.4f}")

    print_statements.append(f"\n{most_impactful_message}")

    # Print all accumulated statements at the end
    for statement in print_statements:
        print(statement)
