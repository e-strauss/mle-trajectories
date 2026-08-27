

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
# This version is made flexible to support ablation of merge strategy
def load_data_ablatable(data_dir, gold_file, merge_how='left'):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation point 1: `merge_how` parameter to change DataFrame merge strategy.
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
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how=merge_how) # Ablation point 1
        
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

# --- Main training and validation process as a function for ablation ---
def run_training_and_validation_ablatable(
    merge_how='left',                       # Ablation 1 parameter: DataFrame merge strategy
    term_sort_ascending=True,               # Ablation 2 parameter: TERM_CODE sorting order
    validation_term_selection='latest',     # Ablation 2 parameter: Which unique term to select for validation
    rf_min_impurity_decrease=0.0            # Ablation 3 parameter: RandomForest min_impurity_decrease
):
    df = load_data_ablatable(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, merge_how=merge_how)

    if df.empty:
        # print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0 # Return 0 for empty DF

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    
    # Ablation point 2: Sorting strategy for time-based split
    df = df.sort_values('TERM_CODE', ascending=term_sort_ascending).reset_index(drop=True)

    # Determine validation set: based on sorted unique_terms
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        # print("Not enough unique terms for a time-based validation split (at least 2 required).")
        return 0.0

    # Ablation point 2 (cont.): Choose validation term based on selection strategy
    if validation_term_selection == 'latest':
        validation_term = unique_terms[-1] 
    elif validation_term_selection == 'earliest':
        validation_term = unique_terms[0]
    else: # Default to latest for any other invalid choice
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
        # print("No usable features available for training after all fallback attempts.")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for empty training data
    if X_train.empty or y_train.empty:
        # print("Training set is empty. Cannot train a model.")
        return 0.0

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        # print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
        pass # Proceeding with training, as this is a valid (though not ideal) training scenario.
    
    # Model Training
    # Ablation point 3: min_impurity_decrease hyperparameter
    model = RandomForestClassifier(random_state=42, n_estimators=100, min_impurity_decrease=rf_min_impurity_decrease)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        # print("Validation set is empty. F1 score cannot be calculated.")
        pass
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # 2. Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            # print(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
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

    # Baseline Scenario
    print("Running Baseline Scenario (Original configuration)...")
    baseline_score = run_training_and_validation_ablatable()
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}\n")

    # Ablation 1: Change Merge Strategy (how='left' to how='inner')
    print("Running Ablation 1: Change DataFrame merge strategy from 'left' to 'inner'...")
    ablation1_score = run_training_and_validation_ablatable(merge_how='inner')
    results['Ablation 1 (Merge how=inner)'] = ablation1_score
    print(f"Ablation 1 F1 Score: {ablation1_score:.4f}\n")

    # Ablation 2: Change Validation Strategy to use the Earliest TERM_CODE
    # By sorting ascending and selecting unique_terms[0], we effectively predict the earliest term.
    print("Running Ablation 2: Validate on the EARLIEST TERM_CODE instead of the latest...")
    ablation2_score = run_training_and_validation_ablatable(validation_term_selection='earliest')
    results['Ablation 2 (Validate on Earliest Term)'] = ablation2_score
    print(f"Ablation 2 F1 Score: {ablation2_score:.4f}\n")

    # Ablation 3: Modify RandomForestClassifier's min_impurity_decrease hyperparameter
    print("Running Ablation 3: RandomForestClassifier with min_impurity_decrease=0.01 (vs default 0.0)...")
    ablation3_score = run_training_and_validation_ablatable(rf_min_impurity_decrease=0.01)
    results['Ablation 3 (RF min_impurity_decrease=0.01)'] = ablation3_score
    print(f"Ablation 3 F1 Score: {ablation3_score:.4f}\n")

    # --- Determine the most impactful component ---
    
    # Calculate performance drops from baseline for components that degraded performance
    performance_changes = {
        name: baseline_score - score
        for name, score in results.items() if name != 'Baseline'
    }

    if not performance_changes:
        print("\nNo performance changes observed in ablations compared to baseline.")
    else:
        # Find the ablation with the largest performance drop
        largest_drop_name = None
        largest_drop_value = 0.0

        # Find the ablation with the largest performance gain
        largest_gain_name = None
        largest_gain_value = 0.0

        for name, drop_value in performance_changes.items():
            if drop_value > largest_drop_value:
                largest_drop_value = drop_value
                largest_drop_name = name
            if -drop_value > largest_gain_value: # -drop_value is the actual gain
                largest_gain_value = -drop_value
                largest_gain_name = name

        if largest_drop_value > 0:
            print(f"\nThe most impactful part of the code, causing the largest performance drop, is: '{largest_drop_name}'")
            print(f"It resulted in an F1 score drop of {largest_drop_value:.4f} (from {baseline_score:.4f} to {results[largest_drop_name]:.4f}).")
        elif largest_gain_value > 0:
            print(f"\nThe most impactful part of the code, causing the largest performance gain, is: '{largest_gain_name}'")
            print(f"It resulted in an F1 score gain of {largest_gain_value:.4f} (from {baseline_score:.4f} to {results[largest_gain_name]:.4f}).")
        else:
            print("\nAll ablations resulted in the same performance as the baseline. The dataset might be too simple or the ablated parts are not critical.")

