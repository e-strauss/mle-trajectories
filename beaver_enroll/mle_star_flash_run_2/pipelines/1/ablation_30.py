
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
def load_data(data_dir, gold_file,
              ablation_no_poly_features=False,
              ablation_no_interaction_features=False,
              ablation_fillna_strategy='zero'): # 'zero' or 'median'

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
        if not ablation_no_poly_features:
            # Select up to 3 features from `numeric_summary_cols` for squaring to avoid feature explosion.
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        if not ablation_no_interaction_features:
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
        if ablation_fillna_strategy == 'zero':
            df[final_feature_cols] = df[final_feature_cols].fillna(0)
        elif ablation_fillna_strategy == 'median':
            for col in final_feature_cols:
                # Only try to fill if the column exists and has NaNs
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    # Fallback to 0 if median is NaN (e.g., column is entirely NaN or empty)
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

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
        if ablation_fillna_strategy == 'zero':
             df[df._feature_cols] = df[df._feature_cols].fillna(0)
        elif ablation_fillna_strategy == 'median':
            for col in df._feature_cols:
                if col in df.columns and df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    return df

# --- Main script ---
def run_training_and_validation(ablation_no_poly_features=False,
                                ablation_no_interaction_features=False,
                                ablation_fillna_strategy='zero'):
    
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                   ablation_no_poly_features,
                   ablation_no_interaction_features,
                   ablation_fillna_strategy)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0 # Return 0 for consistency

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
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
        print("No usable features available for training after all fallback attempts.")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for empty training data
    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        return 0.0

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
        # Proceeding with training, as this is a valid (though not ideal) training scenario.
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        print("Validation set is empty. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

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
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# Ablation study execution
if __name__ == "__main__":
    
    print("Running Ablation Study...")
    
    results = {}

    # Baseline
    print("--- Running Baseline (Default: Zero Imputation, Poly Features, Interaction Features) ---")
    baseline_score = run_training_and_validation()
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}\n")

    # Ablation 1: Change NaN Imputation to Median
    print("--- Running Ablation: NaN Imputation to Median ---")
    ablation1_score = run_training_and_validation(ablation_fillna_strategy='median')
    results['NaN Imputation to Median'] = ablation1_score
    print(f"Ablation (NaN Imputation to Median) F1 Score: {ablation1_score:.4f}\n")

    # Ablation 2: No Polynomial Features
    print("--- Running Ablation: No Polynomial Features ---")
    ablation2_score = run_training_and_validation(ablation_no_poly_features=True)
    results['No Polynomial Features'] = ablation2_score
    print(f"Ablation (No Polynomial Features) F1 Score: {ablation2_score:.4f}\n")

    # Ablation 3: No Interaction Features
    print("--- Running Ablation: No Interaction Features ---")
    ablation3_score = run_training_and_validation(ablation_no_interaction_features=True)
    results['No Interaction Features'] = ablation3_score
    print(f"Ablation (No Interaction Features) F1 Score: {ablation3_score:.4f}\n")

    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")
    
    # Determine the most impactful part from the current study
    most_impactful_current = "None of the tested ablations caused a performance drop in this study, or the dataset is too simple to show meaningful differences from the baseline."
    largest_drop_current = 0.0

    if baseline_score != 0.0: # Only compare if baseline is not 0
        for name, score in results.items():
            if name == 'Baseline':
                continue
            drop = baseline_score - score
            if drop > largest_drop_current:
                largest_drop_current = drop
                most_impactful_current = f"'{name}'"
            elif drop > 0 and drop == largest_drop_current:
                 # If multiple ablations have the same largest drop, list them
                 if not f"'{name}'" in most_impactful_current: # avoid duplicates if already added for first case
                     most_impactful_current += f" and '{name}'"
    
    if largest_drop_current > 0:
        print(f"\nConclusion from current study: The most impactful part(s) identified in this specific ablation study (causing the largest F1 score drop of {largest_drop_current:.4f}) is/are: {most_impactful_current}.")
    else:
        print(f"\nConclusion from current study: {most_impactful_current}")


    # Incorporating insights from ALL previous ablation studies to find the truly most impactful part
    all_impactful_components_known = {
        "Existence of subject_summary.csv features": 0.3333, # F1 drops from 1.0 to 0.6667 (Studies 3, 13, 17)
        "RandomForest min_samples_split": 0.6, # F1 drops from 1.0 to 0.4 (Study 8)
        "RandomForest min_samples_leaf": 0.6, # F1 drops from 1.0 to 0.4 (Study 11)
        "Validation Strategy (Random vs Time-based)": 0.6667, # F1 drops from 1.0 to 0.3333 (Study 21)
        "Validation Strategy (K=2 terms vs 1)": 0.5143, # F1 drops from 1.0 to 0.4857 (Study 10)
        "RandomForest n_estimators (reduced to 1)": 0.3333, # F1 drops from 1.0 to 0.6667 (Study 18)
        "RandomForest max_depth (reduced to 1)": 0.3333, # F1 drops from 1.0 to 0.6667 (Study 24)
        "StandardScaler before PCA": 0.75, # F1 drops from 1.0 to 0.25 (Study 27)
        "Interaction Features (when baseline is low)": 0.0924, # F1 drops from 0.3651 to 0.2727 (Study 30)
        "Disabling Similarity-Based Term Exclusion (improved performance)": -0.75, # F1 improves from 0.25 to 1.0 (Study 26)
        "Changing NaN Imputation to Mean (when baseline is low, improved performance)": -0.75 # F1 improves from 0.25 to 1.0 (Study 26)
    }

    overall_largest_drop_value = largest_drop_current
    overall_most_impactful_component_name = most_impactful_current
    
    for comp, drop_val in all_impactful_components_known.items():
        # Only consider drops, not improvements, for "most impactful" in terms of contribution
        if drop_val > 0:
            if drop_val > overall_largest_drop_value:
                overall_largest_drop_value = drop_val
                overall_most_impactful_component_name = comp
            elif drop_val > 0 and drop_val == overall_largest_drop_value:
                if comp not in overall_most_impactful_component_name:
                    overall_most_impactful_component_name += f" and '{comp}'"

    if overall_largest_drop_value > 0:
        print(f"\nConsidering all previous ablation studies and the current one, the part(s) of the code that contribute the most to the overall performance (causing the largest observed F1 score drop) is/are: {overall_most_impactful_component_name} with an F1 score drop of {overall_largest_drop_value:.4f}.")
    else:
        print("\nConsidering all previous ablation studies and the current one, it appears that for the current experimental setup and dataset, no single component consistently causes a significant performance drop, or the dataset is too simple for differentiation, resulting in consistently high (often perfect) F1 scores across many configurations.")
