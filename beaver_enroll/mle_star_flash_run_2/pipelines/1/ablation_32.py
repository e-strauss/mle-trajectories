
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

# --- Re-implementation of load_data for ablation study ---
# This function will be called with flags to modify its behavior
def load_data_for_ablation(
    data_dir, 
    gold_file,
    ablation_no_existing_numeric_cols=False, 
    ablation_no_polynomial_features=False,
    ablation_no_id_interaction_features=False
):
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        current_feature_cols = []

        # Ablation 1: No existing numerical columns (raw features from subject_summary)
        if not ablation_no_existing_numeric_cols:
            existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
            current_feature_cols.extend(existing_numeric_cols)

        # Encode TERM_CODE and SUBJECT_ID_SORT and add them as numerical features
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        current_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # Ablation 2: No Polynomial Features
        if not ablation_no_polynomial_features:
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
        
        # Ablation 3: No ID-based Interaction Features
        # These are interactions between encoded TERM_CODE/SUBJECT_ID_SORT and numeric summary features
        if not ablation_no_id_interaction_features:
            if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
                col_to_interact = numeric_summary_cols[0] 
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

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

        # Interaction between two distinct numeric summary features (always included as it's not 'ID-based')
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
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        
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

# --- Main ablation study function ---
def run_ablation_scenario(
    data_dir, 
    gold_file, 
    ablation_name,
    ablation_no_existing_numeric_cols=False, 
    ablation_no_polynomial_features=False,
    ablation_no_id_interaction_features=False
):
    df = load_data_for_ablation(
        data_dir, 
        gold_file, 
        ablation_no_existing_numeric_cols, 
        ablation_no_polynomial_features,
        ablation_no_id_interaction_features
    )

    if df.empty:
        print(f"Loaded DataFrame is empty for '{ablation_name}'. Cannot proceed with training.")
        return 0.0

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        print(f"Not enough unique terms for a time-based validation split in '{ablation_name}'.")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    # Retrieve feature columns determined during data loading
    feature_cols = getattr(df, '_feature_cols', [])
    
    # Fallback to identify feature columns if not properly set (should not happen with robust load_data)
    if not feature_cols: 
        warnings.warn(f"Feature columns not correctly identified by load_data for '{ablation_name}'. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print(f"No usable features available for training in '{ablation_name}' after all fallback attempts.")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for empty training data
    if X_train.empty or y_train.empty:
        print(f"Training set is empty for '{ablation_name}'. Cannot train a model.")
        return 0.0

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        warnings.warn(f"Training set target 'HIGH_ENROLLMENT' has only one class in '{ablation_name}': {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        print(f"Validation set is empty for '{ablation_name}'. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # 2. Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            warnings.warn(f"Validation set 'HIGH_ENROLLMENT' has only one class in '{ablation_name}': {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
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

# --- Main ablation study execution ---
if __name__ == "__main__":
    results = {}

    # Baseline
    baseline_name = "Baseline (Full Feature Engineering)"
    print(f"Running {baseline_name}")
    baseline_score = run_ablation_scenario(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, baseline_name)
    results[baseline_name] = baseline_score
    print(f"{baseline_name} F1 Score: {baseline_score:.4f}\n")

    # Ablation 1: No existing numerical columns (raw features from subject_summary)
    ablation1_name = "Ablation 1 (No Existing Numerical Columns)"
    print(f"Running {ablation1_name}")
    ablation1_score = run_ablation_scenario(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, ablation1_name,
                                            ablation_no_existing_numeric_cols=True)
    results[ablation1_name] = ablation1_score
    print(f"{ablation1_name} F1 Score: {ablation1_score:.4f}\n")

    # Ablation 2: No Polynomial Features
    ablation2_name = "Ablation 2 (No Polynomial Features)"
    print(f"Running {ablation2_name}")
    ablation2_score = run_ablation_scenario(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, ablation2_name,
                                            ablation_no_polynomial_features=True)
    results[ablation2_name] = ablation2_score
    print(f"{ablation2_name} F1 Score: {ablation2_score:.4f}\n")

    # Ablation 3: No ID-based Interaction Features
    ablation3_name = "Ablation 3 (No ID-based Interaction Features)"
    print(f"Running {ablation3_name}")
    ablation3_score = run_ablation_scenario(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, ablation3_name,
                                            ablation_no_id_interaction_features=True)
    results[ablation3_name] = ablation3_score
    print(f"{ablation3_name} F1 Score: {ablation3_score:.4f}\n")

    # Determine most impactful part
    print("\n--- Ablation Study Summary ---")
    most_impactful_part = "None of the ablated components caused a performance drop."
    largest_drop = 0.0

    for name, score in results.items():
        if name == baseline_name:
            continue
        drop = baseline_score - score
        print(f"{name}: F1 Score = {score:.4f} (Drop from Baseline: {drop:.4f})")
        if drop > largest_drop:
            largest_drop = drop
            most_impactful_part = name

    print(f"\nConclusion: The part of the code that contributes the most to the overall performance is: {most_impactful_part} (Largest F1 Score drop: {largest_drop:.4f} from Baseline F1 Score: {baseline_score:.4f}).")
