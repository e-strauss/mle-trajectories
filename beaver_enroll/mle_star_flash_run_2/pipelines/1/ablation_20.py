
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features (modified for ablation) ---
def load_data_for_ablation(data_dir, gold_file, include_advanced_fe=True, nan_imputation_value=0):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters:
    - include_advanced_fe: If False, skips polynomial and interaction feature engineering.
    - nan_imputation_value: Value to use for filling NaNs in feature columns.
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
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering (Ablation Point 1) ---
        if include_advanced_fe:
            # 3. Polynomial Features (e.g., squared terms for key numerical features)
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
            
            # 4. Interaction Features (product of two distinct features)
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
            elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1:
                col_to_interact = numeric_summary_cols[0] 
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)

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
        # This applies to all original and newly engineered features. (Ablation Point 2)
        df[final_feature_cols] = df[final_feature_cols].fillna(nan_imputation_value)
        
        # Store feature columns for later use
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        # Fallback to the minimal features if subject_summary.csv is not found
        
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(nan_imputation_value)

    return df

# --- Main script (modified for ablation) ---
def run_training_and_validation_for_ablation(include_advanced_fe=True, nan_imputation_value=0, split_strategy='time_based'):
    """
    Runs the training and validation process with ablation parameters.
    - include_advanced_fe: Passed to load_data_for_ablation.
    - nan_imputation_value: Passed to load_data_for_ablation.
    - split_strategy: 'time_based' for original split, 'random' for train_test_split. (Ablation Point 3)
    """
    df = load_data_for_ablation(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                                 include_advanced_fe=include_advanced_fe,
                                 nan_imputation_value=nan_imputation_value)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

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

    # --- Data Splitting ---
    if split_strategy == 'time_based':
        # Ensure TERM_CODE is treated as string for consistent sorting behavior
        df['TERM_CODE'] = df['TERM_CODE'].astype(str)
        df = df.sort_values('TERM_CODE').reset_index(drop=True)

        unique_terms = df['TERM_CODE'].unique()
        
        if len(unique_terms) < 2:
            print("Not enough unique terms for a time-based validation split (at least 2 required).")
            return 0.0

        validation_term = unique_terms[-1] 
        train_df = df[df['TERM_CODE'] != validation_term]
        val_df = df[df['TERM_CODE'] == validation_term]
    
    elif split_strategy == 'random':
        X = df[feature_cols]
        y = df['HIGH_ENROLLMENT']
        # Using a fixed random_state for reproducibility
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Reconstruct train_df and val_df to maintain consistency with the original script's flow
        # This is more robust than using just X_train/y_train directly, especially for potential future modifications
        train_indices = X_train.index
        val_indices = X_val.index
        train_df = df.loc[train_indices]
        val_df = df.loc[val_indices]
    else:
        raise ValueError("Invalid split_strategy. Must be 'time_based' or 'random'.")

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 
    
    if y_val.empty:
        print("Validation set is empty. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            print(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# --- Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    print("--- Running Baseline (Original Configuration) ---")
    baseline_score = run_training_and_validation_for_ablation(
        include_advanced_fe=True, 
        nan_imputation_value=0, 
        split_strategy='time_based'
    )
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score}\n")

    print("--- Running Ablation 1: No Advanced Feature Engineering ---")
    ablation1_score = run_training_and_validation_for_ablation(
        include_advanced_fe=False, 
        nan_imputation_value=0, 
        split_strategy='time_based'
    )
    results['Ablation 1 (No Advanced FE)'] = ablation1_score
    print(f"Ablation 1 F1 Score: {ablation1_score}\n")

    print("--- Running Ablation 2: NaN Imputation with -999 ---")
    ablation2_score = run_training_and_validation_for_ablation(
        include_advanced_fe=True, 
        nan_imputation_value=-999, 
        split_strategy='time_based'
    )
    results['Ablation 2 (NaN Impute -999)'] = ablation2_score
    print(f"Ablation 2 F1 Score: {ablation2_score}\n")

    print("--- Running Ablation 3: Random Train-Validation Split ---")
    ablation3_score = run_training_and_validation_for_ablation(
        include_advanced_fe=True, 
        nan_imputation_value=0, 
        split_strategy='random'
    )
    results['Ablation 3 (Random Split)'] = ablation3_score
    print(f"Ablation 3 F1 Score: {ablation3_score}\n")

    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score}")

    baseline_f1 = results['Baseline']
    impacts = {}
    for name, score in results.items():
        if name != 'Baseline':
            impacts[name] = abs(baseline_f1 - score)
    
    if not impacts or max(impacts.values()) == 0:
        print("\nConclusion: All ablations performed identically to the baseline (or resulted in 0.0 impact). This suggests that the dataset might be too simple, or these specific components are not critical for achieving optimal performance under the current conditions.")
        print("No single part among the ablated ones contributes 'the most' based on this study's results.")
    else:
        most_impactful_ablation_name = max(impacts, key=impacts.get)
        max_impact_value = impacts[most_impactful_ablation_name]

        impact_description = ""
        if most_impactful_ablation_name == 'Ablation 1 (No Advanced FE)':
            impact_description = "the inclusion of Advanced Feature Engineering (Polynomial and Interaction Features)"
        elif most_impactful_ablation_name == 'Ablation 2 (NaN Impute -999)':
            impact_description = "the choice of NaN imputation value (specifically using 0 instead of -999)"
        elif most_impactful_ablation_name == 'Ablation 3 (Random Split)':
            impact_description = "the use of a time-based validation split (as opposed to a random split)"
        
        print(f"\nConclusion: The part of the code that contributes the most to the overall performance, based on this ablation study, is {impact_description}, which resulted in a performance change of {max_impact_value:.4f} compared to the baseline.")
