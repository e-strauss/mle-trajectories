
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
from sklearn.model_selection import KFold

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Refactored Function to load data and engineer features for ablation ---
def load_data_ablation(data_dir, gold_file, encoding_strategy='target_encoding'):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Accepts an encoding_strategy for 'TERM_CODE' and 'SUBJECT_ID_SORT'.
    Returns DataFrame with feature columns identified, but NaNs in features are not filled here,
    unless in the fallback scenario.
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

        # --- Encoding for TERM_CODE and SUBJECT_ID_SORT ---
        if encoding_strategy == 'target_encoding':
            target_col = 'HIGH_ENROLLMENT'
            # global_mean is computed on the entire input df. For cleaner target encoding,
            # this would ideally be done only on training data after the time-based split.
            # For this ablation study, we maintain consistency with original intent for OOF encoding.
            global_mean = df[target_col].mean()
            NFOLDS = 5
            kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

            def apply_oof_target_encoding(dataframe, column_name, target_name, kf_splitter, global_mean_val):
                new_encoded_col_name = f'{column_name}_ENCODED'
                dataframe[new_encoded_col_name] = np.nan
                # Only iterate if there are valid (non-NaN) target values for encoding
                if dataframe[target_name].notna().any():
                    for trn_idx, val_idx in kf_splitter.split(dataframe):
                        # Calculate target mean on training fold, ignoring NaNs in the target
                        # Ensure trn_idx has enough data and target is not all NaN
                        if not dataframe.loc[trn_idx, target_name].dropna().empty:
                            encoder_map = dataframe.loc[trn_idx].groupby(column_name)[target_name].mean()
                            # Apply mapping to validation fold, handling potential new categories
                            dataframe.loc[val_idx, new_encoded_col_name] = dataframe.loc[val_idx, column_name].map(encoder_map)
                # Fill any remaining NaNs (e.g., categories that only appeared in a validation fold,
                # or rows where target was NaN from the start, or new categories in test data) with global mean
                dataframe[new_encoded_col_name].fillna(global_mean_val, inplace=True)
                return new_encoded_col_name

            encoded_col_term = apply_oof_target_encoding(df, 'TERM_CODE', target_col, kf, global_mean)
            current_feature_cols.append(encoded_col_term)
            encoded_col_subject = apply_oof_target_encoding(df, 'SUBJECT_ID_SORT', target_col, kf, global_mean)
            current_feature_cols.append(encoded_col_subject)
        
        elif encoding_strategy == 'label_encoding':
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            current_feature_cols.append('TERM_CODE_ENCODED')

            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')
        
        else:
            raise ValueError(f"Unknown encoding_strategy: {encoding_strategy}")

        # Identify candidate columns for advanced feature engineering from subject_summary
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = [col for col in summary_cols_added if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        # --- Advanced Feature Engineering ---
        poly_features_to_square = numeric_summary_cols[:3]
        for col in poly_features_to_square:
            if col in df.columns:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
        
        if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
            col_to_interact = numeric_summary_cols[0]
            if col_to_interact in df.columns:
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

        if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
            col_to_interact = numeric_summary_cols[1]
            if col_to_interact in df.columns:
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)
        elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1:
             col_to_interact = numeric_summary_cols[0]
             if col_to_interact in df.columns:
                 new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                 df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                 current_feature_cols.append(new_col_name)

        if len(numeric_summary_cols) >= 2:
            col1 = numeric_summary_cols[0]
            col2 = numeric_summary_cols[1]
            if col1 in df.columns and col2 in df.columns:
                new_col_name = f"{col1}_x_{col2}"
                df[new_col_name] = df[col1] * df[col2]
                current_feature_cols.append(new_col_name)

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(current_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Store feature columns for later use; NaNs will be imputed after train/val split
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        # Fallback will always use LabelEncoder and fill NaNs with 0 as it's a minimal, self-contained path.
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0) # Impute here for fallback path

    return df

# --- Refactored Main script for ablation study ---
def run_ablation_scenario(encoding_strategy='target_encoding', imputation_strategy='zero', rf_max_features='sqrt'):
    """
    Runs a single training and validation scenario with specified ablation parameters.
    Returns the F1 score for the validation set.
    """
    df = load_data_ablation(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, encoding_strategy)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term].copy() # .copy() to avoid SettingWithCopyWarning
    val_df = df[df['TERM_CODE'] == validation_term].copy()   # .copy() to avoid SettingWithCopyWarning

    feature_cols = getattr(df, '_feature_cols', [])
    
    # Fallback for feature_cols if not correctly set by load_data_ablation (should not happen normally)
    if not feature_cols: 
        warnings.warn("Feature columns not correctly identified. Attempting dynamic identification as fallback.")
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

    # --- Imputation Strategy ---
    if imputation_strategy == 'zero':
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
    elif imputation_strategy == 'mean':
        train_means = X_train.mean()
        X_train = X_train.fillna(train_means)
        X_val = X_val.fillna(train_means)
    else:
        raise ValueError(f"Unknown imputation_strategy: {imputation_strategy}")

    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    # max_features='sqrt' is the default for RandomForestClassifier for classification.
    # Passing 1.0 (or None) means considering all features for split.
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_features=rf_max_features)
    model.fit(X_train, y_train)

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

# --- Main ablation study execution ---
if __name__ == "__main__":
    results = {}

    # Baseline: Current best configuration (Target Encoding, Zero Imputation, RF default max_features)
    print("--- Running Baseline ---")
    baseline_score = run_ablation_scenario(
        encoding_strategy='target_encoding', 
        imputation_strategy='zero', 
        rf_max_features='sqrt'
    )
    results['Baseline (Target Encoding, Zero Imputation, RF default max_features)'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score}\n")

    # Ablation 1: Encoding Strategy (Revert to Label Encoding for Identifiers)
    print("--- Running Ablation: Label Encoding for Identifiers ---")
    ablation1_score = run_ablation_scenario(
        encoding_strategy='label_encoding', 
        imputation_strategy='zero', 
        rf_max_features='sqrt'
    )
    results['Ablation 1 (Label Encoding, Zero Imputation, RF default max_features)'] = ablation1_score
    print(f"Ablation 1 F1 Score (Label Encoding): {ablation1_score}\n")

    # Ablation 2: NaN Imputation Strategy (Change to Mean Imputation)
    print("--- Running Ablation: Mean Imputation for NaNs ---")
    ablation2_score = run_ablation_scenario(
        encoding_strategy='target_encoding', 
        imputation_strategy='mean', 
        rf_max_features='sqrt'
    )
    results['Ablation 2 (Target Encoding, Mean Imputation, RF default max_features)'] = ablation2_score
    print(f"Ablation 2 F1 Score (Mean Imputation): {ablation2_score}\n")

    # Ablation 3: RandomForest max_features parameter (Use all features instead of default sqrt)
    print("--- Running Ablation: RF max_features=1.0 (All features) ---")
    ablation3_score = run_ablation_scenario(
        encoding_strategy='target_encoding', 
        imputation_strategy='zero', 
        rf_max_features=1.0 # Equivalent to None for classification, considers all features
    )
    results['Ablation 3 (Target Encoding, Zero Imputation, RF max_features=1.0)'] = ablation3_score
    print(f"Ablation 3 F1 Score (RF max_features=1.0): {ablation3_score}\n")

    # Determine most impactful component
    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score}")

    baseline = results['Baseline (Target Encoding, Zero Imputation, RF default max_features)']
    
    impacts = {}
    if 'Ablation 1 (Label Encoding, Zero Imputation, RF default max_features)' in results:
        impacts['Encoding Strategy (from Target Encoding to Label Encoding)'] = abs(baseline - results['Ablation 1 (Label Encoding, Zero Imputation, RF default max_features)'])
    if 'Ablation 2 (Target Encoding, Mean Imputation, RF default max_features)' in results:
        impacts['Imputation Strategy (from Zero to Mean Imputation)'] = abs(baseline - results['Ablation 2 (Target Encoding, Mean Imputation, RF default max_features)'])
    if 'Ablation 3 (Target Encoding, Zero Imputation, RF max_features=1.0)' in results:
        impacts['RandomForest max_features (from default sqrt to 1.0)'] = abs(baseline - results['Ablation 3 (Target Encoding, Zero Imputation, RF max_features=1.0)'])

    if impacts:
        most_impactful = max(impacts, key=impacts.get)
        print(f"\nThe part of the code that contributes the most to the overall performance is: {most_impactful} (Change in F1 from baseline: {impacts[most_impactful]:.4f})")
    else:
        print("\nCould not determine the most impactful component from the ablations performed.")
