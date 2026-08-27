

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
import itertools

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, enable_encoded_identifiers=True): # Added parameter
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    feature_cols_for_df = [] # Variable to store feature columns

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        # Merge gold_df (which now contains numerical 'HIGH_ENROLLMENT') with subject_summary_df to get features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
        current_feature_cols.extend(existing_numeric_cols)

        # 2. Encode TERM_CODE and SUBJECT_ID_SORT and add them as numerical features, IF enabled
        if enable_encoded_identifiers:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            current_feature_cols.append('TERM_CODE_ENCODED')

            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Identify candidate columns for advanced feature engineering from subject_summary
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # Determine top N numerical features.
        selected_top_numerical_features = [col for col in numeric_summary_cols if col in df.columns][:7] 

        # Determine top N encoded categorical features, only if enabled
        selected_top_categorical_encoded_features = []
        if enable_encoded_identifiers:
            if 'TERM_CODE_ENCODED' in current_feature_cols and 'TERM_CODE_ENCODED' in df.columns:
                selected_top_categorical_encoded_features.append('TERM_CODE_ENCODED')
            if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                selected_top_categorical_encoded_features.append('SUBJECT_ID_SORT_ENCODED')

        # 3. Polynomial Features (degree-2 for selected top numerical features)
        for col in selected_top_numerical_features:
            new_col_name = f"{col}_SQUARED"
            # Check if column exists in df and if it's not already created to avoid KeyError and duplicate columns
            if col in df.columns and new_col_name not in df.columns: 
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)

        # 4. Interaction Features
        # Interaction between pairs of top numerical features
        if len(selected_top_numerical_features) >= 2:
            for col1, col2 in itertools.combinations(selected_top_numerical_features, 2):
                new_col_name = f"{col1}_x_{col2}"
                if col1 in df.columns and col2 in df.columns and new_col_name not in df.columns:
                    df[new_col_name] = df[col1] * df[col2]
                    current_feature_cols.append(new_col_name)

        # Interaction between top numerical and top encoded categorical features
        if selected_top_numerical_features and selected_top_categorical_encoded_features:
            for num_col in selected_top_numerical_features:
                for cat_col in selected_top_categorical_encoded_features:
                    new_col_name = f"{num_col}_x_{cat_col}"
                    if num_col in df.columns and cat_col in df.columns and new_col_name not in df.columns:
                        df[new_col_name] = df[num_col] * df[cat_col]
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
        
        feature_cols_for_df = final_feature_cols # Assign to the common variable

    except FileNotFoundError:
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        # Fallback to the minimal features if subject_summary.csv is not found
        if enable_encoded_identifiers: # Original fallback behavior
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            
            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            
            df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
            
            feature_cols_for_df = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        else: # New fallback for when encoded identifiers are disabled
            df['DUMMY_FEATURE'] = 0 # Simple dummy feature
            feature_cols_for_df = ['DUMMY_FEATURE']
        
        df[feature_cols_for_df] = df[feature_cols_for_df].fillna(0)

    # Store feature columns for later use, after the try-except block finishes
    df._feature_cols = feature_cols_for_df
    return df

# --- F1 Score Calculation Helper ---
def calculate_f1(y_true, y_pred):
    """
    Calculates F1 score with robustness checks for empty or single-class validation sets.
    """
    if y_true.empty:
        return 0.0 # No validation data

    unique_y_true = np.unique(y_true)
    unique_y_pred = np.unique(y_pred)

    if len(unique_y_true) < 2:
        # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
        if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_true[0]:
            return 1.0
        else:
            return 0.0
    else:
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

# --- Main training and validation function ---
def evaluate_model(enable_encoded_identifiers=True, rf_n_estimators=100):
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, enable_encoded_identifiers=enable_encoded_identifiers)

    if df.empty:
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
    # Fallback logic for feature_cols if not correctly set by load_data, consistent with original
    if not feature_cols: 
        warnings.warn("Feature columns not correctly identified by load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            # This specific fallback needs to respect enable_encoded_identifiers
            if enable_encoded_identifiers and 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
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

    if X_train.empty or y_train.empty:
        return 0.0

    # Ensure all feature columns exist in X_val. If not, add missing columns with 0
    missing_cols_in_val = set(X_train.columns) - set(X_val.columns)
    for c in missing_cols_in_val:
        X_val[c] = 0
    # Ensure column order is the same
    X_val = X_val[X_train.columns]

    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=rf_n_estimators)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    final_validation_score = calculate_f1(y_val, y_pred)
    return final_validation_score

# --- Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Baseline
    baseline_score = evaluate_model(enable_encoded_identifiers=True, rf_n_estimators=100)
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}")

    # Ablation 1: No Encoded Identifier Features
    ablation1_score = evaluate_model(enable_encoded_identifiers=False, rf_n_estimators=100)
    results['Ablation 1: No Encoded Identifier Features'] = ablation1_score
    print(f"Ablation 1: No Encoded Identifier Features F1 Score: {ablation1_score:.4f}")

    # Ablation 2: Simplified Model (RF n_estimators=1)
    ablation2_score = evaluate_model(enable_encoded_identifiers=True, rf_n_estimators=1)
    results['Ablation 2: Simplified Model (RF n_estimators=1)'] = ablation2_score
    print(f"Ablation 2: Simplified Model (RF n_estimators=1) F1 Score: {ablation2_score:.4f}")

    # Determine the most impactful part
    most_impactful_part = "None of the specific ablated parts showed a significant performance drop."
    max_drop = 0.0

    for name, score in results.items():
        if name != 'Baseline':
            drop = baseline_score - score
            if drop > max_drop:
                max_drop = drop
                most_impactful_part = name

    if max_drop > 0:
        print(f"The part of the code that contributes the most to the overall performance is: {most_impactful_part}")
    else:
        print(f"The part of the code that contributes the most to the overall performance is: {most_impactful_part}")

