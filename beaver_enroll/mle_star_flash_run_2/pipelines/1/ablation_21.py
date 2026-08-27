
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import os
import warnings
from category_encoders import TargetEncoder 

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, term_code_encoding='ohe', subject_id_encoding='target'):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    
    Args:
        data_dir (str): Directory containing subject_summary.csv.
        gold_file (str): Path to gold_enrollment_train.csv.
        term_code_encoding (str): 'ohe' for OneHotEncoder, 'label' for LabelEncoder.
        subject_id_encoding (str): 'target' for TargetEncoder, 'label' for LabelEncoder.
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

        # Handle TERM_CODE encoding
        term_encoded_feature_for_interaction = None
        if term_code_encoding == 'ohe':
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            term_encoded_array = ohe.fit_transform(df[['TERM_CODE']])
            term_feature_names = ohe.get_feature_names_out(['TERM_CODE'])
            df_term_ohe = pd.DataFrame(term_encoded_array, columns=term_feature_names, index=df.index)
            df = pd.concat([df, df_term_ohe], axis=1)
            current_feature_cols.extend(term_feature_names)
            # For interaction, pick the first OHE feature if available
            if term_feature_names.size > 0:
                term_encoded_feature_for_interaction = term_feature_names[0]
        elif term_code_encoding == 'label':
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            current_feature_cols.append('TERM_CODE_ENCODED')
            term_encoded_feature_for_interaction = 'TERM_CODE_ENCODED'
        else:
            raise ValueError("Invalid term_code_encoding specified.")

        # Handle SUBJECT_ID_SORT encoding
        subject_encoded_feature_for_interaction = None
        if subject_id_encoding == 'target':
            te = TargetEncoder(cols=['SUBJECT_ID_SORT'])
            df['SUBJECT_ID_SORT_TARGET_ENCODED'] = te.fit_transform(df[['SUBJECT_ID_SORT']], df['HIGH_ENROLLMENT'])
            current_feature_cols.append('SUBJECT_ID_SORT_TARGET_ENCODED')
            subject_encoded_feature_for_interaction = 'SUBJECT_ID_SORT_TARGET_ENCODED'
        elif subject_id_encoding == 'label':
            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')
            subject_encoded_feature_for_interaction = 'SUBJECT_ID_SORT_ENCODED'
        else:
            raise ValueError("Invalid subject_id_encoding specified.")

        # Identify candidate columns for advanced feature engineering from subject_summary
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        poly_features_to_square = numeric_summary_cols[:3] # Original behavior
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        
        # Interaction between encoded TERM_CODE and a numeric summary feature
        if term_encoded_feature_for_interaction and len(numeric_summary_cols) >= 1:
            col_to_interact = numeric_summary_cols[0] 
            new_col_name = f"{col_to_interact}_x_TERM_INTERACTION" # Generalized name
            df[new_col_name] = df[col_to_interact] * df[term_encoded_feature_for_interaction]
            current_feature_cols.append(new_col_name)

        # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature (different from above if possible)
        if subject_encoded_feature_for_interaction and len(numeric_summary_cols) >= 2:
            col_to_interact = numeric_summary_cols[1] 
            new_col_name = f"{col_to_interact}_x_SUBJECT_INTERACTION" # Generalized name
            df[new_col_name] = df[col_to_interact] * df[subject_encoded_feature_for_interaction]
            current_feature_cols.append(new_col_name)
        elif subject_encoded_feature_for_interaction and len(numeric_summary_cols) == 1: # Fallback to first if only one
             col_to_interact = numeric_summary_cols[0] 
             new_col_name = f"{col_to_interact}_x_SUBJECT_INTERACTION" # Generalized name
             df[new_col_name] = df[col_to_interact] * df[subject_encoded_feature_for_interaction]
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
        warnings.warn(f"'{subject_summary_path}' not found. Using minimal features from gold_df.")
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

# --- Main script ---
def run_training_and_validation(term_code_encoding='ohe', subject_id_encoding='target', rf_n_estimators=100):
    """
    Runs the training and validation process with configurable feature engineering and model hyperparameters.
    """
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, 
                   term_code_encoding=term_code_encoding, 
                   subject_id_encoding=subject_id_encoding)

    if df.empty:
        return 0.0 # Return 0.0 for F1 score if empty

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

    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=rf_n_estimators)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        pass
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # 2. Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# Main execution for ablation study
if __name__ == "__main__":
    results = {}

    # --- Baseline ---
    print("Running Baseline (TERM_CODE: OHE, SUBJECT_ID_SORT: Target Encoding, RF n_estimators=100)...")
    baseline_score = run_training_and_validation(term_code_encoding='ohe', subject_id_encoding='target', rf_n_estimators=100)
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}")
    print("-" * 30)

    # --- Ablation 1: TERM_CODE encoding from OHE to Label Encoding ---
    print("Running Ablation 1 (TERM_CODE: Label Encoding, SUBJECT_ID_SORT: Target Encoding, RF n_estimators=100)...")
    ablation1_score = run_training_and_validation(term_code_encoding='label', subject_id_encoding='target', rf_n_estimators=100)
    results['Ablation 1: TERM_CODE Label Encoding'] = ablation1_score
    print(f"Ablation 1 F1 Score: {ablation1_score:.4f}")
    print("-" * 30)

    # --- Ablation 2: SUBJECT_ID_SORT encoding from Target Encoding to Label Encoding ---
    print("Running Ablation 2 (TERM_CODE: OHE, SUBJECT_ID_SORT: Label Encoding, RF n_estimators=100)...")
    ablation2_score = run_training_and_validation(term_code_encoding='ohe', subject_id_encoding='label', rf_n_estimators=100)
    results['Ablation 2: SUBJECT_ID_SORT Label Encoding'] = ablation2_score
    print(f"Ablation 2 F1 Score: {ablation2_score:.4f}")
    print("-" * 30)

    # --- Ablation 3: RandomForest n_estimators from 100 to 1 ---
    print("Running Ablation 3 (TERM_CODE: OHE, SUBJECT_ID_SORT: Target Encoding, RF n_estimators=1)...")
    ablation3_score = run_training_and_validation(term_code_encoding='ohe', subject_id_encoding='target', rf_n_estimators=1)
    results['Ablation 3: RF n_estimators=1'] = ablation3_score
    print(f"Ablation 3 F1 Score: {ablation3_score:.4f}")
    print("-" * 30)

    # Determine most impactful component
    most_impactful_component = "None significant"
    largest_drop = 0.0

    for name, score in results.items():
        if name != 'Baseline':
            drop = baseline_score - score
            if drop > largest_drop:
                largest_drop = drop
                most_impactful_component = name

    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: F1 Score = {score:.4f}")
    print("\n" + "=" * 30)

    if largest_drop > 0.001: # A small threshold to consider it a 'significant' drop
        print(f"The most impactful part of the code, causing the largest performance drop ({largest_drop:.4f}) from the baseline, is: {most_impactful_component}.")
    else:
        print("Based on this ablation study, no single ablated component caused a significant performance drop, or all components performed identically to the baseline.")
        print("This might indicate the dataset is too simple or other factors are dominant, as perfect scores are often achieved.")
