
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

# --- Modified Function to load data and engineer features for ablation ---
def load_data_ablated(data_dir, gold_file, include_encoded_identifiers=True, include_advanced_features=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Accepts flags for ablation.
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
        if include_encoded_identifiers:
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
        if include_advanced_features:
            # 3. Polynomial Features (e.g., squared terms for key numerical features)
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
            
            # 4. Interaction Features (product of two distinct features)
            # Create interaction terms between encoded identifiers and some numerical summary features,
            # and between a pair of numerical summary features.
            
            # Interaction between encoded TERM_CODE and a numeric summary feature
            if include_encoded_identifiers and 'TERM_CODE_ENCODED' in df.columns and len(numeric_summary_cols) >= 1:
                col_to_interact = numeric_summary_cols[0] 
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

            # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature (different from above if possible)
            if include_encoded_identifiers and 'SUBJECT_ID_SORT_ENCODED' in df.columns and len(numeric_summary_cols) >= 2:
                col_to_interact = numeric_summary_cols[1] 
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)
            elif include_encoded_identifiers and 'SUBJECT_ID_SORT_ENCODED' in df.columns and len(numeric_summary_cols) == 1: # Fallback to first if only one
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
        temp_feature_cols = []
        if include_encoded_identifiers:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            temp_feature_cols.append('TERM_CODE_ENCODED')
            
            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            temp_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Add a simple dummy numerical feature if no other features exist, or if explicitly requested.
        # Ensure that if encoded identifiers are not included, we still have some feature.
        if not temp_feature_cols: # If no encoded identifiers were added, add a basic dummy
            df['DUMMY_FEATURE'] = 0
            temp_feature_cols = ['DUMMY_FEATURE']
        elif 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns: # If encoded IDs exist, add a more complex dummy
            df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
            temp_feature_cols.append('DUMMY_FEATURE')
        else: # Fallback for edge cases where encoded IDs might not have been fully set up.
            df['DUMMY_FEATURE'] = 0
            temp_feature_cols.append('DUMMY_FEATURE')
        
        df._feature_cols = temp_feature_cols
        # Fill potential NaNs in dummy features (not expected if created from existing data, but for robustness)
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Experiment Runner Function ---
def run_experiment(experiment_name, rf_params, load_data_args):
    """
    Runs a single experiment (baseline or ablation) and returns its F1 score.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")
    df = load_data_ablated(**load_data_args)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

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
    model = RandomForestClassifier(**rf_params)
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

    print(f"Validation Performance for {experiment_name}: {final_validation_score}")
    return final_validation_score

# --- Main Ablation Study ---
if __name__ == "__main__":
    results = {}

    # Common parameters for load_data_ablated
    base_load_data_args = {
        "data_dir": TRAIN_DATA_DIR,
        "gold_file": GOLD_ENROLLMENT_TRAIN_FILE,
        "include_encoded_identifiers": True,
        "include_advanced_features": True
    }

    # Baseline Experiment (Original Solution)
    # RandomForestClassifier(random_state=42, n_estimators=100)
    base_rf_params = {'random_state': 42, 'n_estimators': 100}
    results['Baseline (Full Solution)'] = run_experiment(
        experiment_name='Baseline (Full Solution)', 
        rf_params=base_rf_params, 
        load_data_args=base_load_data_args
    )

    # Ablation 1: Increase min_samples_leaf in RandomForestClassifier
    # This parameter controls the minimum number of samples required to be at a leaf node.
    # Increasing it simplifies the trees and can prevent overfitting.
    abl1_rf_params = base_rf_params.copy()
    abl1_rf_params['min_samples_leaf'] = 5 # Default is 1
    results['Ablation 1: RF min_samples_leaf=5'] = run_experiment(
        experiment_name='Ablation 1: RF min_samples_leaf=5', 
        rf_params=abl1_rf_params, 
        load_data_args=base_load_data_args
    )

    # Ablation 2: Disable LabelEncoder for TERM_CODE and SUBJECT_ID_SORT
    # This removes the direct numerical encoding of these identifier columns as features.
    abl2_load_data_args = base_load_data_args.copy()
    abl2_load_data_args['include_encoded_identifiers'] = False
    results['Ablation 2: No Encoded Identifiers (TERM_CODE, SUBJECT_ID_SORT)'] = run_experiment(
        experiment_name='Ablation 2: No Encoded Identifiers (TERM_CODE, SUBJECT_ID_SORT)', 
        rf_params=base_rf_params, 
        load_data_args=abl2_load_data_args
    )

    # Ablation 3: Remove polynomial and interaction features entirely
    # This disables the "advanced" feature engineering steps.
    abl3_load_data_args = base_load_data_args.copy()
    abl3_load_data_args['include_advanced_features'] = False
    results['Ablation 3: No Advanced Feature Engineering'] = run_experiment(
        experiment_name='Ablation 3: No Advanced Feature Engineering', 
        rf_params=base_rf_params, 
        load_data_args=abl3_load_data_args
    )

    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    # Determine the most impactful part
    baseline_score = results['Baseline (Full Solution)']
    impacts = {}
    for name, score in results.items():
        if name != 'Baseline (Full Solution)':
            impacts[name] = baseline_score - score
    
    if not impacts:
        print("\nNo impactful parts identified (or only baseline run).")
    else:
        most_impactful_part = max(impacts, key=impacts.get)
        max_impact_value = impacts[most_impactful_part]

        if max_impact_value > 0.001: # Threshold for considering it "impactful"
            print(f"\nConclusion: The part that contributes the most to the overall performance is related to '{most_impactful_part}', as its removal/modification resulted in the largest F1 score drop of {max_impact_value:.4f}.")
        else:
            print("\nConclusion: Based on this ablation study, no single part demonstrated a significant impact on performance, or all components performed identically. This might indicate the dataset is too simple or other factors are dominant, as observed in several previous studies.")
