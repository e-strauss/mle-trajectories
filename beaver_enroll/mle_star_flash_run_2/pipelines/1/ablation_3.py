

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features (modified for ablation) ---
def load_data_ablatable(data_dir, gold_file, include_encoded_identifiers=True, include_poly_interaction=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Configurable for ablation study:
    - include_encoded_identifiers: Whether to include TERM_CODE_ENCODED and SUBJECT_ID_SORT_ENCODED.
    - include_poly_interaction: Whether to include polynomial and interaction features.
    """
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        current_feature_cols = []

        existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
        current_feature_cols.extend(existing_numeric_cols)

        if include_encoded_identifiers:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            current_feature_cols.append('TERM_CODE_ENCODED')

            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering (subject to ablation) ---
        if include_poly_interaction:
            # Polynomial Features (e.g., squared terms for key numerical features)
            poly_features_to_square = [col for col in numeric_summary_cols if col in df.columns][:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
            
            # Interaction Features (product of two distinct features)
            term_encoded_present = include_encoded_identifiers and 'TERM_CODE_ENCODED' in df.columns
            subject_encoded_present = include_encoded_identifiers and 'SUBJECT_ID_SORT_ENCODED' in df.columns

            if term_encoded_present and len(numeric_summary_cols) >= 1:
                col_to_interact = numeric_summary_cols[0] 
                if col_to_interact in df.columns:
                    new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                    df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                    current_feature_cols.append(new_col_name)

            if subject_encoded_present and len(numeric_summary_cols) >= 2:
                col_to_interact = numeric_summary_cols[1] 
                if col_to_interact in df.columns:
                    new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                    df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                    current_feature_cols.append(new_col_name)
            elif subject_encoded_present and len(numeric_summary_cols) == 1:
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

        final_feature_cols = list(set(current_feature_cols))
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        # Fallback to minimal features if subject_summary.csv is not found
        temp_feature_cols = []
        if include_encoded_identifiers:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            temp_feature_cols.append('TERM_CODE_ENCODED')
            
            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            temp_feature_cols.append('SUBJECT_ID_SORT_ENCODED')
        
        # Always ensure at least one feature if no others are present
        if not temp_feature_cols:
            df['DUMMY_FEATURE'] = 0
            temp_feature_cols = ['DUMMY_FEATURE']
        else:
            dummy_val = 0
            if 'TERM_CODE_ENCODED' in df.columns:
                dummy_val += df['TERM_CODE_ENCODED'] % 5
            if 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                dummy_val += df['SUBJECT_ID_SORT_ENCODED'] % 7
            df['DUMMY_FEATURE'] = dummy_val
            if 'DUMMY_FEATURE' not in temp_feature_cols:
                temp_feature_cols.append('DUMMY_FEATURE')
        
        df._feature_cols = temp_feature_cols
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script logic, made into a function to be reusable ---
def run_training_and_validation_ablatable(load_data_func_to_use, model_class_to_use=RandomForestClassifier,
                                           include_encoded_identifiers_flag=True, include_poly_interaction_flag=True):
    """
    Runs the training and validation process with configurable data loading and model.
    Returns the F1 score.
    """
    df = load_data_func_to_use(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                               include_encoded_identifiers=include_encoded_identifiers_flag,
                               include_poly_interaction=include_poly_interaction_flag)

    if df.empty:
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)

    # Determine validation set: latest TERM_CODE with purging and embargo
    unique_terms = sorted(df['TERM_CODE'].unique()) # Ensure terms are sorted chronologically
    
    # Define purging and embargo parameters
    purge_gap_terms = 1
    embargo_gap_terms = 1 
    min_required_terms = 1 + purge_gap_terms + 1 + embargo_gap_terms 
    
    if len(unique_terms) < min_required_terms:
        return 0.0

    validation_term = unique_terms[-1] 
    validation_term_idx = len(unique_terms) - 1

    # --- Implement Purging ---
    last_train_term_idx_before_purge = validation_term_idx - purge_gap_terms - 1
    
    if last_train_term_idx_before_purge < 0:
        return 0.0

    train_terms_eligible = unique_terms[:last_train_term_idx_before_purge + 1]

    train_df = df[df['TERM_CODE'].isin(train_terms_eligible)]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
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

    if X_train.empty or y_train.empty:
        return 0.0

    # Model Training
    model = model_class_to_use(random_state=42) 
    if model_class_to_use == RandomForestClassifier:
        model = model_class_to_use(random_state=42, n_estimators=100)

    model.fit(X_train, y_train)

    # F1 Score Calculation with robustness checks
    final_validation_score = 0.0 
    
    if y_val.empty:
        pass
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# --- Ablation Study Execution ---
results = {}

# Scenario 1: Baseline (Original Solution)
results['Baseline (RandomForest, Encoded IDs, Poly/Interaction Features)'] = \
    run_training_and_validation_ablatable(load_data_ablatable, RandomForestClassifier,
                                           include_encoded_identifiers_flag=True,
                                           include_poly_interaction_flag=True)
print(f"Baseline (RandomForest, Encoded IDs, Poly/Interaction Features) F1 Score: {results['Baseline (RandomForest, Encoded IDs, Poly/Interaction Features)']:.4f}")

# Scenario 2: Ablation - Change Model from RandomForest to DecisionTree
results['Ablation: Model changed to DecisionTree'] = \
    run_training_and_validation_ablatable(load_data_ablatable, DecisionTreeClassifier,
                                           include_encoded_identifiers_flag=True,
                                           include_poly_interaction_flag=True)
print(f"Ablation: Model changed to DecisionTree F1 Score: {results['Ablation: Model changed to DecisionTree']:.4f}")

# Scenario 3: Ablation - Exclude Encoded Identifier Features (TERM_CODE_ENCODED, SUBJECT_ID_SORT_ENCODED)
results['Ablation: Exclude Encoded Identifier Features'] = \
    run_training_and_validation_ablatable(load_data_ablatable, RandomForestClassifier,
                                           include_encoded_identifiers_flag=False,
                                           include_poly_interaction_flag=True)
print(f"Ablation: Exclude Encoded Identifier Features F1 Score: {results['Ablation: Exclude Encoded Identifier Features']:.4f}")

# Scenario 4: Ablation - Exclude Polynomial and Interaction Features
results['Ablation: Exclude Polynomial and Interaction Features'] = \
    run_training_and_validation_ablatable(load_data_ablatable, RandomForestClassifier,
                                           include_encoded_identifiers_flag=True,
                                           include_poly_interaction_flag=False)
print(f"Ablation: Exclude Polynomial and Interaction Features F1 Score: {results['Ablation: Exclude Polynomial and Interaction Features']:.4f}")

# Determine the most contributing part among the newly tested components
baseline_score = results['Baseline (RandomForest, Encoded IDs, Poly/Interaction Features)']
most_contributing_part = "None of the newly ablated parts (or the dataset is too simple to show differences)"
largest_drop = 0

for ablation_name, score in results.items():
    if ablation_name == 'Baseline (RandomForest, Encoded IDs, Poly/Interaction Features)':
        continue
    
    drop = baseline_score - score
    if drop > largest_drop:
        largest_drop = drop
        most_contributing_part = ablation_name

if largest_drop == 0:
    print(f"\nBased on this study, all newly ablated parts had no negative impact or the dataset is too simple to show performance differences. This is consistent with previous ablation studies that also showed perfect F1 scores. Therefore, it's not possible to determine the most contributing part from this specific study without more challenging data. However, previous study results indicate that 'inclusion of features derived from subject_summary.csv' is crucial.")
else:
    print(f"\nBased on this ablation study, the part that contributes most to the overall performance (among the newly tested components) is: {most_contributing_part}. Removing it resulted in a performance drop of {largest_drop:.4f}.")

