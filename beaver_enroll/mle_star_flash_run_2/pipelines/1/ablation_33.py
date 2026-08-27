
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

# --- Modified load_data function to support ablations ---
def load_data_ablated(data_dir, gold_file, 
                      include_raw_numeric_summary_features=True, 
                      include_polynomial_features=True,
                      include_numeric_numeric_interaction_features=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters control which features are engineered.
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
        
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        if include_raw_numeric_summary_features:
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
        
        # --- Advanced Feature Engineering ---
        
        if include_polynomial_features:
            # 3. Polynomial Features (e.g., squared terms for key numerical features)
            # Select up to 3 features from `numeric_summary_cols` for squaring to avoid feature explosion.
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
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
        if include_numeric_numeric_interaction_features and len(numeric_summary_cols) >= 2:
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

# --- Modified run_training_and_validation function to support ablations ---
def run_training_and_validation_ablated(
    load_data_func
):
    df = load_data_func(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

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
        pass # Suppress print for cleaner ablation output
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 
    
    if y_val.empty:
        pass # Suppress print
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
if __name__ == "__main__":
    results = {}

    # Baseline
    print("Running Baseline (Full Solution)...")
    baseline_f1 = run_training_and_validation_ablated(
        lambda data_dir, gold_file: load_data_ablated(data_dir, gold_file,
            include_raw_numeric_summary_features=True, 
            include_polynomial_features=True,
            include_numeric_numeric_interaction_features=True
        )
    )
    print(f"Baseline F1 Score: {baseline_f1:.4f}\n")
    results["Baseline"] = baseline_f1

    # Ablation 1: No Raw Numerical Summary Features (direct addition)
    print("Running Ablation 1: No Raw Numerical Summary Features (direct addition)...")
    ablation1_f1 = run_training_and_validation_ablated(
        lambda data_dir, gold_file: load_data_ablated(data_dir, gold_file,
            include_raw_numeric_summary_features=False, # Ablation
            include_polynomial_features=True,
            include_numeric_numeric_interaction_features=True
        )
    )
    print(f"Ablation 1 F1 Score: {ablation1_f1:.4f}\n")
    results["Ablation 1 (No Raw Numeric Summary Features)"] = ablation1_f1

    # Ablation 2: No Polynomial Features
    print("Running Ablation 2: No Polynomial Features...")
    ablation2_f1 = run_training_and_validation_ablated(
        lambda data_dir, gold_file: load_data_ablated(data_dir, gold_file,
            include_raw_numeric_summary_features=True, 
            include_polynomial_features=False, # Ablation
            include_numeric_numeric_interaction_features=True
        )
    )
    print(f"Ablation 2 F1 Score: {ablation2_f1:.4f}\n")
    results["Ablation 2 (No Polynomial Features)"] = ablation2_f1

    # Ablation 3: No Numeric-Numeric Interaction Features (i.e., only Identifier-Numeric interactions)
    print("Running Ablation 3: No Numeric-Numeric Interaction Features...")
    ablation3_f1 = run_training_and_validation_ablated(
        lambda data_dir, gold_file: load_data_ablated(data_dir, gold_file,
            include_raw_numeric_summary_features=True, 
            include_polynomial_features=True,
            include_numeric_numeric_interaction_features=False # Ablation
        )
    )
    print(f"Ablation 3 F1 Score: {ablation3_f1:.4f}\n")
    results["Ablation 3 (No Numeric-Numeric Interaction Features)"] = ablation3_f1


    # Determine the most impactful part
    most_impactful_part = "None of the ablated components had a discernible impact, or the task is too simple."
    largest_abs_change = 0.0
    change_type = "" # "drop" or "increase"
    impactful_name = ""

    for name, score in results.items():
        if name == "Baseline":
            continue
        
        current_change = baseline_f1 - score
        
        if abs(current_change) > largest_abs_change:
            largest_abs_change = abs(current_change)
            impactful_name = name
            if current_change > 0:
                change_type = "dropped by"
            else:
                change_type = "increased by"

    if largest_abs_change > 0.0001: # Threshold for meaningful change
        print(f"The most impactful part identified in this ablation study is: '{impactful_name}' (F1 {change_type} {largest_abs_change:.4f})")
    else:
        print(f"Based on this ablation study, {most_impactful_part}")

