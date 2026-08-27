

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

# --- Baseline load_data function (includes statistical aggregation features from the provided context) ---
# This function is a modified version of the one provided, incorporating the statistical aggregation features
# described in the user's prompt as context from a previous agent's action.
def load_data_baseline(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Augmented with Statistical Aggregation Features.
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

        # 5. Statistical Aggregation Features (ADDED FOR BASELINE FROM AGENT'S CONTEXT)
        # Calculate mean and standard deviation for selected numerical features, grouped by key identifiers.
        # Select up to 3 numerical summary features for aggregation to manage feature count.
        agg_features_to_process = numeric_summary_cols[:3] 
        grouping_identifiers = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']

        for group_col in grouping_identifiers:
            # Check if the grouping column exists in the dataframe (it should be in current_feature_cols already)
            if group_col in current_feature_cols:
                for agg_col in agg_features_to_process:
                    # Ensure the numerical column to aggregate exists
                    if agg_col in df.columns:
                        # Calculate Mean
                        mean_feature_name = f"{agg_col}_MEAN_BY_{group_col}"
                        if mean_feature_name not in df.columns: # Prevent re-creation if already exists
                            df[mean_feature_name] = df.groupby(group_col)[agg_col].transform('mean')
                            current_feature_cols.append(mean_feature_name)
                        
                        # Calculate Standard Deviation
                        std_feature_name = f"{agg_col}_STD_BY_{group_col}"
                        if std_feature_name not in df.columns: # Prevent re-creation if already exists
                            df[std_feature_name] = df.groupby(group_col)[agg_col].transform('std')
                            current_feature_cols.append(std_feature_name)

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

# --- Ablation 1: load_data function without statistical aggregation features ---
def load_data_no_stats_agg(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Statistical Aggregation Features are DISABLED.
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

        # Statistical Aggregation Features - DISABLED FOR THIS ABLATION
        # This block has been intentionally removed to study its impact.

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


# --- Generic run function to allow parameterization for ablations ---
def run_ablation_scenario(load_data_func, sort_by_term=True, rf_criterion='gini'):
    """
    Runs the training and validation process with specified data loading, sorting, and RF criterion.
    """
    df = load_data_func(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0 # Return 0.0 for empty DF

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    
    if sort_by_term:
        df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        # Fallback if sorting is disabled and unique_terms might not be chronological or sufficient
        if not sort_by_term:
            warnings.warn("Not enough unique terms for a time-based validation split, and sorting is disabled. Using a random split as fallback.")
            # Simple random split if time-based is not feasible/reliable
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['HIGH_ENROLLMENT'] if 'HIGH_ENROLLMENT' in df.columns and len(df['HIGH_ENROLLMENT'].unique()) > 1 else None)
        else:
            print("Not enough unique terms for a time-based validation split (at least 2 required).")
            return 0.0

    else: # Normal time-based split
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
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100, criterion=rf_criterion)
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
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score


if __name__ == "__main__":
    scores = {}

    # Baseline Run (with statistical aggregation features, term sorting, gini criterion)
    print("--- Running Baseline ---")
    baseline_score = run_ablation_scenario(load_data_baseline, sort_by_term=True, rf_criterion='gini')
    scores['Baseline (Full Solution)'] = baseline_score
    print(f"Final Validation Performance: {baseline_score:.4f}\n")

    # Ablation 1: Remove Statistical Aggregation Features
    print("--- Ablation: No Statistical Aggregation Features ---")
    score_no_stats_agg = run_ablation_scenario(load_data_no_stats_agg, sort_by_term=True, rf_criterion='gini')
    scores['No Statistical Aggregation Features'] = score_no_stats_agg
    print(f"Final Validation Performance: {score_no_stats_agg:.4f}\n")

    # Ablation 2: Remove Sorting by TERM_CODE
    print("--- Ablation: No Sorting by TERM_CODE for Time-Based Split ---")
    score_no_term_sort = run_ablation_scenario(load_data_baseline, sort_by_term=False, rf_criterion='gini')
    scores['No TERM_CODE Sorting'] = score_no_term_sort
    print(f"Final Validation Performance: {score_no_term_sort:.4f}\n")

    # Ablation 3: Change Random Forest Criterion to 'entropy'
    print("--- Ablation: Random Forest Criterion='entropy' ---")
    score_rf_entropy = run_ablation_scenario(load_data_baseline, sort_by_term=True, rf_criterion='entropy')
    scores['RF Criterion=entropy'] = score_rf_entropy
    print(f"Final Validation Performance: {score_rf_entropy:.4f}\n")

    # Determine the most impactful part
    print("\n--- Ablation Study Summary ---")
    for name, score in scores.items():
        print(f"{name}: {score:.4f}")

    most_impactful_part = "No single component showed a significantly larger impact than others among these ablations."
    
    # Assume baseline is the target, find the largest drop from baseline
    baseline_score = scores['Baseline (Full Solution)']
    max_impact_diff = 0.0
    impact_description = []

    for name, score in scores.items():
        if name == 'Baseline (Full Solution)':
            continue
        
        diff = abs(baseline_score - score)
        if diff > max_impact_diff:
            max_impact_diff = diff
        
        if score < baseline_score:
            impact_description.append(f"Disabling '{name}' decreased F1 score by {baseline_score - score:.4f}")
        elif score > baseline_score:
            impact_description.append(f"Modifying '{name}' increased F1 score by {score - baseline_score:.4f}")
        else:
            impact_description.append(f"'{name}' resulted in no change to F1 score.")

    if max_impact_diff > 0.0001: # Threshold for considering a "significant" change
        # If there are differences, report the most impactful one
        if baseline_score == 1.0: # If baseline was perfect, look for drops
            largest_drop_name = None
            largest_drop_value = 0.0
            for name, score in scores.items():
                if name != 'Baseline (Full Solution)':
                    drop = baseline_score - score
                    if drop > largest_drop_value:
                        largest_drop_value = drop
                        largest_drop_name = name
            if largest_drop_name and largest_drop_value > 0.0001:
                most_impactful_part = f"The 'removal of {largest_drop_name}' showed the most significant impact, causing a drop of {largest_drop_value:.4f} in F1 score from the baseline. This component contributes positively to overall performance."
            else:
                 most_impactful_part = "Among these ablations, there were no significant negative impacts or discernible differences from the baseline. This may suggest the dataset is still too simple or the ablated parts are not critical to this specific problem instance."
        else: # If baseline was not perfect, look for the largest absolute change
            largest_change_name = None
            largest_change_value = 0.0
            for name, score in scores.items():
                if name != 'Baseline (Full Solution)':
                    change = abs(baseline_score - score)
                    if change > largest_change_value:
                        largest_change_value = change
                        largest_change_name = name
            if largest_change_name and largest_change_value > 0.0001:
                most_impactful_part = f"The modification related to '{largest_change_name}' showed the most significant change ({largest_change_value:.4f} F1 score difference) from the baseline, indicating it has the highest impact among the tested components."
    else:
        most_impactful_part = "All ablations yielded identical performance to the baseline. This suggests the dataset might be too simple, the ablated components are not critical, or the current evaluation method is not sensitive enough to detect differences. Further complexity in the dataset or a more challenging evaluation metric might be needed to differentiate contributions."
    
    print(f"\nConclusion: {most_impactful_part}")

