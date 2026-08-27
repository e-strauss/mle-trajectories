
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

# --- Function to load data and engineer features (modified for ablation) ---
def load_data_ablation(data_dir, gold_file, 
                       polynomial_feature_selection_strategy='first_3', # 'first_3', 'last_3'
                       interaction_id_feature_selection_strategy='first_2_with_ids' # 'first_2_with_ids', 'last_2_with_ids'
                      ):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters control specific feature engineering steps related to feature selection.
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
        
        # --- Advanced Feature Engineering ---
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        poly_features_to_square = []
        if polynomial_feature_selection_strategy == 'first_3':
            poly_features_to_square = numeric_summary_cols[:3]
        elif polynomial_feature_selection_strategy == 'last_3':
            # Take last 3 if available, otherwise take all (or fewer if less than 3)
            poly_features_to_square = numeric_summary_cols[-3:] if len(numeric_summary_cols) >= 3 else numeric_summary_cols[:]
        
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        
        # Determine which numeric summary columns to use for interaction with encoded IDs
        cols_for_id_interaction = []
        if interaction_id_feature_selection_strategy == 'first_2_with_ids':
            cols_for_id_interaction = numeric_summary_cols[:2]
        elif interaction_id_feature_selection_strategy == 'last_2_with_ids':
            cols_for_id_interaction = numeric_summary_cols[-2:] if len(numeric_summary_cols) >= 2 else numeric_summary_cols[:]

        # Interaction between encoded TERM_CODE and a numeric summary feature
        if 'TERM_CODE_ENCODED' in current_feature_cols and len(cols_for_id_interaction) >= 1:
            col_to_interact = cols_for_id_interaction[0] 
            new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
            current_feature_cols.append(new_col_name)

        # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature (different from above if possible)
        if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(cols_for_id_interaction) >= 2:
            col_to_interact = cols_for_id_interaction[1] 
            new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
            current_feature_cols.append(new_col_name)
        elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(cols_for_id_interaction) == 1: # Fallback to first if only one
             col_to_interact = cols_for_id_interaction[0] 
             new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
             df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
             current_feature_cols.append(new_col_name)

        # Interaction between two distinct numeric summary features (this part is NOT ablated by strategy for this study)
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

# --- Main training and validation function (modified to accept ablation parameters) ---
def run_ablation_scenario(scenario_name, 
                          polynomial_feature_selection_strategy='first_3', # Default for baseline
                          interaction_id_feature_selection_strategy='first_2_with_ids', # Default for baseline
                          model_random_state=42 # Default for baseline
                         ):
    print(f"\n--- Running scenario: {scenario_name} ---")
    
    # Pass ablation parameters to load_data_ablation
    df = load_data_ablation(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, 
                            polynomial_feature_selection_strategy=polynomial_feature_selection_strategy,
                            interaction_id_feature_selection_strategy=interaction_id_feature_selection_strategy)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return 0.0

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        print("Final Validation Performance: 0.0")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
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
        print("No usable features available for training after all fallback attempts.")
        print("Final Validation Performance: 0.0")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        print("Final Validation Performance: 0.0")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    # Apply model_random_state for ablation
    if model_random_state is not None:
        model = RandomForestClassifier(n_estimators=100, random_state=model_random_state)
    else:
        model = RandomForestClassifier(n_estimators=100) # No random_state for this ablation

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

    print(f"Final Validation Performance: {final_validation_score}")
    return final_validation_score

# --- Run the ablation study ---
if __name__ == "__main__":
    results = {}

    # Baseline (Original Solution's default behavior)
    # Uses first 3 numeric summary columns for polynomial features
    # Uses first 2 numeric summary columns for interaction features with IDs
    # Uses RandomForestClassifier with random_state=42
    results['Baseline (Original Solution)'] = run_ablation_scenario(
        'Baseline (Original Solution)',
        polynomial_feature_selection_strategy='first_3',
        interaction_id_feature_selection_strategy='first_2_with_ids',
        model_random_state=42
    )

    # Ablation 1: Remove random_state from RandomForestClassifier (testing stability)
    # All feature engineering parameters are kept as baseline
    results['Ablation 1: RF without random_state'] = run_ablation_scenario(
        'Ablation 1: RF without random_state',
        polynomial_feature_selection_strategy='first_3',
        interaction_id_feature_selection_strategy='first_2_with_ids',
        model_random_state=None # Ablate random_state
    )

    # Ablation 2: Change polynomial feature selection to 'last_3' (testing specificity of feature choice)
    # This checks if using different specific columns for squaring impacts performance.
    # All other parameters are kept as baseline.
    results['Ablation 2: Polynomial features from last 3 numeric columns'] = run_ablation_scenario(
        'Ablation 2: Polynomial features from last 3 numeric columns',
        polynomial_feature_selection_strategy='last_3', # Modified here
        interaction_id_feature_selection_strategy='first_2_with_ids',
        model_random_state=42
    )

    # Ablation 3: Change interaction feature selection to 'last_2_with_ids' (testing specificity of feature choice)
    # This checks if using different specific columns for interactions with encoded IDs impacts performance.
    # All other parameters are kept as baseline.
    results['Ablation 3: Interaction features from last 2 numeric columns with IDs'] = run_ablation_scenario(
        'Ablation 3: Interaction features from last 2 numeric columns with IDs',
        polynomial_feature_selection_strategy='first_3',
        interaction_id_feature_selection_strategy='last_2_with_ids', # Modified here
        model_random_state=42
    )
    
    # Determine the most impactful part
    baseline_score = results['Baseline (Original Solution)']
    impact = {}
    for scenario, score in results.items():
        if scenario != 'Baseline (Original Solution)':
            impact[scenario] = baseline_score - score # Positive difference means performance drop

    most_impactful_change = None
    max_impact = -np.inf # Initialize with negative infinity to find the largest drop

    # Find the ablation with the largest performance drop
    for scenario, diff in impact.items():
        if diff > max_impact:
            max_impact = diff
            most_impactful_change = scenario
    
    # Conclusion statement based on the results
    if max_impact > 0: # A measurable performance drop was observed
        print(f"\n--- Ablation Study Conclusion ---")
        print(f"The most impactful change, causing a performance drop of {max_impact:.4f}, was: '{most_impactful_change}'.")
        print("This suggests that the ablated part (or the specific configuration it modified) is crucial for the model's performance.")
    elif max_impact < 0: # A measurable performance improvement was observed (baseline was suboptimal)
        # Find the scenario with the largest improvement (smallest negative diff)
        min_impact = np.inf
        most_beneficial_change = None
        for scenario, diff in impact.items():
            if diff < min_impact:
                min_impact = diff
                most_beneficial_change = scenario
        print(f"\n--- Ablation Study Conclusion ---")
        print(f"The most impactful change, leading to a performance *improvement* of {-min_impact:.4f}, was: '{most_beneficial_change}'.")
        print("This suggests that the original component in the baseline might be detrimental or suboptimal, and its modification improved performance.")
    else: # All ablations performed identically to the baseline (diff is 0 for all)
        print(f"\n--- Ablation Study Conclusion ---")
        print("All ablations performed identically to the baseline. No single component caused a measurable performance difference.")
        print("This suggests the dataset might be too simple, or the validation strategy is not challenging enough to differentiate the components' impact.")

