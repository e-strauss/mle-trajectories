
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

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, 
              use_label_encoding=False, 
              limit_adv_fe_summary_cols=False,
              exclude_id_interaction_features=False):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    
    Ablation parameters:
    - use_label_encoding: If True, uses LabelEncoder instead of TargetEncoder for TERM_CODE and SUBJECT_ID_SORT.
    - limit_adv_fe_summary_cols: If True, limits the number of numeric_summary_cols used for polynomial and interaction features.
    - exclude_id_interaction_features: If True, removes interaction features involving TERM_CODE_ENCODED and SUBJECT_ID_SORT_ENCODED.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})
    
    target_col_name = 'HIGH_ENROLLMENT'

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + [target_col_name]]
        current_feature_cols.extend(existing_numeric_cols)

        # 2. Encoding for TERM_CODE and SUBJECT_ID_SORT
        if use_label_encoding:
            # Use LabelEncoder
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            current_feature_cols.append('TERM_CODE_ENCODED')

            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')
        else:
            # Use Target Encoding (original behavior)
            N_SPLITS = 5 
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

            df['TERM_CODE_ENCODED'] = np.nan
            for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
                target_mean_map = df.loc[train_idx].groupby('TERM_CODE')[target_col_name].mean()
                global_mean_train_fold = df.loc[train_idx, target_col_name].mean()
                df.loc[val_idx, 'TERM_CODE_ENCODED'] = df.loc[val_idx, 'TERM_CODE'].map(target_mean_map).fillna(global_mean_train_fold)
            current_feature_cols.append('TERM_CODE_ENCODED')

            df['SUBJECT_ID_SORT_ENCODED'] = np.nan
            # Re-initialize KFold to ensure fresh split for SUBJECT_ID_SORT, consistent with original logic
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42) 
            for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
                target_mean_map = df.loc[train_idx].groupby('SUBJECT_ID_SORT')[target_col_name].mean()
                global_mean_train_fold = df.loc[train_idx, target_col_name].mean()
                df.loc[val_idx, 'SUBJECT_ID_SORT_ENCODED'] = df.loc[val_idx, 'SUBJECT_ID_SORT'].map(target_mean_map).fillna(global_mean_train_fold)
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')


        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # Determine how many summary columns to use for advanced FE
        num_cols_for_poly = 1 if limit_adv_fe_summary_cols else 3
        # For interaction between two distinct features, need at least 2; if limited, we might only use 1, so set to 1
        num_cols_for_distinct_interact = 1 if limit_adv_fe_summary_cols else 2 

        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        poly_features_to_square = numeric_summary_cols[:num_cols_for_poly]
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        
        if not exclude_id_interaction_features:
            # Interaction between encoded TERM_CODE and a numeric summary feature
            if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
                col_to_interact = numeric_summary_cols[0] 
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

            # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature
            if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
                # Use a different column if available and not limited, otherwise use the first
                col_to_interact_idx = 1 if len(numeric_summary_cols) >= 2 and not limit_adv_fe_summary_cols else 0
                col_to_interact = numeric_summary_cols[col_to_interact_idx]
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)

        # Interaction between two distinct numeric summary features
        if len(numeric_summary_cols) >= num_cols_for_distinct_interact:
            col1 = numeric_summary_cols[0]
            col2 = numeric_summary_cols[1] if num_cols_for_distinct_interact > 1 else None # If num_cols_for_distinct_interact is 1, no distinct second column
            if col2 is not None and col1 != col2: # Only if distinct for true interaction
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
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        # Fallback to the minimal features if subject_summary.csv is not found
        
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script ---
def run_ablation_experiment(experiment_name, **load_data_params):
    print(f"--- Running Experiment: {experiment_name} ---")
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, **load_data_params)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return 0.0 

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

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
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
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
    print("-" * 50)
    return final_validation_score

# Run the ablation study
if __name__ == "__main__":
    results = {}

    # Baseline
    results['Baseline (Full Solution, Target Encoding, Full Advanced FE)'] = run_ablation_experiment(
        'Baseline (Full Solution, Target Encoding, Full Advanced FE)'
    )

    # Ablation 1: Use Label Encoding for identifiers instead of Target Encoding
    results['Ablation 1: Label Encoding for Identifiers'] = run_ablation_experiment(
        'Ablation 1: Label Encoding for Identifiers',
        use_label_encoding=True
    )

    # Ablation 2: Limit number of numeric_summary_cols used as source for advanced feature engineering
    # This means poly_features_to_square will be numeric_summary_cols[:1]
    # And interaction features will try to use numeric_summary_cols[0] only or fewer types
    results['Ablation 2: Reduced Numeric Summary Cols for Advanced FE Source'] = run_ablation_experiment(
        'Ablation 2: Reduced Numeric Summary Cols for Advanced FE Source',
        limit_adv_fe_summary_cols=True
    )

    # Ablation 3: Exclude Interaction Features Involving Encoded Identifiers (TERM_CODE_ENCODED, SUBJECT_ID_SORT_ENCODED)
    results['Ablation 3: No Identifier-Numeric Summary Interaction Features'] = run_ablation_experiment(
        'Ablation 3: No Identifier-Numeric Summary Interaction Features',
        exclude_id_interaction_features=True
    )

    # Determine the most impactful component
    baseline_score = results['Baseline (Full Solution, Target Encoding, Full Advanced FE)']
    impacts = {}
    for name, score in results.items():
        if name != 'Baseline (Full Solution, Target Encoding, Full Advanced FE)':
            impacts[name] = baseline_score - score

    if not impacts:
        print("No ablation studies were conducted beyond the baseline.")
    else:
        max_impact_name = ""
        max_impact_value = -float('inf')

        for name, impact in impacts.items():
            if impact > max_impact_value:
                max_impact_value = impact
                max_impact_name = name

        if max_impact_value > 0:
            print(f"\nThe most impactful part of the code (causing the largest performance drop) is: '{max_impact_name}' with a drop of {max_impact_value:.4f}.")
        elif max_impact_value == 0:
            print("\nAll ablations performed identically to the baseline, suggesting the dataset might be too simple to differentiate impacts or the ablated parts are not critical.")
            print("To differentiate, a more complex dataset or a more challenging validation strategy might be required.")
        else: # max_impact_value < 0, meaning an ablation improved performance (unlikely for an ablation study focused on drops)
            print(f"\nSurprisingly, the ablation '{max_impact_name}' led to an improvement of {-max_impact_value:.4f} compared to the baseline, or all ablations had negative impact (improved score).")
            # If all are improvements or no drops:
            all_drops_zero_or_negative = all(impact <= 0 for impact in impacts.values())
            if all_drops_zero_or_negative:
                print("No ablations caused a performance drop, implying the removed/modified components are not critical or even detrimental in the full setup.")
