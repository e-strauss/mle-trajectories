
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
def load_data_ablated(data_dir, gold_file, 
                      include_encoded_id_direct_features=True, 
                      include_raw_numeric_summary_features=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation flags:
    - include_encoded_id_direct_features: If False, TERM_CODE_ENCODED and SUBJECT_ID_SORT_ENCODED
                                          are not added directly as features, but can still be
                                          used for interaction/aggregation feature creation.
    - include_raw_numeric_summary_features: If False, the original numerical columns from
                                            subject_summary.csv are not added directly as features,
                                            but can still be used for polynomial/interaction/aggregation.
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
        
        if include_raw_numeric_summary_features:
            current_feature_cols.extend(existing_numeric_cols)

        # 2. Encode TERM_CODE and SUBJECT_ID_SORT and add them as numerical features
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        if include_encoded_id_direct_features:
            current_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        if include_encoded_id_direct_features:
            current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Identify candidate columns for advanced feature engineering from subject_summary
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        poly_features_to_square = numeric_summary_cols[:3] # Up to 3 features
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        if 'TERM_CODE_ENCODED' in df.columns and len(numeric_summary_cols) >= 1:
            col_to_interact = numeric_summary_cols[0] 
            new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
            current_feature_cols.append(new_col_name)

        if 'SUBJECT_ID_SORT_ENCODED' in df.columns and len(numeric_summary_cols) >= 2:
            col_to_interact = numeric_summary_cols[1] 
            new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
            current_feature_cols.append(new_col_name)
        elif 'SUBJECT_ID_SORT_ENCODED' in df.columns and len(numeric_summary_cols) == 1:
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
        
        # 5. Group-wise Aggregation Features
        agg_stats = ['mean', 'median', 'std', 'min', 'max']

        if 'TERM_CODE_ENCODED' in df.columns and numeric_summary_cols:
            for col in numeric_summary_cols:
                for stat in agg_stats:
                    new_col_name = f"{col}_BY_TERM_CODE_{stat.upper()}"
                    df[new_col_name] = df.groupby('TERM_CODE_ENCODED')[col].transform(stat)
                    current_feature_cols.append(new_col_name)

        if 'SUBJECT_ID_SORT_ENCODED' in df.columns and numeric_summary_cols:
            for col in numeric_summary_cols:
                for stat in agg_stats:
                    new_col_name = f"{col}_BY_SUBJECT_ID_{stat.upper()}"
                    df[new_col_name] = df.groupby('SUBJECT_ID_SORT_ENCODED')[col].transform(stat)
                    current_feature_cols.append(new_col_name)

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(current_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script for running training and validation (modified for ablation) ---
def run_ablation_scenario(
    scenario_name,
    include_encoded_id_direct_features=True,
    include_raw_numeric_summary_features=True,
    rf_params=None
):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    df = load_data_ablated(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                           include_encoded_id_direct_features,
                           include_raw_numeric_summary_features)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print(f"Performance for {scenario_name}: 0.0")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        print(f"Performance for {scenario_name}: 0.0")
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
        print(f"Performance for {scenario_name}: 0.0")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        print(f"Performance for {scenario_name}: 0.0")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    model_params = rf_params if rf_params is not None else {'random_state': 42, 'n_estimators': 100}
    model = RandomForestClassifier(**model_params)
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

    print(f"Performance for {scenario_name}: {final_validation_score}")
    return final_validation_score

# --- Ablation Study Execution ---
results = {}

# Scenario 1: Baseline (Original Solution)
results['Baseline'] = run_ablation_scenario('Baseline')

# Scenario 2: Ablation - Exclude Direct Encoded Identifier Features
# (TERM_CODE_ENCODED, SUBJECT_ID_SORT_ENCODED not added to feature list,
# but still used for interaction/aggregation features)
results['Ablation: No Direct Encoded Identifier Features'] = run_ablation_scenario(
    'Ablation: No Direct Encoded Identifier Features',
    include_encoded_id_direct_features=False
)

# Scenario 3: Ablation - Exclude Direct Numerical Summary Features
# (Original numeric columns from subject_summary.csv not added to feature list,
# but still used for polynomial/interaction/aggregation features)
results['Ablation: No Direct Numerical Summary Features'] = run_ablation_scenario(
    'Ablation: No Direct Numerical Summary Features',
    include_raw_numeric_summary_features=False
)

# Scenario 4: Ablation - Simplify Random Forest (min_samples_split=10)
# Default for min_samples_split is 2, increasing it makes trees simpler.
results['Ablation: RF with min_samples_split=10'] = run_ablation_scenario(
    'Ablation: RF with min_samples_split=10',
    rf_params={'random_state': 42, 'n_estimators': 100, 'min_samples_split': 10}
)

print("\n--- Ablation Study Summary ---")
for scenario, score in results.items():
    print(f"{scenario}: {score:.4f}")

# Determine the most contributing part
# Based on previous ablation studies and the patterns, when F1 scores are consistently 1.0,
# the most impactful change observed was the full exclusion of subject_summary.csv features
# from Study 3. If any of the new ablations cause a drop, that will be the new most impactful.

baseline_score = results['Baseline']
most_impactful_part = "The previous ablation study (Study 3) indicated that the inclusion of features derived from subject_summary.csv is the most crucial part, as its removal led to a significant F1 score drop to 0.6667."
max_score_diff = 0

# Check current ablations for impact
for scenario, score in results.items():
    if scenario == 'Baseline':
        continue
    
    score_diff = baseline_score - score
    if score_diff > max_score_diff:
        max_score_diff = score_diff
        most_impactful_part = f"'{scenario}' led to the largest performance decrease (from {baseline_score:.4f} to {score:.4f})."
    elif score_diff == 0:
        pass # No change

if max_score_diff > 0:
    print(f"\nConclusion: {most_impactful_part}")
else:
    print("\nConclusion: All new ablations maintained performance identical to the baseline (or previous perfect scores)."
          " This suggests that for this specific dataset, these components are not critical, or the dataset is too simple."
          f" {most_impactful_part}")
