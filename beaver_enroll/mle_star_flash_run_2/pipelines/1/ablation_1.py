
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

def _load_data_base(data_dir, gold_file):
    """
    Base function to load data, merge, and perform common steps,
    allowing specific feature engineering and imputation to be customized by callers.
    """
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    if not os.path.exists(subject_summary_path):
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0) # Fallback imputation
        return df

    subject_summary_df_full = pd.read_csv(subject_summary_path)
    identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
    df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
    
    current_feature_cols = []
    existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
    current_feature_cols.extend(existing_numeric_cols)

    le_term = LabelEncoder()
    df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
    current_feature_cols.append('TERM_CODE_ENCODED')

    le_subject = LabelEncoder()
    df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
    current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

    summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
    numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
    
    return df, current_feature_cols, numeric_summary_cols, identifier_cols

def load_data_original(data_dir, gold_file):
    df, current_feature_cols, numeric_summary_cols, identifier_cols = _load_data_base(data_dir, gold_file)
    if not isinstance(df, tuple): return df # Fallback case handled by _load_data_base

    # 3. Polynomial Features (e.g., squared terms for key numerical features)
    poly_features_to_square = numeric_summary_cols[:3]
    for col in poly_features_to_square:
        new_col_name = f"{col}_SQUARED"
        df[new_col_name] = df[col] ** 2
        current_feature_cols.append(new_col_name)
    
    # 4. Interaction Features (product of two distinct features)
    if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
        col_to_interact = numeric_summary_cols[0] 
        new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
        df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
        current_feature_cols.append(new_col_name)

    if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
        col_to_interact = numeric_summary_cols[1] 
        new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
        df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
        current_feature_cols.append(new_col_name)
    elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1:
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

    final_feature_cols = list(set(current_feature_cols))
    final_feature_cols = [col for col in final_feature_cols if col in df.columns]

    if not final_feature_cols:
        warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0 
        final_feature_cols = ['DUMMY_FEATURE']

    df[final_feature_cols] = df[final_feature_cols].fillna(0) # Original imputation
    
    df._feature_cols = final_feature_cols
    return df

def load_data_no_poly_features(data_dir, gold_file):
    df, current_feature_cols, numeric_summary_cols, identifier_cols = _load_data_base(data_dir, gold_file)
    if not isinstance(df, tuple): return df # Fallback case handled by _load_data_base

    # 3. Polynomial Features (DISABLED)
    # No polynomial features added here
    
    # 4. Interaction Features (kept as in original)
    if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
        col_to_interact = numeric_summary_cols[0] 
        new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
        df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
        current_feature_cols.append(new_col_name)

    if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
        col_to_interact = numeric_summary_cols[1] 
        new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
        df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
        current_feature_cols.append(new_col_name)
    elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1:
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

    final_feature_cols = list(set(current_feature_cols))
    final_feature_cols = [col for col in final_feature_cols if col in df.columns]

    if not final_feature_cols:
        warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0 
        final_feature_cols = ['DUMMY_FEATURE']

    df[final_feature_cols] = df[final_feature_cols].fillna(0) # Original imputation
    
    df._feature_cols = final_feature_cols
    return df

def load_data_no_interaction_features(data_dir, gold_file):
    df, current_feature_cols, numeric_summary_cols, identifier_cols = _load_data_base(data_dir, gold_file)
    if not isinstance(df, tuple): return df # Fallback case handled by _load_data_base

    # 3. Polynomial Features (kept as in original)
    poly_features_to_square = numeric_summary_cols[:3]
    for col in poly_features_to_square:
        new_col_name = f"{col}_SQUARED"
        df[new_col_name] = df[col] ** 2
        current_feature_cols.append(new_col_name)
    
    # 4. Interaction Features (DISABLED)
    # No interaction features added here

    final_feature_cols = list(set(current_feature_cols))
    final_feature_cols = [col for col in final_feature_cols if col in df.columns]

    if not final_feature_cols:
        warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0 
        final_feature_cols = ['DUMMY_FEATURE']

    df[final_feature_cols] = df[final_feature_cols].fillna(0) # Original imputation
    
    df._feature_cols = final_feature_cols
    return df

def load_data_median_fillna(data_dir, gold_file):
    df, current_feature_cols, numeric_summary_cols, identifier_cols = _load_data_base(data_dir, gold_file)
    if not isinstance(df, tuple): return df # Fallback case handled by _load_data_base

    # 3. Polynomial Features (kept as in original)
    poly_features_to_square = numeric_summary_cols[:3]
    for col in poly_features_to_square:
        new_col_name = f"{col}_SQUARED"
        df[new_col_name] = df[col] ** 2
        current_feature_cols.append(new_col_name)
    
    # 4. Interaction Features (kept as in original)
    if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
        col_to_interact = numeric_summary_cols[0] 
        new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
        df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
        current_feature_cols.append(new_col_name)

    if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
        col_to_interact = numeric_summary_cols[1] 
        new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
        df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
        current_feature_cols.append(new_col_name)
    elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1:
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

    final_feature_cols = list(set(current_feature_cols))
    final_feature_cols = [col for col in final_feature_cols if col in df.columns]

    if not final_feature_cols:
        warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0 
        final_feature_cols = ['DUMMY_FEATURE']
    
    # Ablation: Change NaN imputation to median
    for col in final_feature_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0) 
    
    df._feature_cols = final_feature_cols
    return df

def run_training_and_validation_generic(load_data_fn, scenario_name):
    print(f"\n--- Running scenario: {scenario_name} ---")
    df = load_data_fn(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty:
        print(f"Loaded DataFrame for {scenario_name} is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print(f"Not enough unique terms for a time-based validation split (at least 2 required) for {scenario_name}.")
        print("Final Validation Performance: 0.0")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
    if not feature_cols: 
        warnings.warn(f"Feature columns not correctly identified by {scenario_name}'s load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print(f"No usable features available for training after all fallback attempts for {scenario_name}.")
        print("Final Validation Performance: 0.0")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print(f"Training set for {scenario_name} is empty. Cannot train a model.")
        print("Final Validation Performance: 0.0")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' for {scenario_name} has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    final_validation_score = 0.0 
    
    if y_val.empty:
        print(f"Validation set for {scenario_name} is empty. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            print(f"Validation set 'HIGH_ENROLLMENT' for {scenario_name} has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance for {scenario_name}: {final_validation_score}")
    return final_validation_score

if __name__ == "__main__":
    results = {}

    results['Original Solution (Baseline)'] = run_training_and_validation_generic(load_data_original, 'Original Solution (Baseline)')
    results['Ablation: No Polynomial Features'] = run_training_and_validation_generic(load_data_no_poly_features, 'Ablation: No Polynomial Features')
    results['Ablation: No Interaction Features'] = run_training_and_validation_generic(load_data_no_interaction_features, 'Ablation: No Interaction Features')
    results['Ablation: NaN Imputation to Median'] = run_training_and_validation_generic(load_data_median_fillna, 'Ablation: NaN Imputation to Median')
    
    print("\n--- Ablation Study Summary ---")
    best_score = -1.0
    best_scenario = ""
    for scenario, score in results.items():
        print(f"{scenario}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_scenario = scenario
    
    baseline_score = results['Original Solution (Baseline)']
    
    if all(score == baseline_score for score in results.values()):
        print("\nAll scenarios, including ablations, achieved the same performance.")
        print("This suggests that the ablated parts are not critical, or the dataset is too simple to show performance differences, similar to the previous study.")
        print("Therefore, it's not possible to definitively state which part contributes the most based on these results.")
    else:
        performance_drops = {}
        for scenario, score in results.items():
            if scenario != 'Original Solution (Baseline)':
                drop = baseline_score - score
                performance_drops[scenario] = drop
        
        if performance_drops:
            most_impactful_ablation = max(performance_drops, key=performance_drops.get)
            max_drop = performance_drops[most_impactful_ablation]
            if max_drop > 0:
                print(f"\nThe part contributing the most to the overall performance appears to be: {most_impactful_ablation.replace('Ablation: ', '')} (removing it caused a performance drop of {max_drop:.4f}).")
            else:
                print("\nNo specific ablation caused a performance drop relative to the baseline.")
                print("This suggests that the ablated parts are not critical or the dataset is too simple to show performance differences.")
                print("Therefore, it's not possible to definitively state which part contributes the most based on these results.")
        else:
            print("\nNo ablations were performed or evaluated for contribution.")

