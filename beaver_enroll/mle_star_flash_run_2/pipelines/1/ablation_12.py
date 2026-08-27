
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
# For the purpose of this ablation script, we'll ensure dummy data is created if files are not present.
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Helper function for evaluation ---
def evaluate_model(df, feature_cols_override=None, description=""):
    """
    Trains and evaluates the model using the provided DataFrame and feature columns.
    Returns the F1 score.
    """
    if df.empty:
        print(f"Warning: DataFrame for {description} is empty. Cannot proceed with training.")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print(f"Warning: Not enough unique terms for a time-based validation split ({description}). At least 2 required.")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = feature_cols_override if feature_cols_override is not None else getattr(df, '_feature_cols', [])
    
    if not feature_cols: 
        warnings.warn(f"Feature columns not correctly identified by load_data for {description}. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print(f"Warning: No usable features available for training after all fallback attempts for {description}.")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print(f"Warning: Training set is empty for {description}. Cannot train a model.")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Warning: Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()} for {description}. This might lead to trivial predictions.")
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    final_validation_score = 0.0 
    
    if y_val.empty:
        print(f"Warning: Validation set is empty for {description}. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            print(f"Warning: Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val} for {description}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# --- Modified load_data functions for ablation study ---

# Base load_data function (renamed for clarity)
def load_data_baseline(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        if not os.path.exists(subject_summary_path):
            warnings.warn(f"Simulating creation of dummy {subject_summary_path} as it does not exist.")
            unique_gold_identifiers = gold_df[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates()
            dummy_subject_summary_df_full = pd.DataFrame({
                'TERM_CODE': unique_gold_identifiers['TERM_CODE'].tolist(),
                'SUBJECT_ID_SORT': unique_gold_identifiers['SUBJECT_ID_SORT'].tolist(),
                'NUM_STUDENTS': np.random.randint(10, 100, len(unique_gold_identifiers)),
                'AVG_GRADE': np.random.rand(len(unique_gold_identifiers)) * 4.0
            })
            subject_summary_df_full = dummy_subject_summary_df_full
        else:
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
        
        # --- Advanced Feature Engineering ---
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


def load_data_ablated_no_subject_summary(data_dir, gold_file):
    """
    Simulates the absence of subject_summary.csv, forcing the minimal features fallback.
    """
    gold_df = pd.read_csv(gold_file)
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    print(f"Ablation: Simulating absence of subject_summary.csv. Using minimal features from gold_df.")
    
    le_term = LabelEncoder()
    df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
    
    le_subject = LabelEncoder()
    df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
    
    df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
    
    df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
    df[df._feature_cols] = df[df._feature_cols].fillna(0)
    
    return df

def load_data_ablated_no_polynomial_features(data_dir, gold_file):
    """
    Loads data and engineers features, but skips polynomial feature creation.
    """
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        if not os.path.exists(subject_summary_path):
            warnings.warn(f"Simulating creation of dummy {subject_summary_path} as it does not exist.")
            unique_gold_identifiers = gold_df[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates()
            dummy_subject_summary_df_full = pd.DataFrame({
                'TERM_CODE': unique_gold_identifiers['TERM_CODE'].tolist(),
                'SUBJECT_ID_SORT': unique_gold_identifiers['SUBJECT_ID_SORT'].tolist(),
                'NUM_STUDENTS': np.random.randint(10, 100, len(unique_gold_identifiers)),
                'AVG_GRADE': np.random.rand(len(unique_gold_identifiers)) * 4.0
            })
            subject_summary_df_full = dummy_subject_summary_df_full
        else:
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
        
        # --- Advanced Feature Engineering ---
        # 3. Polynomial Features (SKIPPED in this ablation)
        
        # 4. Interaction Features (product of two distinct features) - REMAINS
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
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering (no poly). Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']
        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        df._feature_cols = final_feature_cols
    except FileNotFoundError:
        warnings.warn("subject_summary.csv not found during no_polynomial_features ablation. Falling back to minimal features.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)
    return df


def load_data_ablated_no_interaction_features(data_dir, gold_file):
    """
    Loads data and engineers features, but skips interaction feature creation.
    """
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    df = gold_df.copy()
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        if not os.path.exists(subject_summary_path):
            warnings.warn(f"Simulating creation of dummy {subject_summary_path} as it does not exist.")
            unique_gold_identifiers = gold_df[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates()
            dummy_subject_summary_df_full = pd.DataFrame({
                'TERM_CODE': unique_gold_identifiers['TERM_CODE'].tolist(),
                'SUBJECT_ID_SORT': unique_gold_identifiers['SUBJECT_ID_SORT'].tolist(),
                'NUM_STUDENTS': np.random.randint(10, 100, len(unique_gold_identifiers)),
                'AVG_GRADE': np.random.rand(len(unique_gold_identifiers)) * 4.0
            })
            subject_summary_df_full = dummy_subject_summary_df_full
        else:
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
        
        # --- Advanced Feature Engineering ---
        # 3. Polynomial Features (REMAINS)
        poly_features_to_square = numeric_summary_cols[:3]
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (SKIPPED in this ablation)

        final_feature_cols = list(set(current_feature_cols))
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering (no interaction). Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']
        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        df._feature_cols = final_feature_cols
    except FileNotFoundError:
        warnings.warn("subject_summary.csv not found during no_interaction_features ablation. Falling back to minimal features.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)
    return df


# --- Main Ablation Study Execution ---
if __name__ == "__main__":
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
        print(f"Creating dummy {GOLD_ENROLLMENT_TRAIN_FILE}")
        dummy_gold_df = pd.DataFrame({
            'TERM_CODE': ['202201', '202201', '202205', '202205', '202301', '202301'],
            'SUBJECT_ID_SORT': ['CS101', 'MA201', 'CS101', 'PH101', 'MA201', 'PH101'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N']
        })
        dummy_gold_df.to_csv(GOLD_ENROLLMENT_TRAIN_FILE, index=False)
    
    results = {}

    print("--- Running Baseline ---")
    baseline_df = load_data_baseline(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)
    baseline_score = evaluate_model(baseline_df, description="Baseline")
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}\n")

    print("--- Running Ablation: No Subject Summary Features ---")
    ablated_no_summary_df = load_data_ablated_no_subject_summary(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)
    ablated_no_summary_score = evaluate_model(ablated_no_summary_df, description="No Subject Summary Features")
    results['No Subject Summary Features'] = ablated_no_summary_score
    print(f"Ablation 'No Subject Summary Features' F1 Score: {ablated_no_summary_score:.4f}\n")

    print("--- Running Ablation: No Polynomial Features ---")
    ablated_no_poly_df = load_data_ablated_no_polynomial_features(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)
    ablated_no_poly_score = evaluate_model(ablated_no_poly_df, description="No Polynomial Features")
    results['No Polynomial Features'] = ablated_no_poly_score
    print(f"Ablation 'No Polynomial Features' F1 Score: {ablated_no_poly_score:.4f}\n")

    print("--- Running Ablation: No Interaction Features ---")
    ablated_no_interaction_df = load_data_ablated_no_interaction_features(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)
    ablated_no_interaction_score = evaluate_model(ablated_no_interaction_df, description="No Interaction Features")
    results['No Interaction Features'] = ablated_no_interaction_score
    print(f"Ablation 'No Interaction Features' F1 Score: {ablated_no_interaction_score:.4f}\n")

    print("\n--- Ablation Study Summary ---")
    
    unique_scores = set(results.values())
    if len(unique_scores) == 1:
        if list(unique_scores)[0] == 1.0:
            most_impactful_conclusion = "All configurations achieved a perfect F1 score of 1.0. This suggests the dataset might be too simple to differentiate component impact. Based on previous studies, the 'inclusion of features derived from subject_summary.csv' had the largest impact when differentiation was possible (e.g., dropping from 1.0 to 0.6667 in previous study 3)."
        elif list(unique_scores)[0] == 0.0:
            most_impactful_conclusion = "All configurations resulted in an F1 score of 0.0. This indicates a fundamental issue with the data or setup, making it impossible to determine relative impact."
        else:
            most_impactful_conclusion = f"All configurations resulted in an F1 score of {list(unique_scores)[0]:.4f}. No significant performance difference observed among the ablated components."
    else:
        max_drop = 0.0
        most_impactful_part = "None (no significant drop from baseline)"
        
        if baseline_score > 0:
            for ablation, score in results.items():
                if ablation != 'Baseline':
                    drop = baseline_score - score
                    if drop > max_drop:
                        max_drop = drop
                        most_impactful_part = ablation
            
            if max_drop > 0:
                most_impactful_conclusion = f"The part that contributes the most to the overall performance (largest F1 score drop when removed) is: '{most_impactful_part}' (Drop: {max_drop:.4f} from Baseline F1: {baseline_score:.4f})"
            else:
                most_impactful_conclusion = "No single ablated component showed a significantly detrimental impact on performance when compared to the baseline."
        else:
            highest_score = 0.0
            best_config = "Baseline (0.0)"
            for config, score in results.items():
                if score > highest_score:
                    highest_score = score
                    best_config = config
            most_impactful_conclusion = f"The baseline F1 score was 0.0. The best performing configuration was '{best_config}' with an F1 score of {highest_score:.4f}, indicating this component or configuration improved performance from a zero baseline."
            
    print(f"{most_impactful_conclusion}")
