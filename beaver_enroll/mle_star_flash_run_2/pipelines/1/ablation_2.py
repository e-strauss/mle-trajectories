
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
def load_data(data_dir, gold_file, include_summary_features=True, nan_imputation_strategy='zero'):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters:
    - include_summary_features: If False, skips merging subject_summary.csv and its derived features.
    - nan_imputation_strategy: 'zero' or 'mean' for numerical NaN imputation.
    """
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    processed_summary_features = False
    if include_summary_features and os.path.exists(subject_summary_path):
        try:
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
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
            
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

            # Apply chosen NaN imputation strategy
            if nan_imputation_strategy == 'zero':
                df[final_feature_cols] = df[final_feature_cols].fillna(0)
            elif nan_imputation_strategy == 'mean':
                for col in final_feature_cols:
                    if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())

            df._feature_cols = final_feature_cols
            processed_summary_features = True

        except FileNotFoundError:
            print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
            # Fallback will be handled by the subsequent 'else' block if not processed
        except Exception as e:
            print(f"Error during subject_summary processing: {e}. Falling back to minimal features.")


    if not processed_summary_features:
        # Fallback/minimal features path, used if subject_summary is excluded or not found/processed
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        
        # Apply chosen NaN imputation strategy
        if nan_imputation_strategy == 'zero':
            df[df._feature_cols] = df[df._feature_cols].fillna(0)
        elif nan_imputation_strategy == 'mean':
            for col in df._feature_cols:
                if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())

    return df

# --- Main script (modified for ablation) ---
def run_training_and_validation(include_summary_features=True, nan_imputation_strategy='zero', use_class_weight=False):
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                   include_summary_features=include_summary_features,
                   nan_imputation_strategy=nan_imputation_strategy)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
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
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    model_params = {'random_state': 42, 'n_estimators': 100}
    if use_class_weight:
        model_params['class_weight'] = 'balanced'

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

    return final_validation_score

# Run the ablation study
if __name__ == "__main__":
    results = {}

    # Baseline: Original Solution
    print("--- Running Baseline Solution ---")
    results['Baseline'] = run_training_and_validation(
        include_summary_features=True, 
        nan_imputation_strategy='zero', 
        use_class_weight=False
    )
    print(f"Baseline F1 Score: {results['Baseline']:.4f}\n")

    # Ablation 1: Exclude features derived from subject_summary.csv
    print("--- Running Ablation: Exclude Subject Summary Features ---")
    results['Exclude Subject Summary Features'] = run_training_and_validation(
        include_summary_features=False, 
        nan_imputation_strategy='zero', 
        use_class_weight=False
    )
    print(f"Ablation (Exclude Subject Summary Features) F1 Score: {results['Exclude Subject Summary Features']:.4f}\n")

    # Ablation 2: Change NaN imputation strategy from 0 to Mean
    print("--- Running Ablation: NaN Imputation to Mean ---")
    results['NaN Imputation to Mean'] = run_training_and_validation(
        include_summary_features=True, 
        nan_imputation_strategy='mean', 
        use_class_weight=False
    )
    print(f"Ablation (NaN Imputation to Mean) F1 Score: {results['NaN Imputation to Mean']:.4f}\n")

    # Ablation 3: Add class_weight='balanced' to RandomForestClassifier
    print("--- Running Ablation: Add class_weight='balanced' ---")
    results['Add class_weight=balanced'] = run_training_and_validation(
        include_summary_features=True, 
        nan_imputation_strategy='zero', 
        use_class_weight=True
    )
    print(f"Ablation (Add class_weight=balanced) F1 Score: {results['Add class_weight=balanced']:.4f}\n")

    print("--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    # Determine the part that contributes the most
    # Assuming lower score indicates more critical removed component, or higher score indicates critical added component
    baseline_score = results['Baseline']
    
    # Check if all scores are identical (e.g., due to simple dummy data)
    if all(score == baseline_score for score in results.values()):
        print("\nAll ablation scenarios, including the baseline, yielded the same performance.")
        print("This suggests that the dataset used is too simple to reveal significant performance differences or the contributions of the ablated components.")
        print("Therefore, based on this study, it's not possible to definitively determine which part contributes the most.")
    else:
        # If there are differences, find the one that deviates most from baseline
        performance_impacts = {name: abs(score - baseline_score) for name, score in results.items() if name != 'Baseline'}
        if performance_impacts:
            most_impactful_ablation = max(performance_impacts, key=performance_impacts.get)
            print(f"\nThe part of the code that contributes most to the overall performance, based on this ablation study, is related to: '{most_impactful_ablation}'.")
            print(f"Its modification resulted in a performance change of {performance_impacts[most_impactful_ablation]:.4f} compared to the baseline.")
        else:
            print("\nNo ablations were performed or calculated for impact analysis.")

