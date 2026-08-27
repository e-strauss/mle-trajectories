

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
from scipy import stats

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features (modified for ablation) ---
def load_data_ablated(data_dir, gold_file, impute_nan_strategy='zero', exclude_term_code_interactions=False):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Accepts ablation parameters for imputation strategy and interaction features.
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

        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        current_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # Polynomial Features
        poly_features_to_square = numeric_summary_cols[:3]
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # Interaction Features
        # Interaction between encoded TERM_CODE and a numeric summary feature
        if not exclude_term_code_interactions: # Ablation control point
            if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
                col_to_interact = numeric_summary_cols[0] 
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

        # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature
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

        # Interaction between two distinct numeric summary features
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

        # Ablation control for NaN imputation
        if impute_nan_strategy == 'zero':
            df[final_feature_cols] = df[final_feature_cols].fillna(0)
        elif impute_nan_strategy == 'mean':
            # Impute with mean only if the column is numeric and has NaNs
            for col in final_feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
                # Fallback to 0 if mean is NaN (e.g. column was all NaNs or non-numeric after all)
                df[col] = df[col].fillna(0) 
        
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        # Fallback to minimal features if subject_summary.csv is not found
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main training and validation function (modified for ablation) ---
def run_ablation_training(disable_similarity_filter=False, impute_nan_strategy='zero', exclude_term_code_interactions=False):
    """
    Runs the training and validation process with specified ablation configurations.
    Returns the F1 score.
    """
    df = load_data_ablated(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                           impute_nan_strategy=impute_nan_strategy,
                           exclude_term_code_interactions=exclude_term_code_interactions)

    if df.empty:
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        return 0.0

    validation_term = unique_terms[-1] 
    val_df = df[df['TERM_CODE'] == validation_term]

    # --- Ablation Control for Similarity Filtering ---
    if disable_similarity_filter: # Ablation control point
        train_df = df[df['TERM_CODE'] != validation_term]
    else:
        # Identify potential 'key features' for similarity comparison
        # Exclude 'TERM_CODE' itself and the target variable 'HIGH_ENROLLMENT'
        comparison_features = [col for col in df.columns if col not in ['TERM_CODE', 'HIGH_ENROLLMENT']] 

        if not comparison_features:
            # Fallback if no comparison features are available
            train_df = df[df['TERM_CODE'] != validation_term]
        else:
            # --- Similarity Parameters (can be tuned based on dataset characteristics) ---
            NUMERICAL_P_THRESHOLD = 0.7 
            CATEGORICAL_DIFF_THRESHOLD = 0.15 
            TERM_SIMILARITY_RATIO_THRESHOLD = 0.5 
            # --- End Similarity Parameters ---

            historical_terms = [t for t in unique_terms if t != validation_term]
            terms_to_exclude_from_training = []

            for current_term in historical_terms:
                current_term_data = df[df['TERM_CODE'] == current_term]
                if len(current_term_data) == 0:
                    continue

                similar_feature_count = 0
                
                for feature in comparison_features:
                    if feature not in val_df.columns or feature not in current_term_data.columns:
                        continue
                    
                    val_feature_data = val_df[feature].dropna()
                    current_feature_data = current_term_data[feature].dropna()

                    if len(val_feature_data) == 0 or len(current_feature_data) == 0:
                        continue

                    try:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            if val_feature_data.nunique() > 1 and current_feature_data.nunique() > 1:
                                statistic, p_value = stats.ks_2samp(val_feature_data, current_feature_data)
                                if p_value > NUMERICAL_P_THRESHOLD:
                                    similar_feature_count += 1
                            elif val_feature_data.nunique() == 1 and current_feature_data.nunique() == 1:
                                if val_feature_data.iloc[0] == current_feature_data.iloc[0]:
                                    similar_feature_count += 1
                        else: # Categorical feature
                            vc_val = val_feature_data.value_counts(normalize=True).sort_index()
                            vc_curr = current_feature_data.value_counts(normalize=True).sort_index()

                            all_cats = pd.Index(np.union1d(vc_val.index, vc_curr.index))
                            vc_val_aligned = vc_val.reindex(all_cats, fill_value=0)
                            vc_curr_aligned = vc_curr.reindex(all_cats, fill_value=0)

                            diff = (vc_val_aligned - vc_curr_aligned).abs().sum()
                            if diff < CATEGORICAL_DIFF_THRESHOLD:
                                similar_feature_count += 1
                    except Exception as e:
                        pass

                if len(comparison_features) > 0 and \
                   (similar_feature_count / len(comparison_features)) >= TERM_SIMILARITY_RATIO_THRESHOLD:
                    terms_to_exclude_from_training.append(current_term)
            
            train_terms = [t for t in unique_terms if t != validation_term and t not in terms_to_exclude_from_training]
            
            if not train_terms:
                # Fallback: if all historical terms are filtered out, use all terms except validation_term
                train_df = df[df['TERM_CODE'] != validation_term]
            else:
                train_df = df[df['TERM_CODE'].isin(train_terms)]
            
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

    if X_train.empty or y_train.empty:
        return 0.0

    if len(y_train.unique()) < 2:
        pass # Proceeding with training, as this is a valid (though not ideal) training scenario.
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    final_validation_score = 0.0
    
    if y_val.empty:
        pass
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# --- Main ablation study execution ---
if __name__ == "__main__":
    results = {}

    # Baseline: Original code (with similarity filtering, zero imputation, all interaction features)
    print("Running Baseline (full solution)...")
    baseline_score = run_ablation_training(
        disable_similarity_filter=False,
        impute_nan_strategy='zero',
        exclude_term_code_interactions=False
    )
    results['Baseline'] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}")

    # Ablation 1: Disable Similarity-Based Term Exclusion
    print("\nRunning Ablation 1: Disable Similarity-Based Term Exclusion (reverting to all historical terms for training)...")
    ablation1_score = run_ablation_training(
        disable_similarity_filter=True, # This is the change
        impute_nan_strategy='zero',
        exclude_term_code_interactions=False
    )
    results['Ablation 1: Disable Similarity-Based Term Exclusion'] = ablation1_score
    print(f"Ablation 1 F1 Score: {ablation1_score:.4f}")

    # Ablation 2: Change NaN Imputation for Features to Mean (instead of zero)
    print("\nRunning Ablation 2: Change NaN Imputation for Features to Mean...")
    ablation2_score = run_ablation_training(
        disable_similarity_filter=False,
        impute_nan_strategy='mean', # This is the change
        exclude_term_code_interactions=False
    )
    results['Ablation 2: Change NaN Imputation for Features to Mean'] = ablation2_score
    print(f"Ablation 2 F1 Score: {ablation2_score:.4f}")

    # Ablation 3: Remove TERM_CODE_ENCODED Interaction Features
    print("\nRunning Ablation 3: Remove TERM_CODE_ENCODED Interaction Features...")
    ablation3_score = run_ablation_training(
        disable_similarity_filter=False,
        impute_nan_strategy='zero',
        exclude_term_code_interactions=True # This is the change
    )
    results['Ablation 3: Remove TERM_CODE_ENCODED Interaction Features'] = ablation3_score
    print(f"Ablation 3 F1 Score: {ablation3_score:.4f}")

    print("\n--- Ablation Study Results Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    # Determine the most impactful part
    most_impactful_part = "None of the ablated parts showed a significant drop, or the dataset is too simple to differentiate."
    max_drop = 0.0

    # Ensure baseline is not 0.0 before comparing for drops
    if baseline_score > 0.0:
        for name, score in results.items():
            if name == 'Baseline':
                continue
            drop = baseline_score - score
            if drop > max_drop:
                max_drop = drop
                most_impactful_part = name
    
    # Consider a drop of at least 0.01 as significant
    if max_drop > 0.01: 
        print(f"\nThe part of the code that contributes the most to the overall performance (largest F1 score drop from baseline) is: {most_impactful_part} (F1 drop: {max_drop:.4f}).")
    elif baseline_score == 0.0:
        print("\nAll scores are 0.0. This indicates a problem with the data or setup, making it impossible to determine feature impact.")
    else:
        print(f"\nNone of the ablated parts caused a significant performance drop (all F1 scores are {baseline_score:.4f} or higher). This suggests the dataset is too simple, or the validation strategy is not challenging enough to differentiate the components' impact.")

