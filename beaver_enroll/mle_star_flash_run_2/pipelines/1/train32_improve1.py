
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

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
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

# --- Main script ---
def run_training_and_validation():
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = sorted(df['TERM_CODE'].unique()) # Ensure chronological order for time-series split

    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required). No folds generated.")
        rolling_cv_folds = iter([]) # Assign an empty iterable
    else:
        def _generate_rolling_folds_impl(dataframe, term_column, unique_terms_list):
            for i in range(1, len(unique_terms_list)):
                validation_term = unique_terms_list[i]
                train_terms = unique_terms_list[:i]
                
                train_df_fold = dataframe[dataframe[term_column].isin(train_terms)]
                val_df_fold = dataframe[dataframe[term_column] == validation_term]
                
                yield train_df_fold, val_df_fold

        rolling_cv_folds = _generate_rolling_folds_impl(df, 'TERM_CODE', unique_terms)

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
        print("Final Validation Performance: 0.0")
        return

    final_validation_score = 0.0 # Default score if no folds are processed or issues arise
    all_fold_f1_scores = []

    # Iterate over the generated folds for training and validation
    for fold_idx, (train_df_fold, val_df_fold) in enumerate(rolling_cv_folds):
        print(f"Processing fold {fold_idx + 1}...")

        X_train_fold = train_df_fold[feature_cols]
        y_train_fold = train_df_fold['HIGH_ENROLLMENT']
        X_val_fold = val_df_fold[feature_cols]
        y_val_fold = val_df_fold['HIGH_ENROLLMENT']

        # Check for empty training data
        if X_train_fold.empty or y_train_fold.empty:
            print(f"Fold {fold_idx + 1}: Training set is empty. Skipping this fold.")
            continue

        # Check if target variable in training set has only one class
        if len(y_train_fold.unique()) < 2:
            print(f"Fold {fold_idx + 1}: Training set target 'HIGH_ENROLLMENT' has only one class: {y_train_fold.unique()}. This might lead to trivial predictions.")
        
        # Model Training for the current fold
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_fold, y_train_fold)

        # --- F1 Score Calculation with robustness checks for the current fold ---
        current_fold_f1_score = 0.0
        
        if y_val_fold.empty:
            print(f"Fold {fold_idx + 1}: Validation set is empty. F1 score cannot be calculated for this fold.")
        else:
            y_pred_fold = model.predict(X_val_fold)
            
            unique_y_val_fold = np.unique(y_val_fold)
            unique_y_pred_fold = np.unique(y_pred_fold)

            if len(unique_y_val_fold) < 2:
                print(f"Fold {fold_idx + 1}: Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val_fold}. Macro F1 calculation adjusted for this edge case.")
                if len(unique_y_pred_fold) == 1 and unique_y_pred_fold[0] == unique_y_val_fold[0]:
                    current_fold_f1_score = 1.0
                else:
                    current_fold_f1_score = 0.0
            else:
                current_fold_f1_score = f1_score(y_val_fold, y_pred_fold, average='macro', zero_division=0)
        
        all_fold_f1_scores.append(current_fold_f1_score)
        print(f"Fold {fold_idx + 1} Validation Performance: {current_fold_f1_score}")

    if all_fold_f1_scores:
        # As per the task, "latest train years held out", we take the score of the last fold.
        final_validation_score = all_fold_f1_scores[-1]
    else:
        final_validation_score = 0.0 # No folds were processed, or all folds were skipped

    print(f"Final Validation Performance: {final_validation_score}")

# Run the training and validation process
if __name__ == "__main__":
    run_training_and_validation()
