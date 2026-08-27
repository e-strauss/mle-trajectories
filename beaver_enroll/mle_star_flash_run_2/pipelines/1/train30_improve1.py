
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import os
import warnings
from sklearn.impute import SimpleImputer

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
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        print("Not enough unique terms for a time-based validation split (at least 2 required).")
        print("Final Validation Performance: 0.0")
        return

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
        print("No usable features available for training after all fallback attempts.")
        print("Final Validation Performance: 0.0")
        return

    # Preserve the target variables from the original block
    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # Make copies of the feature dataframes to avoid modifying the original dataframes
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    # Define the special categorical column as per the plan
    # Ensure 'TERM_CODE' is handled as string for OHE if it was intended as such.
    # From previous lines: df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    # The 'TERM_CODE_ENCODED' is already numeric, so we use the original 'TERM_CODE' if it's in feature_cols
    # and we want to treat it as a special categorical for OHE.
    
    # If 'TERM_CODE' was in `feature_cols` as a numerical value, it would have been removed from
    # `numerical_cols` in the previous iteration of the code. We need to check if 'TERM_CODE'
    # itself was intended to be one-hot encoded or if 'TERM_CODE_ENCODED' was.
    # Based on the problem description and typical ML workflows, raw TERM_CODE is better for OHE.
    
    term_code_col_raw = 'TERM_CODE' # Use the raw TERM_CODE for OHE
    
    # Ensure that `term_code_col_raw` is available in the X_train/X_val dataframes for OHE.
    # It might not be in `feature_cols` if only `TERM_CODE_ENCODED` was selected as a feature.
    # So we explicitly add it to X_train and X_val if not present for OHE.
    if term_code_col_raw not in X_train.columns:
        X_train[term_code_col_raw] = train_df[term_code_col_raw]
    if term_code_col_raw not in X_val.columns:
        X_val[term_code_col_raw] = val_df[term_code_col_raw]

    # Identify numerical features for imputation and interaction.
    # Exclude 'TERM_CODE' (raw) from numerical features if it's numeric-like but meant as categorical.
    numerical_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X_train[col]) and col != term_code_col_raw]
    
    # Identify other categorical features that are not TERM_CODE.
    # This also handles features like 'SUBJECT_ID_SORT' if it was not numerically encoded.
    other_categorical_cols = [col for col in feature_cols if col not in numerical_cols and col != term_code_col_raw]
    
    # If TERM_CODE_ENCODED is in numerical_cols, it means it's treated as a numerical feature,
    # which is likely the intent for most features unless explicitly one-hot encoded.
    # The current logic will one-hot encode `term_code_col_raw` (original TERM_CODE string).
    # If `TERM_CODE_ENCODED` is also in `numerical_cols`, it will be kept as a numerical feature.

    # 1. Median Imputation for Numerical Features
    imputer = SimpleImputer(strategy='median')

    # Fit on X_train's numerical columns and transform both X_train and X_val
    if numerical_cols: # Only impute if there are numerical columns
        X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
        X_val[numerical_cols] = imputer.transform(X_val[numerical_cols])


    # 2. One-Hot Encode TERM_CODE (raw string)
    # Ensure term_code_col_raw is treated as string for OHE
    X_train[term_code_col_raw] = X_train[term_code_col_raw].astype(str)
    X_val[term_code_col_raw] = X_val[term_code_col_raw].astype(str)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit OHE on the training data's TERM_CODE
    ohe.fit(X_train[[term_code_col_raw]])

    # Transform both training and validation TERM_CODE
    term_code_ohe_train = ohe.transform(X_train[[term_code_col_raw]])
    term_code_ohe_val = ohe.transform(X_val[[term_code_col_raw]])

    # Get feature names for the one-hot encoded columns
    ohe_feature_names = ohe.get_feature_names_out([term_code_col_raw])

    # Create DataFrames from the one-hot encoded arrays, preserving index
    term_code_ohe_df_train = pd.DataFrame(term_code_ohe_train, columns=ohe_feature_names, index=X_train.index)
    term_code_ohe_df_val = pd.DataFrame(term_code_ohe_val, columns=ohe_feature_names, index=X_val.index)


    # 3. Engineer Interaction Features
    # These are interactions between the imputed numerical features and the one-hot encoded TERM_CODE features.
    interaction_features_train = pd.DataFrame(index=X_train.index)
    interaction_features_val = pd.DataFrame(index=X_val.index)

    if numerical_cols and ohe_feature_names.size > 0: # Only create interactions if both exist
        for num_col in numerical_cols:
            for ohe_col in ohe_feature_names:
                # Create a more readable column name for interaction features
                new_col_name = f"{num_col}_x_{ohe_col.replace(term_code_col_raw + '_', '')}"
                interaction_features_train[new_col_name] = X_train[num_col] * term_code_ohe_df_train[ohe_col]
                interaction_features_val[new_col_name] = X_val[num_col] * term_code_ohe_df_val[ohe_col]


    # 4. Combine all processed features
    # Concatenate imputed numerical features, other unchanged categorical features,
    # one-hot encoded TERM_CODE features, and new interaction features.
    
    # Start with the imputed numerical columns
    X_train_final_components = [X_train[numerical_cols]] if numerical_cols else []
    X_val_final_components = [X_val[numerical_cols]] if numerical_cols else []
    
    # Add other categorical columns (if any, unchanged in their original form)
    if other_categorical_cols:
        X_train_final_components.append(X_train[other_categorical_cols])
        X_val_final_components.append(X_val[other_categorical_cols])

    # Add one-hot encoded TERM_CODE features
    X_train_final_components.append(term_code_ohe_df_train)
    X_val_final_components.append(term_code_ohe_df_val)

    # Add new interaction features
    if not interaction_features_train.empty:
        X_train_final_components.append(interaction_features_train)
    if not interaction_features_val.empty:
        X_val_final_components.append(interaction_features_val)

    # Concatenate all components
    X_train = pd.concat(X_train_final_components, axis=1)
    X_val = pd.concat(X_val_final_components, axis=1)


    # Check for empty training data
    if X_train.empty or y_train.empty:
        print("Training set is empty. Cannot train a model.")
        print("Final Validation Performance: 0.0")
        return

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
        # Proceeding with training, as this is a valid (though not ideal) training scenario.
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=100)
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
            # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set
            # zero_division=0 ensures that if a class has no true instances or no predicted instances,
            # its F1 score contribution is 0, preventing division by zero warnings/errors.
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}")

# Run the training and validation process
if __name__ == "__main__":
    run_training_and_validation()
