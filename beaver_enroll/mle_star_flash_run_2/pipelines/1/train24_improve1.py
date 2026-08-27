
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
from scipy import stats # Moved this import to the top

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
    
    # Separate validation set
    val_df = df[df['TERM_CODE'] == validation_term]

    # --- Start of Improvement Plan Implementation ---

    # Identify potential 'key features' for similarity comparison
    # Exclude 'TERM_CODE' itself. If there's a target variable or ID column,
    # they should also be excluded from this list. For this example, we assume
    # all other columns are features.
    comparison_features = [col for col in df.columns if col not in ['TERM_CODE']]

    if not comparison_features:
        # Fallback: If no features are available for comparison, proceed with standard time-based split
        print("Warning: No comparison features identified. Proceeding with standard time-based split.")
        train_df = df[df['TERM_CODE'] != validation_term]
    else:
        # --- Similarity Parameters (can be tuned based on dataset characteristics) ---
        # For numerical features: p-value threshold for Kolmogorov-Smirnov test.
        # A higher p-value indicates that we fail to reject the null hypothesis
        # that the two samples come from the same distribution (i.e., they are similar).
        NUMERICAL_P_THRESHOLD = 0.7 
        
        # For categorical features: Sum of absolute differences threshold for normalized frequency distributions.
        # A lower value indicates more similar distributions. Ranges from 0 (identical) to 2 (completely disjoint).
        CATEGORICAL_DIFF_THRESHOLD = 0.15 
        
        # Proportion of features that must be 'similar' for a historical term to be excluded
        # from the training set.
        TERM_SIMILARITY_RATIO_THRESHOLD = 0.5 
        # --- End Similarity Parameters ---

        historical_terms = [t for t in unique_terms if t != validation_term]
        terms_to_exclude_from_training = []

        for current_term in historical_terms:
            current_term_data = df[df['TERM_CODE'] == current_term]
            
            # Skip comparison if the historical term's data is empty or too small
            if len(current_term_data) == 0:
                continue

            similar_feature_count = 0
            
            for feature in comparison_features:
                # Ensure feature exists in both dataframes and has some data
                if feature not in val_df.columns or feature not in current_term_data.columns:
                    continue
                
                # Drop NAs for comparison and ensure there's enough data for stats
                val_feature_data = val_df[feature].dropna()
                current_feature_data = current_term_data[feature].dropna()

                if len(val_feature_data) == 0 or len(current_feature_data) == 0:
                    continue # Cannot compare empty data

                try:
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        # Numerical feature comparison using Kolmogorov-Smirnov test
                        if val_feature_data.nunique() > 1 and current_feature_data.nunique() > 1:
                            # KS test requires at least two unique values for meaningful comparison
                            statistic, p_value = stats.ks_2samp(val_feature_data, current_feature_data)
                            if p_value > NUMERICAL_P_THRESHOLD:
                                similar_feature_count += 1
                        elif val_feature_data.nunique() == 1 and current_feature_data.nunique() == 1:
                            # If both are constant, check if the constant value is the same
                            if val_feature_data.iloc[0] == current_feature_data.iloc[0]:
                                similar_feature_count += 1
                        # If one is constant and the other is not, or constants are different, they are not considered similar
                    else:
                        # Categorical feature comparison using sum of absolute differences of normalized frequencies
                        vc_val = val_feature_data.value_counts(normalize=True).sort_index()
                        vc_curr = current_feature_data.value_counts(normalize=True).sort_index()

                        # Align indices to compare correctly, handling categories present in one but not the other
                        all_cats = pd.Index(np.union1d(vc_val.index, vc_curr.index))
                        vc_val_aligned = vc_val.reindex(all_cats, fill_value=0)
                        vc_curr_aligned = vc_curr.reindex(all_cats, fill_value=0)

                        diff = (vc_val_aligned - vc_curr_aligned).abs().sum()
                        if diff < CATEGORICAL_DIFF_THRESHOLD:
                            similar_feature_count += 1
                except Exception as e:
                    # Catch potential errors during comparison (e.g., issues with specific feature data)
                    # print(f"Skipping feature '{feature}' for term '{current_term}' due to error: {e}")
                    pass # Continue to the next feature

            # Determine if the entire historical term is too similar to the validation term
            if len(comparison_features) > 0 and \
               (similar_feature_count / len(comparison_features)) >= TERM_SIMILARITY_RATIO_THRESHOLD:
                terms_to_exclude_from_training.append(current_term)
        
        # Construct the training set by excluding the validation term and overly similar historical terms
        train_terms = [t for t in unique_terms if t != validation_term and t not in terms_to_exclude_from_training]
        
        if not train_terms:
            print(f"Warning: After filtering for similarity, no historical terms remain for training.")
            print(f"       Reverting to using all historical terms for training to avoid an empty training set.")
            # Fallback: if all historical terms are filtered out, use all terms except validation_term
            train_df = df[df['TERM_CODE'] != validation_term]
        else:
            train_df = df[df['TERM_CODE'].isin(train_terms)]
            
    # --- End of Improvement Plan Implementation ---

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

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

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
