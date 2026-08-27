
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, use_mi_selection=True, n_top_features_val=7, include_encoded_id_interactions=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Configurable for ablation study.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})
    y = df['HIGH_ENROLLMENT'] # Define y here for mutual_info_regression

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
        N_TOP_FEATURES = n_top_features_val # Use the passed value for ablation

        # Filter numeric_summary_cols to ensure they exist in the current dataframe and are numeric
        available_numeric_cols_for_selection = [
            col for col in numeric_summary_cols
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        top_numeric_features = []
        if available_numeric_cols_for_selection and y is not None:
            if use_mi_selection: # Ablation point 1: Disable MI selection
                # Calculate mutual information for numerical features with the target variable
                try:
                    mi_scores = mutual_info_regression(df[available_numeric_cols_for_selection], y, random_state=42)
                    mi_series = pd.Series(mi_scores, index=available_numeric_cols_for_selection)
                    
                    # Select top N numerical features based on MI score
                    top_numeric_features = mi_series.nlargest(N_TOP_FEATURES).index.tolist()
                except ValueError:
                    # Fallback if MI calculation fails (e.g., target is constant, or features are constant)
                    # Select the first N available numeric features as a fallback
                    top_numeric_features = available_numeric_cols_for_selection[:N_TOP_FEATURES]
            else: # Fallback to simply taking the first N features without MI
                top_numeric_features = available_numeric_cols_for_selection[:N_TOP_FEATURES]
        else:
            # Fallback if no numeric columns are available or target 'y' is missing/None
            top_numeric_features = available_numeric_cols_for_selection[:N_TOP_FEATURES]

        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        # Generate squared terms exclusively for the identified top numerical features.
        for col in top_numeric_features:
            new_col_name = f"{col}_SQUARED"
            if col in df.columns and new_col_name not in df.columns:
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)

        # 4. Interaction Features (product of two distinct features)
        # Create interaction terms based on top numerical features and key encoded categorical features.

        key_encoded_cats = []
        if 'TERM_CODE_ENCODED' in df.columns:
            key_encoded_cats.append('TERM_CODE_ENCODED')
        if 'SUBJECT_ID_SORT_ENCODED' in df.columns:
            key_encoded_cats.append('SUBJECT_ID_SORT_ENCODED')

        if include_encoded_id_interactions: # Ablation point 3: Remove encoded ID interactions
            # Interaction between key encoded categorical features and top numerical features
            for cat_col in key_encoded_cats:
                for num_col in top_numeric_features:
                    new_col_name = f"{num_col}_x_{cat_col}"
                    if num_col in df.columns and cat_col in df.columns and new_col_name not in df.columns:
                        df[new_col_name] = df[num_col] * df[cat_col]
                        current_feature_cols.append(new_col_name)

        # Interaction between two distinct top numerical features
        if len(top_numeric_features) >= 2:
            for col1, col2 in combinations(top_numeric_features, 2):
                new_col_name = f"{col1}_x_{col2}"
                if col1 in df.columns and col2 in df.columns and new_col_name not in df.columns:
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

# --- Main training and validation function, modified to accept ablation parameters ---
def run_ablation_experiment(experiment_name, use_mi_selection=True, n_top_features_val=7, include_encoded_id_interactions=True):
    print(f"--- Running Experiment: {experiment_name} ---")
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                   use_mi_selection=use_mi_selection,
                   n_top_features_val=n_top_features_val,
                   include_encoded_id_interactions=include_encoded_id_interactions)

    if df.empty:
        print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
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

    print(f"Final Validation Performance for {experiment_name}: {final_validation_score}")
    return final_validation_score

# --- Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Baseline
    baseline_score = run_ablation_experiment("Baseline (Full Feature Engineering)",
                                             use_mi_selection=True,
                                             n_top_features_val=7,
                                             include_encoded_id_interactions=True)
    results["Baseline"] = baseline_score

    # Ablation 1: Disable Mutual Information for N_TOP_FEATURES selection
    # Instead of MI, it will just take the first N_TOP_FEATURES available numerical columns.
    ablation1_score = run_ablation_experiment("Ablation 1: No Mutual Information Feature Selection",
                                              use_mi_selection=False,
                                              n_top_features_val=7, # Keep N_TOP_FEATURES same as baseline
                                              include_encoded_id_interactions=True)
    results["Ablation 1: No MI Selection"] = ablation1_score

    # Ablation 2: Reduce N_TOP_FEATURES for polynomial and interaction features
    # Original is 7, let's reduce to 3.
    ablation2_score = run_ablation_experiment("Ablation 2: Reduced N_TOP_FEATURES (to 3)",
                                              use_mi_selection=True,
                                              n_top_features_val=3,
                                              include_encoded_id_interactions=True)
    results["Ablation 2: Reduced N_TOP_FEATURES"] = ablation2_score

    # Ablation 3: Remove interaction features involving encoded identifiers
    ablation3_score = run_ablation_experiment("Ablation 3: No Encoded Identifier Interaction Features",
                                              use_mi_selection=True,
                                              n_top_features_val=7,
                                              include_encoded_id_interactions=False)
    results["Ablation 3: No Encoded ID Interactions"] = ablation3_score


    print("\n--- Ablation Study Summary ---")
    most_impactful_component = ""
    max_drop = 0.0

    for name, score in results.items():
        print(f"{name}: {score:.4f}")
        if name != "Baseline":
            drop = baseline_score - score
            if drop > max_drop:
                max_drop = drop
                most_impactful_component = name

    print("\n--- Conclusion ---")
    if baseline_score == 0.0:
        print("The baseline model achieved an F1 score of 0.0, indicating a potential issue with the data or model setup, making it difficult to draw meaningful conclusions from the ablations.")
    elif max_drop > 0:
        print(f"The part of the code that contributes the most to the overall performance is '{most_impactful_component}', which caused an F1 score drop of {max_drop:.4f}.")
    else:
        print("All ablations performed identically to or better than the baseline, or the baseline itself achieved a perfect score, suggesting that the ablated components are not critical or the dataset is too simple to show significant differences.")
