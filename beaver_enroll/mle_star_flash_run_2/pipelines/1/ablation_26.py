
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_data_ablated(data_dir, gold_file, use_pca=True, pca_n_components=0.95, use_scaler_before_pca=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters:
    - use_pca: If False, PCA is skipped and original (scaled/imputed) numeric features are used.
    - pca_n_components: The n_components parameter for PCA (if use_pca is True).
    - use_scaler_before_pca: If False, StandardScaler is skipped before PCA (if use_pca is True)
                             or before using original numeric features (if use_pca is False).
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        # Identify original numeric feature columns from subject_summary, excluding identifiers
        summary_original_numeric_cols = [
            col for col in subject_summary_df_full.columns 
            if col not in identifier_cols and pd.api.types.is_numeric_dtype(subject_summary_df_full[col])
        ]

        numeric_features_for_advanced_fe = [] # Will hold feature names for poly/interaction

        if len(summary_original_numeric_cols) > 0:
            # Create a temporary DataFrame for processing numerical features and keep identifiers
            subject_features_processed_df = subject_summary_df_full[identifier_cols + summary_original_numeric_cols].copy()
            
            # Handle potential NaNs in numeric features by median imputation
            for col in summary_original_numeric_cols:
                if subject_features_processed_df[col].isnull().any():
                    subject_features_processed_df[col].fillna(subject_features_processed_df[col].median(), inplace=True)

            features_to_transform = subject_features_processed_df[summary_original_numeric_cols].copy()

            # Initialize features_transformed_df to ensure it's always defined
            features_transformed_df = features_to_transform.copy() # Default to unscaled, in case use_scaler_before_pca is False

            if use_scaler_before_pca:
                scaler = StandardScaler()
                features_transformed_array = scaler.fit_transform(features_to_transform)
                features_transformed_df = pd.DataFrame(features_transformed_array, columns=summary_original_numeric_cols)
            # If use_scaler_before_pca is False, features_transformed_df retains the value from its initialization (features_to_transform.copy())
            
            # Merge with identifier columns for a full DataFrame after preprocessing
            processed_subject_df = pd.concat([subject_features_processed_df[identifier_cols].reset_index(drop=True), features_transformed_df.reset_index(drop=True)], axis=1)

            if use_pca:
                # Apply PCA
                # Use the values from features_transformed_df, which is always defined
                features_for_pca = features_transformed_df.values
                
                if features_for_pca.shape[1] == 0:
                    warnings.warn("No features available for PCA after filtering. Skipping PCA.")
                    # Fallback to original features, which would be empty. This path is unlikely given above checks.
                    df = pd.merge(df, processed_subject_df, on=identifier_cols, how='left')
                    numeric_features_for_advanced_fe = []
                else:
                    pca = PCA(n_components=pca_n_components, random_state=42)
                    pca_components = pca.fit_transform(features_for_pca) # Use features_for_pca
                    pca_component_names = [f'pca_component_{i}' for i in range(pca_components.shape[1])]
                    subject_summary_pca_df = pd.DataFrame(data=pca_components, columns=pca_component_names)
                    subject_summary_pca_df = pd.concat([subject_features_processed_df[identifier_cols].reset_index(drop=True), subject_summary_pca_df], axis=1)
                    df = pd.merge(df, subject_summary_pca_df, on=identifier_cols, how='left')
                    numeric_features_for_advanced_fe = pca_component_names
            else: # If not using PCA, use the (scaled/imputed) numeric features directly
                df = pd.merge(df, processed_subject_df, on=identifier_cols, how='left')
                numeric_features_for_advanced_fe = summary_original_numeric_cols
        
        # Initialize the list of feature columns that will be used for model training
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        # This will include any PCA components or original scaled/imputed features if they were merged.
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

        # --- Advanced Feature Engineering ---
        # These operations will use `numeric_features_for_advanced_fe` 
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        poly_features_to_square = [col for col in numeric_features_for_advanced_fe if col in df.columns][:3] # Ensure cols exist in df
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        
        # Interaction between encoded TERM_CODE and a numeric summary feature
        if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_features_for_advanced_fe) >= 1:
            col_to_interact = numeric_features_for_advanced_fe[0] 
            if col_to_interact in df.columns: # Ensure interaction feature exists
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

        # Interaction between encoded SUBJECT_ID_SORT and a numeric summary feature
        if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_features_for_advanced_fe) >= 2:
            col_to_interact = numeric_features_for_advanced_fe[1] 
            if col_to_interact in df.columns:
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)
        elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_features_for_advanced_fe) == 1:
             col_to_interact = numeric_features_for_advanced_fe[0] 
             if col_to_interact in df.columns:
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)

        # Interaction between two distinct numeric summary features
        if len(numeric_features_for_advanced_fe) >= 2:
            col1 = numeric_features_for_advanced_fe[0]
            col2 = numeric_features_for_advanced_fe[1]
            if col1 in df.columns and col2 in df.columns:
                new_col_name = f"{col1}_x_{col2}"
                df[new_col_name] = df[col1] * df[col2]
                current_feature_cols.append(new_col_name)

        # Final list of feature columns, ensuring no duplicates and existence in df
        final_feature_cols = list(set(current_feature_cols))
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found after all engineering, add a dummy feature.
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge or missing values in summary data.
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
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df


# --- Main script ---
def run_training_and_validation(config_name, use_pca, pca_n_components, use_scaler_before_pca):
    df = load_data_ablated(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, use_pca, pca_n_components, use_scaler_before_pca)

    if df.empty:
        print(f"[{config_name}] Loaded DataFrame is empty. Cannot proceed with training.")
        print(f"Final Validation Performance: 0.0") 
        return config_name, 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print(f"[{config_name}] Not enough unique terms for a time-based validation split (at least 2 required).")
        print(f"Final Validation Performance: 0.0") 
        return config_name, 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
    if not feature_cols: 
        warnings.warn(f"[{config_name}] Feature columns not correctly identified by load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print(f"[{config_name}] No usable features available for training after all fallback attempts.")
        print(f"Final Validation Performance: 0.0") 
        return config_name, 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print(f"[{config_name}] Training set is empty. Cannot train a model.")
        print(f"Final Validation Performance: 0.0") 
        return config_name, 0.0

    if len(y_train.unique()) < 2:
        print(f"[{config_name}] Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    final_validation_score = 0.0 
    
    if y_val.empty:
        print(f"[{config_name}] Validation set is empty. F1 score cannot be calculated.")
    else:
        # Ensure that X_val has the same columns as X_train
        missing_cols = set(X_train.columns) - set(X_val.columns)
        for c in missing_cols:
            X_val[c] = 0 # Add missing columns to X_val with default value (e.g., 0)
        X_val = X_val[X_train.columns] # Ensure column order is the same

        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            print(f"[{config_name}] Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}") # Added for parsing
    return config_name, final_validation_score

if __name__ == "__main__":
    results = {}

    # Baseline: Original configuration with PCA, 0.95 variance, and StandardScaler
    print("Running Baseline: PCA (n_components=0.95, StandardScaler)")
    _, score = run_training_and_validation(
        "Baseline", use_pca=True, pca_n_components=0.95, use_scaler_before_pca=True
    )
    results['Baseline'] = score
    print(f"Baseline F1 Score: {results['Baseline']:.4f}")

    # Ablation 1: No PCA - Use original (scaled and imputed) numeric features directly
    print("\nRunning Ablation 1: No PCA (use original scaled features)")
    _, score = run_training_and_validation(
        "Ablation 1 (No PCA)", use_pca=False, pca_n_components=None, use_scaler_before_pca=True
    )
    results['Ablation 1 (No PCA)'] = score
    print(f"Ablation 1 (No PCA) F1 Score: {results['Ablation 1 (No PCA)']:.4f}")

    # Ablation 2: PCA n_components=0.5 - Retain only 50% variance
    print("\nRunning Ablation 2: PCA (n_components=0.5, StandardScaler)")
    _, score = run_training_and_validation(
        "Ablation 2 (PCA n_components=0.5)", use_pca=True, pca_n_components=0.5, use_scaler_before_pca=True
    )
    results['Ablation 2 (PCA n_components=0.5)'] = score
    print(f"Ablation 2 (PCA n_components=0.5) F1 Score: {results['Ablation 2 (PCA n_components=0.5)']:.4f}")

    # Ablation 3: No StandardScaler before PCA
    print("\nRunning Ablation 3: PCA (n_components=0.95, No StandardScaler)")
    _, score = run_training_and_validation(
        "Ablation 3 (No StandardScaler before PCA)", use_pca=True, pca_n_components=0.95, use_scaler_before_pca=False
    )
    results['Ablation 3 (No StandardScaler before PCA)'] = score
    print(f"Ablation 3 (No StandardScaler before PCA) F1 Score: {results['Ablation 3 (No StandardScaler before PCA)']:.4f}")


    # Determine the most impactful component
    baseline_score = results['Baseline']
    
    # Calculate the drop for each ablation
    impacts = {name: baseline_score - score for name, score in results.items() if name != 'Baseline'}

    most_impactful_component = "No single component caused a significant performance drop among those tested, or the dataset is too simple to show strong effects."
    
    if impacts:
        # Find the ablation that caused the largest positive drop (i.e., performance worsened most)
        largest_drop_name = None
        max_drop = 0.0

        for name, drop in impacts.items():
            if drop > max_drop:
                max_drop = drop
                largest_drop_name = name
        
        if largest_drop_name:
            most_impactful_component = (
                f"The most impactful component, causing the largest performance drop, is '{largest_drop_name}'. "
                f"It resulted in an F1 score drop of {max_drop:.4f} compared to the Baseline."
            )
        elif all(drop <= 0 for drop in impacts.values()):
            most_impactful_component = (
                "All ablated components performed identically or slightly better than the baseline. "
                "This suggests they are not critical for performance, or the baseline was suboptimal, "
                "or the dataset is too simple."
            )

    print(f"\nConclusion: {most_impactful_component}")
