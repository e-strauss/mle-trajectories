
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Define paths (consistent with the provided dummy data generation)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")
SUBJECT_FEATURES_FILE = os.path.join(TRAIN_DATA_DIR, "subject_features.csv") # Assuming subject_features.csv replaces subject_summary.csv

# Helper function to create dummy data (taken from the context)
def create_dummy_data_for_ablation():
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
        np.random.seed(42)
        terms = np.array([202010, 202020, 202030, 202110, 202120, 202130, 202210, 202220, 202230, 202310, 202320, 202330, 202410, 202420, 202430])
        subjects = ['CS', 'EE', 'MA', 'PH', 'BI', 'CH', 'HI', 'EN']
        
        gold_data = []
        for term in terms:
            for subject in subjects:
                if subject in ['CS', 'MA'] and term >= 202310:
                    high_enrollment = 'Y' if np.random.rand() < 0.7 else 'N'
                elif subject in ['PH', 'BI'] and term >= 202310:
                    high_enrollment = 'Y' if np.random.rand() < 0.3 else 'N'
                else:
                    high_enrollment = 'Y' if np.random.rand() < 0.5 else 'N'
                gold_data.append({'TERM_CODE': term, 'SUBJECT_ID_SORT': subject, 'HIGH_ENROLLMENT': high_enrollment})
        pd.DataFrame(gold_data).to_csv(GOLD_ENROLLMENT_TRAIN_FILE, index=False)

    if not os.path.exists(SUBJECT_FEATURES_FILE):
        np.random.seed(43)
        subject_features_data = []
        
        if os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
            all_gold_data = pd.read_csv(GOLD_ENROLLMENT_TRAIN_FILE)
            all_terms = all_gold_data['TERM_CODE'].unique()
            all_subjects = all_gold_data['SUBJECT_ID_SORT'].unique()
        else:
            all_terms = np.array([202010, 202110, 202210, 202310, 202410])
            all_subjects = ['CS', 'EE', 'MA']

        for term in all_terms:
            for subject in all_subjects:
                num_courses = np.random.randint(5, 50)
                avg_class_size = np.random.uniform(15, 80)
                faculty_count = np.random.randint(10, 100)
                budget_per_student = np.random.uniform(1000, 5000)
                if np.random.rand() < 0.1:
                    avg_class_size = np.nan
                if np.random.rand() < 0.05:
                    budget_per_student = np.nan

                subject_features_data.append({
                    'TERM_CODE': term,
                    'SUBJECT_ID_SORT': subject,
                    'NUM_COURSES_OFFERED': num_courses,
                    'AVG_CLASS_SIZE': avg_class_size,
                    'FACULTY_COUNT': faculty_count,
                    'BUDGET_PER_STUDENT': budget_per_student
                })
        pd.DataFrame(subject_features_data).to_csv(SUBJECT_FEATURES_FILE, index=False)


# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, subject_summary_file_path, 
              ablation_no_poly_features=False, 
              ablation_no_interaction_features=False,
              ablation_impute_with_mean=False):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation parameters control specific feature engineering steps and imputation.
    """
    gold_df = pd.read_csv(gold_file)
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_file_path)
        
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
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        if not ablation_no_poly_features:
            poly_features_to_square = numeric_summary_cols[:3] # Select up to 3 features
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        if not ablation_no_interaction_features:
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
        if ablation_impute_with_mean:
            for col in final_feature_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val if not pd.isna(mean_val) else 0) 
        else:
            df[final_feature_cols] = df[final_feature_cols].fillna(0)
        
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        print(f"Warning: {subject_summary_file_path} not found. Using minimal features from gold_df.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        
        if ablation_impute_with_mean:
            for col in df._feature_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val if not pd.isna(mean_val) else 0)
        else:
            df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script with ablation parameters ---
def run_training_and_validation(
    ablation_name="Baseline",
    ablation_no_poly_features=False, 
    ablation_no_interaction_features=False,
    ablation_impute_with_mean=False
):
    """
    Executes the training and validation process for a given ablation configuration.
    """
    df = load_data(
        TRAIN_DATA_DIR, 
        GOLD_ENROLLMENT_TRAIN_FILE, 
        SUBJECT_FEATURES_FILE,
        ablation_no_poly_features=ablation_no_poly_features,
        ablation_no_interaction_features=ablation_no_interaction_features,
        ablation_impute_with_mean=ablation_impute_with_mean
    )

    if df.empty:
        print(f"[{ablation_name}] Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print(f"[{ablation_name}] Not enough unique terms for a time-based validation split (at least 2 required).")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
    if not feature_cols: 
        warnings.warn(f"[{ablation_name}] Feature columns not correctly identified by load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print(f"[{ablation_name}] No usable features available for training after all fallback attempts.")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Align columns between train and validation sets
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    if X_train.empty or y_train.empty:
        print(f"[{ablation_name}] Training set is empty. Cannot train a model.")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"[{ablation_name}] Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    final_validation_score = 0.0 
    
    if y_val.empty:
        print(f"[{ablation_name}] Validation set is empty. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            print(f"[{ablation_name}] Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score


if __name__ == "__main__":
    create_dummy_data_for_ablation()
    
    results = {}

    # Run Baseline
    baseline_score = run_training_and_validation(
        ablation_name="Baseline",
        ablation_no_poly_features=False,
        ablation_no_interaction_features=False,
        ablation_impute_with_mean=False
    )
    results["Baseline"] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}")

    # Ablation 1: No Polynomial Features
    ablation1_score = run_training_and_validation(
        ablation_name="Ablation 1 (No Polynomial Features)",
        ablation_no_poly_features=True,
        ablation_no_interaction_features=False,
        ablation_impute_with_mean=False
    )
    results["Ablation 1 (No Polynomial Features)"] = ablation1_score
    print(f"Ablation 1 (No Polynomial Features) F1 Score: {ablation1_score:.4f}")

    # Ablation 2: No Interaction Features
    ablation2_score = run_training_and_validation(
        ablation_name="Ablation 2 (No Interaction Features)",
        ablation_no_poly_features=False,
        ablation_no_interaction_features=True,
        ablation_impute_with_mean=False
    )
    results["Ablation 2 (No Interaction Features)"] = ablation2_score
    print(f"Ablation 2 (No Interaction Features) F1 Score: {ablation2_score:.4f}")

    # Ablation 3: NaN Imputation to Mean (instead of 0)
    ablation3_score = run_training_and_validation(
        ablation_name="Ablation 3 (NaN Imputation to Mean)",
        ablation_no_poly_features=False,
        ablation_no_interaction_features=False,
        ablation_impute_with_mean=True
    )
    results["Ablation 3 (NaN Imputation to Mean)"] = ablation3_score
    print(f"Ablation 3 (NaN Imputation to Mean) F1 Score: {ablation3_score:.4f}")

    # Determine most impactful
    most_impactful_change = "None of the ablations caused a significant performance drop or the dataset is too simple to show meaningful differences."
    max_drop = 0.0

    if baseline_score == 0.0:
        most_impactful_change = "Baseline score is 0.0, which makes it impossible to determine which components contribute most positively."
    else:
        for name, score in results.items():
            if name == "Baseline":
                continue
            drop = baseline_score - score
            if drop > max_drop:
                max_drop = drop
                most_impactful_change = f"'{name}' (F1 Score dropped by {drop:.4f})"
            elif drop < 0:
                print(f"Note: '{name}' improved performance by {-drop:.4f} compared to Baseline.")

        if max_drop == 0 and baseline_score != 1.0:
             most_impactful_change = "No ablation caused a performance drop, but the baseline performance is not perfect, suggesting other factors are limiting performance, or the ablated components are not critical under these conditions."
        elif max_drop == 0 and baseline_score == 1.0:
             most_impactful_change = "All scenarios achieved perfect F1 score, making it impossible to determine the relative impact of the ablated components. The dataset might be too simple."

    print(f"\nThe part of the code that contributes the most to the overall performance (i.e., whose removal/modification caused the largest F1 score drop) is: {most_impactful_change}")

