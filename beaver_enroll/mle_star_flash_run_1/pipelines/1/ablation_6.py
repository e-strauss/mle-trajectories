
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    import subprocess
    import sys
    print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# Define paths
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

# --- Helper Functions (modified for ablation) ---

def load_table_if_exists_ablation(directory, filename):
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        # print(f"Loading {filename} from {directory}") # Suppress verbose output during ablation
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    else:
        # print(f"Warning: {filename} not found at {filepath}. Skipping.") # Suppress verbose output
        return pd.DataFrame() # Return empty DataFrame if file not found

def preprocess_and_feature_engineer_ablated(gold_df,
                                             use_department_code=True,
                                             use_sections_features=True,
                                             use_courses_features=True):
    """
    Preprocesses raw data and engineers features with ablation switches.
    Loads necessary raw data tables internally based on flags.
    """
    df = gold_df.copy()

    # Load potential feature tables locally within this function for flexibility
    # This prevents issues if a raw_data_dict isn't perfectly consistent across runs
    sections_df = load_table_if_exists_ablation(TRAIN_DATA_DIR, 'sections.csv')
    courses_df = load_table_if_exists_ablation(TRAIN_DATA_DIR, 'courses.csv')

    # Create a 'DEPARTMENT_CODE' by extracting alphabetic characters from 'SUBJECT_ID_SORT'
    if use_department_code:
        df['DEPARTMENT_CODE'] = df['SUBJECT_ID_SORT'].astype(str).apply(lambda x: ''.join(filter(str.isalpha, x)).upper())
        df['DEPARTMENT_CODE'] = df['DEPARTMENT_CODE'].replace('', 'UNKNOWN')
    else:
        pass # Ablation: Skipping DEPARTMENT_CODE feature engineering.

    # --- Feature Engineering from sections.csv (if available and not ablated) ---
    if use_sections_features and not sections_df.empty:
        sections_df_copy = sections_df.copy()
        for col in ['TERM_CODE', 'ENROLLMENT', 'MAX_ENROLLMENT']:
            if col in sections_df_copy.columns:
                sections_df_copy[col] = pd.to_numeric(sections_df_copy[col], errors='coerce')
        if 'SUBJECT_ID_SORT' in sections_df_copy.columns:
            sections_df_copy['SUBJECT_ID_SORT'] = sections_df_copy['SUBJECT_ID_SORT'].astype(str)

        sections_df_copy.dropna(subset=['TERM_CODE', 'SUBJECT_ID_SORT'], inplace=True)
        sections_df_copy['ENROLLMENT'] = sections_df_copy['ENROLLMENT'].fillna(0)
        sections_df_copy['MAX_ENROLLMENT'] = sections_df_copy['MAX_ENROLLMENT'].fillna(0)

        course_enrollment_stats = sections_df_copy.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            num_sections=('SECTION_NUMBER', 'nunique'),
            total_enrollment=('ENROLLMENT', 'sum'),
            avg_section_enrollment=('ENROLLMENT', 'mean'),
            max_section_enrollment=('ENROLLMENT', 'max'),
            total_max_enrollment=('MAX_ENROLLMENT', 'sum')
        ).reset_index()

        df = pd.merge(df, course_enrollment_stats, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

        df['enrollment_ratio'] = df['total_enrollment'] / df['total_max_enrollment']
        df['enrollment_ratio'] = df['enrollment_ratio'].replace([np.inf, -np.inf], np.nan)
        df['enrollment_ratio'] = df['enrollment_ratio'].fillna(0)
    elif not use_sections_features:
        pass # Ablation: Skipping sections.csv derived features.


    # --- Feature Engineering from courses.csv (if available and not ablated) ---
    if use_courses_features and not courses_df.empty:
        courses_df_copy = courses_df.copy()
        for col in ['TERM_CODE', 'CREDIT_HOURS']:
            if col in courses_df_copy.columns:
                courses_df_copy[col] = pd.to_numeric(courses_df_copy[col], errors='coerce')
        if 'SUBJECT_ID_SORT' in courses_df_copy.columns:
            courses_df_copy['SUBJECT_ID_SORT'] = courses_df_copy['SUBJECT_ID_SORT'].astype(str)

        courses_df_copy.dropna(subset=['TERM_CODE', 'SUBJECT_ID_SORT'], inplace=True)

        def extract_course_level(s):
            if pd.isna(s): return np.nan
            s = str(s)
            numbers = ''.join(filter(str.isdigit, s))
            if numbers:
                level = int(numbers) // 100
                return min(level, 4)
            return np.nan

        if 'COURSE_NUMBER' in courses_df_copy.columns:
            courses_df_copy['COURSE_LEVEL'] = courses_df_copy['COURSE_NUMBER'].astype(str).apply(extract_course_level)
        else:
            courses_df_copy['COURSE_LEVEL'] = courses_df_copy['SUBJECT_ID_SORT'].apply(extract_course_level)

        courses_cols_to_merge = ['TERM_CODE', 'SUBJECT_ID_SORT']
        if 'CREDIT_HOURS' in courses_df_copy.columns:
            courses_cols_to_merge.append('CREDIT_HOURS')
        if 'COURSE_LEVEL' in courses_df_copy.columns:
            courses_cols_to_merge.append('COURSE_LEVEL')
        courses_df_unique = courses_df_copy[courses_cols_to_merge].drop_duplicates(subset=['TERM_CODE', 'SUBJECT_ID_SORT'])

        df = pd.merge(df, courses_df_unique, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    elif not use_courses_features:
        pass # Ablation: Skipping courses.csv derived features.

    # --- General Features (TERM_YEAR, TERM_SEASON) ---
    df['TERM_YEAR'] = df['TERM_CODE'] // 100
    df['TERM_SEASON'] = df['TERM_CODE'] % 100

    # Fill NaN values for numerical features before scaling (ensure all expected numerical cols are listed)
    # This list needs to be comprehensive of all possible numerical features that might be generated
    numerical_cols = [
        'num_sections', 'total_enrollment', 'avg_section_enrollment',
        'max_section_enrollment', 'total_max_enrollment', 'enrollment_ratio',
        'CREDIT_HOURS', 'COURSE_LEVEL', 'TERM_YEAR', 'TERM_SEASON'
    ]
    for col in numerical_cols:
        if col in df.columns: # Only fill if the column actually exists (might be ablated)
            df[col] = df[col].fillna(df[col].median())
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)


    # --- Categorical Feature Encoding ---
    categorical_cols = []
    if use_department_code: # Only add if we're using this feature
        categorical_cols.append('DEPARTMENT_CODE')

    # TERM_SEASON is always generated but might be excluded from encoding if only one unique value
    if 'TERM_SEASON' in df.columns and len(df['TERM_SEASON'].unique()) > 1:
        categorical_cols.append('TERM_SEASON')

    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)

    # Handle the target variable
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    return df

def train_validate_model_ablated(data):
    """Trains a RandomForestClassifier and evaluates it using a time-based validation split."""

    data = data.copy().sort_values(by='TERM_CODE').reset_index(drop=True)

    X = data.drop(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'], errors='ignore')
    y = data['HIGH_ENROLLMENT']

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # Exclude dummy variables if they were created from numerical columns like TERM_SEASON
    numerical_features = [
        f for f in numerical_features
        if not f.startswith('DEPARTMENT_CODE_') and not f.startswith('TERM_SEASON_')
    ]

    scaler = StandardScaler()
    if numerical_features:
        # Check if numerical features actually exist in X before scaling
        features_to_scale = [f for f in numerical_features if f in X.columns]
        if features_to_scale:
            X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

    unique_terms = sorted(data['TERM_CODE'].unique())
    if len(unique_terms) < 2:
        # Fallback to random split if time-based split is not possible
        # print("Not enough unique terms for time-based split. Falling back to random split.")
        if len(y.unique()) < 2:
            print("Cannot perform stratified split with only one class in target.")
            return None, 0.0
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Determine validation terms
        if len(unique_terms) < 5:
            validation_terms = [unique_terms[-1]]
        else:
            split_term_idx = int(len(unique_terms) * 0.8)
            validation_terms = unique_terms[split_term_idx:]

        train_indices = data[~data['TERM_CODE'].isin(validation_terms)].index
        val_indices = data[data['TERM_CODE'].isin(validation_terms)].index

        X_train, X_val = X.loc[train_indices], X.loc[val_indices]
        y_train, y_val = y.loc[train_indices], y.loc[val_indices]

        # Handle potential empty sets after time-based split, fallback to random if needed
        if X_train.empty or X_val.empty:
            # print("Time-based split resulted in an empty train or validation set. Falling back to random split.")
            if len(y.unique()) < 2:
                print("Cannot perform stratified split with only one class in target.")
                return None, 0.0
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Handle case where stratified split is not possible due to single class in y.
        elif len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            # print("Target has only one class in train/val after time split. Falling back to random split if possible.")
            if len(y.unique()) < 2:
                print("Cannot perform stratified split with only one class in target.")
                return None, 0.0
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Ensure all columns are numeric
    X_train = X_train.select_dtypes(include=np.number)
    X_val = X_val.select_dtypes(include=np.number)

    # Align columns in train and validation sets
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)

    missing_in_val = list(train_cols - val_cols)
    for col in missing_in_val:
        X_val[col] = 0

    missing_in_train = list(val_cols - train_cols)
    for col in missing_in_train:
        X_train[col] = 0

    X_val = X_val[list(X_train.columns)] # Ensure column order is the same

    # Check for empty sets or single-class target before training
    if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        # print("Error: Training or validation set is empty, or target has only one class. Returning 0.0.")
        return None, 0.0 # Return 0.0 if training is not possible

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    # print(f"Validation Macro F1 Score: {f1_macro:.4f}") # Suppress verbose output

    return model, f1_macro

def run_ablation_experiment(experiment_name, gold_enrollment_df,
                            use_department_code=True,
                            use_sections_features=True,
                            use_courses_features=True):
    print(f"--- Running Experiment: {experiment_name} ---")
    try:
        processed_df = preprocess_and_feature_engineer_ablated(
            gold_enrollment_df,
            use_department_code=use_department_code,
            use_sections_features=use_sections_features,
            use_courses_features=use_courses_features
        )

        # Final check for NaN values in features before training
        for col in processed_df.columns:
            if processed_df[col].dtype == object or processed_df[col].dtype == bool:
                pass
            elif processed_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col] = processed_df[col].fillna(0)
                else:
                    processed_df[col] = processed_df[col].fillna('MISSING')

        if processed_df.empty or 'HIGH_ENROLLMENT' not in processed_df.columns or not processed_df['HIGH_ENROLLMENT'].nunique() > 1:
            print("Processed DataFrame is empty, missing 'HIGH_ENROLLMENT' target column, or target has only one unique value. Returning 0.0.")
            return 0.0
        
        # Ensure there are features remaining after dropping identifiers and target
        features_df = processed_df.drop(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'], errors='ignore')
        if features_df.empty or features_df.select_dtypes(include=np.number).empty:
             print("Processed DataFrame has no valid features for training after dropping identifiers and target. Returning 0.0.")
             return 0.0


        _, score = train_validate_model_ablated(processed_df)
        return score
    except Exception as e:
        print(f"Error during experiment '{experiment_name}': {e}")
        return 0.0 # Return 0.0 on error for consistent reporting

# --- Main ablation execution block ---
if __name__ == "__main__":
    results = {}

    try:
        # Load gold labels
        if not os.path.exists(GOLD_ENROLLMENT_TRAIN_PATH):
            raise FileNotFoundError(f"Gold enrollment labels file not found at {GOLD_ENROLLMENT_TRAIN_PATH}")
        gold_enrollment_df = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        print("Gold enrollment labels loaded.")
        if gold_enrollment_df.empty:
            raise ValueError("Gold enrollment labels file is empty.")

        # Baseline
        results['Baseline (All Features)'] = run_ablation_experiment(
            'Baseline (All Features)', gold_enrollment_df,
            use_department_code=True, use_sections_features=True, use_courses_features=True
        )

        # Ablation 1: No DEPARTMENT_CODE feature
        results['Ablation: No DEPARTMENT_CODE'] = run_ablation_experiment(
            'Ablation: No DEPARTMENT_CODE', gold_enrollment_df,
            use_department_code=False, use_sections_features=True, use_courses_features=True
        )

        # Ablation 2: No sections.csv derived features
        results['Ablation: No Sections Features'] = run_ablation_experiment(
            'Ablation: No Sections Features', gold_enrollment_df,
            use_department_code=True, use_sections_features=False, use_courses_features=True
        )

        # Ablation 3: No courses.csv derived features
        results['Ablation: No Courses Features'] = run_ablation_experiment(
            'Ablation: No Courses Features', gold_enrollment_df,
            use_department_code=True, use_sections_features=True, use_courses_features=False
        )

        print("\n--- Ablation Study Results ---")
        for experiment, score in results.items():
            print(f"{experiment}: Macro F1 Score = {score:.4f}")

        # Determine the most impactful part
        baseline_score = results.get('Baseline (All Features)', 0.0) # Use .get with default 0.0 for safety
        impacts = {}
        for experiment, score in results.items():
            if experiment != 'Baseline (All Features)':
                impacts[experiment] = baseline_score - score

        if not impacts:
            print("\n--- Conclusion ---")
            print("No ablation experiments were run or produced valid scores beyond baseline.")
        else:
            # Filter out non-numeric or NaN scores if any, in case of errors
            valid_impacts = {k: v for k, v in impacts.items() if pd.notna(v)}
            
            if not valid_impacts:
                print("\n--- Conclusion ---")
                print("No valid scores were obtained from ablation experiments.")
            else:
                most_impactful_change_key = max(valid_impacts, key=valid_impacts.get)
                most_impactful_value = valid_impacts[most_impactful_change_key]

                print("\n--- Conclusion ---")
                if most_impactful_value > 0:
                    contributing_part = most_impactful_change_key.replace('Ablation: No ', '')
                    print(f"The part of the code that contributes the most to the overall performance is '{contributing_part}', as its removal caused the largest drop in Macro F1 score ({most_impactful_value:.4f}).")
                elif most_impactful_value < 0:
                    detrimental_part = most_impactful_change_key.replace('Ablation: No ', '')
                    print(f"The removal of '{detrimental_part}' actually *improved* performance by {-most_impactful_value:.4f}. This suggests it might be detrimental or redundant.")
                else:
                    print("No significant change in performance observed for the ablated components.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data is in the correct directory.")
    except ValueError as e:
        print(f"Data Processing or Model Training Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

