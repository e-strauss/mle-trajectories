
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
from imblearn.over_sampling import SMOTE

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_data(data_dir, gold_file, use_advanced_features=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    
    Args:
        data_dir (str): Directory containing subject_summary.csv.
        gold_file (str): Path to gold_enrollment_train.csv.
        use_advanced_features (bool): If True, polynomial and interaction features are created.
                                       Otherwise, only base numeric and encoded identifier features are used.
    Returns:
        pd.DataFrame: DataFrame with loaded data and engineered features.
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
        
        # --- Advanced Feature Engineering (Ablation Point) ---
        if use_advanced_features:
            # Polynomial Features (e.g., squared terms for key numerical features)
            poly_features_to_square = numeric_summary_cols[:3] # Select up to 3 features
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
            
            # Interaction Features (product of two distinct features)
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
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

def train_and_evaluate(use_smote, n_estimators_rf, use_advanced_features):
    """
    Runs the training and validation process with specified ablation settings.
    Returns the final validation F1 score.
    """
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, use_advanced_features=use_advanced_features)

    if df.empty:
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
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
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        return 0.0

    # Handle cases where training set target has only one class
    if len(y_train.unique()) < 2:
        # If the training set has only one class, the model will effectively learn to predict only that class.
        # Predict the dominant class from training for all validation samples.
        dominant_class = y_train.mode()[0]
        y_pred = np.array([dominant_class] * len(y_val)) 
    else:
        # Model Training
        X_train_processed, y_train_processed = X_train, y_train
        if use_smote:
            smote = SMOTE(random_state=42)
            try:
                # SMOTE can raise ValueError if n_samples < n_neighbors. Catch this.
                X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
            except ValueError as e:
                warnings.warn(f"SMOTE failed: {e}. Proceeding without SMOTE.")
                
        model = RandomForestClassifier(random_state=42, n_estimators=n_estimators_rf)
        model.fit(X_train_processed, y_train_processed)
        y_pred = model.predict(X_val)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 
    
    if y_val.empty:
        pass # Score remains 0.0
    else:
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            # F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# Main ablation study execution
if __name__ == "__main__":
    scores = {}

    # 1. Baseline: Original solution with SMOTE, 100 n_estimators, advanced features
    print("Running Baseline (Full solution with SMOTE, RF n_estimators=100, advanced features)...")
    baseline_score = train_and_evaluate(use_smote=True, n_estimators_rf=100, use_advanced_features=True)
    scores["Baseline"] = baseline_score
    print(f"Baseline F1 Score: {baseline_score:.4f}\n")

    # 2. Ablation: No SMOTE
    print("Running Ablation: No SMOTE...")
    no_smote_score = train_and_evaluate(use_smote=False, n_estimators_rf=100, use_advanced_features=True)
    scores["No SMOTE"] = no_smote_score
    print(f"No SMOTE F1 Score: {no_smote_score:.4f}\n")

    # 3. Ablation: Reduced n_estimators for RandomForestClassifier (from 100 to 10)
    print("Running Ablation: Reduced RandomForest n_estimators (to 10)...")
    reduced_estimators_score = train_and_evaluate(use_smote=True, n_estimators_rf=10, use_advanced_features=True)
    scores["Reduced RF n_estimators (10)"] = reduced_estimators_score
    print(f"Reduced RF n_estimators (10) F1 Score: {reduced_estimators_score:.4f}\n")

    # 4. Ablation: No Advanced Feature Engineering (No polynomial or interaction features)
    print("Running Ablation: No Advanced Feature Engineering (No poly/interaction features)...")
    no_advanced_features_score = train_and_evaluate(use_smote=True, n_estimators_rf=100, use_advanced_features=False)
    scores["No Advanced Features"] = no_advanced_features_score
    print(f"No Advanced Features F1 Score: {no_advanced_features_score:.4f}\n")

    # Determine what contributes the most
    print("\n--- Ablation Study Summary ---")
    most_impactful_change = "N/A"
    largest_drop = 0.0
    
    if baseline_score > 0:
        for ablation_name, score in scores.items():
            if ablation_name != "Baseline":
                drop = baseline_score - score
                if drop > largest_drop:
                    largest_drop = drop
                    most_impactful_change = ablation_name
        
        if largest_drop > 0:
            print(f"The most impactful part, causing the largest performance drop ({largest_drop:.4f}), was the presence of: {most_impactful_change}")
        else:
            print("All ablations performed similarly to or better than the baseline, or the baseline score was 0.0, indicating no clear single most impactful component identified from these ablations.")
    else:
        print("Baseline F1 score is 0.0, which makes it difficult to assess relative contributions. All components might be essential or the data itself is problematic for this setup.")

