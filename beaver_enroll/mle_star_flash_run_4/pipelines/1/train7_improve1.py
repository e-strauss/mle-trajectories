
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback
import sys
import subprocess

# Define constants based on the problem description
BASE_INPUT_DIR = './input'
DEFAULT_TRAIN_DIR = './input'
GOLD_LABELS_PATH = os.path.join(BASE_INPUT_DIR, 'gold_enrollment_train.csv')

def load_data(train_dir):
    """
    Loads all necessary CSV files from both base and reference solutions into pandas DataFrames.
    Includes error handling for missing files by creating empty DataFrames.
    """
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'instructor_attributes': os.path.join(train_dir, 'instructor_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': GOLD_LABELS_PATH
    }
    
    dataframes = {}
    for name, path in paths.items():
        try:
            if os.path.exists(path):
                dataframes[name] = pd.read_csv(path)
            else:
                print(f"Warning: {name}.csv not found at {path}. Proceeding without this data.", file=sys.stderr)
                dataframes[name] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading {name} data: {e}", file=sys.stderr)
            dataframes[name] = pd.DataFrame()

    return dataframes

def feature_engineering(dfs):
    """
    Merges dataframes and creates a combined set of features from both base and reference solutions.
    """
    subject_summary = dfs.get('subject_summary')
    gold_labels = dfs.get('gold_labels')
    course_attributes = dfs.get('course_attributes')
    course_summary = dfs.get('course_summary')
    faculty_summary = dfs.get('faculty_summary')

    if subject_summary.empty or gold_labels.empty:
        raise ValueError("Core data files (subject_summary or gold_enrollment_train) are missing or empty.")

    # --- Start with core data ---
    # Ensure TERM_CODE is consistent for merging
    for df in [subject_summary, gold_labels, course_summary, faculty_summary]:
        if df is not None and 'TERM_CODE' in df.columns:
            df['TERM_CODE'] = pd.to_numeric(df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    # Merge summary with gold labels
    data = pd.merge(subject_summary, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # --- Base Solution Feature Engineering ---
    if not course_attributes.empty:
         data = pd.merge(data, course_attributes.add_suffix('_attr'), on='SUBJECT_ID_SORT', how='left')
    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]

    # --- Reference Solution Feature Engineering ---
    if not course_summary.empty:
        course_agg = course_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_COURSES=('COURSE_ID', 'nunique'),
            TOTAL_SEATS=('MAX_ENROLLMENT', 'sum'),
            AVG_SEATS=('MAX_ENROLLMENT', 'mean')
        ).reset_index()
        data = pd.merge(data, course_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not faculty_summary.empty and 'FACULTY_ID' in faculty_summary.columns and 'NUM_COURSES_TAUGHT' in faculty_summary.columns:
        faculty_agg = faculty_summary.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
            NUM_FACULTY=('FACULTY_ID', 'nunique'),
            AVG_FACULTY_LOAD=('NUM_COURSES_TAUGHT', 'mean')
        ).reset_index()
        data = pd.merge(data, faculty_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    else:
        # Create empty columns if faculty data is missing
        data['NUM_FACULTY'] = np.nan
        data['AVG_FACULTY_LOAD'] = np.nan

    # New features from reference solution
    data['SEATS_PER_COURSE'] = data['TOTAL_SEATS'] / data['NUM_COURSES']
    data['COURSES_PER_FACULTY'] = data['NUM_COURSES'] / data['NUM_FACULTY']
    
    # --- Final Processing ---
    # Handle infinities from division by zero and NaNs created
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create sortable integer term code
    data['TERM_CODE_INT'] = data['TERM_CODE']
    
    # Convert target variable 'Y'/'N' to binary 1/0, drop rows where target is missing
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def main():
    """
    Main function to execute the integrated training and validation pipeline.
    """
    parser = argparse.ArgumentParser(description='Predict high enrollment for courses.')
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DIR,
                        help='Directory containing training data tables.')
    args = parser.parse_args()

    # --- 0. Dependency Installation ---
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error: Failed to install or import dependencies. {e}", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    # --- 1. Data Loading ---
    dataframes = load_data(args.train_data_dir)

    # --- 2. Feature Engineering ---
    try:
        data = feature_engineering(dataframes)
    except ValueError as e:
        print(f"Feature engineering failed: {e}", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    # --- 3. Data Splitting (Multi-Split Time-based) ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())

    # Fallback to random split if not enough terms for a robust time-based split
    if len(sorted_terms) < 2:
        print("Warning: Less than two terms available. Using random split for validation.", file=sys.stderr)
        if data.empty or data['HIGH_ENROLLMENT'].nunique() < 2:
             print("Final dataset is too small to split or train. Cannot train or validate.", file=sys.stderr)
             print('Final Validation Performance: 0.0')
             return
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data.get('HIGH_ENROLLMENT'))
        # The original code will now be replaced by the robust time-series validation below
    else:
        # The new strategy: Use the last term for validation, and create multiple historical training sets.
        pass # The logic is now integrated into the main flow

    # Define the single validation set using the most recent term
    if len(sorted_terms) >= 2:
        validation_term = sorted_terms[-1]
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
    else: # This handles the random split case from above
        val_df = val_df

    # Check for invalid validation set early
    if val_df.empty or val_df['HIGH_ENROLLMENT'].nunique() < 2:
        print("Validation set is unusable (empty or single class). Cannot validate.", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    y_val = val_df['HIGH_ENROLLMENT']

    # Define multiple historical training sets
    train_dfs = []
    if len(sorted_terms) >= 2:
        num_splits = min(3, len(sorted_terms) - 1)  # Use up to 3 historical splits
        for i in range(1, num_splits + 1):
            cutoff_term = sorted_terms[-i]
            train_df_split = data[data['TERM_CODE_INT'] < cutoff_term].copy()
            if not train_df_split.empty and train_df_split['HIGH_ENROLLMENT'].nunique() > 1:
                train_dfs.append(train_df_split)
    else: # Random split case
        train_dfs.append(train_df)

    if not train_dfs:
        print("No valid historical training sets could be created.", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    # --- 4. Model Training & Prediction Loop ---
    pipeline_cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    pipeline_num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    tabpfn_cat_features = ['SUBJECT_ID_SORT', 'CAMPUS_ID_DESC', 'DEPARTMENT_ID_DESC']
    tabpfn_num_features = ['NUM_COURSES', 'TOTAL_SEATS', 'AVG_SEATS', 'NUM_FACULTY', 'AVG_FACULTY_LOAD', 'SEATS_PER_COURSE', 'COURSES_PER_FACULTY']
    tabpfn_cat_features = [f for f in tabpfn_cat_features if f in data.columns]
    tabpfn_num_features = [f for f in tabpfn_num_features if f in data.columns]
    all_tabpfn_features = tabpfn_num_features + tabpfn_cat_features
    
    if len(all_tabpfn_features) > 100:
        print(f"Warning: Number of features ({len(all_tabpfn_features)}) for TabPFN exceeds 100. Truncating.", file=sys.stderr)
        all_tabpfn_features = all_tabpfn_features[:100]

    split_probas_rf, split_probas_lgbm, split_probas_tabpfn = [], [], []

    for train_df in train_dfs:
        y_train = train_df['HIGH_ENROLLMENT']
        
        # --- Models 1 & 2: RF and LGBM ---
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_rf = build_classifier_pipeline(rf_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_rf.fit(train_df, y_train)
        split_probas_rf.append(pipeline_rf.predict_proba(val_df)[:, 1])

        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        pipeline_lgbm = build_classifier_pipeline(lgbm_classifier, pipeline_num_features, pipeline_cat_features)
        pipeline_lgbm.fit(train_df, y_train)
        split_probas_lgbm.append(pipeline_lgbm.predict_proba(val_df)[:, 1])

        # --- Model 3: TabPFN ---
        train_df_tabpfn, val_df_tabpfn = train_df.copy(), val_df.copy()
        for col in tabpfn_cat_features:
            codes, uniques = pd.factorize(train_df_tabpfn[col])
            train_df_tabpfn[col] = codes
            mapping = {label: i for i, label in enumerate(uniques)}
            val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

        X_train_tabpfn = train_df_tabpfn[all_tabpfn_features].fillna(0)
        X_val_tabpfn = val_df_tabpfn[all_tabpfn_features].fillna(0)
        
        if X_train_tabpfn.shape[1] > 0 and X_train_tabpfn.shape[0] < 1024 and X_train_tabpfn.shape[0] > 0:
            clf_tabpfn = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            clf_tabpfn.fit(X_train_tabpfn, y_train, overwrite_warning=True)
            split_probas_tabpfn.append(clf_tabpfn.predict_proba(X_val_tabpfn)[:, 1])
        else:
            # Append neutral prediction if TabPFN cannot be trained
            split_probas_tabpfn.append(np.full(len(val_df), 0.5))

    # --- 5. Validation and Ensembling ---
    # Weighted averaging of predictions from different historical splits
    # Weights are based on recency (most recent data gets highest weight)
    num_valid_splits = len(train_dfs)
    if num_valid_splits > 1:
        weights = np.arange(num_valid_splits, 0, -1) / np.sum(np.arange(num_valid_splits, 0, -1))
    else:
        weights = [1.0]

    # Average predictions for each model type across the splits using weights
    avg_proba_rf = np.average(np.array(split_probas_rf), axis=0, weights=weights)
    avg_proba_lgbm = np.average(np.array(split_probas_lgbm), axis=0, weights=weights)
    avg_proba_tabpfn = np.average(np.array(split_probas_tabpfn), axis=0, weights=weights)

    # Simple averaging ensemble of the weighted-averaged models
    proba_ensemble = (avg_proba_rf + avg_proba_lgbm + avg_proba_tabpfn) / 3
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)

    final_validation_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
    print(f'Final Validation Performance: {final_validation_score}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred in the main execution block: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # To ensure the evaluation system receives a score even on failure
        print('Final Validation Performance: 0.0')
