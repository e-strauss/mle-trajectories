
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
        from sklearn.metrics import f1_score
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

    # --- 3. Data Splitting (Time-based with Holdout and Checks) ---
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    
    train_df, val_df, holdout_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if len(sorted_terms) < 2:
        print("Warning: Only one term available. Using random split for validation.", file=sys.stderr)
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data.get('HIGH_ENROLLMENT'))
    else:
        validation_term = sorted_terms[-1]
        train_holdout_df = data[data['TERM_CODE_INT'] < validation_term].copy()
        val_df = data[data['TERM_CODE_INT'] == validation_term].copy()

        # Crucial check to ensure validation set has both classes
        if val_df.empty or val_df['HIGH_ENROLLMENT'].nunique() < 2:
            print("Warning: Time-based split resulted in an invalid validation set. Using random split.", file=sys.stderr)
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data.get('HIGH_ENROLLMENT'))
        else:
            # Create a hold-out set from the end of the training data for weighting
            train_holdout_terms = sorted(train_holdout_df['TERM_CODE_INT'].unique())
            if len(train_holdout_terms) > 1:
                holdout_term = train_holdout_terms[-1]
                train_df = train_holdout_df[train_holdout_df['TERM_CODE_INT'] < holdout_term].copy()
                holdout_df = train_holdout_df[train_holdout_df['TERM_CODE_INT'] == holdout_term].copy()
                # Ensure holdout is useful for evaluation
                if holdout_df.empty or holdout_df['HIGH_ENROLLMENT'].nunique() < 2:
                    train_df = train_holdout_df.copy() # Cannot use holdout, merge back
                    holdout_df = pd.DataFrame()
            else:
                train_df = train_holdout_df.copy() # Not enough terms to create holdout

    if train_df.empty or val_df.empty or train_df['HIGH_ENROLLMENT'].nunique() < 2:
        print("Final dataset is too small to split or train. Cannot train or validate.", file=sys.stderr)
        print('Final Validation Performance: 0.0')
        return

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']
    y_holdout = holdout_df['HIGH_ENROLLMENT'] if not holdout_df.empty else pd.Series()

    # --- 4. Unified Feature Processing (Target Encoding) ---
    cat_features = [col for col in ['DEPARTMENT'] + [c for c in data.columns if c.endswith('_attr') and data[c].dtype == 'object'] if col in data.columns]
    num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]

    # Apply target encoding
    global_mean = y_train.mean()
    for col in cat_features:
        mapping = train_df.groupby(col)['HIGH_ENROLLMENT'].mean()
        train_df[col] = train_df[col].map(mapping).fillna(global_mean)
        val_df[col] = val_df[col].map(mapping).fillna(global_mean)
        if not holdout_df.empty:
            holdout_df[col] = holdout_df[col].map(mapping).fillna(global_mean)

    features = num_features + cat_features
    X_train = train_df[features].fillna(0)
    X_val = val_df[features].fillna(0)
    X_holdout = holdout_df[features].fillna(0) if not holdout_df.empty else pd.DataFrame()

    # --- 5. Model Training ---
    models = {}
    
    # Model 1: RandomForest
    models['rf'] = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Model 2: LightGBM
    models['lgbm'] = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # Train RF and LGBM
    for name, model in models.items():
        model.fit(X_train, y_train)

    # Model 3: TabPFN
    X_train_tabpfn = X_train.copy()
    X_val_tabpfn = X_val.copy()
    X_holdout_tabpfn = X_holdout.copy()

    if X_train_tabpfn.shape[1] > 100:
        print(f"Warning: Number of features ({X_train_tabpfn.shape[1]}) for TabPFN exceeds 100. Truncating.", file=sys.stderr)
        top_100_features = X_train_tabpfn.columns[:100]
        X_train_tabpfn = X_train_tabpfn[top_100_features]
        X_val_tabpfn = X_val_tabpfn[top_100_features]
        if not X_holdout_tabpfn.empty:
            X_holdout_tabpfn = X_holdout_tabpfn[top_100_features]
    
    models['tabpfn'] = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    models['tabpfn'].fit(X_train_tabpfn, y_train, overwrite_warning=True)

    # --- 6. Validation and Performance-Weighted Ensembling ---
    weights = {'rf': 1.0, 'lgbm': 1.0, 'tabpfn': 1.0}
    
    # Calculate weights from holdout set if available
    if not holdout_df.empty and not y_holdout.empty:
        print("Calculating model weights on holdout set.")
        model_scores = {}
        for name, model in models.items():
            X_h = X_holdout_tabpfn if name == 'tabpfn' else X_holdout
            probas_h = model.predict_proba(X_h)[:, 1]
            preds_h = (probas_h >= 0.5).astype(int)
            score = f1_score(y_holdout, preds_h, average='macro', zero_division=0)
            model_scores[name] = score
        
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {name: score / total_score for name, score in model_scores.items()}
            print(f"Ensemble weights: {weights}")
        else:
            print("Holdout scores are all zero, using equal weights.")
    else:
        print("No holdout set available, using equal weights for ensembling.")

    # Get probabilities from all models on the final validation set
    if not y_val.empty:
        probas = {}
        probas['rf'] = models['rf'].predict_proba(X_val)[:, 1]
        probas['lgbm'] = models['lgbm'].predict_proba(X_val)[:, 1]
        probas['tabpfn'] = models['tabpfn'].predict_proba(X_val_tabpfn)[:, 1]

        # Performance-weighted averaging ensemble
        proba_ensemble = (probas['rf'] * weights['rf'] +
                          probas['lgbm'] * weights['lgbm'] +
                          probas['tabpfn'] * weights['tabpfn'])
        
        # The weights should already sum to 1 if calculated, otherwise we're dividing by 3 implicitly
        if not (not holdout_df.empty and not y_holdout.empty and sum(weights.values()) > 0):
             proba_ensemble /= sum(weights.values())

        pred_ensemble = (proba_ensemble >= 0.5).astype(int)

        final_validation_score = f1_score(y_val, pred_ensemble, average='macro', zero_division=0)
        print(f'Final Validation Performance: {final_validation_score}')
    else:
        print("Validation set is empty. Cannot compute performance.", file=sys.stderr)
        print('Final Validation Performance: 0.0')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"A critical error occurred in the main execution block: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # To ensure the evaluation system receives a score even on failure
        print('Final Validation Performance: 0.0')
