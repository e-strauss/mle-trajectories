
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import warnings

# Correctly placed imports at the top of the script
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# --- Feature Engineering ---
def feature_engineer(df, is_train=True, scalers=None, encoders=None):
    """Applies feature engineering to the dataframe."""
    if is_train:
        scalers = {}
        encoders = {}

    # 1. Term-based features
    df['TERM_SEASON'] = df['TERM_DESC'].apply(lambda x: x.split(' ')[0])
    if is_train:
        encoders['TERM_SEASON'] = LabelEncoder()
        df['TERM_SEASON'] = encoders['TERM_SEASON'].fit_transform(df['TERM_SEASON'])
    else:
        # Handle unseen categories in test data
        df['TERM_SEASON'] = df['TERM_SEASON'].map(lambda s: s if s in encoders['TERM_SEASON'].classes_ else -1)
        known_mask = df['TERM_SEASON'] != -1
        if np.any(known_mask):
             df.loc[known_mask, 'TERM_SEASON'] = encoders['TERM_SEASON'].transform(df.loc[known_mask, 'TERM_SEASON'])

    # 2. Course-level features
    df['COURSE_LEVEL'] = df['CATALOG_NBR'].str[:1]
    if is_train:
        encoders['COURSE_LEVEL'] = LabelEncoder()
        df['COURSE_LEVEL'] = encoders['COURSE_LEVEL'].fit_transform(df['COURSE_LEVEL'])
    else:
        df['COURSE_LEVEL'] = df['COURSE_LEVEL'].map(lambda s: s if s in encoders['COURSE_LEVEL'].classes_ else -1)
        known_mask = df['COURSE_LEVEL'] != -1
        if np.any(known_mask):
            df.loc[known_mask, 'COURSE_LEVEL'] = encoders['COURSE_LEVEL'].transform(df.loc[known_mask, 'COURSE_LEVEL'])


    # 3. Interaction features (Example)
    df['DEPT_LEVEL_INTERACTION'] = df['SUBJECT_ID_SORT'] + '_' + df['COURSE_LEVEL'].astype(str)
    if is_train:
        encoders['DEPT_LEVEL_INTERACTION'] = LabelEncoder()
        df['DEPT_LEVEL_INTERACTION'] = encoders['DEPT_LEVEL_INTERACTION'].fit_transform(df['DEPT_LEVEL_INTERACTION'])
    else:
        df['DEPT_LEVEL_INTERACTION'] = df['DEPT_LEVEL_INTERACTION'].map(lambda s: s if s in encoders['DEPT_LEVEL_INTERACTION'].classes_ else -1)
        known_mask = df['DEPT_LEVEL_INTERACTION'] != -1
        if np.any(known_mask):
            df.loc[known_mask, 'DEPT_LEVEL_INTERACTION'] = encoders['DEPT_LEVEL_INTERACTION'].transform(df.loc[known_mask, 'DEPT_LEVEL_INTERACTION'])

    # Scale numerical features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in ['TERM_CODE', 'SUBJECT_ID_SORT']: # remove identifiers
        if col in numeric_cols:
            numeric_cols.remove(col)

    if is_train:
        for col in numeric_cols:
            scalers[col] = StandardScaler()
            df[col] = scalers[col].fit_transform(df[[col]])
    else:
        for col in numeric_cols:
            if col in scalers:
                df[col] = scalers[col].transform(df[[col]])

    return df, scalers, encoders

# --- Data Loading and Merging ---
def load_and_merge_data(data_dir):
    """Loads and merges all relevant tables from the data directory."""
    try:
        courses_df = pd.read_csv(os.path.join(data_dir, 'subject_summary.csv'))
        terms_df = pd.read_csv(os.path.join(data_dir, 'term_summary.csv'))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure you are in the correct directory.")
        return None

    # Merge courses and terms
    df = pd.merge(courses_df, terms_df, on='TERM_CODE')

    # Add more data sources if necessary (e.g., instructor data)
    # For example:
    # instructors_df = pd.read_csv(os.path.join(data_dir, 'instructors.csv'))
    # df = pd.merge(df, instructors_df, on=['TERM_CODE', 'COURSE_ID'], how='left')

    return df

# --- Main Pipeline ---
def run_pipeline(model_name, train_dir, gold_enrollment_path, test_dir=None, predict_test=False):
    """Runs the full training and prediction pipeline."""
    print("1. Loading and merging data...")
    train_df = load_and_merge_data(train_dir)
    if train_df is None:
        return

    print("2. Loading gold labels...")
    gold_df = pd.read_csv(gold_enrollment_path)
    train_df = pd.merge(train_df, gold_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'])

    # Handle subsampling if dataset is large (for faster iteration)
    if len(train_df) > 50000:
         print("Subsampling data...")
         train_df = train_df.sample(n=50000, random_state=42)


    print("3. Feature Engineering...")
    train_df, scalers, encoders = feature_engineer(train_df)

    # Convert target to binary
    train_df['HIGH_ENROLLMENT'] = train_df['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    # Define features and target
    features = [col for col in train_df.columns if col not in ['TERM_DESC', 'SUBJECT_DESCR', 'HIGH_ENROLLMENT', 'SUBJECT_ID_SORT']]
    # Ensure all feature columns are numeric, handle potential non-numeric dtypes
    non_numeric_features = train_df[features].select_dtypes(exclude=np.number).columns
    if len(non_numeric_features) > 0:
        print(f"Warning: Non-numeric features detected and will be dropped: {list(non_numeric_features)}")
        features = [f for f in features if f not in non_numeric_features]

    X = train_df[features]
    y = train_df['HIGH_ENROLLMENT']

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)


    print("4. Splitting data for validation...")
    # Time-based split: use last term for validation
    last_term = train_df['TERM_CODE'].max()
    train_indices = train_df[train_df['TERM_CODE'] < last_term].index
    val_indices = train_df[train_df['TERM_CODE'] == last_term].index

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

    if len(X_val) == 0:
        print("Validation set is empty. Using random split for validation.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    print(f"5. Training model: {model_name}...")
    if model_name == "lgbm":
        model = LGBMClassifier(random_state=42)
    elif model_name == "tabpfn":
        # Note: TabPFN is best for small datasets (<10k samples)
        # It may be slow or ineffective on larger datasets.
        model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    else:
        raise ValueError("Invalid model name specified.")

    model.fit(X_train, y_train)


    print("6. Evaluating model...")
    y_pred_val = model.predict(X_val)
    final_validation_score = f1_score(y_val, y_pred_val, average='macro')
    print(f'Final Validation Performance: {final_validation_score}')


    if predict_test and test_dir:
        print("\n7. Making predictions on test data...")
        test_df = load_and_merge_data(test_dir)
        if test_df is None: return

        # Store keys for final output
        test_keys = test_df[['TERM_CODE', 'SUBJECT_ID_SORT']]

        test_df, _, _ = feature_engineer(test_df, is_train=False, scalers=scalers, encoders=encoders)
        X_test = test_df[features]
        X_test = imputer.transform(X_test)

        predictions = model.predict(X_test)
        predictions = ['Y' if p == 1 else 'N' for p in predictions]

        # Create submission file
        submission_df = test_keys
        submission_df['HIGH_ENROLLMENT'] = predictions
        submission_df.to_csv('prediction.csv', index=False)
        print("Submission file 'prediction.csv' created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lgbm', choices=['lgbm', 'tabpfn'],
                        help='Model to train.')
    parser.add_argument('--train_dir', type=str, default='./input',
                        help='Directory for training data tables.')
    parser.add_argument('--gold_path', type=str, default='./input/gold_enrollment_train.csv',
                        help='Path to the gold enrollment file for training.')
    # The following arguments are placeholders for a future test phase
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory for held-out test tables.')
    parser.add_argument('--predict_test', action='store_true',
                        help='Flag to run prediction on the test set.')

    args = parser.parse_args()

    # The NameError was caused by these imports being inside the main block.
    # They have been moved to the top of the file to be in the global scope.
    # from lightgbm import LGBMClassifier
    # from tabpfn import TabPFNClassifier

    run_pipeline(
        model_name=args.model,
        train_dir=args.train_dir,
        gold_enrollment_path=args.gold_path,
        test_dir=args.test_dir,
        predict_test=args.predict_test
    )
