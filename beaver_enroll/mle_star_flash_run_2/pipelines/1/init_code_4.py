
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_DIR = './input'
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, 'table_splits', 'train')
GOLD_ENROLLMENT_FILE = os.path.join(TRAIN_DATA_DIR, 'gold_enrollment_train.csv')

# --- Helper Functions ---
def load_data(directory, file_name):
    """Loads a CSV file from a specified directory."""
    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Required data file not found: {file_path}")
    logging.info(f"Loading data from {file_path}")
    # Using low_memory=False to avoid DtypeWarning for mixed types in potentially large files
    return pd.read_csv(file_path, low_memory=False)

def load_all_train_data(train_data_dir):
    """Loads all relevant tables from the training data directory."""
    data_frames = {}
    
    # Try to find a central table containing course offering details
    # Prioritize 'subject_summary.csv' as mentioned in problem description context,
    # falling back to 'course_offerings.csv' if not found.
    potential_main_tables = ['subject_summary.csv', 'course_offerings.csv']
    main_df_found = False
    
    for table_name in potential_main_tables:
        try:
            df = load_data(train_data_dir, table_name)
            data_frames['main_table'] = df
            logging.info(f"Loaded {table_name} with {len(df)} rows as main_table.")
            main_df_found = True
            break
        except FileNotFoundError:
            logging.warning(f"{table_name} not found in {train_data_dir}.")

    if not main_df_found:
        logging.error(f"None of the expected main tables ({', '.join(potential_main_tables)}) were found in {train_data_dir}. Cannot proceed without core data.")
        raise FileNotFoundError("No core course data table found.")
        
    return data_frames

def preprocess_features(df):
    """
    Generate features from the raw data and handle initial data cleaning.
    """
    processed_df = df.copy()

    # Ensure TERM_CODE is treated as string for slicing
    if 'TERM_CODE' in processed_df.columns:
        processed_df['TERM_CODE_STR'] = processed_df['TERM_CODE'].astype(str)
        processed_df['TERM_YEAR'] = processed_df['TERM_CODE_STR'].str[:4].astype(int)
        processed_df['TERM_SEMESTER'] = processed_df['TERM_CODE_STR'].str[4:].astype(int)
        logging.info("Generated 'TERM_YEAR' and 'TERM_SEMESTER' features.")
    else:
        logging.warning("'TERM_CODE' column not found, cannot generate time-based features.")

    # Example feature engineering based on common academic data
    # Handle cases where columns might not exist
    if 'ENROLLMENT_COUNT' in processed_df.columns and 'CAPACITY' in processed_df.columns:
        # Ensure 'CAPACITY' is not zero to avoid division by zero
        processed_df['CAPACITY_ADJ'] = processed_df['CAPACITY'].replace(0, np.nan) 
        processed_df['FILL_RATE'] = processed_df['ENROLLMENT_COUNT'] / processed_df['CAPACITY_ADJ']
        processed_df['FILL_RATE'] = processed_df['FILL_RATE'].replace([np.inf, -np.inf], np.nan) # Ensure no inf values
        processed_df.drop(columns=['CAPACITY_ADJ'], inplace=True, errors='ignore') # Drop temporary column
        logging.info("Created 'FILL_RATE' feature.")
    else:
        logging.warning("Columns 'ENROLLMENT_COUNT' or 'CAPACITY' not found for 'FILL_RATE' feature generation.")

    # Convert object columns that contain mostly numeric data to numeric
    for col in processed_df.select_dtypes(include='object').columns:
        # Check if a significant portion of non-null values can be converted to numeric
        # This helps in cases where numbers are stored as strings.
        is_numeric_like = pd.to_numeric(processed_df[col], errors='coerce').notna().sum() / processed_df[col].count() > 0.8
        if is_numeric_like:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            logging.info(f"Converted '{col}' to numeric type.")
        
    return processed_df

# --- Main Script ---
def main():
    logging.info("Starting training process.")

    # 1. Load Data
    try:
        train_dfs = load_all_train_data(TRAIN_DATA_DIR)
        gold_labels = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_FILE)
    except FileNotFoundError as e:
        logging.error(f"Critical error during data loading: {e}")
        return

    main_df = train_dfs.get('main_table')
    if main_df is None:
        logging.error("Main course offerings table not found after loading. Exiting.")
        return

    # Merge gold labels with the main dataframe
    # Ensure join keys are present in both dataframes
    required_join_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']
    for key in required_join_keys:
        if key not in main_df.columns:
            logging.error(f"Required join key '{key}' not found in the main data frame. Cannot merge with gold labels. Exiting.")
            return
        if key not in gold_labels.columns:
            logging.error(f"Required join key '{key}' not found in gold labels. Cannot merge. Exiting.")
            return

    # Ensure gold_labels keys are unique for merging if there are duplicates in gold_enrollment_train.csv
    gold_labels_merged = gold_labels.drop_duplicates(subset=required_join_keys)
    
    df = pd.merge(main_df, gold_labels_merged, on=required_join_keys, how='inner')
    logging.info(f"Merged data has {len(df)} rows.")

    if df.empty:
        logging.error("Merged DataFrame is empty. This might indicate issues with join keys or missing data after merge. Exiting.")
        return

    # 2. Preprocessing and Feature Engineering
    df = preprocess_features(df)
    
    # Drop rows where HIGH_ENROLLMENT is missing (shouldn't happen with inner join but good practice)
    df.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    
    # Convert target to numeric (Y=1, N=0)
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})
    
    # Define features (X) and target (y)
    # Exclude ID columns, target, and temporary helper columns from features
    id_cols = ['TERM_CODE', 'SUBJECT_ID_SORT', 'TERM_CODE_STR'] # TERM_CODE_STR is temporary
    target_col = 'HIGH_ENROLLMENT'
    
    # Automatically identify features for the model
    # Exclude columns with too many unique values (potential IDs) or non-numeric/non-categorical types
    candidate_features = [
        col for col in df.columns 
        if col not in id_cols + [target_col] 
        and not pd.api.types.is_datetime64_any_dtype(df[col])
        and df[col].nunique() > 1 # Exclude columns with only one unique value (constant columns)
    ]
    
    # Refine numerical and categorical features after preprocessing
    numerical_features = df[candidate_features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[candidate_features].select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out high cardinality categorical features that might cause OHE to blow up memory
    # Heuristic: if unique values > 50 or > 10% of total rows, consider it high cardinality
    initial_categorical_features_count = len(categorical_features)
    categorical_features = [
        f for f in categorical_features 
        if df[f].nunique() <= 50 and df[f].nunique() < len(df) * 0.1
    ]
    if len(categorical_features) < initial_categorical_features_count:
        logging.warning(f"Filtered out {initial_categorical_features_count - len(categorical_features)} high cardinality categorical features.")

    # If no features are identified, add a dummy feature to prevent pipeline errors
    if not numerical_features and not categorical_features:
        logging.warning("No suitable numerical or categorical features identified. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0
        numerical_features.append('DUMMY_FEATURE') # Make it a numerical feature

    X = df[numerical_features + categorical_features]
    y = df[target_col]

    logging.info(f"Final Numerical features used: {numerical_features}")
    logging.info(f"Final Categorical features used: {categorical_features}")

    if X.empty:
        logging.error("Feature DataFrame (X) is empty after selection. Cannot proceed. Exiting.")
        return
    if y.empty:
        logging.error("Target Series (y) is empty. Cannot proceed. Exiting.")
        return

    # 3. Time-based Validation Split
    df_for_split = df.copy() # Use the preprocessed df for splitting
    
    # Ensure TERM_CODE is numeric for proper sorting
    if 'TERM_CODE' in df_for_split.columns:
        df_for_split['TERM_CODE_INT'] = pd.to_numeric(df_for_split['TERM_CODE'], errors='coerce')
        df_sorted = df_for_split.sort_values(by='TERM_CODE_INT').reset_index(drop=True)
    else:
        logging.warning("'TERM_CODE' not found for time-based split, using current order for sorting.")
        df_sorted = df_for_split.reset_index(drop=True)

    unique_terms = df_sorted['TERM_CODE'].dropna().unique() # Get unique terms after sorting
    
    X_train_val, X_test_val, y_train_val, y_test_val = None, None, None, None

    if len(unique_terms) < 2:
        logging.warning("Not enough unique terms for a time-based split. Using random split for validation (20%).")
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
    else:
        # Use the latest term as validation set
        validation_term = unique_terms[-1]
        logging.info(f"Using TERM_CODE {validation_term} for validation set.")
        
        train_val_mask = df_sorted['TERM_CODE'] < validation_term
        test_val_mask = df_sorted['TERM_CODE'] == validation_term
        
        X_train_val = df_sorted[train_val_mask][numerical_features + categorical_features]
        y_train_val = df_sorted[train_val_mask][target_col]
        
        X_test_val = df_sorted[test_val_mask][numerical_features + categorical_features]
        y_test_val = df_sorted[test_val_mask][target_col]

        if X_test_val.empty or y_test_val.empty or X_train_val.empty or y_train_val.empty:
            logging.warning(f"Time-based split resulted in an empty train or test set for term {validation_term}. Falling back to random split.")
            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )

    logging.info(f"Training set size: {len(X_train_val)} (positive: {y_train_val.sum()}, negative: {len(y_train_val) - y_train_val.sum()})")
    logging.info(f"Validation set size: {len(X_test_val)} (positive: {y_test_val.sum()}, negative: {len(y_test_val) - y_test_val.sum()})")
    
    if X_train_val.empty or X_test_val.empty:
        logging.error("Training or validation set is empty after splitting. Cannot proceed. Exiting.")
        return
    if y_train_val.nunique() < 2 or y_test_val.nunique() < 2:
        logging.warning("Training or validation target set contains less than 2 unique classes. This can lead to model training or evaluation errors.")

    # 4. Create Preprocessing Pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Only include transformers if there are features for them
    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        logging.error("No features for ColumnTransformer. Cannot proceed with model training. Exiting.")
        return

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # Drop other columns not explicitly transformed
    )

    # 5. Model Training
    # Using RandomForestClassifier as a good general-purpose model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

    logging.info("Starting model training.")
    try:
        model.fit(X_train_val, y_train_val)
        logging.info("Model training complete.")
    except Exception as e:
        logging.error(f"Error during model fitting: {e}. Exiting.")
        return

    # 6. Evaluation
    y_pred = model.predict(X_test_val)
    
    # Calculate macro F1 score
    if len(np.unique(y_test_val)) < 2:
        # If validation set has only one class, F1 score calculation might fail or be misleading.
        logging.warning("Validation set contains only one class. Macro F1 score might be undefined or misleading. Setting to 0.0.")
        final_validation_score = 0.0 # Assign a default score to indicate the issue
    else:
        final_validation_score = f1_score(y_test_val, y_pred, average='macro')
    
    logging.info(f"Final Validation Performance: {final_validation_score}")
    print(f"Final Validation Performance: {final_validation_score}")

if __name__ == '__main__':
    main()
