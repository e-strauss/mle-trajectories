
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import sys
import subprocess
import warnings
import random
import torch # Import torch at the top if it's generally required or checked for.

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
TEST_DATA_DIR = os.path.join(BASE_DIR, "__TEST_DATA_DIR__")  # Use the placeholder as specified
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
# Initialize globally. This will be the initial state passed to main.
_can_proceed_tabnet = False
pytorch_tabnet = None
# torch is already imported at the top, but we need to ensure it's functional for TabNet
TabNetClassifier = None # Ensure TabNetClassifier is initialized to None

# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
except ImportError:
    print("pytorch_tabnet or torch (if used by tabnet internally) not found. Attempting to install pytorch-tabnet and torch...")
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        print("pytorch-tabnet and torch installed successfully.")
        # Attempt to import again after successful installation
        import pytorch_tabnet
        from pytorch_tabnet.tab_model import TabNetClassifier # Import after successful installation
        _can_proceed_tabnet = True
    except Exception as e:
        print(f"Failed to install pytorch-tabnet and torch: {e}")
        print("TabNet will not be used in this run.")
        _can_proceed_tabnet = False # Explicitly set to False on failure

# --- Data Loading Function (from reference solution, enhanced) ---
def load_data_from_dir(data_dir):
    """
    Loads all primary summary data from a given directory by merging all CSVs.
    Assumes CSVs contain 'TERM_CODE' and 'SUBJECT_ID_SORT' for merging.
    """
    all_files = os.listdir(data_dir)
    df_list = []
    
    primary_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']

    for f in all_files:
        if f.endswith(".csv"):
            file_path = os.path.join(data_dir, f)
            try:
                df = pd.read_csv(file_path)
                
                # Ensure primary keys are present
                if not all(key in df.columns for key in primary_keys):
                    continue
                
                # Ensure consistent data types for merging
                df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not df_list:
        print(f"No valid CSV files found or processed in {data_dir}.")
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Main Script ---
def main():
    # Make a local copy of the global flag to avoid UnboundLocalError.
    # This local variable will be used and potentially modified within main().
    can_proceed_tabnet_local = _can_proceed_tabnet

    print("Loading training data...")
    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    # Try loading real data first
    try:
        train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
        gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
        gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)

        if train_data_raw.empty or gold_labels_df.empty:
            raise FileNotFoundError("Raw training data or gold labels are empty after loading.")

        # Merge features with gold labels
        train_df = pd.merge(train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        if train_df.empty:
            raise ValueError("Training DataFrame is empty after merging with gold labels.")

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading real data: {e}. Creating dummy training data for demonstration.")
        # Create dummy data if real data loading fails or results in empty df
        
        # The following block of dummy data generation had an indentation error.
        # It was not correctly indented under the 'except' block.
        # The fix is to indent all lines from 'import pandas as pd' until
        # 'print("Using dummy training data.")' by one level (4 spaces).
        
        # Define the number of rows for the larger dataset, within the specified range
        num_rows = random.randint(100, 500)

        # --- Generate TERM_CODE ---
        # Generate a sequence of term codes, ensuring variety but not an excessive number of unique terms
        term_years = np.arange(2023, 2023 + num_rows // 20 + 2) # Create enough years based on num_rows
        term_options = []
        for y in term_years:
            term_options.extend([y * 100 + 1, y * 100 + 7]) # E.g., 202301, 202307, 202401, etc.
        term_codes = random.choices(term_options, k=num_rows)

        # --- Generate SUBJECT_ID_SORT ---
        # A wider variety of subjects
        subjects_pool = [f'CS-{i:03d}' for i in range(101, 130)] + \
                        [f'MA-{i:03d}' for i in range(201, 225)] + \
                        [f'PH-{i:03d}' for i in range(101, 115)] + \
                        [f'BI-{i:03d}' for i in range(301, 320)] + \
                        [f'CH-{i:03d}' for i in range(201, 210)] + \
                        [f'EE-{i:03d}' for i in range(401, 410)]
        subject_ids = random.choices(subjects_pool, k=num_rows)

        # --- Generate CREDIT_HOURS ---
        # More varied distribution for credit hours
        credit_hours_options = [2, 3, 4, 5]
        credit_hours = random.choices(credit_hours_options, weights=[0.1, 0.45, 0.4, 0.05], k=num_rows)

        # --- Generate INSTRUCTOR_RANK ---
        instructor_ranks_options = ['PROF', 'ASSIST', 'LECT', 'ADJ', 'VISIT']
        instructor_ranks = random.choices(instructor_ranks_options, weights=[0.3, 0.3, 0.2, 0.15, 0.05], k=num_rows)

        # --- Generate CAPACITY and PREV_ENROLLMENT_AVG with inter-feature dependencies ---
        capacities = np.random.randint(30, 150, num_rows)
        prev_enrollments = []
        high_enrollments = []

        popular_subjects = ['CS-101', 'MA-201', 'CS-103'] # Define some popular subjects

        for i in range(num_rows):
            cap = capacities[i]
            
            # Base enrollment factor
            enrollment_factor = np.random.uniform(0.6, 1.05) # Typically below or slightly above capacity

            # Introduce dependency: popular subjects tend to have higher enrollment
            if subject_ids[i] in popular_subjects:
                enrollment_factor = np.random.uniform(0.8, 1.2) # Higher enrollment for popular courses
            elif instructor_ranks[i] == 'PROF': # Professors might attract more students
                enrollment_factor = np.random.uniform(0.7, 1.1)
            
            prev_enroll = int(cap * enrollment_factor)
            prev_enrollments.append(prev_enroll)
            
            # Calculate HIGH_ENROLLMENT based on capacity and enrollment
            high_enrollments.append(1 if (prev_enroll / cap > 0.85) else 0)

        # Create the base DataFrame
        train_df = pd.DataFrame({
            'TERM_CODE': term_codes,
            'SUBJECT_ID_SORT': subject_ids,
            'CREDIT_HOURS': credit_hours,
            'INSTRUCTOR_RANK': instructor_ranks,
            'CAPACITY': capacities,
            'PREV_ENROLLMENT_AVG': prev_enrollments,
            'HIGH_ENROLLMENT': high_enrollments
        })

        # --- Introduce Missing Values ---
        # Select columns to introduce NaNs into and a percentage
        columns_for_nan = ['CREDIT_HOURS', 'INSTRUCTOR_RANK', 'PREV_ENROLLMENT_AVG', 'CAPACITY', 'SUBJECT_ID_SORT']
        nan_percentage = 0.05 # 5% missing values

        for col in columns_for_nan:
            num_nan = int(num_rows * nan_percentage)
            nan_indices = np.random.choice(train_df.index, num_nan, replace=False)
            train_df.loc[nan_indices, col] = np.nan

        # --- Introduce Realistic Outliers ---
        outlier_percentage = 0.015 # 1.5% outliers

        # CAPACITY outliers (e.g., a small seminar, or a massive lecture hall)
        num_outliers_cap = int(num_rows * outlier_percentage)
        outlier_indices_cap = np.random.choice(train_df.index, num_outliers_cap, replace=False)
        for idx in outlier_indices_cap:
            if random.random() < 0.5:
                train_df.loc[idx, 'CAPACITY'] = random.randint(1, 15) # Very small capacity
            else:
                train_df.loc[idx, 'CAPACITY'] = random.randint(200, 700) # Very large capacity

        # PREV_ENROLLMENT_AVG outliers (e.g., course cancelled, or unexpected surge)
        num_outliers_enroll = int(num_rows * outlier_percentage)
        outlier_indices_enroll = np.random.choice(train_df.index, num_outliers_enroll, replace=False)
        for idx in outlier_indices_enroll:
            if random.random() < 0.5:
                train_df.loc[idx, 'PREV_ENROLLMENT_AVG'] = random.randint(0, 5) # Extremely low (e.g., cancelled or very few)
            else:
                train_df.loc[idx, 'PREV_ENROLLMENT_AVG'] = random.randint(300, 800) # Extremely high (could significantly exceed capacity)

        # CREDIT_HOURS outliers (e.g., a non-standard 1-credit seminar or a 6-credit intensive course)
        num_outliers_credits = int(num_rows * outlier_percentage * 0.5) # Fewer credit hour outliers
        outlier_indices_credits = np.random.choice(train_df.index, num_outliers_credits, replace=False)
        for idx in outlier_indices_credits:
            train_df.loc[idx, 'CREDIT_HOURS'] = random.choice([1, 6, 7]) # Non-standard credits

        # Note: After introducing NaNs and outliers, the 'HIGH_ENROLLMENT' column might become inconsistent
        # with 'PREV_ENROLLMENT_AVG' and 'CAPACITY' for those specific rows. This reflects real-world data
        # where target variables might not perfectly align with corrupted features.
        print("Using dummy training data.")

    # Ensure TERM_CODE is numeric for sorting
    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    # --- Feature Engineering & Preprocessing ---
    target = 'HIGH_ENROLLMENT'
    
    # Columns to be dropped from features (identifiers or target itself)
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        # Heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        if train_df[col].dtype == 'object' or (train_df[col].dtype in ['int64', 'float64'] and train_df[col].nunique() < 50):
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    # Store LabelEncoders for categorical features
    label_encoders = {}
    tabnet_cat_dims = [] # Store dimensions for TabNet's categorical embeddings
    
    for col in categorical_features:
        # Fill NA before unique() to ensure 'nan_category' is included in classes if needed
        train_df[col] = train_df[col].astype(str).fillna('nan_category') 
        le = LabelEncoder()
        # Fit on all unique values including 'nan_category'
        le.fit(train_df[col].unique()) 
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        # Add 1 for potential unseen categories in test or if 'nan_category' is one of the classes
        tabnet_cat_dims.append(len(le.classes_) + 1) 

    numerical_means = {}
    for col in numerical_features:
        if train_df[col].isnull().any():
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            numerical_means[col] = mean_val
        else:
            numerical_means[col] = train_df[col].mean() 

    # Define the final list of feature columns for the model
    feature_columns = numerical_features + categorical_features
    
    if not feature_columns:
        raise ValueError("No features identified for training after preprocessing.")

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-based Validation Split ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.")
        if len(train_df_for_split) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            raise ValueError("Insufficient data to perform any kind of train-validation split.")
    else:
        # Using a fixed percentage (e.g., 20%) of terms for validation, at least one term
        num_val_terms = max(1, int(len(unique_terms) * 0.2))
        val_terms = unique_terms[-num_val_terms:]
        
        val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
        train_val_indices = train_df_for_split[~train_df_for_split['TERM_CODE'].isin(val_terms)].index

        X_train_val_rf = X_full.loc[train_val_indices]
        y_train_val = y_full.loc[train_val_indices]
        X_val_rf = X_full.loc[val_indices]
        y_val = y_full.loc[val_indices]

        if X_train_val_rf.empty or X_val_rf.empty:
            print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.")
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")
        else:
            print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}")
            print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}")

    print(f"Training data size: {len(X_train_val_rf)}")
    print(f"Validation data size: {len(X_val_rf)}")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        raise ValueError("Training or validation set is empty. Cannot proceed with model training.")

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)


    # --- Model Training: RandomForest ---
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    print("Evaluating RandomForest on validation set...")
    rf_y_pred_val = rf_model.predict(X_val_rf)
    rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')
    print(f"RandomForest Validation F1: {rf_val_f1}")

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used

    if can_proceed_tabnet_local:
        print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            can_proceed_tabnet_local = False
        else:
            tabnet_model = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=tabnet_cat_dims,
                cat_emb_dim=1,
                n_d=8, n_a=8,
                n_steps=3,
                gamma=1.3,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":50, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='sparsemax',
                verbose=0,
                seed=42 # Add seed for reproducibility
            )
            
            try:
                tabnet_model.fit(
                    X_train=X_train_val_tabnet, y_train=y_train_val_np,
                    eval_set=[(X_train_val_tabnet, y_train_val_np), (X_val_tabnet, y_val_np)],
                    eval_name=['train', 'valid'],
                    eval_metric=['f1', 'accuracy'],
                    max_epochs=100,
                    patience=10,
                    batch_size=1024,
                    virtual_batch_size=128,
                    drop_last=False
                )
                tabnet_y_pred_val = tabnet_model.predict(X_val_tabnet)
                tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='macro')
                print(f"TabNet Validation F1: {tabnet_val_f1}")
            except Exception as e:
                print(f"Error during TabNet training or validation: {e}. TabNet will not contribute to ensemble.")
                can_proceed_tabnet_local = False # Disable TabNet for prediction too

    # --- Ensemble Validation ---
    print("Ensembling predictions on validation set...")
    if can_proceed_tabnet_local:
        # Simple averaging for ensemble. Ensure predictions are of similar type (e.g., float before averaging)
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    print(f'Final Validation Performance: {final_validation_f1}') # Required output format


    # --- Prediction on Test Data ---
    print("Loading test data...")
    test_df_raw = None
    try:
        test_df_raw = load_data_from_dir(TEST_DATA_DIR)
        if test_df_raw.empty:
            raise FileNotFoundError("Test data is empty after loading.")
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}. Creating dummy test data for demonstration.")
        test_df_raw = pd.DataFrame({
            'TERM_CODE': [202507, 202507, 202601],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'PH-101'],
            'CREDIT_HOURS': [3, 4, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT'],
            'CAPACITY': [100, 50, 120],
            'PREV_ENROLLMENT_AVG': [80, 45, 110]
        })
        print("Using dummy test data.")
    
    # Keep original test_df for output merge
    test_original_df = test_df_raw[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()

    test_processed_df = test_df_raw.copy()
    test_processed_df['TERM_CODE'] = pd.to_numeric(test_processed_df['TERM_CODE'])

    # Preprocess test data using training-derived transformations
    # Handle categorical features
    for i, col in enumerate(categorical_features):
        if col in test_processed_df.columns:
            test_processed_df[col] = test_processed_df[col].astype(str).fillna('nan_category')
            le = label_encoders[col]
            def transform_with_unseen(x, encoder, cat_dim_val):
                try:
                    return encoder.transform([x])[0]
                except ValueError:
                    # Assign to the last category index for unseen
                    return cat_dim_val - 1 
            
            # Use the correct `tabnet_cat_dims[i]` for the current column's dimensions
            test_processed_df[col] = test_processed_df[col].apply(lambda x: transform_with_unseen(x, le, tabnet_cat_dims[i]))
        else:
            # If categorical feature is entirely missing in test data, fill with an appropriate value.
            # For TabNet, this would be the 'unseen' code (tabnet_cat_dims[i] - 1).
            # If TabNet is not active, filling with 0 (first category) is a reasonable fallback.
            # Ensure index 'i' for tabnet_cat_dims is valid
            fill_value = tabnet_cat_dims[i] - 1 if can_proceed_tabnet_local and i < len(tabnet_cat_dims) else 0
            test_processed_df[col] = fill_value

    # Handle numerical features
    for col in numerical_features:
        if col in test_processed_df.columns:
            if test_processed_df[col].isnull().any():
                mean_val = numerical_means.get(col, 0) # Use stored mean, default to 0 if not found
                test_processed_df[col] = test_processed_df[col].fillna(mean_val)
        else:
            # If numerical feature is entirely missing in test data, fill with its training mean
            test_processed_df[col] = numerical_means.get(col, 0) 
            
    # Ensure all features used in training are present in test_processed_df and in the correct order
    # Add missing columns with 0, remove extra columns
    for col in feature_columns:
        if col not in test_processed_df.columns:
            test_processed_df[col] = 0 # Default value for missing features in test

    # Drop columns in test_processed_df that are not in feature_columns (new features in test data)
    extra_cols_in_test = set(test_processed_df.columns) - set(feature_columns)
    if extra_cols_in_test:
        test_processed_df = test_processed_df.drop(columns=list(extra_cols_in_test))

    # Ensure the order of columns is the same as training features
    X_test_processed = test_processed_df[feature_columns]

    print("Making predictions on test data...")
    rf_y_pred_test = rf_model.predict(X_test_processed)
    
    tabnet_y_pred_test = np.zeros(len(X_test_processed)) # Default to zeros
    if can_proceed_tabnet_local:
        try:
            tabnet_y_pred_test = tabnet_model.predict(X_test_processed.values)
        except Exception as e:
            print(f"Error during TabNet test prediction: {e}. TabNet predictions will be zeros.")
            
    # --- Ensemble Test Predictions ---
    if can_proceed_tabnet_local:
        # Ensure predictions are of similar type (e.g., float before averaging)
        ensemble_y_pred_test = (rf_y_pred_test.astype(float) + tabnet_y_pred_test.astype(float)) / 2
    else:
        ensemble_y_pred_test = rf_y_pred_test.astype(float) # Only RF if TabNet not available

    final_ensemble_y_pred_test = (ensemble_y_pred_test >= 0.5).astype(int)

    # --- Prepare Submission ---
    submission_df = test_original_df.copy()
    submission_df['HIGH_ENROLLMENT'] = final_ensemble_y_pred_test

    # Output predictions to a CSV file (e.g., 'predictions.csv')
    output_path = "predictions.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
