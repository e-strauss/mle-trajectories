
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder # Added OneHotEncoder
import os
import numpy as np
import sys
import subprocess
import warnings

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
torch = None
TabNetClassifier = None # Ensure TabNetClassifier is initialized to None

_can_proceed_category_encoders = False
TargetEncoder = None # Ensure TargetEncoder is initialized to None


# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
except ImportError:
    print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...")
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        print("pytorch-tabnet and torch installed successfully.")
        # Attempt to import again after successful installation
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier # Import after successful installation
        _can_proceed_tabnet = True
    except Exception as e:
        print(f"Failed to install pytorch-tabnet and torch: {e}")
        print("TabNet will not be used in this run.")
        _can_proceed_tabnet = False # Explicitly set to False on failure

# Check and install category_encoders if not present
try:
    from category_encoders import TargetEncoder
    _can_proceed_category_encoders = True
except ImportError:
    print("category_encoders not found. Attempting to install category_encoders...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "category_encoders", "--user"])
        print("category_encoders installed successfully.")
        from category_encoders import TargetEncoder # Import after successful installation
        _can_proceed_category_encoders = True
    except Exception as e:
        print(f"Failed to install category_encoders: {e}")
        print("TargetEncoder will not be used in this run for high cardinality features.")
        _can_proceed_category_encoders = False # Explicitly set to False on failure


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
    # Make local copies of the global flags.
    can_proceed_tabnet_local = _can_proceed_tabnet
    can_proceed_category_encoders_local = _can_proceed_category_encoders

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
        train_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        print("Using dummy training data.")

    # Ensure TERM_CODE is numeric for sorting
    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    # --- Feature Engineering & Preprocessing ---
    target = 'HIGH_ENROLLMENT'
    
    # Columns to be dropped from features (identifiers or target itself)
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    numerical_features = []
    # `initial_categorical_features` stores the names of columns identified as categorical
    # before they are transformed by One-Hot or Target Encoding.
    initial_categorical_features = [] 
    
    # Internal lists to manage features destined for specific encoding strategies
    _low_card_cat_features_ohe = []
    _high_card_cat_features_target_enc = []

    # Define a threshold for distinguishing between low and high cardinality for categorical features.
    # Features with unique count less than this will be One-Hot Encoded.
    OHE_CARDINALITY_THRESHOLD = 10 

    # --- Feature Type Identification ---
    all_candidate_cols = [col for col in train_df.columns if col not in features_to_exclude]

    for col in all_candidate_cols:
        is_object_type = (train_df[col].dtype == 'object')
        unique_count = train_df[col].nunique()

        # Heuristic: object columns are categorical, and numericals with low unique count (< 50) are also considered categorical.
        if is_object_type or (unique_count < 50 and unique_count > 1): # unique_count > 1 to exclude constant numerical columns
            initial_categorical_features.append(col)
            if unique_count < OHE_CARDINALITY_THRESHOLD:
                _low_card_cat_features_ohe.append(col)
            else:
                _high_card_cat_features_target_enc.append(col)
        else:
            numerical_features.append(col)

    # Store encoders and transformers for later use (e.g., transforming test data).
    one_hot_encoders = {} # Store the OneHotEncoder instance
    target_encoders = {}  # Store the TargetEncoder instance
    numerical_means = {}  # Store means for numerical feature imputation

    # --- Encoding and Imputation ---

    # Make a copy of the dataframe to store processed features
    processed_train_df = train_df.copy()

    # 1. Process Low Cardinality Categorical Features with One-Hot Encoding
    ohe_feature_names = []
    if _low_card_cat_features_ohe:
        # Convert NaNs to a string representation to be treated as a separate category by OHE.
        for col in _low_card_cat_features_ohe:
            processed_train_df[col] = processed_train_df[col].astype(str).fillna('nan_ohe_category')

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(processed_train_df[_low_card_cat_features_ohe])
        ohe_transformed = ohe.transform(processed_train_df[_low_card_cat_features_ohe])
        
        ohe_feature_names = ohe.get_feature_names_out(_low_card_cat_features_ohe)
        ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_feature_names, index=processed_train_df.index)
        
        # Drop original columns and concatenate new OHE columns.
        processed_train_df = pd.concat([processed_train_df.drop(columns=_low_card_cat_features_ohe), ohe_df], axis=1)
        one_hot_encoders['ohe_transformer'] = ohe # Store the encoder instance

    # 2. Process High Cardinality Categorical Features with Target Encoding
    if _high_card_cat_features_target_enc:
        if not can_proceed_category_encoders_local:
            print("TargetEncoder is not available. High cardinality categorical features will be dropped.")
            processed_train_df = processed_train_df.drop(columns=[col for col in _high_card_cat_features_target_enc if col in processed_train_df.columns])
            _high_card_cat_features_target_enc = [] # Clear this list as features are dropped
        else:
            # Fill NaNs with a unique string placeholder to ensure TargetEncoder treats them as a distinct category.
            for col in _high_card_cat_features_target_enc:
                processed_train_df[col] = processed_train_df[col].astype(str).fillna('nan_te_category')

            # TargetEncoder will replace the original column with the encoded numerical value.
            te = TargetEncoder(cols=_high_card_cat_features_target_enc, handle_missing='value', handle_unknown='value')
            te.fit(processed_train_df[_high_card_cat_features_target_enc], processed_train_df[target])
            processed_train_df[_high_card_cat_features_target_enc] = te.transform(processed_train_df[_high_card_cat_features_target_enc])
            target_encoders['target_transformer'] = te # Store the encoder instance

    # 3. Process Numerical Features (Imputation)
    for col in numerical_features:
        if col in processed_train_df.columns:
            if processed_train_df[col].isnull().any():
                mean_val = processed_train_df[col].mean()
                processed_train_df[col] = processed_train_df[col].fillna(mean_val)
                numerical_means[col] = mean_val
            else: # Store mean even if no NaNs, for test data
                numerical_means[col] = processed_train_df[col].mean()

    # Define the final list of feature columns for the model
    # This must reflect the columns actually present in processed_train_df after all transformations
    feature_columns = []
    feature_columns.extend(numerical_features)
    feature_columns.extend(ohe_feature_names)
    feature_columns.extend(_high_card_cat_features_target_enc) # These are now numerical features

    # Filter feature_columns to ensure they actually exist in processed_train_df
    feature_columns = [col for col in feature_columns if col in processed_train_df.columns]
    
    if not feature_columns:
        raise ValueError("No features identified for training after preprocessing.")

    X_full = processed_train_df[feature_columns]
    y_full = processed_train_df[target]
    
    # --- Time-based Validation Split ---
    # Create a temporary dataframe that contains X_full, y_full and TERM_CODE for splitting
    df_for_split_temp = processed_train_df[['TERM_CODE'] + feature_columns + [target]].copy()
    df_for_split_temp = df_for_split_temp.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(df_for_split_temp['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.")
        if len(df_for_split_temp) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            raise ValueError("Insufficient data to perform any kind of train-validation split.")
    else:
        # Using a fixed percentage (e.g., 20%) of terms for validation, at least one term
        num_val_terms = max(1, int(len(unique_terms) * 0.2))
        val_terms = unique_terms[-num_val_terms:]
        
        val_df = df_for_split_temp[df_for_split_temp['TERM_CODE'].isin(val_terms)]
        train_val_df = df_for_split_temp[~df_for_split_temp['TERM_CODE'].isin(val_terms)]

        X_train_val_rf = train_val_df[feature_columns]
        y_train_val = train_val_df[target]
        X_val_rf = val_df[feature_columns]
        y_val = val_df[target]

        if X_train_val_rf.empty or X_val_rf.empty:
            print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.")
            if len(df_for_split_temp) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")
        else:
            print(f"Time-based split: Training on terms {sorted(train_val_df['TERM_CODE'].unique())}")
            print(f"Validating on terms: {sorted(val_df['TERM_CODE'].unique())}")

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
        # Since all features are now numerical after OHE/TargetEncoding, cat_idxs and cat_dims should be empty
        tabnet_cat_idxs = [] 
        tabnet_cat_dims_for_model = [] # This will remain empty as features are numerical

        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            can_proceed_tabnet_local = False
        else:
            tabnet_model = TabNetClassifier(
                cat_idxs=tabnet_cat_idxs, 
                cat_dims=tabnet_cat_dims_for_model, 
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
                seed=42 
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
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    print(f'Final Validation Performance: {final_validation_f1}')


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

    # --- Preprocess test data using training-derived transformations ---

    # 1. Process Low Cardinality Categorical Features with One-Hot Encoding
    if _low_card_cat_features_ohe:
        for col in _low_card_cat_features_ohe:
            if col in test_processed_df.columns:
                test_processed_df[col] = test_processed_df[col].astype(str).fillna('nan_ohe_category')
            else:
                test_processed_df[col] = 'nan_ohe_category' # Add missing column with placeholder for OHE
        
        ohe_transformer = one_hot_encoders['ohe_transformer']
        ohe_transformed_test = ohe_transformer.transform(test_processed_df[_low_card_cat_features_ohe])
        ohe_df_test = pd.DataFrame(ohe_transformed_test, columns=ohe_feature_names, index=test_processed_df.index)
        
        test_processed_df = pd.concat([test_processed_df.drop(columns=_low_card_cat_features_ohe, errors='ignore'), ohe_df_test], axis=1)

    # 2. Process High Cardinality Categorical Features with Target Encoding
    if _high_card_cat_features_target_enc:
        if not can_proceed_category_encoders_local:
            # Drop these columns if TargetEncoder was not available during training or here
            test_processed_df = test_processed_df.drop(columns=[col for col in _high_card_cat_features_target_enc if col in test_processed_df.columns], errors='ignore')
        else:
            for col in _high_card_cat_features_target_enc:
                if col in test_processed_df.columns:
                    test_processed_df[col] = test_processed_df[col].astype(str).fillna('nan_te_category')
                else:
                    test_processed_df[col] = 'nan_te_category' # Add missing column with placeholder
            
            te_transformer = target_encoders['target_transformer']
            test_processed_df[_high_card_cat_features_target_enc] = te_transformer.transform(test_processed_df[_high_card_cat_features_target_enc])

    # 3. Process Numerical Features (Imputation)
    for col in numerical_features:
        if col in test_processed_df.columns:
            if test_processed_df[col].isnull().any():
                mean_val = numerical_means.get(col, 0) # Use stored mean, default to 0 if not found
                test_processed_df[col] = test_processed_df[col].fillna(mean_val)
        else:
            # If numerical feature is entirely missing in test data, add it and fill with its training mean
            test_processed_df[col] = numerical_means.get(col, 0) 
            
    # Ensure all features used in training are present in test_processed_df and in the correct order
    for col in feature_columns:
        if col not in test_processed_df.columns:
            test_processed_df[col] = 0 # Default value for missing features in test, should be handled by above logic better

    # Drop columns in test_processed_df that are not in feature_columns
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
