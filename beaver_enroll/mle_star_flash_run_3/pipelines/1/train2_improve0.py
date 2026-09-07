
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Added OneHotEncoder
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

# Define thresholds for cardinality to guide feature type inference and encoding strategy
OHE_CARDINALITY_THRESHOLD = 15  # Features with up to this many unique values will be One-Hot Encoded
FREQ_ENCODING_CARDINALITY_THRESHOLD = 500 # Features with up to this many unique values will be Frequency Encoded

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
    
    # --- Feature Type Identification ---
    # These lists hold the original column names before encoding
    rf_numerical_features = []
    rf_ohe_categorical_features = [] 
    rf_freq_categorical_features = [] 

    tabnet_numerical_features = [] 
    tabnet_categorical_features = [] 

    for col in train_df.columns:
        if col in features_to_exclude:
            continue

        current_col_nunique = train_df[col].nunique(dropna=False)

        if pd.api.types.is_object_dtype(train_df[col]) or pd.api.types.is_categorical_dtype(train_df[col]):
            if current_col_nunique <= OHE_CARDINALITY_THRESHOLD:
                rf_ohe_categorical_features.append(col)
                tabnet_categorical_features.append(col) 
            else:
                rf_freq_categorical_features.append(col)
                tabnet_categorical_features.append(col) 
        else: # Numerical or other dtypes
            if current_col_nunique <= OHE_CARDINALITY_THRESHOLD:
                # Numerical column with very low unique values, treat as categorical for both
                rf_ohe_categorical_features.append(col)
                tabnet_categorical_features.append(col)
            elif current_col_nunique <= FREQ_ENCODING_CARDINALITY_THRESHOLD:
                # Numerical column with medium cardinality, treat as categorical for both
                rf_freq_categorical_features.append(col)
                tabnet_categorical_features.append(col)
            else:
                # Truly numerical feature with high unique values for both
                rf_numerical_features.append(col)
                tabnet_numerical_features.append(col) 
    
    # --- Preprocessing for RandomForest ---
    train_df_rf = train_df.copy()
    rf_final_feature_columns = []

    # Imputation for Numerical Features (for RF)
    rf_numerical_means = {}
    for col in rf_numerical_features:
        if train_df_rf[col].isnull().any():
            mean_val = train_df_rf[col].mean()
            train_df_rf[col] = train_df_rf[col].fillna(mean_val)
            rf_numerical_means[col] = mean_val
        else:
            rf_numerical_means[col] = train_df_rf[col].mean()
        rf_final_feature_columns.append(col)

    # One-Hot Encoding for low-cardinality features (for RF)
    rf_ohe_encoder = None
    rf_ohe_feature_names_out = None
    if rf_ohe_categorical_features:
        for col in rf_ohe_categorical_features:
            train_df_rf[col] = train_df_rf[col].astype(str).fillna('nan_category')

        rf_ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_encoded_features = rf_ohe_encoder.fit_transform(train_df_rf[rf_ohe_categorical_features])
        rf_ohe_feature_names_out = rf_ohe_encoder.get_feature_names_out(rf_ohe_categorical_features)
        
        ohe_df = pd.DataFrame(ohe_encoded_features, columns=rf_ohe_feature_names_out, index=train_df_rf.index)
        train_df_rf = pd.concat([train_df_rf.drop(columns=rf_ohe_categorical_features), ohe_df], axis=1)
        rf_final_feature_columns.extend(list(rf_ohe_feature_names_out))

    # Frequency Encoding for high-cardinality features (for RF)
    rf_frequency_maps = {}
    for col in rf_freq_categorical_features:
        train_df_rf[col] = train_df_rf[col].astype(str).fillna('nan_category')

        freq_map = train_df_rf[col].value_counts(normalize=True).to_dict()
        train_df_rf[f'{col}_freq'] = train_df_rf[col].map(freq_map)
        rf_frequency_maps[col] = freq_map
        train_df_rf.drop(columns=[col], inplace=True)
        rf_final_feature_columns.append(f'{col}_freq')
    
    # Define X_full for RandomForest
    X_full_rf = train_df_rf[rf_final_feature_columns]
    y_full = train_df[target] # Target is common

    if not rf_final_feature_columns:
        raise ValueError("No features identified for RandomForest training after preprocessing.")

    # --- Preprocessing for TabNet (if enabled) ---
    X_full_tabnet = None
    tabnet_cat_idxs = []
    tabnet_cat_dims = []
    tabnet_label_encoders = {} # Store encoders for test set
    tabnet_final_feature_columns = [] # To store column names for TabNet

    if can_proceed_tabnet_local:
        train_df_tabnet = train_df.copy()

        # Impute numerical features (for TabNet - reuse RF means for consistency)
        for col in tabnet_numerical_features:
             if train_df_tabnet[col].isnull().any():
                mean_val = rf_numerical_means.get(col, train_df_tabnet[col].mean())
                train_df_tabnet[col] = train_df_tabnet[col].fillna(mean_val)
             tabnet_final_feature_columns.append(col)

        # Label Encoding for categorical features (for TabNet)
        for col in tabnet_categorical_features:
            train_df_tabnet[col] = train_df_tabnet[col].astype(str).fillna('nan_category')

            le = LabelEncoder()
            train_df_tabnet[col] = le.fit_transform(train_df_tabnet[col])
            tabnet_label_encoders[col] = le

            tabnet_cat_dims.append(len(le.classes_))
            tabnet_cat_idxs.append(len(tabnet_final_feature_columns)) 
            tabnet_final_feature_columns.append(col)
        
        # Ensure correct order of feature columns for TabNet
        X_full_tabnet = train_df_tabnet[tabnet_final_feature_columns].values 

        if not tabnet_final_feature_columns:
            print("Warning: No features identified for TabNet training after preprocessing. TabNet will be disabled.")
            can_proceed_tabnet_local = False
            X_full_tabnet = None 


    # --- Time-based Validation Split ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    X_train_val_tabnet, X_val_tabnet = None, None # Initialize for TabNet split results

    if len(unique_terms) < 2:
        print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.")
        if len(train_df_for_split) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full_rf, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
            if can_proceed_tabnet_local and X_full_tabnet is not None:
                X_train_val_tabnet, X_val_tabnet, _, _ = train_test_split(
                    X_full_tabnet, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
        else:
            raise ValueError("Insufficient data to perform any kind of train-validation split.")
    else:
        num_val_terms = max(1, int(len(unique_terms) * 0.2))
        val_terms = unique_terms[-num_val_terms:]
        
        val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
        train_val_indices = train_df_for_split[~train_df_for_split['TERM_CODE'].isin(val_terms)].index

        X_train_val_rf = X_full_rf.loc[train_val_indices]
        y_train_val = y_full.loc[train_val_indices]
        X_val_rf = X_full_rf.loc[val_indices]
        y_val = y_full.loc[val_indices]

        if can_proceed_tabnet_local and X_full_tabnet is not None:
            X_train_val_tabnet = X_full_tabnet[train_val_indices] 
            X_val_tabnet = X_full_tabnet[val_indices]

        if X_train_val_rf.empty or X_val_rf.empty:
            print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.")
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full_rf, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
                if can_proceed_tabnet_local and X_full_tabnet is not None:
                    X_train_val_tabnet, X_val_tabnet, _, _ = train_test_split(
                        X_full_tabnet, y_full, test_size=0.2, random_state=42, stratify=y_full
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

    if can_proceed_tabnet_local and TabNetClassifier is not None and X_full_tabnet is not None:
        print("Training TabNet model...")
        
        # If no categorical features, TabNet can still run but cat_idxs/dims should be empty
        if not tabnet_cat_dims and tabnet_categorical_features: # If there were features but dims empty, likely an issue
            print("Warning: Categorical features identified for TabNet but tabnet_cat_dims is empty. TabNet may not perform optimally.")

        tabnet_model = TabNetClassifier(
            cat_idxs=tabnet_cat_idxs,
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
            can_proceed_tabnet_local = False 
    else:
        print("TabNet is not enabled or its dependencies are not met.")
        can_proceed_tabnet_local = False 

    # --- Ensemble Validation ---
    print("Ensembling predictions on validation set...")
    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) 
    
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

    test_processed_df_rf = test_df_raw.copy()
    test_processed_df_rf['TERM_CODE'] = pd.to_numeric(test_processed_df_rf['TERM_CODE'])

    # --- Preprocess test data for RandomForest ---
    for col in rf_numerical_features:
        if col in test_processed_df_rf.columns:
            if test_processed_df_rf[col].isnull().any():
                mean_val = rf_numerical_means.get(col, 0)
                test_processed_df_rf[col] = test_processed_df_rf[col].fillna(mean_val)
        else:
            test_processed_df_rf[col] = rf_numerical_means.get(col, 0)

    # OHE for RF
    if rf_ohe_categorical_features:
        for col in rf_ohe_categorical_features:
            if col in test_processed_df_rf.columns:
                test_processed_df_rf[col] = test_processed_df_rf[col].astype(str).fillna('nan_category')
            else:
                test_processed_df_rf[col] = 'nan_category' 

        ohe_encoded_test_features = rf_ohe_encoder.transform(test_processed_df_rf[rf_ohe_categorical_features])
        ohe_test_df = pd.DataFrame(ohe_encoded_test_features, columns=rf_ohe_feature_names_out, index=test_processed_df_rf.index)
        test_processed_df_rf = pd.concat([test_processed_df_rf.drop(columns=rf_ohe_categorical_features), ohe_test_df], axis=1)

    # Frequency Encoding for RF
    for col in rf_freq_categorical_features:
        if col in test_processed_df_rf.columns:
            test_processed_df_rf[col] = test_processed_df_rf[col].astype(str).fillna('nan_category')
            test_processed_df_rf[f'{col}_freq'] = test_processed_df_rf[col].map(rf_frequency_maps.get(col, {})).fillna(0) 
            test_processed_df_rf.drop(columns=[col], inplace=True)
        else:
            test_processed_df_rf[f'{col}_freq'] = 0 

    X_test_processed_rf = test_processed_df_rf[rf_final_feature_columns]

    # --- Preprocess test data for TabNet (if enabled) ---
    X_test_processed_tabnet = None
    if can_proceed_tabnet_local:
        test_processed_df_tabnet = test_df_raw.copy()
        test_processed_df_tabnet['TERM_CODE'] = pd.to_numeric(test_processed_df_tabnet['TERM_CODE'])

        for col in tabnet_numerical_features:
            if col in test_processed_df_tabnet.columns:
                if test_processed_df_tabnet[col].isnull().any():
                    mean_val = rf_numerical_means.get(col, 0)
                    test_processed_df_tabnet[col] = test_processed_df_tabnet[col].fillna(mean_val)
            else:
                test_processed_df_tabnet[col] = rf_numerical_means.get(col, 0) 

        for col in tabnet_categorical_features:
            if col in test_processed_df_tabnet.columns:
                test_processed_df_tabnet[col] = test_processed_df_tabnet[col].astype(str).fillna('nan_category')
                le = tabnet_label_encoders.get(col)
                if le:
                    known_classes = set(le.classes_)
                    test_processed_df_tabnet[col] = test_processed_df_tabnet[col].apply(
                        lambda x: le.transform([x])[0] if x in known_classes else (len(le.classes_) - 1 if len(le.classes_) > 0 else 0)
                    )
                else: 
                    test_processed_df_tabnet[col] = 0
            else: 
                test_processed_df_tabnet[col] = 0
        
        X_test_processed_tabnet = test_processed_df_tabnet[tabnet_final_feature_columns].values


    print("Making predictions on test data...")
    rf_y_pred_test = rf_model.predict(X_test_processed_rf)
    
    # Initialize with RF length to ensure consistency, if TabNet is not used or fails.
    tabnet_y_pred_test = np.zeros(len(X_test_processed_rf)) 
    if can_proceed_tabnet_local and X_test_processed_tabnet is not None:
        try:
            tabnet_y_pred_test = tabnet_model.predict(X_test_processed_tabnet)
        except Exception as e:
            print(f"Error during TabNet test prediction: {e}. TabNet predictions will be zeros.")
            
    # --- Ensemble Test Predictions ---
    if can_proceed_tabnet_local:
        ensemble_y_pred_test = (rf_y_pred_test.astype(float) + tabnet_y_pred_test.astype(float)) / 2
    else:
        ensemble_y_pred_test = rf_y_pred_test.astype(float)

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
