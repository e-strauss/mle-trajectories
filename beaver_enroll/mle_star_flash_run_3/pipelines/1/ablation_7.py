

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

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv") # TEST_DATA_DIR not needed for ablation script

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

# --- Data Loading Function (from plan_implement_agent_1, enhanced) ---
def load_data_from_dir(data_dir):
    """
    Loads all primary summary data from a given directory by merging all CSVs.
    Constructs a comprehensive base index from all unique (TERM_CODE, SUBJECT_ID_SORT)
    pairs across all files, then individually left-joins each feature dataframe
    onto this consolidated index.
    """
    all_files = os.listdir(data_dir)
    primary_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']

    df_list_for_merging = [] # Stores actual dataframes to be merged
    all_unique_keys_dfs = [] # Stores only primary keys for creating the base index

    # First pass: Load dataframes, extract keys, ensure types
    for f in all_files:
        if f.endswith(".csv"):
            file_path = os.path.join(data_dir, f)
            try:
                df = pd.read_csv(file_path)
                
                # Ensure primary keys are present
                if not all(key in df.columns for key in primary_keys):
                    print(f"Skipping {file_path}: Missing one or more primary keys {primary_keys}.")
                    continue
                
                # Ensure consistent data types for merging
                df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                
                # Store the full dataframe for later merging
                df_list_for_merging.append(df)
                
                # Collect unique primary keys from this dataframe
                all_unique_keys_dfs.append(df[primary_keys].drop_duplicates())
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not df_list_for_merging:
        print(f"No valid CSV files found or processed in {data_dir}.")
        return pd.DataFrame()

    # Step 1: Create the comprehensive base index
    # Concatenate all unique key combinations and drop duplicates to get the full key space
    # .reset_index(drop=True) ensures a clean index for the base_index_df
    base_index_df = pd.concat(all_unique_keys_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # Step 2: Individually left-join each feature dataframe onto the consolidated index
    # Initialize the merged dataframe with the base index (containing all unique primary keys)
    merged_df = base_index_df.copy()

    # Iterate through the collected dataframes and left-join each onto the growing merged_df
    # Suffixes are added to disambiguate columns that appear in multiple feature files.
    for i, df_to_merge in enumerate(df_list_for_merging):
        merged_df = pd.merge(merged_df, df_to_merge, on=primary_keys, how='left', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Ablation Study Function ---
def run_ablation_scenario(
    categorical_detection_strategy='nunique_50', # 'nunique_50' or 'object_only'
    rf_class_weight_balanced=True,
    numerical_imputation_strategy='mean', # 'mean' or 'median'
    can_proceed_tabnet_flag=False # Flag from global check
):
    print(f"Running scenario with: cat_strategy={categorical_detection_strategy}, rf_class_weight={rf_class_weight_balanced}, num_imputation={numerical_imputation_strategy}")

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
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        
        # Ablation point 1: Categorical feature detection strategy
        if categorical_detection_strategy == 'object_only':
            if train_df[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        elif categorical_detection_strategy == 'nunique_50':
            if train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        else:
            raise ValueError(f"Unknown categorical_detection_strategy: {categorical_detection_strategy}")

    # Store LabelEncoders for categorical features
    label_encoders = {}
    tabnet_cat_dims = [] # Store dimensions for TabNet's categorical embeddings
    
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('nan_category') 
        le = LabelEncoder()
        le.fit(train_df[col].unique()) 
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        tabnet_cat_dims.append(len(le.classes_) + 1) # +1 for potential unseen categories in test

    numerical_fill_values = {} # Store mean/median for numerical imputation
    for col in numerical_features:
        if train_df[col].isnull().any():
            if numerical_imputation_strategy == 'mean':
                fill_val = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                fill_val = train_df[col].median()
            else:
                raise ValueError(f"Unknown numerical_imputation_strategy: {numerical_imputation_strategy}")
            train_df[col] = train_df[col].fillna(fill_val)
            numerical_fill_values[col] = fill_val
        else:
            if numerical_imputation_strategy == 'mean':
                numerical_fill_values[col] = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                numerical_fill_values[col] = train_df[col].median()

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
    rf_params = {'random_state': 42}
    if rf_class_weight_balanced: # Ablation point 2: RF class_weight
        rf_params['class_weight'] = 'balanced'
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)
    rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')
    print(f"RandomForest Validation F1: {rf_val_f1}")

    # --- Model Training: TabNet (if can_proceed_tabnet_flag) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used
    tabnet_successful = False

    if can_proceed_tabnet_flag:
        print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            can_proceed_tabnet_flag = False
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
                tabnet_successful = True
            except Exception as e:
                print(f"Error during TabNet training or validation: {e}. TabNet will not contribute to ensemble.")
                can_proceed_tabnet_flag = False

    # --- Ensemble Validation ---
    print("Ensembling predictions on validation set...")
    if can_proceed_tabnet_flag and tabnet_successful:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available or failed
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    return final_validation_f1


if __name__ == "__main__":
    results = {}

    # Baseline Scenario (Original configuration)
    print("\n--- Running Baseline Scenario (Original Configuration) ---")
    baseline_f1 = run_ablation_scenario(
        can_proceed_tabnet_flag=_can_proceed_tabnet
    )
    results['Baseline'] = baseline_f1
    print(f"Baseline F1-score: {baseline_f1}")

    # Ablation 1: Categorical Feature Detection (Object Dtype Only)
    print("\n--- Running Ablation 1: Categorical Detection (Object Dtype Only) ---")
    ablation1_f1 = run_ablation_scenario(
        categorical_detection_strategy='object_only',
        rf_class_weight_balanced=True,
        numerical_imputation_strategy='mean',
        can_proceed_tabnet_flag=_can_proceed_tabnet
    )
    results['Ablation 1: Categorical Detection (Object Dtype Only)'] = ablation1_f1
    print(f"Ablation 1 F1-score: {ablation1_f1}")
    print(f"Modification effect for Ablation 1: {ablation1_f1 - baseline_f1:.4f}")

    # Ablation 2: RandomForestClassifier without `class_weight='balanced'`
    print("\n--- Running Ablation 2: RF No Class Weight ---")
    ablation2_f1 = run_ablation_scenario(
        categorical_detection_strategy='nunique_50',
        rf_class_weight_balanced=False,
        numerical_imputation_strategy='mean',
        can_proceed_tabnet_flag=_can_proceed_tabnet
    )
    results['Ablation 2: RF No Class Weight'] = ablation2_f1
    print(f"Ablation 2 F1-score: {ablation2_f1}")
    print(f"Modification effect for Ablation 2: {ablation2_f1 - baseline_f1:.4f}")

    # Ablation 3: Numerical Imputation Strategy changed to Median
    print("\n--- Running Ablation 3: Numerical Imputation (Median) ---")
    ablation3_f1 = run_ablation_scenario(
        categorical_detection_strategy='nunique_50',
        rf_class_weight_balanced=True,
        numerical_imputation_strategy='median',
        can_proceed_tabnet_flag=_can_proceed_tabnet
    )
    results['Ablation 3: Numerical Imputation (Median)'] = ablation3_f1
    print(f"Ablation 3 F1-score: {ablation3_f1}")
    print(f"Modification effect for Ablation 3: {ablation3_f1 - baseline_f1:.4f}")

    print("\n--- Ablation Study Summary ---")
    best_f1 = baseline_f1
    best_scenario = 'Baseline'
    
    for scenario, f1_score in results.items():
        if f1_score > best_f1:
            best_f1 = f1_score
            best_scenario = scenario
        print(f"{scenario}: F1-score = {f1_score:.4f} (Change from Baseline: {f1_score - baseline_f1:.4f})")

    if best_scenario == 'Baseline':
        print("\nNo specific modification showed a positive contribution. The Baseline configuration performed the best or equally well.")
    else:
        print(f"\nResult: The '{best_scenario}' part of the code contributes the most to the overall performance, achieving an F1-score of {best_f1:.4f}.")
    if _can_proceed_tabnet == False:
        print("\nNote: TabNet was disabled in all runs due to installation issues. All results are effectively for RandomForest only.")

