
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
import copy # To deepcopy configurations

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
# Use fixed paths relative to the script for consistency in ablation.
BASE_DIR = "./input" # Assuming 'input' folder is next to the script
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Global flag and module imports for TabNet (as in original script)
_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None

# Check and install pytorch_tabnet if not present - ONLY ONCE FOR THE SCRIPT
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
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        _can_proceed_tabnet = True
    except Exception as e:
        print(f"Failed to install pytorch-tabnet and torch: {e}")
        print("TabNet will not be used in this run.")
        _can_proceed_tabnet = False

# --- Data Loading Function (from reference solution) ---
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

# Store loaded data globally to avoid re-loading in each ablation run
_cached_train_data_raw = None
_cached_gold_labels_df = None

def load_data_once():
    """Attempts to load real data once and caches it. Returns True if successful, False otherwise."""
    global _cached_train_data_raw, _cached_gold_labels_df
    if _cached_train_data_raw is None or _cached_gold_labels_df is None:
        print("Attempting to load real training data and gold labels...")
        try:
            train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
            gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
            gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
            gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)

            if train_data_raw.empty or gold_labels_df.empty:
                raise FileNotFoundError("Raw training data or gold labels are empty after loading.")
            
            _cached_train_data_raw = train_data_raw
            _cached_gold_labels_df = gold_labels_df
            print("Real data loaded successfully for ablation study.")
            return True

        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"Error loading real data: {e}.")
            return False
    return True # Already loaded successfully

def get_dummy_data():
    """Generates dummy data for fallback."""
    train_df = pd.DataFrame({
        'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
        'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
        'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
        'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT'], # Added 'LECT' to match length 10
        'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
        'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
        'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })
    print("Using dummy training data.")
    return train_df


# --- Ablation Study Function ---
def run_ablation_scenario(config, scenario_name):
    print(f"\n--- Running Ablation Scenario: {scenario_name} ---")

    can_proceed_tabnet_local = _can_proceed_tabnet # Inherit global state for TabNet availability

    # Prepare training data based on whether real data loaded or dummy is used
    if load_data_once():
        train_data_raw = _cached_train_data_raw
        gold_labels_df = _cached_gold_labels_df
        train_df = pd.merge(train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        if train_df.empty:
            print("Merged DataFrame is empty, falling back to dummy data.")
            train_df = get_dummy_data()
    else:
        train_df = get_dummy_data()

    # Ensure TERM_CODE is numeric for sorting
    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    # --- Feature Engineering & Preprocessing ---
    target = 'HIGH_ENROLLMENT'
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        # Heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    label_encoders = {}
    tabnet_cat_dims = [] 
    
    for col in categorical_features:
        # Apply ablation for categorical NaN fill strategy
        train_df[col] = train_df[col].astype(str).fillna(config['cat_nan_fill_value'])
             
        le = LabelEncoder()
        le.fit(train_df[col].unique()) 
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        tabnet_cat_dims.append(len(le.classes_) + 1) # +1 for potential unseen categories in test

    numerical_means = {}
    for col in numerical_features:
        if train_df[col].isnull().any():
            # Apply ablation for numerical imputation strategy
            if config['num_imputation_strategy'] == 'mean':
                imputation_value = train_df[col].mean()
            elif config['num_imputation_strategy'] == 'zero':
                imputation_value = 0
            elif config['num_imputation_strategy'] == 'median':
                imputation_value = train_df[col].median()
            else: # Fallback to mean
                imputation_value = train_df[col].mean()
            
            train_df[col] = train_df[col].fillna(imputation_value)
            numerical_means[col] = imputation_value
        else:
            numerical_means[col] = train_df[col].mean() # Still store for consistency


    feature_columns = numerical_features + categorical_features
    
    if not feature_columns:
        raise ValueError("No features identified for training after preprocessing.")

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-based Validation Split (fixed as per previous study results indicate its importance) ---
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
        num_val_terms = max(1, int(len(unique_terms) * config['validation_split_ratio']))
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

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    print("Training RandomForestClassifier...")
    # Apply ablation for RandomForest hyperparameters
    rf_model = RandomForestClassifier(
        random_state=42,
        class_weight=config['rf_class_weight'],
        n_estimators=config['rf_n_estimators']
    )
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)
    rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')
    print(f"RandomForest Validation F1: {rf_val_f1}")

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np)
    
    # Check if TabNet should be used based on config AND global import success
    if config['use_tabnet'] and can_proceed_tabnet_local:
        print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            config['use_tabnet'] = False # Effectively disable for this run
        else:
            try:
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
                    seed=42
                )
                
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
                config['use_tabnet'] = False # Disable for prediction too

    # --- Ensemble Validation ---
    print("Ensembling predictions on validation set...")
    if config['use_tabnet'] and can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    print(f'Final Validation Performance: {final_validation_f1}')

    return final_validation_f1


if __name__ == "__main__":
    
    results = {}

    # --- Define Ablation Configurations ---

    # Baseline configuration (as per the provided train.py)
    baseline_config = {
        'num_imputation_strategy': 'mean',    # Original: fills with mean
        'rf_n_estimators': 100,               # Original: default is 100
        'rf_class_weight': 'balanced',        # Original: 'balanced' (kept fixed as previously studied)
        'cat_nan_fill_value': 'nan_category', # Original: fills with 'nan_category'
        'use_tabnet': True,                   # Original: attempts to use TabNet
        'validation_split_ratio': 0.2,        # Original: 20% of terms for validation (kept fixed)
    }
    results['Baseline'] = run_ablation_scenario(baseline_config, 'Baseline')

    # Ablation 1: Numerical Imputation Strategy (mean -> zero)
    ablation1_config = copy.deepcopy(baseline_config)
    ablation1_config['num_imputation_strategy'] = 'zero'
    results['Ablation 1: Numerical Imputation (Zero)'] = run_ablation_scenario(ablation1_config, 'Numerical Imputation (Zero)')

    # Ablation 2: RandomForest n_estimators (100 -> 10)
    ablation2_config = copy.deepcopy(baseline_config)
    ablation2_config['rf_n_estimators'] = 10
    results['Ablation 2: RandomForest n_estimators (10)'] = run_ablation_scenario(ablation2_config, 'RandomForest n_estimators (10)')
    
    # Ablation 3: Categorical NaN Handling (nan_category -> unknown_cat)
    ablation3_config = copy.deepcopy(baseline_config)
    ablation3_config['cat_nan_fill_value'] = 'unknown_cat'
    results['Ablation 3: Categorical NaN Fill (unknown_cat)'] = run_ablation_scenario(ablation3_config, 'Categorical NaN Fill (unknown_cat)')

    # --- Summarize Results ---
    print("\n--- Ablation Study Summary ---")
    
    for scenario, f1 in results.items():
        print(f"{scenario}: F1-Score = {f1:.4f}")

    # Determine the most contributing part based on absolute difference from baseline
    baseline_f1 = results['Baseline']
    
    # Store tuples of (absolute_diff, scenario_name, original_feature_name)
    ablation_diffs_info = []

    # Calculate absolute differences for each ablation
    diff_num_imputation = results['Ablation 1: Numerical Imputation (Zero)'] - baseline_f1
    ablation_diffs_info.append((abs(diff_num_imputation), 'Ablation 1: Numerical Imputation (Zero)', 'Numerical Imputation Strategy', diff_num_imputation))

    diff_rf_estimators = results['Ablation 2: RandomForest n_estimators (10)'] - baseline_f1
    ablation_diffs_info.append((abs(diff_rf_estimators), 'Ablation 2: RandomForest n_estimators (10)', 'RandomForest n_estimators', diff_rf_estimators))
    
    diff_cat_nan = results['Ablation 3: Categorical NaN Fill (unknown_cat)'] - baseline_f1
    ablation_diffs_info.append((abs(diff_cat_nan), 'Ablation 3: Categorical NaN Fill (unknown_cat)', 'Categorical NaN Fill Value', diff_cat_nan))

    if ablation_diffs_info:
        # Sort by absolute difference in descending order
        ablation_diffs_info.sort(key=lambda x: x[0], reverse=True)
        
        # The first element is the one with the largest absolute difference
        largest_abs_diff, scenario_causing_diff, feature_name, original_diff_value = ablation_diffs_info[0]
        
        print(f"\nThe part of the code that causes the most significant change in performance is: {feature_name}.")
        print(f"Baseline F1: {baseline_f1:.4f}, F1 with change: {results[scenario_causing_diff]:.4f}, Difference: {original_diff_value:.4f}.")
    else:
        print("No ablation scenarios processed or significant differences found.")
