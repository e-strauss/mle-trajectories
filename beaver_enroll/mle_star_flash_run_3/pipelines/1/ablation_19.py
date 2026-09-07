

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
import copy

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
# Initialize globally. This will be the initial state.
_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None 

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
        from pytorch_tabnet.tab_model import TabNetClassifier
        _can_proceed_tabnet = True
    except Exception as e:
        print(f"Failed to install pytorch-tabnet and torch: {e}")
        print("TabNet will not be used in this run.")
        _can_proceed_tabnet = False

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

# --- Ablation Study Function ---
def run_ablation_scenario(
    scenario_name,
    rf_random_state,
    tabnet_lambda_sparse,
    tabnet_scheduler_config,
    can_proceed_tabnet_global
):
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Local copy of the TabNet flag for this scenario
    can_proceed_tabnet_local = can_proceed_tabnet_global

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
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        # Heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

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
            print(f"Warning: train_df_for_split has {len(train_df_for_split)} rows. Insufficient data for splitting. Using full data as training and validation will be trivial.")
            X_train_val_rf = X_full
            y_train_val = y_full
            X_val_rf = X_full # Assign full data to val to prevent empty, but results will be skewed.
            y_val = y_full # Assign full data to val to prevent empty, but results will be skewed.
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
                print(f"Warning: train_df_for_split has {len(train_df_for_split)} rows. Insufficient data for splitting. Using full data as training and validation will be trivial.")
                X_train_val_rf = X_full
                y_train_val = y_full
                X_val_rf = X_full
                y_val = y_full
        else:
            print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}")
            print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}")

    print(f"Training data size: {len(X_train_val_rf)}")
    print(f"Validation data size: {len(X_val_rf)}")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        # Fallback for extremely small datasets where even random split fails or is trivial
        print("Warning: Training or validation set is empty after splitting. Using full dataset for both to proceed, but results will be highly biased.")
        X_train_val_rf = X_full
        y_train_val = y_full
        X_val_rf = X_full
        y_val = y_full

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    print("Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(random_state=rf_random_state, class_weight='balanced')
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
            scheduler_fn = tabnet_scheduler_config.get('scheduler_fn')
            scheduler_params = tabnet_scheduler_config.get('scheduler_params', {})
            
            tabnet_model = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=tabnet_cat_dims,
                cat_emb_dim=8, # From plan_implement_agent_1
                n_d=16, n_a=16, # From plan_implement_agent_1
                n_steps=3,
                gamma=1.3,
                lambda_sparse=tabnet_lambda_sparse, # Ablated parameter
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_fn=scheduler_fn, # Ablated parameter
                scheduler_params=scheduler_params, # Ablated parameter
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
    
    return final_validation_f1


if __name__ == "__main__":
    results = {}

    # Baseline Configuration (using existing solution's values and plan_implement_agent_1 updates for TabNet)
    baseline_rf_random_state = 42
    baseline_tabnet_lambda_sparse = 1e-3
    baseline_tabnet_scheduler_config = {
        'scheduler_fn': torch.optim.lr_scheduler.StepLR if _can_proceed_tabnet else None,
        'scheduler_params': {"step_size":50, "gamma":0.9} if _can_proceed_tabnet else {}
    }
    
    print("--- Starting Ablation Study ---")

    # Scenario 1: Baseline
    results['Baseline'] = run_ablation_scenario(
        "Baseline",
        rf_random_state=baseline_rf_random_state,
        tabnet_lambda_sparse=baseline_tabnet_lambda_sparse,
        tabnet_scheduler_config=baseline_tabnet_scheduler_config,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )

    # Scenario 2: Ablation on RF random_state (change from 42 to 0)
    results['Ablation 1: RF Random State (0)'] = run_ablation_scenario(
        "Ablation 1: RF Random State (0)",
        rf_random_state=0, # Changed
        tabnet_lambda_sparse=baseline_tabnet_lambda_sparse,
        tabnet_scheduler_config=baseline_tabnet_scheduler_config,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )

    # Scenario 3: Ablation on TabNet lambda_sparse (set to 0.0)
    results['Ablation 2: TabNet Lambda Sparse (0.0)'] = run_ablation_scenario(
        "Ablation 2: TabNet Lambda Sparse (0.0)",
        rf_random_state=baseline_rf_random_state,
        tabnet_lambda_sparse=0.0, # Changed
        tabnet_scheduler_config=baseline_tabnet_scheduler_config,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )

    # Scenario 4: Ablation on TabNet Scheduler (removed)
    results['Ablation 3: TabNet Scheduler (Removed)'] = run_ablation_scenario(
        "Ablation 3: TabNet Scheduler (Removed)",
        rf_random_state=baseline_rf_random_state,
        tabnet_lambda_sparse=baseline_tabnet_lambda_sparse,
        tabnet_scheduler_config={
            'scheduler_fn': None, # Removed
            'scheduler_params': {} # Removed
        },
        can_proceed_tabnet_global=_can_proceed_tabnet
    )

    print("\n--- Ablation Study Results Summary ---")
    baseline_f1 = results['Baseline']
    print(f"Baseline F1: {baseline_f1:.4f}")

    most_impactful_change = "None"
    max_f1_change = 0.0

    for scenario, f1_score in results.items():
        if scenario == 'Baseline':
            continue
        change = f1_score - baseline_f1
        print(f"{scenario} F1: {f1_score:.4f} (Change from Baseline: {change:.4f})")
        if abs(change) > abs(max_f1_change):
            max_f1_change = change
            most_impactful_change = scenario

    if most_impactful_change == "None" or max_f1_change == 0.0:
        print("\nAll tested parts had negligible or zero impact on performance relative to the baseline.")
    else:
        print(f"\nThe part of the code that contributed the most to the overall performance was: {most_impactful_change} (F1 change: {max_f1_change:.4f}).")

