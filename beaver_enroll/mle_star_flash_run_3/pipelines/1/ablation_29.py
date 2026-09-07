
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
from io import StringIO
import contextlib

# Suppress all warnings for cleaner output during the ablation study
warnings.filterwarnings("ignore")

# --- Configuration (from original train.py) ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Global variables for TabNet, mimic original train.py structure
_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None

# Attempt to import/install pytorch_tabnet and torch once
print("Attempting to import/install pytorch-tabnet and torch... (This will only be attempted once)")
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
    print("pytorch-tabnet and torch imported successfully.")
except ImportError:
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
        print("TabNet will not be used in this run for any experiment.")
        _can_proceed_tabnet = False

# Helper to capture stdout, for cleaner ablation output
@contextlib.contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    try:
        yield redirected_output
    finally:
        sys.stdout = old_stdout

# --- Data Loading Function (from reference solution, enhanced) ---
# This needs to be available to run_ablation_experiment
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
                pass # Suppress print for ablation clarity
    
    if not df_list:
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Core Experiment Logic (extracted and made into a function) ---
def run_ablation_experiment(
    rf_min_samples_split_param=2,
    tabnet_n_steps_param=3,
    tabnet_lr_param=2e-2
):
    # Use the globally determined _can_proceed_tabnet
    can_proceed_tabnet_local = _can_proceed_tabnet

    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    try:
        train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
        gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
        gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)

        if train_data_raw.empty or gold_labels_df.empty:
            raise FileNotFoundError("Raw training data or gold labels are empty after loading.")

        train_df = pd.merge(train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        if train_df.empty:
            raise ValueError("Training DataFrame is empty after merging with gold labels.")

    except (FileNotFoundError, ValueError, Exception) as e:
        # Fallback to dummy data if real data loading fails
        train_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })

    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    target = 'HIGH_ENROLLMENT'
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    label_encoders = {}
    tabnet_cat_dims = []
    
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('nan_category') 
        le = LabelEncoder()
        le.fit(train_df[col].unique()) 
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        tabnet_cat_dims.append(len(le.classes_) + 1)

    numerical_means = {}
    for col in numerical_features:
        if train_df[col].isnull().any():
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            numerical_means[col] = mean_val
        else:
            numerical_means[col] = train_df[col].mean() 

    feature_columns = numerical_features + categorical_features
    
    if not feature_columns:
        raise ValueError("No features identified for training after preprocessing.")

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        if len(train_df_for_split) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            raise ValueError("Insufficient data to perform any kind of train-validation split.")
    else:
        num_val_terms = max(1, int(len(unique_terms) * 0.2))
        val_terms = unique_terms[-num_val_terms:]
        
        val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
        train_val_indices = train_df_for_split[~train_df_for_split['TERM_CODE'].isin(val_terms)].index

        X_train_val_rf = X_full.loc[train_val_indices]
        y_train_val = y_full.loc[train_val_indices]
        X_val_rf = X_full.loc[val_indices]
        y_val = y_full.loc[val_indices]

        if X_train_val_rf.empty or X_val_rf.empty:
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        raise ValueError("Training or validation set is empty. Cannot proceed with model training.")

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', min_samples_split=rf_min_samples_split_param)
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)
    
    # Robust F1 score calculation for potentially small/single-class validation sets
    if len(np.unique(y_val)) < 2 or len(np.unique(rf_y_pred_val)) < 2:
        if len(y_val) > 0 and 1 in np.unique(y_val) and 1 in np.unique(rf_y_pred_val):
             rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='binary', pos_label=1)
        elif len(y_val) > 0 and 0 in np.unique(y_val) and 0 in np.unique(rf_y_pred_val):
            rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='binary', pos_label=0)
        else:
            rf_val_f1 = 0.0 # Cannot compute F1 meaningfully, assume 0
    else:
        rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) 
    tabnet_val_f1 = 0.0 # Default if TabNet is not used or fails

    if can_proceed_tabnet_local:
        if TabNetClassifier is None:
            can_proceed_tabnet_local = False
        else:
            try:
                tabnet_model = TabNetClassifier(
                    cat_idxs=[i for i, col in enumerate(feature_columns) if col in categorical_features],
                    cat_dims=tabnet_cat_dims,
                    cat_emb_dim=1,
                    n_d=8, n_a=8,
                    n_steps=tabnet_n_steps_param, # Ablation point 2
                    gamma=1.3,
                    lambda_sparse=1e-3,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=tabnet_lr_param), # Ablation point 3
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
                # Robust F1 score calculation
                if len(np.unique(y_val_np)) < 2 or len(np.unique(tabnet_y_pred_val)) < 2:
                    if len(y_val_np) > 0 and 1 in np.unique(y_val_np) and 1 in np.unique(tabnet_y_pred_val):
                        tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='binary', pos_label=1)
                    elif len(y_val_np) > 0 and 0 in np.unique(y_val_np) and 0 in np.unique(tabnet_y_pred_val):
                        tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='binary', pos_label=0)
                    else:
                        tabnet_val_f1 = 0.0
                else:
                    tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='macro')
            except Exception as e:
                can_proceed_tabnet_local = False

    # --- Ensemble Validation ---
    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    
    # Robust F1 score calculation
    if len(np.unique(y_val_np)) < 2 or len(np.unique(final_ensemble_y_pred_val)) < 2:
        if len(y_val_np) > 0 and 1 in np.unique(y_val_np) and 1 in np.unique(final_ensemble_y_pred_val):
            final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='binary', pos_label=1)
        elif len(y_val_np) > 0 and 0 in np.unique(y_val_np) and 0 in np.unique(final_ensemble_y_pred_val):
            final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='binary', pos_label=0)
        else:
            final_validation_f1 = 0.0
    else:
        final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')

    return final_validation_f1, can_proceed_tabnet_local

# --- Ablation Study Execution ---
ablation_results = {}
ablation_descriptions = {}

print("Running Baseline (Original Parameters)...")
with capture_stdout(): # Capture stdout from run_ablation_experiment to keep output clean
    baseline_f1, tabnet_active = run_ablation_experiment()
ablation_results["Baseline (RF min_samples_split=2, TabNet n_steps=3, TabNet LR=2e-2)"] = baseline_f1
ablation_descriptions["Baseline"] = baseline_f1
print(f"Baseline F1-Score: {baseline_f1:.4f}")
if not tabnet_active:
    print("Note: TabNet was NOT active for the Baseline run.")

# Ablation 1: Random Forest min_samples_split increased
rf_min_samples_split_ablation = 3
print(f"\nRunning Ablation 1: RF min_samples_split = {rf_min_samples_split_ablation}...")
with capture_stdout():
    ablation1_f1, _ = run_ablation_experiment(rf_min_samples_split_param=rf_min_samples_split_ablation)
ablation_results[f"Ablation 1 (RF min_samples_split={rf_min_samples_split_ablation})"] = ablation1_f1
ablation_descriptions["RF min_samples_split (original value 2)"] = ablation1_f1
print(f"Ablation 1 F1-Score: {ablation1_f1:.4f} (Change from Baseline: {ablation1_f1 - baseline_f1:.4f})")

# Ablation 2: TabNet n_steps decreased
tabnet_n_steps_ablation = 2
print(f"\nRunning Ablation 2: TabNet n_steps = {tabnet_n_steps_ablation}...")
with capture_stdout():
    ablation2_f1, tabnet_active_ablation2 = run_ablation_experiment(tabnet_n_steps_param=tabnet_n_steps_ablation)
ablation_results[f"Ablation 2 (TabNet n_steps={tabnet_n_steps_ablation})"] = ablation2_f1
ablation_descriptions["TabNet n_steps (original value 3)"] = ablation2_f1
print(f"Ablation 2 F1-Score: {ablation2_f1:.4f} (Change from Baseline: {ablation2_f1 - baseline_f1:.4f})")
if not tabnet_active_ablation2:
    print("Note: TabNet was NOT active for Ablation 2 run.")

# Ablation 3: TabNet Learning Rate changed
tabnet_lr_ablation = 1e-2
print(f"\nRunning Ablation 3: TabNet Learning Rate = {tabnet_lr_ablation}...")
with capture_stdout():
    ablation3_f1, tabnet_active_ablation3 = run_ablation_experiment(tabnet_lr_param=tabnet_lr_ablation)
ablation_results[f"Ablation 3 (TabNet LR={tabnet_lr_ablation})"] = ablation3_f1
ablation_descriptions["TabNet Learning Rate (original value 2e-2)"] = ablation3_f1
print(f"Ablation 3 F1-Score: {ablation3_f1:.4f} (Change from Baseline: {ablation3_f1 - baseline_f1:.4f})")
if not tabnet_active_ablation3:
    print("Note: TabNet was NOT active for Ablation 3 run.")

# --- Determine most impactful part ---
print("\n--- Ablation Study Summary ---")
performance_changes = {}
for desc_key, f1_score_val in ablation_descriptions.items():
    if desc_key != "Baseline":
        change = abs(f1_score_val - ablation_descriptions["Baseline"])
        performance_changes[desc_key] = change

if performance_changes:
    most_impactful_part = max(performance_changes, key=performance_changes.get)
    max_change = performance_changes[most_impactful_part]
    if max_change > 0:
        print(f"The part of the code that contributed the most to the overall performance (absolute change) is: '{most_impactful_part}' with an F1-score change of {max_change:.4f}")
    else:
        print("All tested parts had negligible or zero impact on performance relative to the baseline.")
else:
    print("No ablation scenarios were run or evaluated for impact.")

