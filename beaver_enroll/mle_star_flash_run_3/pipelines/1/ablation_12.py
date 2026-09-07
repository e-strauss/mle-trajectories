
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

# --- Original Data Loading Function (used as default in ablation) ---
def load_data_from_dir_original(data_dir):
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

# Helper function for dummy data generation
def generate_dummy_train_data():
    n_samples = 150
    term_codes = np.random.choice([202301, 202307, 202401, 202407, 202501, 202507, 202601], n_samples)
    subjects = np.random.choice(['CS-101', 'MA-201', 'CS-102', 'PH-101', 'BI-301', 'CH-201', 'EE-305', 'ME-200', 'LA-101', 'AR-203', 'PS-150', 'SO-210'], n_samples)
    credit_hours = np.random.choice([2, 3, 4], n_samples)
    instructor_ranks = np.random.choice(['PROF', 'ASSIST', 'LECT', 'ADJ'], n_samples)
    capacities = np.random.randint(30, 150, n_samples)
    enrollment_ratio = np.random.uniform(0.6, 1.15, n_samples)
    prev_enrollment_avgs = (capacities * enrollment_ratio).astype(int)
    prev_enrollment_avgs = np.maximum(1, prev_enrollment_avgs)
    prev_enrollment_avgs = np.minimum((capacities * 1.2).astype(int), prev_enrollment_avgs)

    train_df = pd.DataFrame({
        'TERM_CODE': term_codes,
        'SUBJECT_ID_SORT': subjects,
        'CREDIT_HOURS': credit_hours,
        'INSTRUCTOR_RANK': instructor_ranks,
        'CAPACITY': capacities,
        'PREV_ENROLLMENT_AVG': prev_enrollment_avgs
    })

    enrollment_fill_rate = train_df['PREV_ENROLLMENT_AVG'] / train_df['CAPACITY']
    high_enrollment_mask = (
        (enrollment_fill_rate > 0.95) & (train_df['INSTRUCTOR_RANK'].isin(['PROF', 'ASSIST']))
    ) | (
        (enrollment_fill_rate > 1.05)
    )
    train_df['HIGH_ENROLLMENT'] = high_enrollment_mask.astype(int)

    noise_percentage = 0.1
    num_flips = int(n_samples * noise_percentage)
    flip_indices = np.random.choice(train_df.index, num_flips, replace=False)
    train_df.loc[flip_indices, 'HIGH_ENROLLMENT'] = 1 - train_df.loc[flip_indices, 'HIGH_ENROLLMENT']
    return train_df

# --- Ablation Study Function ---
def run_ablation_scenario(
    scenario_name,
    categorical_nan_filler='nan_category',
    load_data_merge_how='outer',
    can_proceed_tabnet_global_flag=_can_proceed_tabnet
):
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Local function for data loading, potentially ablated for merge strategy
    def load_data_from_dir_ablated_merge(data_dir):
        all_files = os.listdir(data_dir)
        df_list = []
        primary_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']

        for f in all_files:
            if f.endswith(".csv"):
                file_path = os.path.join(data_dir, f)
                try:
                    df = pd.read_csv(file_path)
                    if not all(key in df.columns for key in primary_keys):
                        continue
                    df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                    df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        if not df_list:
            print(f"No valid CSV files found or processed in {data_dir}.")
            return pd.DataFrame()

        merged_df = df_list[0]
        for i in range(1, len(df_list)):
            merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how=load_data_merge_how, suffixes=('', f'_{i}'))
        return merged_df

    # Use the ablated or original load_data_from_dir based on parameter
    current_load_data_from_dir = load_data_from_dir_ablated_merge if load_data_merge_how != 'outer' else load_data_from_dir_original

    # Make a local copy of the global flag to avoid UnboundLocalError.
    can_proceed_tabnet_local = can_proceed_tabnet_global_flag

    print("Loading training data...")
    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    # Try loading real data first
    try:
        train_data_raw = current_load_data_from_dir(TRAIN_DATA_DIR)
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
        train_df = generate_dummy_train_data()
        print("Using dummy training data.")

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
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    label_encoders = {}
    tabnet_cat_dims = []
    
    for col in categorical_features:
        # Ablation Point 1: Categorical NaN Filler
        train_df[col] = train_df[col].astype(str).fillna(categorical_nan_filler) 
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
    tabnet_y_pred_val = np.zeros_like(y_val_np)

    if can_proceed_tabnet_local:
        print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
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

    return final_validation_f1

# --- Main Execution for Ablation Study ---
if __name__ == "__main__":
    np.random.seed(42) # Ensure dummy data generation and splits are reproducible

    results = {}

    # Baseline Scenario
    baseline_f1 = run_ablation_scenario(
        scenario_name="Baseline (Original Logic)",
        categorical_nan_filler='nan_category',
        load_data_merge_how='outer',
        can_proceed_tabnet_global_flag=_can_proceed_tabnet
    )
    results["Baseline (Original Logic)"] = baseline_f1

    # Ablation 1: Change Categorical NaN Filler from 'nan_category' to 'UNKNOWN_CAT_VALUE'
    ablation1_f1 = run_ablation_scenario(
        scenario_name="Ablation 1: Categorical NaN Filler = 'UNKNOWN_CAT_VALUE'",
        categorical_nan_filler='UNKNOWN_CAT_VALUE',
        load_data_merge_how='outer',
        can_proceed_tabnet_global_flag=_can_proceed_tabnet
    )
    results["Ablation 1: Categorical NaN Filler = 'UNKNOWN_CAT_VALUE'"] = ablation1_f1

    # Ablation 2: Change Data Loading Merge Strategy from 'outer' to 'left'
    ablation2_f1 = run_ablation_scenario(
        scenario_name="Ablation 2: Data Loading Merge Strategy = 'left'",
        categorical_nan_filler='nan_category',
        load_data_merge_how='left',
        can_proceed_tabnet_global_flag=_can_proceed_tabnet
    )
    results["Ablation 2: Data Loading Merge Strategy = 'left'"] = ablation2_f1

    # Print all results
    print("\n--- Ablation Study Results ---")
    for scenario, f1 in results.items():
        print(f"{scenario}: F1 Score = {f1:.4f}")

    # Determine the most impactful part
    most_impactful_part = "No specific part showed a positive contribution to overall performance or all contributions were zero/negative."
    max_diff = 0.0

    contributions = {}
    for scenario, f1 in results.items():
        if scenario != "Baseline (Original Logic)":
            diff = abs(f1 - baseline_f1)
            contributions[scenario] = diff
            
            # Identify the most impactful part based on absolute difference
            if diff > max_diff:
                max_diff = diff
                # Extract a concise description of the ablated part
                part_description = scenario.replace("Ablation ", "").split(':')[0].strip()
                most_impactful_part = f"'{part_description}' (F1 change from baseline: {f1 - baseline_f1:.4f})"

    print("\n--- Contribution Analysis (Absolute Change from Baseline) ---")
    for scenario, diff in contributions.items():
        print(f"{scenario}: Absolute F1 Change = {diff:.4f}")

    print(f"\nThe part of the code that contributed the most to the overall performance is: {most_impactful_part}")
