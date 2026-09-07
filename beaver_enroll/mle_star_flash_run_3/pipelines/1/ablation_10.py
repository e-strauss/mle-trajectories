
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import numpy as np
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

# --- Configuration (from original solution) ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None

# Attempt to install and import pytorch_tabnet and torch
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
except ImportError:
    print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...")
    try:
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

# --- Data Loading Function (from original solution) ---
def load_data_from_dir(data_dir):
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
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Ablation Study Function ---
def run_ablation_scenario(
    scenario_name,
    rf_criterion='gini',
    rf_max_features='sqrt',
    apply_numerical_scaling=False,
    enable_tabnet=False # Explicitly disable tabnet for this study due to common installation issues
):
    print(f"\n--- Running Scenario: {scenario_name} ---")

    can_proceed_tabnet_local = _can_proceed_tabnet and enable_tabnet

    # --- Data Loading ---
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
        print(f"Error loading real data: {e}. Creating dummy training data for demonstration.")
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
        train_df[col] = train_df[col].astype(str).fillna('nan_category') 
        le = LabelEncoder()
        le.fit(train_df[col].unique()) 
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        tabnet_cat_dims.append(len(le.classes_) + 1)

    # Impute numerical features
    numerical_means = {}
    for col in numerical_features:
        mean_val = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean_val)
        numerical_means[col] = mean_val
    
    # Apply numerical scaling if enabled
    if apply_numerical_scaling and numerical_features:
        print("Applying StandardScaler to numerical features.")
        scaler = StandardScaler()
        train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])

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
            print("Insufficient data to perform any kind of train-validation split. Returning F1=0.")
            return 0.0
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
                print("Insufficient data to perform any kind of train-validation split even with random split. Returning F1=0.")
                return 0.0
        else:
            print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}")
            print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}")

    print(f"Training data size: {len(X_train_val_rf)}")
    print(f"Validation data size: {len(X_val_rf)}")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        print("Training or validation set is empty. Cannot proceed with model training. Returning F1=0.")
        return 0.0

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    print(f"Training RandomForestClassifier with criterion={rf_criterion}, max_features={rf_max_features}...")
    rf_model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        criterion=rf_criterion, # Ablation parameter 1
        max_features=rf_max_features # Ablation parameter 2
    )
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)
    
    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) 

    if can_proceed_tabnet_local:
        print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            can_proceed_tabnet_local = False
        else:
            try:
                tabnet_model = TabNetClassifier(
                    cat_idxs=cat_idxs,
                    cat_dims=tabnet_cat_dims,
                    cat_emb_dim=1, n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                    scheduler_params={"step_size":50, "gamma":0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
                    mask_type='sparsemax', verbose=0, seed=42
                )
                
                tabnet_model.fit(
                    X_train=X_train_val_tabnet, y_train=y_train_val_np,
                    eval_set=[(X_train_val_tabnet, y_train_val_np), (X_val_tabnet, y_val_np)],
                    eval_name=['train', 'valid'], eval_metric=['f1', 'accuracy'],
                    max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128, drop_last=False
                )
                tabnet_y_pred_val = tabnet_model.predict(X_val_tabnet)
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
    print(f'Validation F1 Score for {scenario_name}: {final_validation_f1}')
    return final_validation_f1

# --- Main Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Scenario 1: Baseline (Original RandomForest parameters and no scaling)
    # rf_criterion default is 'gini', rf_max_features default is 'sqrt'.
    results['Baseline'] = run_ablation_scenario(
        'Baseline',
        rf_criterion='gini',
        rf_max_features='sqrt',
        apply_numerical_scaling=False,
        enable_tabnet=False
    )

    # Scenario 2: Ablation on RandomForest criterion (change to 'entropy')
    results['Ablation: RF Criterion=entropy'] = run_ablation_scenario(
        'Ablation: RF Criterion=entropy',
        rf_criterion='entropy',
        rf_max_features='sqrt',
        apply_numerical_scaling=False,
        enable_tabnet=False
    )

    # Scenario 3: Ablation on RandomForest max_features (change to 'log2')
    results['Ablation: RF Max_Features=log2'] = run_ablation_scenario(
        'Ablation: RF Max_Features=log2',
        rf_criterion='gini',
        rf_max_features='log2',
        apply_numerical_scaling=False,
        enable_tabnet=False
    )

    # Scenario 4: Ablation on Numerical Scaling (apply StandardScaler)
    results['Ablation: Numerical StandardScaler'] = run_ablation_scenario(
        'Ablation: Numerical StandardScaler',
        rf_criterion='gini',
        rf_max_features='sqrt',
        apply_numerical_scaling=True,
        enable_tabnet=False
    )

    # Print all results and determine the most impactful part
    print("\n--- Ablation Study Results Summary ---")
    baseline_f1 = results['Baseline']
    print(f"Baseline F1 Score: {baseline_f1}")

    impacts = {}
    for scenario, f1 in results.items():
        if scenario != 'Baseline':
            f1_diff = f1 - baseline_f1
            impacts[scenario] = f1_diff
            print(f"{scenario}: F1 Score = {f1} (Change from Baseline: {f1_diff:.4f})")

    if impacts:
        most_impactful_scenario = max(impacts, key=lambda k: abs(impacts[k]))
        print(f"\nThe part of the code that contributed the most to the overall performance is: '{most_impactful_scenario}' "
              f"with an F1 score change of {impacts[most_impactful_scenario]:.4f}.")
    else:
        print("No ablation scenarios were run or no impact calculated.")
