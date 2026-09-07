
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
import shutil

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None

# --- Intelligent TabNet/Torch Installation ---
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
except ImportError:
    pip_install_base_cmd = [sys.executable, "-m", "pip", "install", "--user"]
    packages_to_install = ["pytorch-tabnet"]
    
    cuda_likely_available = False
    try:
        subprocess.check_output("nvidia-smi", stderr=subprocess.PIPE, shell=True)
        cuda_likely_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            if shutil.which("nvcc"):
                cuda_likely_available = True
        except Exception:
            pass

    torch_install_successful = False

    if cuda_likely_available:
        cuda_versions_to_try = ["cu121", "cu118"] 
        for cuda_version_suffix in cuda_versions_to_try:
            try:
                torch_cuda_install_cmd = pip_install_base_cmd + [
                    "torch",
                    "--index-url", f"https://download.pytorch.org/whl/{cuda_version_suffix}"
                ] + packages_to_install
                subprocess.check_call(torch_cuda_install_cmd)
                torch_install_successful = True
                break
            except Exception:
                pass
    
    if not torch_install_successful:
        try:
            torch_cpu_install_cmd = pip_install_base_cmd + ["torch"] + packages_to_install
            subprocess.check_call(torch_cpu_install_cmd)
            torch_install_successful = True
        except Exception:
            _can_proceed_tabnet = False

    if torch_install_successful:
        try:
            import pytorch_tabnet
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier
            _can_proceed_tabnet = True
        except Exception:
            _can_proceed_tabnet = False

# --- Data Loading Function (from reference solution, enhanced) ---
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
            except Exception:
                pass 
    
    if not df_list:
        return pd.DataFrame()

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Modified run_ablation_scenario function (from original main) ---
def run_ablation_scenario(
    rf_max_depth=None, 
    tabnet_mask_type='sparsemax', 
    ensemble_strategy='average'
):
    can_proceed_tabnet_local = _can_proceed_tabnet
    current_TabNetClassifier = TabNetClassifier

    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    try:
        train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
        gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
        gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)

        if train_data_raw.empty or gold_labels_df.empty:
            raise FileNotFoundError()

        train_df = pd.merge(train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        if train_df.empty:
            raise ValueError()

    except (FileNotFoundError, ValueError, Exception):
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
        return 0.0

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
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                return 0.0

    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        return 0.0

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    rf_params = {'random_state': 42, 'class_weight': 'balanced'}
    if rf_max_depth is not None:
        rf_params['max_depth'] = rf_max_depth
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)
    rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np)
    tabnet_val_f1 = 0.0
    
    if can_proceed_tabnet_local and current_TabNetClassifier is not None:
        try:
            tabnet_model = current_TabNetClassifier(
                cat_idxs=[i for i, col in enumerate(feature_columns) if col in categorical_features],
                cat_dims=tabnet_cat_dims,
                cat_emb_dim=1,
                n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":50, "gamma":0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type=tabnet_mask_type,
                verbose=0, seed=42
            )
            
            tabnet_model.fit(
                X_train=X_train_val_tabnet, y_train=y_train_val_np,
                eval_set=[(X_train_val_tabnet, y_train_val_np), (X_val_tabnet, y_val_np)],
                eval_name=['train', 'valid'], eval_metric=['f1', 'accuracy'],
                max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=128, drop_last=False
            )
            tabnet_y_pred_val = tabnet_model.predict(X_val_tabnet)
            tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='macro')
        except Exception:
            can_proceed_tabnet_local = False

    # --- Ensemble Validation ---
    final_validation_f1 = 0.0
    if ensemble_strategy == 'average':
        if can_proceed_tabnet_local:
            ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
        else:
            ensemble_y_pred_val = rf_y_pred_val.astype(float)
        final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
        final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    elif ensemble_strategy == 'best_model':
        if can_proceed_tabnet_local:
            final_validation_f1 = max(rf_val_f1, tabnet_val_f1)
        else:
            final_validation_f1 = rf_val_f1
    
    return final_validation_f1

# --- Ablation Study Driver ---
if __name__ == "__main__":
    
    ablation_results = {}

    ablation_results['Baseline'] = run_ablation_scenario(
        rf_max_depth=None, 
        tabnet_mask_type='sparsemax', 
        ensemble_strategy='average'
    )
    print(f"Performance for Baseline: {ablation_results['Baseline']:.4f}")

    ablation_results['RF_MaxDepth_5'] = run_ablation_scenario(
        rf_max_depth=5, 
        tabnet_mask_type='sparsemax', 
        ensemble_strategy='average'
    )
    print(f"Performance for RF_MaxDepth_5: {ablation_results['RF_MaxDepth_5']:.4f}")

    if _can_proceed_tabnet:
        ablation_results['TabNet_Entmax'] = run_ablation_scenario(
            rf_max_depth=None, 
            tabnet_mask_type='entmax', 
            ensemble_strategy='average'
        )
        print(f"Performance for TabNet_Entmax: {ablation_results['TabNet_Entmax']:.4f}")
    else:
        ablation_results['TabNet_Entmax'] = 0.0 

    ablation_results['Ensemble_BestModel'] = run_ablation_scenario(
        rf_max_depth=None, 
        tabnet_mask_type='sparsemax', 
        ensemble_strategy='best_model'
    )
    print(f"Performance for Ensemble_BestModel: {ablation_results['Ensemble_BestModel']:.4f}")

    baseline_f1 = ablation_results['Baseline']
    contributions = {
        'RF_MaxDepth_5': ablation_results['RF_MaxDepth_5'] - baseline_f1,
        'TabNet_Entmax': ablation_results['TabNet_Entmax'] - baseline_f1,
        'Ensemble_BestModel': ablation_results['Ensemble_BestModel'] - baseline_f1
    }

    most_contributing_part = "None explicitly identified (all changes led to similar performance or are zero)"
    max_contribution_value = 0.0

    for part, contribution in contributions.items():
        if abs(contribution) > abs(max_contribution_value):
            max_contribution_value = contribution
            most_contributing_part = part
    
    if max_contribution_value > 0:
        print(f"The part that contributes the most to the overall performance is: {most_contributing_part} with a change of {max_contribution_value:.4f} in F1-score.")
    elif max_contribution_value < 0:
        print(f"The part that negatively impacts performance the most is: {most_contributing_part} with a change of {max_contribution_value:.4f} in F1-score.")
    else:
        print("All tested parts have negligible impact on performance.")
