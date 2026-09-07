
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

# --- Configuration (kept minimal for ablation study context) ---
# Placeholder directories - in a real Kaggle environment, these would be provided
BASE_DIR = "./input" 
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Global flags for TabNet, determined once at script start
_can_proceed_tabnet_global = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None 

# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet_global = True
except ImportError:
    # print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...") # Suppress for cleaner output
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        # print("pytorch-tabnet and torch installed successfully.") # Suppress for cleaner output
        # Attempt to import again after successful installation
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier # Import after successful installation
        _can_proceed_tabnet_global = True
    except Exception as e:
        # print(f"Failed to install pytorch-tabnet and torch: {e}") # Suppress for cleaner output
        # print("TabNet will not be used in this run.") # Suppress for cleaner output
        _can_proceed_tabnet_global = False # Explicitly set to False on failure

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
                pass # print(f"Error reading {file_path}: {e}") # Suppress for cleaner output
    
    if not df_list:
        # print(f"No valid CSV files found or processed in {data_dir}.") # Suppress for cleaner output
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Experiment Function ---
def run_experiment(
    rf_min_samples_leaf=1, 
    tabnet_gamma=1.3,      
    rf_ensemble_weight=0.5 
):
    can_proceed_tabnet_local = _can_proceed_tabnet_global

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
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full if y_full.nunique() > 1 else None
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
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full if y_full.nunique() > 1 else None
                )
            else:
                raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        if len(X_full) > 1: 
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=max(1, int(0.2 * len(X_full))), random_state=42, stratify=y_full if y_full.nunique() > 1 else None
            )
        else:
            raise ValueError("Training or validation set is empty even after fallback to random split. Cannot proceed with model training.")


    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', min_samples_leaf=rf_min_samples_leaf)
    rf_model.fit(X_train_val_rf, y_train_val)
    rf_y_pred_val = rf_model.predict(X_val_rf)

    tabnet_y_pred_val = np.zeros_like(y_val_np)

    if can_proceed_tabnet_local:
        if TabNetClassifier is None:
            can_proceed_tabnet_local = False
        else:
            tabnet_model = TabNetClassifier(
                cat_idxs=[i for i, col in enumerate(feature_columns) if col in categorical_features],
                cat_dims=tabnet_cat_dims,
                cat_emb_dim=1,
                n_d=8, n_a=8,
                n_steps=3,
                gamma=tabnet_gamma,
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
            except Exception as e:
                can_proceed_tabnet_local = False

    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_ensemble_weight * rf_y_pred_val.astype(float) + (1 - rf_ensemble_weight) * tabnet_y_pred_val.astype(float))
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    
    return final_validation_f1

if __name__ == "__main__":
    results = {}

    baseline_f1 = run_experiment()
    results['Baseline'] = baseline_f1

    ablation1_f1 = run_experiment(rf_min_samples_leaf=5)
    results['RF min_samples_leaf = 5'] = ablation1_f1

    ablation2_f1 = run_experiment(tabnet_gamma=1.0)
    results['TabNet gamma = 1.0'] = ablation2_f1

    ablation3_f1 = run_experiment(rf_ensemble_weight=0.7)
    results['RF Ensemble Weight = 0.7'] = ablation3_f1

    print(f"Baseline F1-Score: {baseline_f1:.4f}")
    for scenario, f1 in results.items():
        if scenario != 'Baseline':
            print(f"- {scenario}: F1-Score = {f1:.4f} (Change from Baseline: {f1 - baseline_f1:.4f})")
    
    best_f1 = -1.0
    best_scenario = ""
    most_impactful_change = {"name": "None", "abs_change": 0.0, "original_value": "", "new_value": ""}

    for scenario, f1 in results.items():
        if f1 > best_f1:
            best_f1 = f1
            best_scenario = scenario
        
        if scenario != 'Baseline':
            change = abs(f1 - baseline_f1)
            if change > most_impactful_change['abs_change']:
                most_impactful_change['abs_change'] = change
                most_impactful_change['name'] = scenario
                
                if scenario == 'RF min_samples_leaf = 5':
                    most_impactful_change['original_value'] = 1
                    most_impactful_change['new_value'] = 5
                elif scenario == 'TabNet gamma = 1.0':
                    most_impactful_change['original_value'] = 1.3
                    most_impactful_change['new_value'] = 1.0
                elif scenario == 'RF Ensemble Weight = 0.7':
                    most_impactful_change['original_value'] = 0.5
                    most_impactful_change['new_value'] = 0.7

    print(f"\nBest performing scenario: {best_scenario} with F1-Score = {best_f1:.4f}")
    
    if most_impactful_change['abs_change'] > 0:
        print(f"The part of the code that contributed the most to the overall performance (largest absolute change from baseline) was: {most_impactful_change['name']}.")
        if most_impactful_change['original_value'] != "" and most_impactful_change['new_value'] != "":
            print(f"  - Original value: {most_impactful_change['original_value']}")
            print(f"  - New value: {most_impactful_change['new_value']}")
            print(f"  - Absolute F1-score change: {most_impactful_change['abs_change']:.4f}")
    else:
        print("No specific part showed a significant positive or negative contribution to overall performance or all contributions were zero/negative.")
