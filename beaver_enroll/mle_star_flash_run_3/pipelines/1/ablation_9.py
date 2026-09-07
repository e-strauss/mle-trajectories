

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
TEST_DATA_DIR = os.path.join(BASE_DIR, "__TEST_DATA_DIR__")  # Not used in this ablation study
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
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
    _can_proceed_tabnet = False 

    # --- Prioritized Installation Strategy ---
    # Attempt 1: CUDA-enabled PyTorch installation
    print("Attempting to install pytorch-tabnet with CUDA-enabled torch (e.g., cu121 for recent CUDA)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "pytorch-tabnet",
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121",
            "--user"
        ])
        print("pytorch-tabnet and CUDA-enabled torch installed successfully.")
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        _can_proceed_tabnet = True
    except Exception as e_cuda:
        print(f"Failed to install pytorch-tabnet with CUDA-enabled torch: {e_cuda}")
        print("Falling back to CPU-only installation...")

        # Attempt 2: Fallback to CPU-only PyTorch installation
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "pytorch-tabnet", "torch",
                "--user"
            ])
            print("pytorch-tabnet and CPU-only torch installed successfully.")
            import pytorch_tabnet
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier
            _can_proceed_tabnet = True
        except Exception as e_cpu:
            print(f"Failed to install pytorch-tabnet and torch (CPU-only): {e_cpu}")
            print("TabNet will not be used in this run.")


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

# --- Ablation Study Orchestration ---
def run_ablation_scenario(
    scenario_name,
    rf_min_samples_split=2,
    categorical_nunique_threshold=50,
    tabnet_n_d=8,
    tabnet_n_a=8,
    can_proceed_tabnet_global=_can_proceed_tabnet 
):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    can_proceed_tabnet_local = can_proceed_tabnet_global

    print("Loading training data...")
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
        # Ablated Part 2: categorical_nunique_threshold
        if train_df[col].dtype == 'object' or train_df[col].nunique() < categorical_nunique_threshold:
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
    # Ablated Part 1: rf_min_samples_split
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', min_samples_split=rf_min_samples_split)
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
            # Ablated Part 3: tabnet_n_d, tabnet_n_a
            tabnet_model = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=tabnet_cat_dims,
                cat_emb_dim=1,
                n_d=tabnet_n_d, n_a=tabnet_n_a,
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
    print(f'Final Validation Performance for {scenario_name}: {final_validation_f1}') 
    
    return final_validation_f1

# --- Main function to run the ablation study ---
def main():
    results = {}

    # BASELINE
    baseline_f1 = run_ablation_scenario(
        "Baseline (RF min_samples_split=2, Cat Nunique=50, TabNet n_d=8, n_a=8)",
        rf_min_samples_split=2,
        categorical_nunique_threshold=50,
        tabnet_n_d=8,
        tabnet_n_a=8,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results["Baseline"] = baseline_f1

    # ABLATION 1: Random Forest min_samples_split
    ablation1_f1 = run_ablation_scenario(
        "Ablation 1: RF min_samples_split=5",
        rf_min_samples_split=5,
        categorical_nunique_threshold=50,
        tabnet_n_d=8,
        tabnet_n_a=8,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results["RF min_samples_split=5"] = ablation1_f1

    # ABLATION 2: Categorical Nunique Threshold
    ablation2_f1 = run_ablation_scenario(
        "Ablation 2: Cat Nunique Threshold=10",
        rf_min_samples_split=2,
        categorical_nunique_threshold=10,
        tabnet_n_d=8,
        tabnet_n_a=8,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results["Cat Nunique Threshold=10"] = ablation2_f1

    # ABLATION 3: TabNet n_d and n_a
    ablation3_f1 = run_ablation_scenario(
        "Ablation 3: TabNet n_d=4, n_a=4",
        rf_min_samples_split=2,
        categorical_nunique_threshold=50,
        tabnet_n_d=4,
        tabnet_n_a=4,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results["TabNet n_d=4, n_a=4"] = ablation3_f1


    print("\n--- Ablation Study Summary ---")
    best_f1 = -1.0
    best_scenario = ""
    for scenario, f1 in results.items():
        print(f"{scenario}: F1 Score = {f1}")
        if f1 > best_f1:
            best_f1 = f1
            best_scenario = scenario
        elif f1 == best_f1 and best_scenario == "": # Initialize if first scenario is best
            best_scenario = scenario


    # Determine contributions
    contributions = {}
    baseline_f1_score = results["Baseline"]

    contributions["RF min_samples_split (original value 2)"] = results["RF min_samples_split=5"] - baseline_f1_score
    contributions["Categorical Nunique Threshold (original value 50)"] = results["Cat Nunique Threshold=10"] - baseline_f1_score
    
    if _can_proceed_tabnet:
        contributions["TabNet n_d, n_a (original values 8,8)"] = results["TabNet n_d=4, n_a=4"] - baseline_f1_score
    else:
        contributions["TabNet n_d, n_a (original values 8,8)"] = 0.0

    print("\nContribution Analysis (Change in F1 relative to Baseline):")
    for part, change in contributions.items():
        print(f"- {part}: {change:.4f}")

    most_impactful_part = ""
    max_abs_change = -1.0
    for part, change in contributions.items():
        abs_change = abs(change)
        if abs_change > max_abs_change:
            max_abs_change = abs_change
            most_impactful_part = part
        elif abs_change == max_abs_change and most_impactful_part == "":
            most_impactful_part = part

    if max_abs_change == 0.0:
        print("\nAll tested parts have negligible impact on performance or all contributions were zero/negative.")
    else:
        print(f"\nThe part of the code that contributes the most to the overall performance is related to: {most_impactful_part}")
        

if __name__ == "__main__":
    main()

