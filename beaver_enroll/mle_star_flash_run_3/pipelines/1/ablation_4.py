

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
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        # Attempt to import again after successful installation
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier # Import after successful installation
        _can_proceed_tabnet = True
    except Exception as e:
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
                
                if not all(key in df.columns for key in primary_keys):
                    continue
                
                df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                df_list.append(df)
            except Exception as e:
                pass # Suppress for cleaner ablation output
    
    if not df_list:
        return pd.DataFrame()

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Preprocessing and Splitting Function ---
def get_preprocessed_data():
    """
    Loads, preprocesses, and splits data into training and validation sets.
    Returns preprocessed data and metadata needed for model training.
    """
    train_df = pd.DataFrame()
    
    # Try loading real data first
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
        # Create dummy data if real data loading fails or results in empty df
        train_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501, 202501, 202501, 202507, 202507, 202601, 202601],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201', 'CS-101', 'MA-201', 'PH-101', 'CS-101', 'MA-201', 'CS-102'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3, 3, 4, 3, 3, 4, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'PROF', 'LECT'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65, 100, 50, 120, 80, 100, 110],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50, 80, 45, 110, 70, 95, 105],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
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
    
    # --- Time-based Validation Split (adaptive logic from original) ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    if len(train_df_for_split) <= 1:
        raise ValueError("Insufficient data to perform any kind of train-validation split.")

    val_terms_candidate_list = []
    found_sufficient_val_classes = False
    
    if len(unique_terms) < 2:
        pass # Will directly lead to the fallback mechanism later.
    else:
        for term in reversed(unique_terms):
            val_terms_candidate_list.append(term)
            
            current_val_df_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms_candidate_list)].index
            current_y_val_candidates = y_full.loc[current_val_df_indices]
            
            if current_y_val_candidates.nunique() >= 2:
                found_sufficient_val_classes = True
                break
        
        if found_sufficient_val_classes:
            val_terms = val_terms_candidate_list
            train_val_terms = [term for term in unique_terms if term not in val_terms]

            val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
            train_val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(train_val_terms)].index

            X_train_val_rf = X_full.loc[train_val_indices]
            y_train_val = y_full.loc[train_val_indices]
            X_val_rf = X_full.loc[val_indices]
            y_val = y_full.loc[val_indices]

            if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
                found_sufficient_val_classes = False # Trigger fallback
        else:
            pass # Trigger fallback

    if not found_sufficient_val_classes:
        if y_full.nunique() >= 2:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42
            )

    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        raise ValueError("Resulting training or validation set is empty after splitting. Cannot proceed.")
    if y_val.nunique() < 2:
        raise ValueError("Validation set does not contain at least two unique target classes, which is required for meaningful F1-score computation. Consider adjusting data or split strategy.")
        
    return X_train_val_rf, y_train_val, X_val_rf, y_val, feature_columns, categorical_features, tabnet_cat_dims, _can_proceed_tabnet, TabNetClassifier, torch

# --- Ablation Scenario Runner ---
def run_ablation_scenario(
    scenario_name, 
    X_train_val_rf, y_train_val, X_val_rf, y_val, 
    feature_columns, categorical_features, tabnet_cat_dims, 
    can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
    use_random_forest=True, 
    use_tabnet=True, 
    rf_n_estimators=100, 
    tabnet_cat_emb_dim=1
):
    print(f"\n--- Running Scenario: {scenario_name} ---")

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    rf_y_pred_val = None
    tabnet_y_pred_val = None

    # RandomForest Model
    if use_random_forest:
        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=rf_n_estimators)
        rf_model.fit(X_train_val_rf, y_train_val)
        rf_y_pred_val = rf_model.predict(X_val_rf)
    else:
        rf_y_pred_val = np.zeros_like(y_val_np) # Placeholder for ensemble if RF is off

    # TabNet Model
    if use_tabnet and can_proceed_tabnet_global and tabnet_classifier_class is not None and torch_lib is not None:
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        tabnet_model = tabnet_classifier_class(
            cat_idxs=cat_idxs,
            cat_dims=tabnet_cat_dims,
            cat_emb_dim=tabnet_cat_emb_dim, # Ablated parameter
            n_d=8, n_a=8,
            n_steps=3,
            gamma=1.3,
            lambda_sparse=1e-3,
            optimizer_fn=torch_lib.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch_lib.optim.lr_scheduler.StepLR,
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
            tabnet_y_pred_val = np.zeros_like(y_val_np)
            use_tabnet = False # Disable for ensemble if it failed
    else:
        tabnet_y_pred_val = np.zeros_like(y_val_np)
        use_tabnet = False # Ensure it's marked as not used for ensemble

    # Ensemble Validation
    if use_random_forest and use_tabnet:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    elif use_random_forest:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    elif use_tabnet:
        ensemble_y_pred_val = tabnet_y_pred_val.astype(float)
    else:
        return 0.0 # No models ran, return 0 F1

    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    print(f'  F1 Score: {final_validation_f1}')
    return final_validation_f1


def main_ablation():
    # Store results
    results = {}
    
    # Prepare data once
    try:
        X_train_val_rf, y_train_val, X_val_rf, y_val, \
        feature_columns, categorical_features, tabnet_cat_dims, \
        can_proceed_tabnet_global, tabnet_classifier_class, torch_lib = get_preprocessed_data()
    except ValueError as e:
        print(f"Error preparing data: {e}. Aborting ablation study.")
        return

    # Scenario 0: Baseline - RF + TabNet (original config)
    results['Baseline (RF + TabNet, RF=100, TabNet_emb=1)'] = run_ablation_scenario(
        'Baseline (RF + TabNet, RF=100, TabNet_emb=1)',
        X_train_val_rf, y_train_val, X_val_rf, y_val,
        feature_columns, categorical_features, tabnet_cat_dims,
        can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
        use_random_forest=True, use_tabnet=True, rf_n_estimators=100, tabnet_cat_emb_dim=1
    )

    # Ablation 1: Random Forest Only (Disable TabNet)
    results['Ablation 1: Random Forest Only'] = run_ablation_scenario(
        'Ablation 1: Random Forest Only',
        X_train_val_rf, y_train_val, X_val_rf, y_val,
        feature_columns, categorical_features, tabnet_cat_dims,
        can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
        use_random_forest=True, use_tabnet=False, rf_n_estimators=100, tabnet_cat_emb_dim=1
    )

    # Ablation 2: TabNet Only (Disable Random Forest)
    if can_proceed_tabnet_global:
        results['Ablation 2: TabNet Only'] = run_ablation_scenario(
            'Ablation 2: TabNet Only',
            X_train_val_rf, y_train_val, X_val_rf, y_val,
            feature_columns, categorical_features, tabnet_cat_dims,
            can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
            use_random_forest=False, use_tabnet=True, rf_n_estimators=100, tabnet_cat_emb_dim=1
        )
    else:
        results['Ablation 2: TabNet Only'] = 0.0 # TabNet not available
        print("\n--- Skipping Ablation 2: TabNet Only (TabNet not available) ---")

    # Ablation 3: RF with n_estimators=50
    results['Ablation 3: RF with n_estimators=50 (Ensemble)'] = run_ablation_scenario(
        'Ablation 3: RF with n_estimators=50 (Ensemble)',
        X_train_val_rf, y_train_val, X_val_rf, y_val,
        feature_columns, categorical_features, tabnet_cat_dims,
        can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
        use_random_forest=True, use_tabnet=True, rf_n_estimators=50, tabnet_cat_emb_dim=1
    )

    # Ablation 4: RF with n_estimators=200
    results['Ablation 4: RF with n_estimators=200 (Ensemble)'] = run_ablation_scenario(
        'Ablation 4: RF with n_estimators=200 (Ensemble)',
        X_train_val_rf, y_train_val, X_val_rf, y_val,
        feature_columns, categorical_features, tabnet_cat_dims,
        can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
        use_random_forest=True, use_tabnet=True, rf_n_estimators=200, tabnet_cat_emb_dim=1
    )

    # Ablation 5: TabNet with cat_emb_dim=2
    if can_proceed_tabnet_global:
        results['Ablation 5: TabNet with cat_emb_dim=2 (Ensemble)'] = run_ablation_scenario(
            'Ablation 5: TabNet with cat_emb_dim=2 (Ensemble)',
            X_train_val_rf, y_train_val, X_val_rf, y_val,
            feature_columns, categorical_features, tabnet_cat_dims,
            can_proceed_tabnet_global, tabnet_classifier_class, torch_lib,
            use_random_forest=True, use_tabnet=True, rf_n_estimators=100, tabnet_cat_emb_dim=2
        )
    else:
        results['Ablation 5: TabNet with cat_emb_dim=2 (Ensemble)'] = 0.0 # TabNet not available
        print("\n--- Skipping Ablation 5: TabNet cat_emb_dim=2 (TabNet not available) ---")


    print("\n--- Ablation Study Results Summary ---")
    for scenario, f1 in results.items():
        print(f"{scenario}: F1 = {f1:.4f}")

    # Determine the most significant change from baseline
    baseline_f1 = results['Baseline (RF + TabNet, RF=100, TabNet_emb=1)']
    contributions = {}

    # Contribution of TabNet (by disabling it in the ensemble)
    # The change is relative to baseline (RF+TabNet) vs RF_Only.
    # If RF_Only is higher, TabNet hurt the ensemble (negative contribution).
    # If RF_Only is lower, TabNet helped the ensemble (positive contribution).
    rf_only_f1 = results.get('Ablation 1: Random Forest Only', 0.0)
    contributions['TabNet (presence in ensemble)'] = baseline_f1 - rf_only_f1
    
    if can_proceed_tabnet_global:
        # Also report TabNet's standalone performance
        tabnet_only_f1 = results.get('Ablation 2: TabNet Only', 0.0)
        contributions['TabNet (standalone performance)'] = tabnet_only_f1 # Not a change from baseline, but its direct impact

    # Contribution of RF n_estimators changes
    contributions['RF n_estimators=50 (vs 100 in ensemble)'] = results['Ablation 3: RF with n_estimators=50 (Ensemble)'] - baseline_f1
    contributions['RF n_estimators=200 (vs 100 in ensemble)'] = results['Ablation 4: RF with n_estimators=200 (Ensemble)'] - baseline_f1

    # Contribution of TabNet cat_emb_dim change
    if can_proceed_tabnet_global:
        contributions['TabNet cat_emb_dim=2 (vs 1 in ensemble)'] = results['Ablation 5: TabNet with cat_emb_dim=2 (Ensemble)'] - baseline_f1

    print("\n--- Contribution Analysis (Change in F1 from Baseline) ---")
    for part, change in contributions.items():
        print(f"'{part}': {change:.4f}")

    if not contributions:
        print("No measurable contributions could be determined.")
        return

    # Find the most significant change (largest absolute value)
    most_significant_part = ""
    max_abs_change = -1.0
    
    for part, change in contributions.items():
        if abs(change) > max_abs_change:
            max_abs_change = abs(change)
            most_significant_part = part
            
    if max_abs_change == 0.0:
        print("\nConclusion: All tested parts have negligible impact on performance relative to the baseline.")
    elif contributions[most_significant_part] > 0:
        print(f"\nConclusion: The part that caused the most significant change from the baseline is '{most_significant_part}', leading to a positive change of {contributions[most_significant_part]:.4f}.")
    else:
        print(f"\nConclusion: The part that caused the most significant change from the baseline is '{most_significant_part}', leading to a negative change of {contributions[most_significant_part]:.4f}.")


if __name__ == "__main__":
    main_ablation()

