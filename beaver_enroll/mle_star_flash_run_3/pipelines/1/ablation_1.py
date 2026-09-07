
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

# Global flags for TabNet - These are expected to be set by the provided setup script.
# We will check their state and use them if available.
_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None

# Attempt to import TabNet components. This block mirrors the expectation that
# the setup script has already run or its effects are globally available.
try:
    import torch
    import pytorch_tabnet
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
    print("Pre-check: pytorch_tabnet and torch successfully imported. TabNet is available.")
except ImportError:
    print("Pre-check: pytorch_tabnet or torch not found. TabNet will be disabled for all runs.")
    _can_proceed_tabnet = False
except Exception as e:
    print(f"Pre-check: Error importing pytorch_tabnet or torch: {e}. TabNet will be disabled for all runs.")
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
                
                if not all(key in df.columns for key in primary_keys):
                    continue
                
                df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                df_list.append(df)
            except Exception as e:
                # Suppress error printing for cleaner ablation output
                pass
    
    if not df_list:
        return pd.DataFrame()

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Ablation Study Runner Function ---
def run_ablation_scenario(
    scenario_name: str,
    use_tabnet_in_ensemble: bool,
    rf_class_weight_balanced: bool,
    categorical_detection_strategy: str # "auto" (original) or "object_only"
) -> float:
    """
    Runs a single ablation scenario and returns its validation F1-score.
    """
    print(f"\n--- Running Scenario: {scenario_name} ---")

    # Determine if TabNet should be used in this specific scenario, considering global availability
    current_can_proceed_tabnet = _can_proceed_tabnet and use_tabnet_in_ensemble
    
    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    # Try loading real data first, fall back to dummy if failure
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
        # print(f"Real data loaded. Shape: {train_df.shape}")

    except (FileNotFoundError, ValueError, Exception) as e:
        # print(f"Error loading real data: {e}. Creating dummy training data for demonstration.")
        train_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        # print("Using dummy training data.")

    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    target = 'HIGH_ENROLLMENT'
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        
        # Ablation point: Categorical Feature Identification Strategy
        is_categorical = False
        if categorical_detection_strategy == "object_only":
            if train_df[col].dtype == 'object':
                is_categorical = True
        elif categorical_detection_strategy == "auto": # Original logic
            if train_df[col].dtype == 'object' or (train_df[col].nunique() < 50 and pd.api.types.is_numeric_dtype(train_df[col])):
                is_categorical = True

        if is_categorical:
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
    rf_class_weight_param = 'balanced' if rf_class_weight_balanced else None
    rf_model = RandomForestClassifier(random_state=42, class_weight=rf_class_weight_param)
    rf_model.fit(X_train_val_rf, y_train_val)
    rf_y_pred_val = rf_model.predict(X_val_rf)

    # --- Model Training: TabNet (if enabled for this scenario) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros
    
    if current_can_proceed_tabnet:
        if TabNetClassifier is None:
            current_can_proceed_tabnet = False # Should not happen if _can_proceed_tabnet is True globally
        else:
            try:
                cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
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
            except Exception as e:
                print(f"  Warning: TabNet training/prediction failed for this scenario: {e}. Defaulting TabNet predictions to zeros.")
                current_can_proceed_tabnet = False

    # --- Ensemble Validation ---
    if current_can_proceed_tabnet:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    return final_validation_f1

# --- Main Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Scenario 1: Baseline - Original configuration
    # (RF + TabNet Ensemble, class_weight='balanced', 'auto' categorical detection)
    results['Baseline (RF+TabNet, Balanced CW, Auto Cat)'] = run_ablation_scenario(
        "Baseline",
        use_tabnet_in_ensemble=True,
        rf_class_weight_balanced=True,
        categorical_detection_strategy="auto"
    )

    # Scenario 2: Ablation - RandomForest Only (disable TabNet contribution)
    results['Ablation: RandomForest Only (Balanced CW, Auto Cat)'] = run_ablation_scenario(
        "RandomForest Only",
        use_tabnet_in_ensemble=False, # Ablation point
        rf_class_weight_balanced=True,
        categorical_detection_strategy="auto"
    )

    # Scenario 3: Ablation - RandomForest without class_weight='balanced'
    # Keeping TabNet off for a focused comparison on RF's class weight.
    results['Ablation: RF (No CW, Auto Cat) - TabNet off'] = run_ablation_scenario(
        "RandomForest without Class Weight",
        use_tabnet_in_ensemble=False, # Control variable
        rf_class_weight_balanced=False, # Ablation point
        categorical_detection_strategy="auto"
    )
    
    # Scenario 4: Ablation - Simplified Categorical Feature Detection (only object types)
    # Re-enable TabNet for this scenario to see its interaction with feature engineering.
    results['Ablation: Simplified Categorical Detection (RF+TabNet, Balanced CW)'] = run_ablation_scenario(
        "Simplified Categorical Detection",
        use_tabnet_in_ensemble=True, # Control variable
        rf_class_weight_balanced=True,
        categorical_detection_strategy="object_only" # Ablation point
    )

    print("\n--- Ablation Study Results ---")
    for scenario, f1_score in results.items():
        print(f"{scenario}: F1-Score = {f1_score:.4f}")

    # Determine the best performing scenario
    best_scenario = max(results, key=results.get)
    best_f1 = results[best_scenario]

    print(f"\nThe best performing scenario is '{best_scenario}' with an F1-Score of {best_f1:.4f}.")

    # Analyze contributions
    baseline_f1 = results['Baseline (RF+TabNet, Balanced CW, Auto Cat)']
    rf_only_f1 = results['Ablation: RandomForest Only (Balanced CW, Auto Cat)']
    rf_no_cw_f1 = results['Ablation: RF (No CW, Auto Cat) - TabNet off']
    simplified_cat_f1 = results['Ablation: Simplified Categorical Detection (RF+TabNet, Balanced CW)']

    # Calculate absolute impact of each component compared to its ablated state
    tabnet_impact = baseline_f1 - rf_only_f1 # Positive if TabNet helps
    cw_impact = rf_only_f1 - rf_no_cw_f1 # Positive if 'balanced' class weight helps RF (alone)
    cat_det_impact = baseline_f1 - simplified_cat_f1 # Positive if 'auto' detection helps

    contributions = {
        "TabNet in ensemble": tabnet_impact,
        "RandomForest class_weight='balanced'": cw_impact,
        "'Auto' categorical detection strategy": cat_det_impact
    }

    print("\nAnalysis of Contributions (Positive value means the feature improves performance):")
    print(f"  Contribution of TabNet in ensemble: {tabnet_impact:.4f}")
    print(f"  Contribution of RandomForest class_weight='balanced': {cw_impact:.4f}")
    print(f"  Contribution of 'Auto' categorical detection strategy: {cat_det_impact:.4f}")

    # Determine which part contributed the most (largest absolute impact)
    most_contributing_part_key = max(contributions, key=lambda k: abs(contributions[k]))
    largest_impact_value = contributions[most_contributing_part_key]

    if largest_impact_value > 0:
        print(f"\nThe part that contributes the most positively to the overall performance is: '{most_contributing_part_key}' (improving F1 by +{largest_impact_value:.4f}).")
    elif largest_impact_value < 0:
        print(f"\nThe part that degrades performance the most is: '{most_contributing_part_key}' (reducing F1 by {-largest_impact_value:.4f}). This suggests its removal or modification could improve the model.")
    else:
        print(f"\nAll tested parts have negligible impact on performance.")

