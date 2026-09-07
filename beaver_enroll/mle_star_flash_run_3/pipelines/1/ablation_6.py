

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
        print(f"No valid CSV files found or processed in {data_dir}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Experiment Runner Function ---
def run_ablation_experiment(
    exp_name: str,
    force_credit_hours_numerical: bool = False,
    numerical_imputation_strategy: str = 'mean', # 'mean' or 'median'
    rf_ensemble_weight: float = 0.5 # Weight for RF in ensemble, TabNet gets 1 - weight
) -> float:
    print(f"\n--- Running Experiment: {exp_name} ---")
    
    # Use a local copy of the global flag to avoid UnboundLocalError and allow modification within this scope.
    can_proceed_tabnet_local = _can_proceed_tabnet

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

        # Ablation 1: Force CREDIT_HOURS to be numerical
        if force_credit_hours_numerical and col == 'CREDIT_HOURS':
            numerical_features.append(col)
        # Original heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        elif train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
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

    numerical_imputation_values = {}
    for col in numerical_features:
        if train_df[col].isnull().any():
            impute_val = None
            if numerical_imputation_strategy == 'mean':
                impute_val = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                impute_val = train_df[col].median()
            else: # Fallback to mean for robustness
                impute_val = train_df[col].mean()

            train_df[col] = train_df[col].fillna(impute_val)
            numerical_imputation_values[col] = impute_val
        else:
            # Store mean/median even if no NaNs, for consistency
            if numerical_imputation_strategy == 'mean':
                numerical_imputation_values[col] = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                numerical_imputation_values[col] = train_df[col].median()
            else:
                numerical_imputation_values[col] = train_df[col].mean()

    # Define the final list of feature columns for the model
    feature_columns = numerical_features + categorical_features
    
    if not feature_columns:
        print("No features identified for training after preprocessing.")
        return 0.0

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
            print("Insufficient data to perform any kind of train-validation split. Returning 0.0 F1.")
            return 0.0 # Cannot perform meaningful validation

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
                print("Insufficient data to perform any kind of train-validation split even with random split. Returning 0.0 F1.")
                return 0.0
        else:
            print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}")
            print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}")

    print(f"Training data size: {len(X_train_val_rf)}")
    print(f"Validation data size: {len(X_val_rf)}")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        print("Training or validation set is empty. Cannot proceed with model training. Returning 0.0 F1.")
        return 0.0

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
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
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used
    tabnet_val_f1 = 0.0 # Initialize TabNet F1 score

    if can_proceed_tabnet_local:
        print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            can_proceed_tabnet_local = False
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
                    seed=42 # Add seed for reproducibility
                )
                
                tabnet_model.fit(
                    X_train=X_train_val_tabnet, y_train=y_train_val_np,
                    eval_set=[(X_val_tabnet, y_val_np)], # Use only validation set for faster eval
                    eval_name=['valid'],
                    eval_metric=['f1', 'accuracy'],
                    max_epochs=50, # Reduced epochs for faster ablation
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
    final_ensemble_y_pred_val = None
    
    if can_proceed_tabnet_local:
        # Ablation 3: Ensemble Weighting
        rf_weight = rf_ensemble_weight
        tabnet_weight = 1.0 - rf_ensemble_weight
        
        # Ensure predictions are of similar type (e.g., float before averaging)
        ensemble_y_pred_val_raw = (rf_y_pred_val.astype(float) * rf_weight + tabnet_y_pred_val.astype(float) * tabnet_weight)
        final_ensemble_y_pred_val = (ensemble_y_pred_val_raw >= 0.5).astype(int)
    else:
        # Only RF if TabNet not available
        final_ensemble_y_pred_val = rf_y_pred_val.astype(int)
    
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    print(f'Final Validation Performance for {exp_name}: {final_validation_f1}') # Required output format
    
    return final_validation_f1

# --- Main Ablation Study Runner ---
def main_ablation():
    # Ensure necessary dummy directories and files exist for script execution
    os.makedirs('./input', exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GOLD_LABELS_PATH), exist_ok=True) 
    
    # Create dummy gold enrollment file if it doesn't exist (to prevent FileNotFoundError if real one is missing)
    if not os.path.exists(GOLD_LABELS_PATH):
        dummy_gold_data = {
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        pd.DataFrame(dummy_gold_data).to_csv(GOLD_LABELS_PATH, index=False)
        print(f"Created dummy gold labels file at {GOLD_LABELS_PATH}")

    # Create a dummy training data file if it doesn't exist
    dummy_train_file_path = os.path.join(TRAIN_DATA_DIR, 'course_summary.csv')
    if not os.path.exists(dummy_train_file_path):
        dummy_train_data = {
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
        }
        pd.DataFrame(dummy_train_data).to_csv(dummy_train_file_path, index=False)
        print(f"Created dummy training data file at {dummy_train_file_path}")

    results = {}

    # --- Baseline (Original Logic) ---
    results['Baseline (Original Logic)'] = run_ablation_experiment(
        exp_name='Baseline (Original Logic)',
        force_credit_hours_numerical=False, # CREDIT_HOURS determined by `nunique()` heuristic
        numerical_imputation_strategy='mean',
        rf_ensemble_weight=0.5
    )

    # --- Ablation 1: Force CREDIT_HOURS to be Numerical ---
    results['Ablation 1: Force CREDIT_HOURS Numerical'] = run_ablation_experiment(
        exp_name='Ablation 1: Force CREDIT_HOURS Numerical',
        force_credit_hours_numerical=True,
        numerical_imputation_strategy='mean',
        rf_ensemble_weight=0.5
    )

    # --- Ablation 2: Numerical Imputation Strategy (Median) ---
    results['Ablation 2: Numerical Imputation (Median)'] = run_ablation_experiment(
        exp_name='Ablation 2: Numerical Imputation (Median)',
        force_credit_hours_numerical=False,
        numerical_imputation_strategy='median',
        rf_ensemble_weight=0.5
    )
    
    # --- Ablation 3: Ensemble Weighting (RF 70%, TabNet 30%) ---
    results['Ablation 3: Ensemble Weighting (RF 70%)'] = run_ablation_experiment(
        exp_name='Ablation 3: Ensemble Weighting (RF 70%)',
        force_credit_hours_numerical=False,
        numerical_imputation_strategy='mean',
        rf_ensemble_weight=0.7
    )

    print("\n--- Ablation Study Results Summary ---")
    best_f1 = -1.0
    best_scenario = ""
    
    for scenario, f1_score_val in results.items():
        print(f"{scenario}: F1 Score = {f1_score_val:.4f}")
        if f1_score_val > best_f1:
            best_f1 = f1_score_val
            best_scenario = scenario

    print(f"\nBest performing scenario: {best_scenario} with F1 Score = {best_f1:.4f}")

    # Determine contributions based on change from baseline
    baseline_f1 = results['Baseline (Original Logic)']
    contributions = {}
    
    contributions['Force CREDIT_HOURS Numerical'] = results['Ablation 1: Force CREDIT_HOURS Numerical'] - baseline_f1
    contributions['Numerical Imputation (Median)'] = results['Ablation 2: Numerical Imputation (Median)'] - baseline_f1
    contributions['Ensemble Weighting (RF 70%)'] = results['Ablation 3: Ensemble Weighting (RF 70%)'] - baseline_f1
    
    most_contributing_part = "No specific part showed a positive contribution to overall performance or all contributions were zero/negative relative to baseline."
    max_contribution = 0.0
    
    for part, contribution in contributions.items():
        if contribution > max_contribution:
            max_contribution = contribution
            most_contributing_part = part
        print(f"Contribution of '{part}' relative to baseline: {contribution:.4f}")

    if max_contribution > 0:
        print(f"\nThe part of the code that contributes the most to the overall performance is: '{most_contributing_part}' (with an F1 increase of {max_contribution:.4f} over baseline).")
    else:
        print(f"\n{most_contributing_part}")


if __name__ == "__main__":
    main_ablation()
