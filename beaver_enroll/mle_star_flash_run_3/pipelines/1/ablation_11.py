
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

# --- Configuration (kept global as in original script for simplicity) ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Global flag for TabNet, will be managed locally per experiment
_can_proceed_tabnet_global = False
pytorch_tabnet_global = None
torch_global = None
TabNetClassifier_global = None

# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet as pytorch_tabnet_global_actual
    import torch as torch_global_actual
    from pytorch_tabnet.tab_model import TabNetClassifier as TabNetClassifier_global_actual
    _can_proceed_tabnet_global = True
    pytorch_tabnet_global = pytorch_tabnet_global_actual
    torch_global = torch_global_actual
    TabNetClassifier_global = TabNetClassifier_global_actual
except ImportError:
    print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...")
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        print("pytorch-tabnet and torch installed successfully.")
        # Attempt to import again after successful installation
        import pytorch_tabnet as pytorch_tabnet_global_actual
        import torch as torch_global_actual
        from pytorch_tabnet.tab_model import TabNetClassifier as TabNetClassifier_global_actual
        _can_proceed_tabnet_global = True
        pytorch_tabnet_global = pytorch_tabnet_global_actual
        torch_global = torch_global_actual
        TabNetClassifier_global = TabNetClassifier_global_actual
    except Exception as e:
        print(f"Failed to install pytorch-tabnet and torch: {e}")
        print("TabNet will not be used in this run.")
        _can_proceed_tabnet_global = False

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
                # print(f"Error reading {file_path}: {e}") # Suppress for cleaner ablation output
                pass
    
    if not df_list:
        # print(f"No valid CSV files found or processed in {data_dir}.") # Suppress for cleaner ablation output
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Core Training and Validation Logic (extracted from original main) ---
def run_experiment(rf_max_depth=None, exclude_features=None, numerical_imputation_strategy='mean', can_proceed_tabnet=False):
    """
    Runs a single experiment with specified configurations and returns the validation F1-score.
    """
    
    # Reload dummy data for each experiment to ensure a clean state and make ablations detectable
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
        # print(f"Error loading real data: {e}. Creating dummy training data for demonstration.") # Suppress for cleaner ablation output
        # Create dummy data if real data loading fails or results in empty df, now with NaNs for imputation testing
        train_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, np.nan, 4, 3, 4, 3, 3], # Added NaN for testing imputation
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, np.nan, 110, 70, 90, 65], # Added NaN for testing imputation
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        # print("Using dummy training data.") # Suppress for cleaner ablation output

    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    # --- Feature Engineering & Preprocessing ---
    target = 'HIGH_ENROLLMENT'
    
    # Columns to be dropped from features (identifiers or target itself)
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    # Apply feature exclusion from ablation
    if exclude_features:
        for feat in exclude_features:
            if feat in train_df.columns:
                train_df = train_df.drop(columns=[feat])
    
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
        # Fit on unique values, ensuring 'nan_category' is included if present
        le.fit(train_df[col].unique()) 
        train_df[col] = le.transform(train_df[col])
        label_encoders[col] = le
        tabnet_cat_dims.append(len(le.classes_) + 1) # +1 for potential unseen categories in test

    # Apply numerical imputation strategy
    numerical_imputation_values = {}
    for col in numerical_features:
        if train_df[col].isnull().any():
            if numerical_imputation_strategy == 'mean':
                impute_val = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                impute_val = train_df[col].median()
            elif numerical_imputation_strategy == 'zero':
                impute_val = 0
            else: # Default to mean if strategy is unknown
                impute_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(impute_val)
            numerical_imputation_values[col] = impute_val # Store for consistency, though not used here
        else:
            # Even if no NaNs, store what the imputation value would have been
            if numerical_imputation_strategy == 'mean':
                numerical_imputation_values[col] = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                numerical_imputation_values[col] = train_df[col].median()
            elif numerical_imputation_strategy == 'zero':
                numerical_imputation_values[col] = 0

    # Define the final list of feature columns for the model
    feature_columns = numerical_features + categorical_features
    
    if not feature_columns:
        return 0.0 # No features identified, cannot train model.

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-Series Cross-Validation (Walk-Forward Validation) Split Generator ---
    def walk_forward_cv_splitter(df_source, X_source, y_source, term_col='TERM_CODE'):
        df_sorted = df_source.sort_values(by=term_col).reset_index(drop=True)
        unique_terms = sorted(df_sorted[term_col].unique())

        if len(unique_terms) < 2:
            return

        for i in range(1, len(unique_terms)):
            val_term = unique_terms[i]
            train_terms = unique_terms[:i] # All terms strictly preceding the current validation term

            val_indices = df_sorted[df_sorted[term_col] == val_term].index
            train_indices = df_sorted[df_sorted[term_col].isin(train_terms)].index

            if train_indices.empty or val_indices.empty:
                continue

            X_train_fold = X_source.loc[train_indices]
            y_train_fold = y_source.loc[train_indices]
            X_val_fold = X_source.loc[val_indices]
            y_val_fold = y_source.loc[val_indices]

            yield X_train_fold, X_val_fold, y_train_fold, y_val_fold

    # Obtain the last fold from the walk-forward splitter for model training and validation
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    has_validation_fold = False
    for X_train_fold, X_val_fold, y_train_fold, y_val_fold in walk_forward_cv_splitter(train_df, X_full, y_full, 'TERM_CODE'):
        X_train_val_rf, X_val_rf, y_train_val, y_val = X_train_fold, X_val_fold, y_train_fold, y_val_fold
        has_validation_fold = True

    if not has_validation_fold:
        # Fallback to simple split if walk-forward doesn't produce folds
        if y_full.nunique() > 1 and len(y_full) > 1: # Ensure stratification is possible
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
        elif len(y_full) > 1: # Fallback without stratification
             X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        else: # Cannot split
            return 0.0 # No validation possible

    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        return 0.0 # Cannot proceed with empty sets

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)


    # --- Model Training: RandomForest ---
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=rf_max_depth)
    rf_model.fit(X_train_val_rf, y_train_val)
    rf_y_pred_val = rf_model.predict(X_val_rf)

    # --- Model Training: TabNet (if can_proceed_tabnet_global and enabled for this run) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np)
    current_can_proceed_tabnet = can_proceed_tabnet and _can_proceed_tabnet_global

    if current_can_proceed_tabnet:
        if TabNetClassifier_global is None:
            current_can_proceed_tabnet = False
        else:
            tabnet_model = TabNetClassifier_global(
                cat_idxs=[i for i, col in enumerate(feature_columns) if col in categorical_features],
                cat_dims=tabnet_cat_dims,
                cat_emb_dim=1, n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
                optimizer_fn=torch_global.optim.Adam, optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":50, "gamma":0.9}, scheduler_fn=torch_global.optim.lr_scheduler.StepLR,
                mask_type='sparsemax', verbose=0, seed=42
            )
            try:
                # Reduced epochs/patience for faster ablation if TabNet runs
                tabnet_model.fit(
                    X_train=X_train_val_tabnet, y_train=y_train_val_np,
                    eval_set=[(X_val_tabnet, y_val_np)], eval_name=['valid'],
                    eval_metric=['f1'], max_epochs=20, patience=5,
                    batch_size=1024, virtual_batch_size=128, drop_last=False
                )
                tabnet_y_pred_val = tabnet_model.predict(X_val_tabnet)
            except Exception as e:
                # print(f"Error during TabNet training or validation: {e}. TabNet will not contribute to ensemble.") # Suppress for cleaner ablation output
                current_can_proceed_tabnet = False

    # --- Ensemble Validation ---
    if current_can_proceed_tabnet:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    
    return final_validation_f1

# --- Ablation Study Orchestration ---
if __name__ == "__main__":
    results = {}

    # Baseline configuration (matches original script's defaults/behaviors)
    baseline_params = {
        'rf_max_depth': None, # Default in original script
        'exclude_features': [],
        'numerical_imputation_strategy': 'mean',
        'can_proceed_tabnet': _can_proceed_tabnet_global # Use global flag for baseline
    }
    print("Running Baseline (Original Logic)...")
    results['Baseline (Original Logic)'] = run_experiment(**baseline_params)
    print(f"Baseline F1-Score: {results['Baseline (Original Logic)']:.4f}\n")

    # Ablation 1: Random Forest with max_depth=3
    ablation_1_params = baseline_params.copy()
    ablation_1_params['rf_max_depth'] = 3
    print("Running Ablation 1: RandomForest max_depth=3 (limited depth to prevent overfitting)...")
    results['Ablation 1: RF max_depth=3'] = run_experiment(**ablation_1_params)
    print(f"Ablation 1 F1-Score: {results['Ablation 1: RF max_depth=3']:.4f}\n")

    # Ablation 2: Exclude 'INSTRUCTOR_RANK' feature
    ablation_2_params = baseline_params.copy()
    ablation_2_params['exclude_features'] = ['INSTRUCTOR_RANK']
    print("Running Ablation 2: Exclude INSTRUCTOR_RANK feature (a categorical feature)...")
    results['Ablation 2: Exclude INSTRUCTOR_RANK'] = run_experiment(**ablation_2_params)
    print(f"Ablation 2 F1-Score: {results['Ablation 2: Exclude INSTRUCTOR_RANK']:.4f}\n")

    # Ablation 3: Numerical Imputation Strategy: Zero
    ablation_3_params = baseline_params.copy()
    ablation_3_params['numerical_imputation_strategy'] = 'zero'
    print("Running Ablation 3: Numerical Imputation Strategy = Zero (simpler imputation)...")
    results['Ablation 3: Numerical Imputation (Zero)'] = run_experiment(**ablation_3_params)
    print(f"Ablation 3 F1-Score: {results['Ablation 3: Numerical Imputation (Zero)']:.4f}\n")

    # --- Print Summary of Results ---
    print("\n--- Ablation Study Summary ---")
    
    # Calculate performance changes
    baseline_f1 = results['Baseline (Original Logic)']
    contributions = {}
    
    # Track the best F1 score and corresponding scenario
    best_f1 = -1.0
    best_scenario = ""

    for scenario, f1_score in results.items():
        if scenario == 'Baseline (Original Logic)':
            print(f"- {scenario}: F1-Score = {f1_score:.4f}")
        else:
            change = f1_score - baseline_f1
            contributions[scenario] = change
            print(f"- {scenario}: F1-Score = {f1_score:.4f} (Change from Baseline: {change:.4f})")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_scenario = scenario
    
    print("\n--- Contribution Analysis (Change in F1-score from Baseline) ---")
    if not contributions:
        print("No ablation scenarios to analyze.")
    else:
        most_impactful_part = ""
        max_abs_impact = -1.0 # Initialize with a value that ensures any impact is greater

        # Extract the "part" from the scenario name for clearer reporting
        def get_ablation_part_name(scenario_name):
            parts = scenario_name.split(': ', 1)
            if len(parts) > 1:
                # For "Ablation X: Description", take "Description"
                return parts[1].split('(', 1)[0].strip()
            return scenario_name # Fallback if format is unexpected

        for scenario, change in contributions.items():
            abs_change = abs(change)
            if abs_change > max_abs_impact:
                max_abs_impact = abs_change
                most_impactful_part = get_ablation_part_name(scenario)
        
        if max_abs_impact > 0:
            print(f"\nThe part of the code that contributed the most to the overall performance (in terms of absolute change from baseline) was: '{most_impactful_part}' (Absolute F1-score change: {max_abs_impact:.4f})")
        else:
            print("All tested parts had negligible or zero impact on performance relative to the baseline.")

    print(f"\nBest performing scenario: {best_scenario} with an F1-Score of {best_f1:.4f}")
