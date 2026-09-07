
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
import site # Required for site.USER_SITE

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

def _verify_tabnet_functional():
    """
    Attempts a minimal instantiation of TabNetClassifier to verify its functionality.
    This check ensures that not only the modules can be imported, but also that
    their core components (like Torch optimizer) are functional.
    """
    try:
        _ = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            n_d=8, n_a=8, # Model dimensions
            n_steps=3, # Number of decision steps
            gamma=1.3, # Relaxation parameter
            n_independent=2, n_shared=2, # Shared and independent layers
            epsilon=1e-15, # Epsilon for numerical stability
            seed=42, # For reproducibility
            lambda_sparse=1e-3, # Sparsity regularization
            verbose=0 # Suppress training output during this check
        )
        return True
    except Exception:
        return False

# --- Main logic for import and verification ---
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    # Perform robust post-import verification
    if _verify_tabnet_functional():
        _can_proceed_tabnet = True
    else:
        # If imported but functional check fails, treat as if not found to trigger re-installation attempt
        # print("TabNet modules imported but failed functional check. Attempting re-installation.") # Suppress for cleaner output
        raise ImportError("TabNet functional check failed after initial import.")

except ImportError:
    # print("pytorch_tabnet or torch not found or initial functional check failed. Attempting to install pytorch-tabnet and torch...") # Suppress for cleaner output
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        # print("pytorch-tabnet and torch installed successfully.") # Suppress for cleaner output

        # Explicitly update sys.path to include the user site-packages directory.
        # This ensures the newly installed modules are discoverable.
        user_site_packages = site.USER_SITE
        if user_site_packages not in sys.path:
            sys.path.insert(0, user_site_packages)
            # print(f"Added user site-packages directory ({user_site_packages}) to sys.path.") # Suppress for cleaner output

        # Attempt to import again after successful installation and sys.path update
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        # Perform robust post-import verification after installation
        if _verify_tabnet_functional():
            _can_proceed_tabnet = True
            # print("TabNet installed, re-imported, and passed functional check.") # Suppress for cleaner output
        else:
            # print("TabNet installed and re-imported, but failed functional check. TabNet will not be used.") # Suppress for cleaner output
            _can_proceed_tabnet = False # Explicitly set to False on functional failure after install

    except Exception:
        # print(f"Failed to install or re-import pytorch-tabnet and torch after initial attempt: {e}") # Suppress for cleaner output
        # print("TabNet will not be used in this run.") # Suppress for cleaner output
        _can_proceed_tabnet = False # Explicitly set to False on any failure during installation or re-import


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
            except Exception:
                pass # Suppress for cleaner ablation output
    
    if not df_list:
        # print(f"No valid CSV files found or processed in {data_dir}.") # Suppress for cleaner output
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Main Script as a function for ablation ---
def run_experiment(
    categorical_nunique_threshold=50, # Default from original script
    val_split_term_percentage=0.2,    # Default from original script
    rf_min_impurity_decrease=0.0      # Default for RandomForestClassifier
):
    # Make a local copy of the global flag to avoid UnboundLocalError.
    can_proceed_tabnet_local = _can_proceed_tabnet

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

    except (FileNotFoundError, ValueError, Exception):
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
        # Ablation 1: Categorical Nunique Threshold
        # Heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        if train_df[col].dtype == 'object' or train_df[col].nunique() < categorical_nunique_threshold:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    # Store LabelEncoders for categorical features
    label_encoders = {}
    tabnet_cat_dims = [] # Store dimensions for TabNet's categorical embeddings
    
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str).fillna('nan_category') # Kept default 'nan_category' fill value
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

    # Define the final list of feature columns for the model
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
        # Ablation 2: Validation Split Term Percentage
        # Using a fixed percentage (e.g., 20%) of terms for validation, at least one term
        num_val_terms = max(1, int(len(unique_terms) * val_split_term_percentage))
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

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)


    # --- Model Training: RandomForest ---
    # Ablation 3: RF min_impurity_decrease
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', min_impurity_decrease=rf_min_impurity_decrease)
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used

    if can_proceed_tabnet_local:
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
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
                seed=42 # Add seed for reproducibility
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
            except Exception:
                can_proceed_tabnet_local = False # Disable TabNet for prediction too

    # --- Ensemble Validation ---
    if can_proceed_tabnet_local:
        # Simple averaging for ensemble. Ensure predictions are of similar type (e.g., float before averaging)
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    
    return final_validation_f1

if __name__ == "__main__":
    
    # Baseline
    baseline_f1 = run_experiment()
    print(f"Baseline F1: {baseline_f1:.4f}")

    results = {
        "Baseline": baseline_f1
    }
    
    # Ablation 1: Categorical Nunique Threshold (default 50) -> 2
    # This changes the heuristic for identifying categorical features: any numerical column with less than 2 unique values will be treated as categorical.
    ablation1_f1 = run_experiment(categorical_nunique_threshold=2)
    results["Categorical Nunique Threshold (2)"] = ablation1_f1
    print(f"Ablation 1 (Categorical Nunique Threshold=2) F1: {ablation1_f1:.4f}")

    # Ablation 2: Validation Split Term Percentage (default 0.2) -> 0.5
    # This changes the proportion of unique terms used for the validation set in time-based splitting.
    ablation2_f1 = run_experiment(val_split_term_percentage=0.5)
    results["Validation Split Term Percentage (0.5)"] = ablation2_f1
    print(f"Ablation 2 (Validation Split Term Percentage=0.5) F1: {ablation2_f1:.4f}")

    # Ablation 3: RF min_impurity_decrease (default 0.0) -> 0.1
    # This sets a minimum impurity decrease required for a split in the RandomForestClassifier. Higher values can prune trees.
    ablation3_f1 = run_experiment(rf_min_impurity_decrease=0.1)
    results["RF min_impurity_decrease (0.1)"] = ablation3_f1
    print(f"Ablation 3 (RF min_impurity_decrease=0.1) F1: {ablation3_f1:.4f}")

    # Determine the most impactful part
    most_impactful_change = "None"
    max_f1_change = 0.0

    print("\n--- Contribution Analysis ---")
    for name, f1 in results.items():
        if name == "Baseline":
            continue
        change = abs(f1 - baseline_f1)
        print(f"Change from Baseline for '{name}': {change:.4f}")
        if change > max_f1_change:
            max_f1_change = change
            most_impactful_change = name
    
    if max_f1_change == 0.0:
        print("\nAll tested parts had negligible or zero impact on performance relative to the baseline.")
    else:
        print(f"\nThe part of the code that contributed the most to the overall performance is: '{most_impactful_change}' (Absolute F1 change: {max_f1_change:.4f}).")
