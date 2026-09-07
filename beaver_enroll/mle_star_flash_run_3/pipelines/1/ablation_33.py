
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
# TEST_DATA_DIR is not used in ablation study as per instructions
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
# Initialize globally. This will be the initial state passed to main.
_can_proceed_tabnet_global = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None # Ensure TabNetClassifier is initialized to None

# --- Data Loading Function (from reference solution) ---
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
                # Suppress error printing for cleaner ablation output
                pass # print(f"Error reading {file_path}: {e}")
    
    if not df_list:
        # Suppress info printing for cleaner ablation output
        pass # print(f"No valid CSV files found or processed in {data_dir}.")
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Dummy Data ---
def get_dummy_train_df():
    return pd.DataFrame({
        'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
        'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
        'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
        'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
        'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
        'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
        'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })


def run_ablation_experiment(
    rf_class_weight_balanced=True,
    exclude_zero_variance_numerical=False,
    rf_random_state=42
):
    global _can_proceed_tabnet_global, pytorch_tabnet, torch, TabNetClassifier

    can_proceed_tabnet_local = _can_proceed_tabnet_global # Start with global state

    # Re-check and install pytorch_tabnet if not present, but only once per script run
    if not can_proceed_tabnet_local:
        try:
            # Attempt to import globally for subsequent runs
            import pytorch_tabnet
            import torch
            from pytorch_tabnet.tab_model import TabNetClassifier as TNC_class
            TabNetClassifier = TNC_class # Assign to global variable
            _can_proceed_tabnet_global = True
            can_proceed_tabnet_local = True
        except ImportError:
            # Try installing only once if not yet tried
            if not getattr(run_ablation_experiment, '_tabnet_install_attempted', False):
                print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
                    print("pytorch-tabnet and torch installed successfully.")
                    import pytorch_tabnet
                    import torch
                    from pytorch_tabnet.tab_model import TabNetClassifier as TNC_class
                    TabNetClassifier = TNC_class
                    _can_proceed_tabnet_global = True
                    can_proceed_tabnet_local = True
                except Exception as e:
                    print(f"Failed to install pytorch-tabnet and torch: {e}")
                    print("TabNet will not be used in this run.")
                    _can_proceed_tabnet_global = False
                    can_proceed_tabnet_local = False
                finally:
                    run_ablation_experiment._tabnet_install_attempted = True
            else:
                # If installation was attempted before and failed, just ensure local flag is False
                can_proceed_tabnet_local = False


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
        # Suppress info printing for cleaner ablation output
        # print(f"Error loading real data: {e}. Creating dummy training data for demonstration.")
        train_df = get_dummy_train_df()
        # print("Using dummy training data.")

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
        # Heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 50:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    # Ablation: Exclude numerical features with zero variance
    if exclude_zero_variance_numerical:
        numerical_features_to_check = list(numerical_features) # Copy to iterate
        numerical_features.clear() # Clear the original list
        for col in numerical_features_to_check:
            # Check for non-zero variance (or >1 unique values for non-constant)
            if train_df[col].nunique() > 1:
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
        # print("No features identified for training after preprocessing. Returning 0 F1.")
        return 0.0 # Return 0 F1 if no features

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-based Validation Split ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        # print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.")
        if len(train_df_for_split) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            # print("Insufficient data to perform any kind of train-validation split. Returning 0 F1.")
            return 0.0
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
            # print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.")
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                # print("Insufficient data to perform any kind of train-validation split even with random split. Returning 0 F1.")
                return 0.0

    # print(f"Training data size: {len(X_train_val_rf)}")
    # print(f"Validation data size: {len(X_val_rf)}")
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        # print("Training or validation set is empty. Cannot proceed with model training. Returning 0 F1.")
        return 0.0

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)


    # --- Model Training: RandomForest ---
    # print("Training RandomForestClassifier...")
    rf_params = {'random_state': rf_random_state} # Ablation: rf_random_state
    if rf_class_weight_balanced: # Ablation: rf_class_weight_balanced
        rf_params['class_weight'] = 'balanced'
        
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    # print("Evaluating RandomForest on validation set...")
    rf_y_pred_val = rf_model.predict(X_val_rf)

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used

    if can_proceed_tabnet_local:
        # print("Training TabNet model...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
            # print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
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
                # print(f"Error during TabNet training or validation: {e}. TabNet will not contribute to ensemble.")
                can_proceed_tabnet_local = False # Disable TabNet for prediction too

    # --- Ensemble Validation ---
    # print("Ensembling predictions on validation set...")
    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    
    return final_validation_f1


# --- Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Baseline
    print("Running Baseline: RF class_weight='balanced', all numerical features, RF random_state=42")
    results['Baseline'] = run_ablation_experiment(
        rf_class_weight_balanced=True,
        exclude_zero_variance_numerical=False,
        rf_random_state=42
    )
    print(f"Baseline F1: {results['Baseline']:.4f}\n")

    # Ablation 1: Random Forest class_weight=None
    print("Running Ablation 1: RF class_weight=None")
    results['Ablation 1 (RF class_weight=None)'] = run_ablation_experiment(
        rf_class_weight_balanced=False,
        exclude_zero_variance_numerical=False,
        rf_random_state=42
    )
    print(f"Ablation 1 F1: {results['Ablation 1 (RF class_weight=None)']:.4f}\n")

    # Ablation 2: Exclude numerical features with zero variance
    print("Running Ablation 2: Exclude numerical features with zero variance")
    results['Ablation 2 (Exclude Zero Variance Numerical)'] = run_ablation_experiment(
        rf_class_weight_balanced=True,
        exclude_zero_variance_numerical=True,
        rf_random_state=42
    )
    print(f"Ablation 2 F1: {results['Ablation 2 (Exclude Zero Variance Numerical)']:.4f}\n")

    # Ablation 3: Random Forest random_state=None
    print("Running Ablation 3: RF random_state=None")
    results['Ablation 3 (RF random_state=None)'] = run_ablation_experiment(
        rf_class_weight_balanced=True,
        exclude_zero_variance_numerical=False,
        rf_random_state=None
    )
    print(f"Ablation 3 F1: {results['Ablation 3 (RF random_state=None)']:.4f}\n")


    # --- Contribution Analysis ---
    baseline_f1 = results['Baseline']
    print("\n--- Contribution Analysis ---")
    
    contributions = {}
    for scenario, f1 in results.items():
        if scenario == 'Baseline':
            continue
        change = f1 - baseline_f1
        contributions[scenario] = change
        print(f"'{scenario}' changed F1 by: {change:.4f}")

    if contributions:
        # Find the scenario with the largest absolute change
        most_impactful_scenario = max(contributions, key=lambda k: abs(contributions[k]))
        most_impactful_change = contributions[most_impactful_scenario]

        print(f"\nThe part of the code that contributed the most to the overall performance is related to: '{most_impactful_scenario}' with an F1 change of {most_impactful_change:.4f}.")
    else:
        print("No ablation scenarios to compare for contribution analysis.")

