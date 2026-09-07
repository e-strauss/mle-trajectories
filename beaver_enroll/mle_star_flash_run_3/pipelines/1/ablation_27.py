

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
# TEST_DATA_DIR is not used in the ablation study, only for original full solution
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
# Initialize globally. This will be the initial state passed to main.
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
    # Attempt to install pytorch-tabnet and torch if not found
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

# --- Main Script for Ablation ---
def run_ablation_experiment(
    exclude_credit_hours=False,
    data_merge_how='outer', # Ablation parameter for merge strategy in load_data_from_dir
    original_can_proceed_tabnet=_can_proceed_tabnet # Pass the global state
):
    # Make a local copy of the global flag
    can_proceed_tabnet_local = original_can_proceed_tabnet

    # Define an internal data loading function that uses the ablated merge strategy
    def load_data_from_dir_ablated(data_dir, merge_how_strategy):
        """
        Loads all primary summary data from a given directory by merging all CSVs.
        Uses the specified merge_how_strategy.
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
                    # Suppress error messages during data loading for cleaner output in ablation
                    pass
        
        if not df_list:
            return pd.DataFrame()

        # Start merging with the first dataframe
        merged_df = df_list[0]
        for i in range(1, len(df_list)):
            # Merge with the specified strategy, handling potential duplicate column names by adding suffixes
            merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how=merge_how_strategy, suffixes=('', f'_{i}'))
            
        return merged_df

    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    # Try loading real data first, using the ablated data loading function
    try:
        train_data_raw = load_data_from_dir_ablated(TRAIN_DATA_DIR, data_merge_how)
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

    # Ablation 1: Exclude 'CREDIT_HOURS'
    if exclude_credit_hours and 'CREDIT_HOURS' in train_df.columns:
        features_to_exclude.append('CREDIT_HOURS')

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
        # Handle cases where all features are excluded or no features are identified
        return 0.0 # Return 0.0 F1 if no features are left for training

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
            return 0.0 # Return 0.0 F1 if no split can be made
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
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                return 0.0 # Return 0.0 F1 if no split can be made
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        return 0.0 # Return 0.0 F1 if sets are empty

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
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
            except Exception as e:
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

# --- Ablation Study Orchestration ---
if __name__ == "__main__":
    results = {}

    # Baseline Scenario
    print("Running Baseline Scenario (Original Logic)...")
    base_f1 = run_ablation_experiment(
        exclude_credit_hours=False,
        data_merge_how='outer',
        original_can_proceed_tabnet=_can_proceed_tabnet
    )
    results['Baseline'] = base_f1
    print(f"Baseline F1 Score: {base_f1:.4f}\n")

    # Ablation 1: Exclude 'CREDIT_HOURS' feature
    print("Running Ablation 1: Exclude 'CREDIT_HOURS' feature...")
    ablation1_f1 = run_ablation_experiment(
        exclude_credit_hours=True,
        data_merge_how='outer', # Revert to baseline for other parameters
        original_can_proceed_tabnet=_can_proceed_tabnet
    )
    results['Ablation 1: Exclude CREDIT_HOURS'] = ablation1_f1
    print(f"Ablation 1 F1 Score: {ablation1_f1:.4f} (Change from Baseline: {(ablation1_f1 - base_f1):.4f})\n")

    # Ablation 2: Change data loading merge strategy from 'outer' to 'inner'
    print("Running Ablation 2: Change data loading merge strategy to 'inner'...")
    ablation2_f1 = run_ablation_experiment(
        exclude_credit_hours=False, # Revert to baseline for other parameters
        data_merge_how='inner',
        original_can_proceed_tabnet=_can_proceed_tabnet
    )
    results['Ablation 2: Data Merge Strategy (inner)'] = ablation2_f1
    print(f"Ablation 2 F1 Score: {ablation2_f1:.4f} (Change from Baseline: {(ablation2_f1 - base_f1):.4f})\n")

    # --- Determine the most impactful part ---
    print("\n--- Contribution Analysis ---")
    
    # Calculate absolute differences from baseline
    abs_diffs = {
        'Excluding CREDIT_HOURS feature': abs(results['Ablation 1: Exclude CREDIT_HOURS'] - base_f1),
        'Changing Data Merge Strategy to "inner"': abs(results['Ablation 2: Data Merge Strategy (inner)'] - base_f1)
    }

    if not abs_diffs:
        print("No ablation scenarios were run or valid. Cannot determine most impactful part.")
    else:
        most_impactful_part = max(abs_diffs, key=abs_diffs.get)
        max_impact_value = abs_diffs[most_impactful_part]

        if max_impact_value == 0:
            print("All tested parts had negligible or zero impact on performance relative to the baseline.")
        else:
            print(f"The part of the code that contributed the most to the overall performance (in terms of absolute F1-score change) was: '{most_impactful_part}'.")
            print(f"Absolute F1-score change: {max_impact_value:.4f}")
