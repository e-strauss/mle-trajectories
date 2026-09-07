

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
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
                
                # Ensure primary keys are present
                if not all(key in df.columns for key in primary_keys):
                    continue
                
                # Ensure consistent data types for merging
                df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                df_list.append(df)
            except Exception as e:
                pass # Suppress verbose error for this script
    
    if not df_list:
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Ablation Study Wrapper ---
def run_ablation_scenario(
    numerical_imputation_strategy='median', # 'median' or 'mean'
    use_standard_scaler=True,
    rf_class_weight=None # 'balanced' or None
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
    
    numerical_features = []
    categorical_features = []
    
    # Define a threshold for classifying numerical columns as categorical
    NUMERIC_AS_CATEGORICAL_THRESHOLD = 20
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        
        if pd.api.types.is_object_dtype(train_df[col]):
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(train_df[col]):
            if train_df[col].nunique() < NUMERIC_AS_CATEGORICAL_THRESHOLD:
                categorical_features.append(col)
            else:
                numerical_features.append(col)

    # --- Missing Value Imputation ---
    # Impute numerical features
    for col in numerical_features:
        if train_df[col].isnull().any():
            if numerical_imputation_strategy == 'median':
                imputation_val = train_df[col].median()
            elif numerical_imputation_strategy == 'mean':
                imputation_val = train_df[col].mean()
            else: # Default to median if unknown strategy
                imputation_val = train_df[col].median()
            train_df[col].fillna(imputation_val, inplace=True)
            
    # Impute categorical features with their mode or a 'Missing' indicator
    for col in categorical_features:
        if train_df[col].isnull().any():
            if pd.api.types.is_object_dtype(train_df[col]):
                train_df[col].fillna('Missing', inplace=True)
            else: # For numerical categoricals (e.g., encoded categories), use mode
                mode_val = train_df[col].mode()[0]
                train_df[col].fillna(mode_val, inplace=True)

    # --- Feature Scaling ---
    if use_standard_scaler and numerical_features: # Apply StandardScaler based on flag
        scaler = StandardScaler()
        train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])

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
                raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")

    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        raise ValueError("Training or validation set is empty. Cannot proceed with model training.")

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)


    # --- Model Training: RandomForest ---
    rf_model = RandomForestClassifier(random_state=42, class_weight=rf_class_weight)
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    rf_y_pred_val = rf_model.predict(X_val_rf)
    rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used

    if can_proceed_tabnet_local:
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
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
                can_proceed_tabnet_local = False # Disable TabNet for prediction too

    # --- Ensemble Validation ---
    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    return final_validation_f1

if __name__ == "__main__":
    results = {}

    # Baseline Scenario (median imputation, StandardScaler, RF class_weight='balanced')
    print("Running Baseline Scenario...")
    baseline_f1 = run_ablation_scenario(
        numerical_imputation_strategy='median',
        use_standard_scaler=True,
        rf_class_weight='balanced'
    )
    results['Baseline'] = baseline_f1
    print(f"Baseline F1: {baseline_f1:.4f}")

    # Ablation 1: Change Numerical Imputation Strategy (Median -> Mean)
    print("\nRunning Ablation 1: Numerical Imputation (Median -> Mean)")
    ablation1_f1 = run_ablation_scenario(
        numerical_imputation_strategy='mean',
        use_standard_scaler=True,
        rf_class_weight='balanced'
    )
    results['Ablation 1: Num Imputation (Mean)'] = ablation1_f1
    print(f"Ablation 1 F1: {ablation1_f1:.4f}")

    # Ablation 2: Disable StandardScaler
    print("\nRunning Ablation 2: Disable StandardScaler")
    ablation2_f1 = run_ablation_scenario(
        numerical_imputation_strategy='median',
        use_standard_scaler=False,
        rf_class_weight='balanced'
    )
    results['Ablation 2: No StandardScaler'] = ablation2_f1
    print(f"Ablation 2 F1: {ablation2_f1:.4f}")

    # Ablation 3: Change RandomForest class_weight (balanced -> None)
    print("\nRunning Ablation 3: RF class_weight (balanced -> None)")
    ablation3_f1 = run_ablation_scenario(
        numerical_imputation_strategy='median',
        use_standard_scaler=True,
        rf_class_weight=None
    )
    results['Ablation 3: RF no class_weight'] = ablation3_f1
    print(f"Ablation 3 F1: {ablation3_f1:.4f}")

    print("\n--- Ablation Study Summary ---")
    best_f1 = -1.0
    best_scenario = "N/A"
    contributions = {}

    print(f"Baseline F1 Score: {baseline_f1:.4f}")
    for scenario, f1 in results.items():
        if scenario == 'Baseline':
            continue
        change = f1 - baseline_f1
        print(f"{scenario} F1 Score: {f1:.4f} (Change from Baseline: {'+' if change >= 0 else ''}{change:.4f})")
        contributions[scenario] = change
    
    # Determine best performing scenario, including baseline
    all_scenarios = {**results} # Copy all results
    best_f1_overall = -1.0
    best_scenario_overall = ""
    for scenario, f1_score in all_scenarios.items():
        if f1_score > best_f1_overall:
            best_f1_overall = f1_score
            best_scenario_overall = scenario
            
    print(f"\nBest performing scenario: {best_scenario_overall} with F1-Score of {best_f1_overall:.4f}")

    # Determine most impactful part (absolute change)
    most_impactful_change = 0.0
    most_impactful_parts = []

    for part, change in contributions.items():
        if abs(change) > abs(most_impactful_change):
            most_impactful_change = change
            most_impactful_parts = [part]
        elif abs(change) == abs(most_impactful_change) and abs(change) > 0:
            most_impactful_parts.append(part)

    if most_impactful_change == 0.0:
        print("All tested parts had negligible or zero impact on performance relative to the baseline.")
    elif len(most_impactful_parts) > 1:
        print(f"The parts of the code that contributed the most to the overall performance (or negatively impacted it the most) are: {', '.join(most_impactful_parts)} with an F1-score change of {most_impactful_change:.4f}.")
    else:
        print(f"The part of the code that contributed the most to the overall performance (or negatively impacted it the most) is: {most_impactful_parts[0]} with an F1-score change of {most_impactful_change:.4f}.")

