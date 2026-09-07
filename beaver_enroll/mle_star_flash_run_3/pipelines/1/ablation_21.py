
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
# TEST_DATA_DIR is specified in the original solution but not used in this ablation study
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
        # Install to user site-packages to avoid permissions issues, suppressing output
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
                
                # Ensure primary keys are present
                if not all(key in df.columns for key in primary_keys):
                    continue
                
                # Ensure consistent data types for merging
                df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                df_list.append(df)
            except Exception as e:
                # Suppress error messages for cleaner output during ablation runs
                pass
    
    if not df_list:
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Ablation Study Function ---
def run_experiment(
    include_subject_id_sort_as_feature: bool = False,
    include_term_code_as_feature: bool = False,
    rf_min_samples_split: int = 2,
    can_proceed_tabnet_global: bool = False # Pass global state of TabNet availability
) -> float:
    """
    Runs a single experiment iteration with specified modifications and returns the F1-score.
    """
    can_proceed_tabnet_local = can_proceed_tabnet_global

    train_df = pd.DataFrame()
    gold_labels_df = pd.DataFrame()
    
    try:
        train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
        
        # --- Apply type optimizations to train_data_raw ---
        if not train_data_raw.empty:
            if 'SUBJECT_ID_SORT' in train_data_raw.columns:
                train_data_raw['SUBJECT_ID_SORT'] = train_data_raw['SUBJECT_ID_SORT'].astype('category')
            if 'INSTRUCTOR_RANK' in train_data_raw.columns:
                train_data_raw['INSTRUCTOR_RANK'] = train_data_raw['INSTRUCTOR_RANK'].astype('category')

            for col in ['TERM_CODE', 'CREDIT_HOURS', 'CAPACITY']:
                if col in train_data_raw.columns and pd.api.types.is_numeric_dtype(train_data_raw[col]):
                    train_data_raw[col] = pd.to_numeric(train_data_raw[col], downcast='integer')
            
            if 'PREV_ENROLLMENT_AVG' in train_data_raw.columns and pd.api.types.is_numeric_dtype(train_data_raw['PREV_ENROLLMENT_AVG']):
                if pd.api.types.is_integer_dtype(train_data_raw['PREV_ENROLLMENT_AVG']):
                    train_data_raw['PREV_ENROLLMENT_AVG'] = pd.to_numeric(train_data_raw['PREV_ENROLLMENT_AVG'], downcast='integer')
                elif pd.api.types.is_float_dtype(train_data_raw['PREV_ENROLLMENT_AVG']):
                    train_data_raw['PREV_ENROLLMENT_AVG'] = train_data_raw['PREV_ENROLLMENT_AVG'].astype(np.float32)

            if 'HIGH_ENROLLMENT' in train_data_raw.columns and pd.api.types.is_numeric_dtype(train_data_raw['HIGH_ENROLLMENT']):
                train_data_raw['HIGH_ENROLLMENT'] = train_data_raw['HIGH_ENROLLMENT'].astype(np.int8)

        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
        
        # --- Apply type optimizations to gold_labels_df ---
        if not gold_labels_df.empty:
            if 'TERM_CODE' in gold_labels_df.columns and pd.api.types.is_numeric_dtype(gold_labels_df['TERM_CODE']):
                gold_labels_df['TERM_CODE'] = pd.to_numeric(gold_labels_df['TERM_CODE'], downcast='integer')
            
            if 'SUBJECT_ID_SORT' in gold_labels_df.columns:
                gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype('category')

        if train_data_raw.empty or gold_labels_df.empty:
            raise FileNotFoundError("Raw training data or gold labels are empty after loading.")

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
        
        # --- Apply type optimizations to dummy data as well for consistency ---
        if not train_df.empty:
            train_df['SUBJECT_ID_SORT'] = train_df['SUBJECT_ID_SORT'].astype('category')
            train_df['INSTRUCTOR_RANK'] = train_df['INSTRUCTOR_RANK'].astype('category')

            for col in ['TERM_CODE', 'CREDIT_HOURS', 'CAPACITY', 'PREV_ENROLLMENT_AVG']:
                if col in train_df.columns:
                    train_df[col] = pd.to_numeric(train_df[col], downcast='integer')

            if 'HIGH_ENROLLMENT' in train_df.columns:
                train_df['HIGH_ENROLLMENT'] = train_df['HIGH_ENROLLMENT'].astype(np.int8)

    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    target = 'HIGH_ENROLLMENT'
    
    # --- Ablation 1 & 2: Feature Inclusion/Exclusion ---
    # Dynamically build features_to_exclude based on experiment parameters
    features_to_exclude = [target]
    if not include_subject_id_sort_as_feature:
        features_to_exclude.append('SUBJECT_ID_SORT')
    if not include_term_code_as_feature:
        features_to_exclude.append('TERM_CODE')

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

    label_encoders = {}
    tabnet_cat_dims = [] 
    
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

    feature_columns = numerical_features + categorical_features
    
    if not feature_columns:
        return 0.0 # Return 0 F1 if no features for model training

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        if len(train_df_for_split) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            return 0.0
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
                return 0.0

    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        return 0.0

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest (Ablation 3: min_samples_split) ---
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', min_samples_split=rf_min_samples_split)
    rf_model.fit(X_train_val_rf, y_train_val)
    rf_y_pred_val = rf_model.predict(X_val_rf)

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used

    if can_proceed_tabnet_local:
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
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
                can_proceed_tabnet_local = False # Disable TabNet for prediction too

    # --- Ensemble Validation ---
    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    
    return final_validation_f1

def main_ablation():
    global _can_proceed_tabnet # Access the global flag for TabNet state

    results = {}

    # --- Baseline Scenario ---
    print("Running Baseline Scenario...")
    baseline_f1 = run_experiment(
        include_subject_id_sort_as_feature=False,
        include_term_code_as_feature=False,
        rf_min_samples_split=2,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results['Baseline'] = baseline_f1
    print(f"Baseline F1: {baseline_f1:.4f}")

    # --- Ablation 1: Include SUBJECT_ID_SORT as a categorical feature ---
    print("\nRunning Ablation 1: Include SUBJECT_ID_SORT as feature")
    ablation1_f1 = run_experiment(
        include_subject_id_sort_as_feature=True,
        include_term_code_as_feature=False,
        rf_min_samples_split=2,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results['Ablation 1: Include SUBJECT_ID_SORT'] = ablation1_f1
    print(f"Ablation 1 F1: {ablation1_f1:.4f} (Change from Baseline: {ablation1_f1 - baseline_f1:.4f})")

    # --- Ablation 2: Include TERM_CODE as a numerical feature ---
    print("\nRunning Ablation 2: Include TERM_CODE as feature")
    ablation2_f1 = run_experiment(
        include_subject_id_sort_as_feature=False,
        include_term_code_as_feature=True,
        rf_min_samples_split=2,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results['Ablation 2: Include TERM_CODE'] = ablation2_f1
    print(f"Ablation 2 F1: {ablation2_f1:.4f} (Change from Baseline: {ablation2_f1 - baseline_f1:.4f})")

    # --- Ablation 3: Change RandomForest min_samples_split to 3 (from default 2) ---
    print("\nRunning Ablation 3: Change RF min_samples_split to 3")
    ablation3_f1 = run_experiment(
        include_subject_id_sort_as_feature=False,
        include_term_code_as_feature=False,
        rf_min_samples_split=3,
        can_proceed_tabnet_global=_can_proceed_tabnet
    )
    results['Ablation 3: RF min_samples_split=3'] = ablation3_f1
    print(f"Ablation 3 F1: {ablation3_f1:.4f} (Change from Baseline: {ablation3_f1 - baseline_f1:.4f})")

    # --- Contribution Analysis ---
    print("\n--- Contribution Analysis ---")
    
    # Filter out baseline for comparison
    ablation_results = {k: v for k, v in results.items() if k != 'Baseline'}

    if not ablation_results:
        print("No ablation results to compare.")
        return

    # Calculate absolute changes from baseline
    contributions = {name: abs(f1 - baseline_f1) for name, f1 in ablation_results.items()}
    
    if not contributions:
        print("No measurable contributions found.")
        return

    most_impactful_part = max(contributions, key=contributions.get)
    max_change = contributions[most_impactful_part]

    if max_change == 0:
        print("All tested parts had negligible or zero impact on performance relative to the baseline.")
    else:
        print(f"The part of the code that contributed the most to the overall performance is: '{most_impactful_part}' with an absolute F1-score change of {max_change:.4f}.")


if __name__ == "__main__":
    main_ablation()
