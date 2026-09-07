
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
import random
import torch # Import torch at the top if it's generally required or checked for.

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Global Configuration and TabNet Check ---
BASE_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "table_splits/train")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

_can_proceed_tabnet = False
pytorch_tabnet = None
TabNetClassifier = None 

# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
except ImportError:
    # Attempt to install, but keep the global flag for the ablation study
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        import pytorch_tabnet
        from pytorch_tabnet.tab_model import TabNetClassifier
        _can_proceed_tabnet = True
    except Exception as e:
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
                # Suppress printing errors for cleaner ablation study output
                pass 
    
    if not df_list:
        return pd.DataFrame()

    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# Function to generate dummy data for consistent use across ablation runs
def generate_dummy_data(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    num_rows = random.randint(100, 500)

    term_years = np.arange(2023, 2023 + num_rows // 20 + 2)
    term_options = []
    for y in term_years:
        term_options.extend([y * 100 + 1, y * 100 + 7])
    term_codes = random.choices(term_options, k=num_rows)

    subjects_pool = [f'CS-{i:03d}' for i in range(101, 130)] + \
                    [f'MA-{i:03d}' for i in range(201, 225)] + \
                    [f'PH-{i:03d}' for i in range(101, 115)] + \
                    [f'BI-{i:03d}' for i in range(301, 320)] + \
                    [f'CH-{i:03d}' for i in range(201, 210)] + \
                    [f'EE-{i:03d}' for i in range(401, 410)]
    subject_ids = random.choices(subjects_pool, k=num_rows)

    credit_hours_options = [2, 3, 4, 5]
    credit_hours = random.choices(credit_hours_options, weights=[0.1, 0.45, 0.4, 0.05], k=num_rows)

    instructor_ranks_options = ['PROF', 'ASSIST', 'LECT', 'ADJ', 'VISIT']
    instructor_ranks = random.choices(instructor_ranks_options, weights=[0.3, 0.3, 0.2, 0.15, 0.05], k=num_rows)

    capacities = np.random.randint(30, 150, num_rows)
    prev_enrollments = []
    high_enrollments = []

    popular_subjects = ['CS-101', 'MA-201', 'CS-103']

    for i in range(num_rows):
        cap = capacities[i]
        enrollment_factor = np.random.uniform(0.6, 1.05)
        if subject_ids[i] in popular_subjects:
            enrollment_factor = np.random.uniform(0.8, 1.2)
        elif instructor_ranks[i] == 'PROF':
            enrollment_factor = np.random.uniform(0.7, 1.1)
        
        prev_enroll = int(cap * enrollment_factor)
        prev_enrollments.append(prev_enroll)
        high_enrollments.append(1 if (prev_enroll / cap > 0.85) else 0)

    train_df = pd.DataFrame({
        'TERM_CODE': term_codes,
        'SUBJECT_ID_SORT': subject_ids,
        'CREDIT_HOURS': credit_hours,
        'INSTRUCTOR_RANK': instructor_ranks,
        'CAPACITY': capacities,
        'PREV_ENROLLMENT_AVG': prev_enrollments,
        'HIGH_ENROLLMENT': high_enrollments
    })

    columns_for_nan = ['CREDIT_HOURS', 'INSTRUCTOR_RANK', 'PREV_ENROLLMENT_AVG', 'CAPACITY', 'SUBJECT_ID_SORT']
    nan_percentage = 0.05
    for col in columns_for_nan:
        num_nan = int(num_rows * nan_percentage)
        nan_indices = np.random.choice(train_df.index, num_nan, replace=False)
        train_df.loc[nan_indices, col] = np.nan

    outlier_percentage = 0.015
    num_outliers_cap = int(num_rows * outlier_percentage)
    outlier_indices_cap = np.random.choice(train_df.index, num_outliers_cap, replace=False)
    for idx in outlier_indices_cap:
        if random.random() < 0.5:
            train_df.loc[idx, 'CAPACITY'] = random.randint(1, 15)
        else:
            train_df.loc[idx, 'CAPACITY'] = random.randint(200, 700)

    num_outliers_enroll = int(num_rows * outlier_percentage)
    outlier_indices_enroll = np.random.choice(train_df.index, num_outliers_enroll, replace=False)
    for idx in outlier_indices_enroll:
        if random.random() < 0.5:
            train_df.loc[idx, 'PREV_ENROLLMENT_AVG'] = random.randint(0, 5)
        else:
            train_df.loc[idx, 'PREV_ENROLLMENT_AVG'] = random.randint(300, 800)

    num_outliers_credits = int(num_rows * outlier_percentage * 0.5)
    outlier_indices_credits = np.random.choice(train_df.index, num_outliers_credits, replace=False)
    for idx in outlier_indices_credits:
        train_df.loc[idx, 'CREDIT_HOURS'] = random.choice([1, 6, 7])
    
    return train_df

def run_ablation_experiment(
    initial_train_df, # Pass the same data for all runs
    rf_n_estimators=100,
    val_split_term_percentage=0.2,
    numerical_imputation_strategy='mean',
    tabnet_n_d=8, # For TabNet, if active
    tabnet_n_a=8 # For TabNet, if active
):
    train_df = initial_train_df.copy() # Work on a copy

    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    target = 'HIGH_ENROLLMENT'
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        if train_df[col].dtype == 'object' or (train_df[col].dtype in ['int64', 'float64'] and train_df[col].nunique() < 50):
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

    numerical_imputation_values = {}
    for col in numerical_features:
        if train_df[col].isnull().any():
            if numerical_imputation_strategy == 'mean':
                impute_val = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                impute_val = train_df[col].median()
            else: # Fallback to mean if strategy is unknown
                impute_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(impute_val)
            numerical_imputation_values[col] = impute_val
        else: # Store values even if no NaNs, for potential test data
            if numerical_imputation_strategy == 'mean':
                numerical_imputation_values[col] = train_df[col].mean()
            elif numerical_imputation_strategy == 'median':
                numerical_imputation_values[col] = train_df[col].median()

    feature_columns = numerical_features + categorical_features
    if not feature_columns:
        return 0.0 # No features to train on

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-based Validation Split ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        # Fallback to random split if not enough unique terms, ensure target has at least 2 classes
        if len(train_df_for_split) > 1 and len(np.unique(y_full)) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=val_split_term_percentage, random_state=42, stratify=y_full
            )
        else: # Cannot split meaningfully
            return 0.0
    else:
        num_val_terms = max(1, int(len(unique_terms) * val_split_term_percentage))
        val_terms = unique_terms[-num_val_terms:]
        
        val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
        train_val_indices = train_df_for_split[~train_df_for_split['TERM_CODE'].isin(val_terms)].index

        X_train_val_rf = X_full.loc[train_val_indices]
        y_train_val = y_full.loc[train_val_indices]
        X_val_rf = X_full.loc[val_indices]
        y_val = y_full.loc[val_indices]

        # Fallback to random split if time-based split creates empty or single-class sets
        if X_train_val_rf.empty or X_val_rf.empty or len(np.unique(y_train_val)) < 2 or len(np.unique(y_val)) < 2:
            if len(X_full) > 1 and len(np.unique(y_full)) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=val_split_term_percentage, random_state=42, stratify=y_full
                )
            else:
                return 0.0

    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        return 0.0

    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=rf_n_estimators)
    rf_model.fit(X_train_val_rf, y_train_val)
    rf_y_pred_val = rf_model.predict(X_val_rf)

    # --- Model Training: TabNet (if active) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np)
    can_proceed_tabnet_local = _can_proceed_tabnet # Use global flag

    if can_proceed_tabnet_local:
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        if TabNetClassifier is None:
            can_proceed_tabnet_local = False
        else:
            try:
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
                # print(f"TabNet training/prediction error: {e}. Disabling TabNet for this run.")
                can_proceed_tabnet_local = False

    # --- Ensemble Validation ---
    if can_proceed_tabnet_local:
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float)
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    
    # Check if y_val_np has at least two unique classes before computing F1 score
    if len(np.unique(y_val_np)) < 2:
        return 0.0 # Cannot compute F1 if validation set has only one class
    
    final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    return final_validation_f1

# --- Main Ablation Study Logic ---
if __name__ == "__main__":
    print("Starting ablation study.")

    # Load real data or generate consistent dummy data once for all runs
    initial_data = pd.DataFrame()
    try:
        real_train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
        gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
        gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)
        
        initial_data = pd.merge(real_train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        
        if initial_data.empty:
            raise FileNotFoundError("Merged real data is empty after loading or failed to merge.")
        print("Using real training data.")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading real data for ablation study: {e}. Generating consistent dummy data.")
        initial_data = generate_dummy_data(seed=42)
        print("Using dummy training data.")

    if initial_data.empty:
        print("No data available for ablation study. Exiting.")
        sys.exit(1)

    ablation_results = {}
    
    # Baseline
    print("\n--- Running Baseline ---")
    baseline_f1 = run_ablation_experiment(
        initial_train_df=initial_data,
        rf_n_estimators=100,
        val_split_term_percentage=0.2,
        numerical_imputation_strategy='mean'
    )
    ablation_results['Baseline'] = baseline_f1
    print(f"Baseline F1-score: {baseline_f1:.4f}")

    # Ablation 1: Random Forest n_estimators (50)
    print("\n--- Running Ablation 1: RF n_estimators = 50 ---")
    ablation1_f1 = run_ablation_experiment(
        initial_train_df=initial_data,
        rf_n_estimators=50,
        val_split_term_percentage=0.2,
        numerical_imputation_strategy='mean'
    )
    ablation_results['RF n_estimators (50)'] = ablation1_f1
    print(f"Ablation 1 F1-score (RF n_estimators=50): {ablation1_f1:.4f}")

    # Ablation 2: Validation Split Term Percentage (0.5)
    print("\n--- Running Ablation 2: Validation Split Term Percentage = 0.5 ---")
    ablation2_f1 = run_ablation_experiment(
        initial_train_df=initial_data,
        rf_n_estimators=100,
        val_split_term_percentage=0.5,
        numerical_imputation_strategy='mean'
    )
    ablation_results['Validation Split Term Percentage (0.5)'] = ablation2_f1
    print(f"Ablation 2 F1-score (Validation Split Term Percentage=0.5): {ablation2_f1:.4f}")

    # Ablation 3: Numerical Imputation Strategy (Median)
    print("\n--- Running Ablation 3: Numerical Imputation Strategy = Median ---")
    ablation3_f1 = run_ablation_experiment(
        initial_train_df=initial_data,
        rf_n_estimators=100,
        val_split_term_percentage=0.2,
        numerical_imputation_strategy='median'
    )
    ablation_results['Numerical Imputation Strategy (Median)'] = ablation3_f1
    print(f"Ablation 3 F1-score (Numerical Imputation Strategy=Median): {ablation3_f1:.4f}")

    # --- Contribution Analysis ---
    print("\n--- Ablation Study Summary ---")
    print(f"Baseline F1-score: {baseline_f1:.4f}")

    contributions = {}
    for name, f1 in ablation_results.items():
        if name == 'Baseline':
            continue
        change = f1 - baseline_f1
        contributions[name] = change
        print(f"  {name}: F1-score = {f1:.4f} (Change from Baseline: {change:+.4f})")

    if contributions:
        # Find the ablation with the largest absolute impact
        most_impactful_change_name = max(contributions, key=lambda k: abs(contributions[k]))
        most_impactful_change_value = contributions[most_impactful_change_name]

        print(f"\nMost significant absolute impact on performance was from: '{most_impactful_change_name}'")
        print(f"This change resulted in an F1-score difference of {most_impactful_change_value:+.4f} from the baseline.")
    else:
        print("\nNo ablations were performed or no changes observed.")

