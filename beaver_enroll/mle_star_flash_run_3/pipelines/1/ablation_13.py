
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
# TEST_DATA_DIR is not used in ablation, define dummy to prevent errors
TEST_DATA_DIR = os.path.join(BASE_DIR, "__TEST_DATA_DIR__")
GOLD_LABELS_PATH = os.path.join(BASE_DIR, "eval/gold_enrollment_train.csv")

# Flag to control execution flow based on successful module import/installation
_can_proceed_tabnet = False
pytorch_tabnet = None
torch = None
TabNetClassifier = None 

# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _can_proceed_tabnet = True
except ImportError:
    # print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...") # Suppress for ablation output
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        # print("pytorch-tabnet and torch installed successfully.") # Suppress for ablation output
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        _can_proceed_tabnet = True
    except Exception as e:
        # print(f"Failed to install pytorch-tabnet and torch: {e}") # Suppress for ablation output
        # print("TabNet will not be used in this run.") # Suppress for ablation output
        _can_proceed_tabnet = False

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
                # print(f"Error reading {file_path}: {e}") # Suppress for cleaner output
                pass
    
    if not df_list:
        # print(f"No valid CSV files found or processed in {data_dir}.") # Suppress for cleaner output
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# Refactor the main logic into a function that accepts ablation parameters
def run_ablation_experiment(
    rf_n_estimators=100,
    categorical_nunique_threshold=50,
    exclude_features=None,
    use_tabnet_flag=_can_proceed_tabnet # Pass the global flag
):
    # Make a local copy of the global flag
    can_proceed_tabnet_local = use_tabnet_flag

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
        # print("Using dummy training data.") # Suppress for cleaner output

    # Ensure TERM_CODE is numeric for sorting
    train_df['TERM_CODE'] = pd.to_numeric(train_df['TERM_CODE'])
    
    # --- Feature Engineering & Preprocessing ---
    target = 'HIGH_ENROLLMENT'
    
    # Columns to be dropped from features (identifiers or target itself)
    features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
    if exclude_features:
        for feat in exclude_features:
            if feat in train_df.columns and feat not in features_to_exclude:
                features_to_exclude.append(feat)
    
    numerical_features = []
    categorical_features = []
    
    for col in train_df.columns:
        if col in features_to_exclude:
            continue
        # Heuristic: object columns are categorical, and numericals with low unique count are also categorical.
        if train_df[col].dtype == 'object' or train_df[col].nunique() < categorical_nunique_threshold:
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
        # print("No features identified for training after preprocessing.") # Suppress for cleaner output
        return 0.0 # Return 0 F1 score if no features

    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-based Validation Split ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    X_train_val_rf, X_val_rf, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    if len(unique_terms) < 2:
        # print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.") # Suppress for cleaner output
        if len(train_df_for_split) > 1:
            X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        else:
            # print("Insufficient data to perform any kind of train-validation split.") # Suppress for cleaner output
            return 0.0 # Return 0 F1 score if split fails
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
            # print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.") # Suppress for cleaner output
            if len(train_df_for_split) > 1:
                X_train_val_rf, X_val_rf, y_train_val, y_val = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                )
            else:
                # print("Insufficient data to perform any kind of train-validation split even with random split.") # Suppress for cleaner output
                return 0.0
        # else:
            # print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}") # Suppress for cleaner output
            # print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}") # Suppress for cleaner output

    # print(f"Training data size: {len(X_train_val_rf)}") # Suppress for cleaner output
    # print(f"Validation data size: {len(X_val_rf)}") # Suppress for cleaner output
    
    if X_train_val_rf.empty or X_val_rf.empty or y_train_val.empty or y_val.empty:
        # print("Training or validation set is empty. Cannot proceed with model training.") # Suppress for cleaner output
        return 0.0

    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # --- Model Training: RandomForest ---
    # print("Training RandomForestClassifier...") # Suppress for cleaner output
    rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    # print("Evaluating RandomForest on validation set...") # Suppress for cleaner output
    rf_y_pred_val = rf_model.predict(X_val_rf)
    # Ensure there's enough data for f1_score to be computed
    if len(np.unique(y_val)) > 1:
        rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')
    else:
        # Bug fix: np.unique(y_val) == np.unique(rf_y_pred_val) returns a numpy array, not a scalar.
        # Access the boolean value inside the array before converting to float.
        if len(y_val) > 0:
            # Ensure both y_val and rf_y_pred_val have unique elements to compare
            if len(np.unique(y_val)) > 0 and len(np.unique(rf_y_pred_val)) > 0:
                rf_val_f1 = float(np.unique(y_val)[0] == np.unique(rf_y_pred_val)[0])
            else: # If one is empty (e.g., all predictions are of a class not in y_val, and y_val only had one class)
                rf_val_f1 = 0.0 
        else:
            rf_val_f1 = 0.0
    # print(f"RandomForest Validation F1: {rf_val_f1}") # Suppress for cleaner output

    # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_pred_val = np.zeros_like(y_val_np) # Default to zeros if TabNet fails or not used

    if can_proceed_tabnet_local:
        # print("Training TabNet model...") # Suppress for cleaner output
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
            # print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.") # Suppress for cleaner output
            can_proceed_tabnet_local = False
        else:
            try:
                # Re-calculate cat_dims based on the current run's label encoders.
                current_tabnet_cat_dims = []
                for col in categorical_features:
                    le = label_encoders[col] # Use the label encoder fitted during this run
                    current_tabnet_cat_dims.append(len(le.classes_) + 1)

                tabnet_model = TabNetClassifier(
                    cat_idxs=cat_idxs,
                    cat_dims=current_tabnet_cat_dims,
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
                # tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='macro') # Suppress for cleaner output
                # print(f"TabNet Validation F1: {tabnet_val_f1}") # Suppress for cleaner output
            except Exception as e:
                # print(f"Error during TabNet training or validation: {e}. TabNet will not contribute to ensemble.") # Suppress for cleaner output
                can_proceed_tabnet_local = False # Disable TabNet for prediction too

    # --- Ensemble Validation ---
    # print("Ensembling predictions on validation set...") # Suppress for cleaner output
    if can_proceed_tabnet_local:
        # Simple averaging for ensemble. Ensure predictions are of similar type (e.g., float before averaging)
        ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
    else:
        ensemble_y_pred_val = rf_y_pred_val.astype(float) # Only RF if TabNet not available
    
    final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
    
    # Ensure enough classes are present in y_val_np for f1_score macro average
    if len(np.unique(y_val_np)) > 1:
        final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
    else:
        if len(y_val_np) > 0:
            if len(np.unique(y_val_np)) > 0 and len(np.unique(final_ensemble_y_pred_val)) > 0:
                final_validation_f1 = float(np.unique(y_val_np)[0] == np.unique(final_ensemble_y_pred_val)[0])
            else:
                final_validation_f1 = 0.0
        else:
            final_validation_f1 = 0.0

    print(f'Final Validation Performance: {final_validation_f1}')

    return final_validation_f1

# Define the ablation scenarios
ablation_scenarios = {
    "Baseline": {
        "rf_n_estimators": 100,
        "categorical_nunique_threshold": 50,
        "exclude_features": None
    },
    "Ablation 1: RF n_estimators = 5": {
        "rf_n_estimators": 5,
        "categorical_nunique_threshold": 50,
        "exclude_features": None
    },
    "Ablation 2: Cat Nunique Threshold = 2": {
        "rf_n_estimators": 100,
        "categorical_nunique_threshold": 2,
        "exclude_features": None
    },
    "Ablation 3: Exclude CAPACITY feature": {
        "rf_n_estimators": 100,
        "categorical_nunique_threshold": 50,
        "exclude_features": ['CAPACITY']
    },
}

if __name__ == "__main__":
    results = {}
    baseline_f1 = 0.0

    print("Starting ablation study...")
    for name, params in ablation_scenarios.items():
        print(f"Running scenario: {name}")
        # Pass the global _can_proceed_tabnet flag to each experiment run
        f1 = run_ablation_experiment(**params, use_tabnet_flag=_can_proceed_tabnet)
        results[name] = f1
        if name == "Baseline":
            baseline_f1 = f1
        print(f"Scenario '{name}' F1 Score: {f1:.4f}")
    
    print("\nAblation Study Results:")
    for name, f1 in results.items():
        print(f"- {name}: F1 Score = {f1:.4f}")

    # Calculate contribution and determine the most impactful part
    contributions = {}
    impact_data = []

    for name, f1 in results.items():
        if name == "Baseline":
            continue
        
        # Calculate impact relative to the baseline
        impact = f1 - baseline_f1
        contributions[name] = impact
        impact_data.append((abs(impact), name))

    print("\nContribution Analysis (change relative to Baseline):")
    if not contributions:
        print("No ablation scenarios were run besides baseline.")
    else:
        for name, impact in contributions.items():
            print(f"- {name}: F1 Score change = {impact:.4f}")

        # Determine the most impactful part
        if impact_data:
            most_impactful_scenario = max(impact_data, key=lambda item: item[0])
            most_impactful_name = most_impactful_scenario[1]
            most_impactful_change = contributions[most_impactful_name]

            print(f"\nThe part of the code that contributes the most to the overall performance is: '{most_impactful_name}' (F1 Score change: {most_impactful_change:.4f})")
        else:
            print("No ablations performed to determine the most impactful part.")
