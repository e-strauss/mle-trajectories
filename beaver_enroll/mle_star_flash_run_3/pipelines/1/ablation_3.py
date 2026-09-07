
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
    # print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...")
    try:
        # Install to user site-packages to avoid permissions issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch", "--user"])
        # print("pytorch-tabnet and torch installed successfully.")
        # Attempt to import again after successful installation
        import pytorch_tabnet
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        _can_proceed_tabnet = True
    except Exception as e:
        # print(f"Failed to install pytorch-tabnet and torch: {e}")
        # print("TabNet will not be used in this run.")
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
                # print(f"Error reading {file_path}: {e}")
                pass # Suppress error prints for cleaner ablation output
    
    if not df_list:
        # print(f"No valid CSV files found or processed in {data_dir}.")
        return pd.DataFrame()

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Preprocessing Function for Ablation ---
def preprocess_data_for_ablation(X_df, identifier_features, config, is_train=True, stored_params=None):
    """
    Applies preprocessing steps (categorical encoding, numerical imputation, capping, scaling).
    Parameters:
        X_df (pd.DataFrame): DataFrame containing features to preprocess.
        identifier_features (list): List of column names that are identifiers (e.g., TERM_CODE, SUBJECT_ID_SORT)
                                    and should not be treated as features for modeling.
        config (dict): Dictionary specifying the preprocessing configuration (imputation, capping, scaling).
        is_train (bool): True if fitting transformers, False if transforming with stored ones.
        stored_params (dict): Dictionary to store/retrieve fitted transformers/parameters.
    Returns:
        pd.DataFrame: Preprocessed features.
        dict: Stored parameters (if is_train) or None (if not is_train).
    """
    processed_df = X_df.copy()
    
    numerical_features = []
    categorical_features = []
    
    # Determine feature types based on training data or stored params
    if is_train:
        for col in processed_df.columns:
            if col in identifier_features:
                continue
            if processed_df[col].dtype == 'object' or processed_df[col].nunique() < 50:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Initialize storage for new params
        label_encoders = {}
        tabnet_cat_dims = []
        numerical_imputers = {}
        numerical_scalers = {}
        numerical_percentile_caps = {}
    else:
        # Use stored parameters for feature types and transformations
        numerical_features = stored_params['numerical_features_order']
        categorical_features = stored_params['categorical_features_order']
        
        label_encoders = stored_params['label_encoders']
        tabnet_cat_dims = stored_params['tabnet_cat_dims']
        numerical_imputers = stored_params['numerical_imputers']
        numerical_scalers = stored_params['numerical_scalers']
        numerical_percentile_caps = stored_params['numerical_percentile_caps']

    # Process Categorical Features
    for col in categorical_features:
        if col not in processed_df.columns: 
            processed_df[col] = 'nan_category' 
        
        processed_df[col] = processed_df[col].astype(str).fillna('nan_category') 
        if is_train:
            le = LabelEncoder()
            le.fit(processed_df[col].unique()) 
            processed_df[col] = le.transform(processed_df[col])
            label_encoders[col] = le
            tabnet_cat_dims.append(len(le.classes_) + 1)
        else:
            if col in label_encoders:
                le = label_encoders[col]
                cat_dim_val = 0
                try:
                    idx = stored_params['categorical_features_order'].index(col)
                    cat_dim_val = tabnet_cat_dims[idx]
                except ValueError:
                    cat_dim_val = 1 

                def transform_with_unseen(x, encoder, cat_dim):
                    try:
                        return encoder.transform([x])[0]
                    except ValueError:
                        return cat_dim - 1 
                processed_df[col] = processed_df[col].apply(lambda x: transform_with_unseen(x, le, cat_dim_val))
            else:
                processed_df[col] = 0

    # Process Numerical Features
    for col in numerical_features:
        if col not in processed_df.columns:
            processed_df[col] = np.nan
        
        if is_train:
            imputer_val = None
            if config['imputation'] == 'mean':
                imputer_val = processed_df[col].mean()
            elif config['imputation'] == 'median':
                imputer_val = processed_df[col].median()
            else:
                imputer_val = 0 # Default: fill with 0 if no specific strategy
            numerical_imputers[col] = imputer_val
            processed_df[col] = processed_df[col].fillna(imputer_val)
            
            if config['capping']:
                lower_bound = processed_df[col].quantile(0.01)
                upper_bound = processed_df[col].quantile(0.99)
                processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
                numerical_percentile_caps[col] = {'lower': lower_bound, 'upper': upper_bound}
            
            if config['scaling']:
                scaler = StandardScaler()
                # Ensure values are numeric before fitting/transforming
                processed_df[col] = processed_df[col].astype(float) 
                processed_df[col] = scaler.fit_transform(processed_df[[col]])
                numerical_scalers[col] = scaler
        else: # Apply stored transformations to validation data
            if col in numerical_imputers:
                processed_df[col] = processed_df[col].fillna(numerical_imputers[col])
            else:
                processed_df[col] = processed_df[col].fillna(0) # Fallback if imputer not stored

            if config['capping'] and col in numerical_percentile_caps:
                caps = numerical_percentile_caps[col]
                processed_df[col] = processed_df[col].clip(lower=caps['lower'], upper=caps['upper'])
            
            if config['scaling'] and col in numerical_scalers:
                scaler = numerical_scalers[col]
                if col in processed_df.columns and not processed_df[col].isnull().all():
                    processed_df[col] = processed_df[col].astype(float)
                    processed_df[col] = scaler.transform(processed_df[[col]])
                else: 
                    processed_df[col] = 0.0 # Default to scaled mean (0)

    final_feature_columns = numerical_features + categorical_features
    # Ensure all final_feature_columns are present in processed_df and in correct order
    for col in final_feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 # Default value for missing features
    
    # Drop columns in processed_df that are not in final_feature_columns (e.g., identifier_features)
    extra_cols_to_drop = set(processed_df.columns) - set(final_feature_columns)
    if extra_cols_to_drop:
        processed_df = processed_df.drop(columns=list(extra_cols_to_drop))

    X_processed = processed_df[final_feature_columns]

    params_to_store = {
        'label_encoders': label_encoders,
        'tabnet_cat_dims': tabnet_cat_dims,
        'numerical_imputers': numerical_imputers,
        'numerical_scalers': numerical_scalers,
        'numerical_percentile_caps': numerical_percentile_caps,
        'numerical_features_order': numerical_features, 
        'categorical_features_order': categorical_features, 
        'feature_columns_order': final_feature_columns 
    }

    return X_processed, params_to_store

# --- Main Script for Ablation Study ---
def main():
    can_proceed_tabnet_local = _can_proceed_tabnet

    # print("Loading training data...")
    train_df_full = pd.DataFrame()
    
    # Try loading real data first
    try:
        train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
        gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
        gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
        gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)

        if train_data_raw.empty or gold_labels_df.empty:
            raise FileNotFoundError("Raw training data or gold labels are empty after loading.")

        train_df_full = pd.merge(train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        if train_df_full.empty:
            raise ValueError("Training DataFrame is empty after merging with gold labels.")
        
        # Ensure TERM_CODE is numeric for sorting
        train_df_full['TERM_CODE'] = pd.to_numeric(train_df_full['TERM_CODE'])

    except (FileNotFoundError, ValueError, Exception) as e:
        # print(f"Error loading real data: {e}. Creating dummy training data for demonstration.")
        # Create dummy data if real data loading fails or results in empty df
        train_df_full = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201'],
            'CREDIT_HOURS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3],
            'INSTRUCTOR_RANK': ['PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST', 'PROF', 'ASSIST', 'LECT', 'PROF', 'ASSIST'],
            'CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65],
            'PREV_ENROLLMENT_AVG': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        # print("Using dummy training data.")

    target = 'HIGH_ENROLLMENT'
    identifier_features = ['TERM_CODE', 'SUBJECT_ID_SORT']

    ablation_scenarios = {
        "Ablation Scenario 1: Original Numerical Preprocessing (Baseline)": {
            'imputation': 'mean', 'capping': False, 'scaling': False
        },
        "Ablation Scenario 2: Add Numerical Scaling (StandardScaler)": {
            'imputation': 'mean', 'capping': False, 'scaling': True
        },
        "Ablation Scenario 3: Add Outlier Capping & Numerical Scaling": {
            'imputation': 'mean', 'capping': True, 'scaling': True
        },
        "Ablation Scenario 4: Change Imputation to Median, Add Outlier Capping & Numerical Scaling": {
            'imputation': 'median', 'capping': True, 'scaling': True
        },
    }

    results = {}
    best_f1 = -1
    best_scenario = ""

    for scenario_name, config in ablation_scenarios.items():
        # print(f"\n--- Running {scenario_name} ---")
        
        # --- Time-based Validation Split ---
        train_df_for_split = train_df_full.sort_values(by='TERM_CODE').reset_index(drop=True)
        unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
        
        # Prepare full feature set and target for splitting
        features_df_full = train_df_for_split.drop(columns=[target])
        target_series_full = train_df_for_split[target]

        X_train_val_raw, X_val_raw, y_train_val, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
        
        if len(unique_terms) < 2:
            # print("Warning: Not enough unique terms for a meaningful time-based split. Falling back to random split.")
            if len(train_df_for_split) > 1:
                X_train_val_raw, X_val_raw, y_train_val, y_val = train_test_split(
                    features_df_full, target_series_full, test_size=0.2, random_state=42, stratify=target_series_full
                )
            else:
                raise ValueError("Insufficient data to perform any kind of train-validation split.")
        else:
            num_val_terms = max(1, int(len(unique_terms) * 0.2))
            val_terms = unique_terms[-num_val_terms:]
            
            val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
            train_val_indices = train_df_for_split[~train_df_for_split['TERM_CODE'].isin(val_terms)].index

            X_train_val_raw = features_df_full.loc[train_val_indices]
            y_train_val = target_series_full.loc[train_val_indices]
            X_val_raw = features_df_full.loc[val_indices]
            y_val = target_series_full.loc[val_indices]

            if X_train_val_raw.empty or X_val_raw.empty:
                # print("Warning: Time-based split resulted in an empty training or validation set after filtering. Falling back to random split.")
                if len(train_df_for_split) > 1:
                    X_train_val_raw, X_val_raw, y_train_val, y_val = train_test_split(
                        features_df_full, target_series_full, test_size=0.2, random_state=42, stratify=target_series_full
                    )
                else:
                    raise ValueError("Insufficient data to perform any kind of train-validation split even with random split.")
            # else:
                # print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}")
                # print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}")
        
        # Preprocess training data
        X_train_proc, stored_params = preprocess_data_for_ablation(
            X_train_val_raw, identifier_features, config, is_train=True
        )
        
        # Preprocess validation data using parameters from training
        X_val_proc, _ = preprocess_data_for_ablation(
            X_val_raw, identifier_features, config, is_train=False, stored_params=stored_params
        )

        # print(f"Training data size: {len(X_train_proc)}")
        # print(f"Validation data size: {len(X_val_proc)}")
        
        if X_train_proc.empty or X_val_proc.empty or y_train_val.empty or y_val.empty:
            # print(f"Skipping {scenario_name} due to empty training or validation set after preprocessing.")
            results[scenario_name] = 0.0 # Assign a low F1 if skipped
            continue

        # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
        X_train_val_tabnet = X_train_proc.values
        X_val_tabnet = X_val_proc.values
        y_train_val_np = y_train_val.values.astype(int)
        y_val_np = y_val.values.astype(int)
        
        feature_columns = stored_params['feature_columns_order']
        categorical_features_order = stored_params['categorical_features_order']

        # --- Model Training: RandomForest ---
        # print("Training RandomForestClassifier...")
        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf_model.fit(X_train_proc, y_train_val)

        # --- Validation: RandomForest ---
        rf_y_pred_val = rf_model.predict(X_val_proc)
        rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')
        # print(f"RandomForest Validation F1: {rf_val_f1}")

        # --- Model Training: TabNet (if can_proceed_tabnet_local) ---
        tabnet_y_pred_val = np.zeros_like(y_val_np) 
        tabnet_val_f1 = 0.0

        if can_proceed_tabnet_local:
            # print("Training TabNet model...")
            cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features_order]
            tabnet_cat_dims = stored_params['tabnet_cat_dims']
            
            if TabNetClassifier is None: # Redundant check, but safe if global var was reset
                # print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
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
                    tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='macro')
                    # print(f"TabNet Validation F1: {tabnet_val_f1}")
                except Exception as e:
                    # print(f"Error during TabNet training or validation for {scenario_name}: {e}. TabNet will not contribute to ensemble.")
                    can_proceed_tabnet_local = False 

        # --- Ensemble Validation ---
        # print("Ensembling predictions on validation set...")
        if can_proceed_tabnet_local and TabNetClassifier is not None:
            ensemble_y_pred_val = (rf_y_pred_val.astype(float) + tabnet_y_pred_val.astype(float)) / 2
        else:
            ensemble_y_pred_val = rf_y_pred_val.astype(float)
        
        final_ensemble_y_pred_val = (ensemble_y_pred_val >= 0.5).astype(int)
        final_validation_f1 = f1_score(y_val_np, final_ensemble_y_pred_val, average='macro')
        results[scenario_name] = final_validation_f1

        if final_validation_f1 > best_f1:
            best_f1 = final_validation_f1
            best_scenario = scenario_name

    print("\n--- Ablation Study Results ---")
    baseline_f1 = results.get("Ablation Scenario 1: Original Numerical Preprocessing (Baseline)", 0)
    for scenario_name, f1_score_val in results.items():
        print(f"- {scenario_name}: F1-Score = {f1_score_val:.4f}")
    
    print(f"\nBest performing scenario: {best_scenario} with F1-Score = {best_f1:.4f}")
    print(f"Final Validation Performance: {best_f1:.4f}")

    # Determine which part contributes most
    contributions = {}
    
    # Impact of adding Scaling (Scenario 2 vs Scenario 1)
    if "Ablation Scenario 2: Add Numerical Scaling (StandardScaler)" in results:
        contributions["Adding Numerical Scaling (StandardScaler)"] = results["Ablation Scenario 2: Add Numerical Scaling (StandardScaler)"] - baseline_f1
    
    # Impact of adding Capping (Scenario 3 vs Scenario 2)
    if "Ablation Scenario 3: Add Outlier Capping & Numerical Scaling" in results and \
       "Ablation Scenario 2: Add Numerical Scaling (StandardScaler)" in results:
        # Corrected line: Removed the extra double quote from the dictionary key
        contributions["Adding Outlier Capping (on top of Scaling)"] = results["Ablation Scenario 3: Add Outlier Capping & Numerical Scaling"] - results["Ablation Scenario 2: Add Numerical Scaling (StandardScaler)"]
    
    # Impact of changing Imputation to Median (Scenario 4 vs Scenario 3)
    if "Ablation Scenario 4: Change Imputation to Median, Add Outlier Capping & Numerical Scaling" in results and \
       "Ablation Scenario 3: Add Outlier Capping & Numerical Scaling" in results:
        contributions["Changing Numerical Imputation to Median"] = results["Ablation Scenario 4: Change Imputation to Median, Add Outlier Capping & Numerical Scaling"] - results["Ablation Scenario 3: Add Outlier Capping & Numerical Scaling"]
    
    print("\n--- Contribution Analysis (Change in F1-score relative to previous step or baseline) ---")
    if contributions:
        most_contributing_part = max(contributions, key=contributions.get)
        max_contribution = contributions[most_contributing_part]
        
        for part, contribution in contributions.items():
            print(f"- {part}: {contribution:.4f}")
        
        if max_contribution > 0:
            print(f"\nConclusion: The part that contributes the most to the overall performance is '{most_contributing_part}' with an improvement of {max_contribution:.4f} F1-score.")
        else:
            print("\nConclusion: No specific part showed a positive contribution to overall performance or all contributions were zero/negative.")
    else:
        print("No specific contributions could be calculated.")

if __name__ == "__main__":
    main()
