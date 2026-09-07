
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
# Fix: Make paths robust by resolving relative to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join("/Users/USER/Documents/UNI/SS26/BEAVER/")

TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, "table_splits/train")
TEST_DATA_DIR = os.path.join(BASE_DATA_DIR, "test")
GOLD_LABELS_PATH = os.path.join(BASE_DATA_DIR, "eval/gold_enrollment_train.csv")

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
    # Ensure the directory exists before listing its contents
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return pd.DataFrame() # Return empty DataFrame if directory does not exist

    all_files = os.listdir(data_dir)
    df_list = []
    
    primary_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']

    for f in all_files:
        if f.endswith(".csv"):
            file_path = os.path.join(data_dir, f)
            # Assuming data integrity as per task description, removed specific error handling for each file.
            df = pd.read_csv(file_path)
            
            # Ensure primary keys are present
            if not all(key in df.columns for key in primary_keys):
                print(f"Skipping {file_path} due to missing primary keys.")
                continue
            
            # Ensure consistent data types for merging
            df['TERM_CODE'] = df['TERM_CODE'].astype(int)
            df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
            df_list.append(df)
    
    if not df_list:
        return pd.DataFrame() # Return empty DataFrame if no valid files were processed

    # Start merging with the first dataframe, using outer merge to keep all course offerings
    merged_df = df_list[0]
    for i in range(1, len(df_list)):
        # Merge, handling potential duplicate column names by adding suffixes
        merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
        
    return merged_df

# --- Main Script ---
def main():
    # Make a local copy of the global flag to avoid UnboundLocalError.
    # This local variable will be used and potentially modified within main().
    can_proceed_tabnet_local = _can_proceed_tabnet

    print("Loading training data...")
    # Load real data directly as per task description (data guaranteed to exist)
    train_data_raw = load_data_from_dir(TRAIN_DATA_DIR)
    
    # Handle case where train_data_raw might be empty if directory not found
    if train_data_raw.empty:
        print("Training data could not be loaded. Exiting.")
        return # Exit if no training data

    gold_labels_df = pd.read_csv(GOLD_LABELS_PATH)
    gold_labels_df['TERM_CODE'] = gold_labels_df['TERM_CODE'].astype(int)
    gold_labels_df['SUBJECT_ID_SORT'] = gold_labels_df['SUBJECT_ID_SORT'].astype(str)

    # Merge features with gold labels
    train_df = pd.merge(train_data_raw, gold_labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
    
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
        # Assuming numerical features always have some data or can be filled with mean
        if train_df[col].isnull().any():
            mean_val = train_df[col].mean()
            train_df[col] = train_df[col].fillna(mean_val)
            numerical_means[col] = mean_val
        else:
            numerical_means[col] = train_df[col].mean() 

    # Define the final list of feature columns for the model
    feature_columns = numerical_features + categorical_features
    
    X_full = train_df[feature_columns]
    y_full = train_df[target]
    
    # --- Time-based Validation Split (for F1 reporting only) ---
    train_df_for_split = train_df.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = sorted(train_df_for_split['TERM_CODE'].unique())
    
    # Using a fixed percentage (e.g., 20%) of terms for validation, at least one term
    num_val_terms = max(1, int(len(unique_terms) * 0.2))
    val_terms = unique_terms[-num_val_terms:]
    
    val_indices = train_df_for_split[train_df_for_split['TERM_CODE'].isin(val_terms)].index
    train_val_indices = train_df_for_split[~train_df_for_split['TERM_CODE'].isin(val_terms)].index

    X_train_val_rf = X_full.loc[train_val_indices]
    y_train_val = y_full.loc[train_val_indices]
    X_val_rf = X_full.loc[val_indices]
    y_val = y_full.loc[val_indices]

    print(f"Time-based split: Training on terms {sorted(train_df_for_split.loc[train_val_indices, 'TERM_CODE'].unique())}")
    print(f"Validating on terms: {sorted(train_df_for_split.loc[val_indices, 'TERM_CODE'].unique())}")

    print(f"Training data size (for validation benchmark): {len(X_train_val_rf)}")
    print(f"Validation data size: {len(X_val_rf)}")
    
    # Convert to numpy arrays for TabNet (RandomForest can take DataFrames)
    X_train_val_tabnet = X_train_val_rf.values
    X_val_tabnet = X_val_rf.values
    y_train_val_np = y_train_val.values.astype(int)
    y_val_np = y_val.values.astype(int)


    # --- Model Training (for validation F1 benchmark): RandomForest ---
    print("Training RandomForestClassifier for validation benchmark...")
    rf_model_val = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_model_val.fit(X_train_val_rf, y_train_val)

    # --- Validation: RandomForest ---
    print("Evaluating RandomForest on validation set...")
    rf_y_pred_val = rf_model_val.predict(X_val_rf)
    rf_y_proba_val = rf_model_val.predict_proba(X_val_rf)[:, 1] # Get probabilities for class 1
    rf_val_f1 = f1_score(y_val, rf_y_pred_val, average='macro')
    print(f"RandomForest Validation F1: {rf_val_f1}")

    # --- Model Training (for validation F1 benchmark): TabNet (if can_proceed_tabnet_local) ---
    tabnet_y_proba_val = np.zeros_like(y_val_np, dtype=float) # Default to zeros
    tabnet_val_f1 = 0.0 # Default F1 for TabNet

    if can_proceed_tabnet_local:
        print("Training TabNet model for validation benchmark...")
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        
        # Additional check to ensure TabNetClassifier was actually loaded
        if TabNetClassifier is None:
            print("TabNetClassifier was not successfully imported. Disabling TabNet for this run.")
            can_proceed_tabnet_local = False
        else:
            tabnet_model_val = TabNetClassifier(
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
            
            try: # Use a try-except to catch potential TabNet specific errors during training
                tabnet_model_val.fit(
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
                tabnet_y_proba_val = tabnet_model_val.predict_proba(X_val_tabnet)[:, 1] # Get probabilities for class 1
                tabnet_y_pred_val = (tabnet_y_proba_val >= 0.5).astype(int) # For calculating TabNet's individual F1
                tabnet_val_f1 = f1_score(y_val_np, tabnet_y_pred_val, average='macro')
                print(f"TabNet Validation F1: {tabnet_val_f1}")
            except Exception as e:
                print(f"Error during TabNet training or validation: {e}. TabNet will not contribute to ensemble for validation.")
                can_proceed_tabnet_local = False # Disable TabNet for subsequent test prediction too if it failed
                tabnet_val_f1 = 0.0 # Reset F1 if training failed
                tabnet_y_proba_val = np.zeros_like(y_val_np, dtype=float) # Reset probabilities to zeros

    # --- Ensemble Validation ---
    print("Ensembling predictions on validation set for F1 reporting...")
    
    # Determine weights based on individual model F1 scores
    total_f1 = rf_val_f1
    if can_proceed_tabnet_local:
        total_f1 += tabnet_val_f1
    
    # Handle case where total F1 for weighting is zero (e.g., if F1 for both models is 0, or just RF and its F1 is 0)
    if total_f1 == 0:
        weight_rf = 1.0 # Default to RF only if no F1 can be calculated (or both are 0)
        weight_tabnet = 0.0
        print("Warning: Total F1 for weighting is zero. Defaulting to RF only weights.")
    else:
        weight_rf = rf_val_f1 / total_f1
        weight_tabnet = tabnet_val_f1 / total_f1 if can_proceed_tabnet_local else 0.0

    print(f"Ensemble weights (derived from validation F1): RF={weight_rf:.3f}, TabNet={weight_tabnet:.3f}")

    # Generate weighted ensemble probabilities for validation set
    ensemble_y_proba_val = (weight_rf * rf_y_proba_val) + (weight_tabnet * tabnet_y_proba_val)

    # --- Optimize Threshold on Validation Set ---
    best_f1 = -1.0
    optimal_threshold = 0.5 # Default to 0.5 if no better threshold is found

    threshold_range = np.arange(0.0, 1.01, 0.01) # Check thresholds from 0.0 to 1.0 with 0.01 step
    for threshold in threshold_range:
        temp_y_pred_val = (ensemble_y_proba_val >= threshold).astype(int)
        current_f1 = f1_score(y_val_np, temp_y_pred_val, average='macro')
        if current_f1 > best_f1:
            best_f1 = current_f1
            optimal_threshold = threshold
            
    print(f"Optimal threshold found on validation set: {optimal_threshold:.2f} with F1: {best_f1:.4f}")
    
    final_validation_f1 = best_f1
    print(f'Final Validation Performance: {final_validation_f1}') # Required output format


    # --- Retrain models on FULL training data for final predictions ---
    print("\nRetraining models on full training data (X_full, y_full) for final predictions...")
    
    # RandomForest
    final_rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    final_rf_model.fit(X_full, y_full)
    print("RandomForestClassifier trained on full data.")

    # TabNet
    final_tabnet_model = None
    if can_proceed_tabnet_local:
        cat_idxs_full = [i for i, col in enumerate(feature_columns) if col in categorical_features]
        final_tabnet_model = TabNetClassifier(
            cat_idxs=cat_idxs_full,
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
            # Use X_full.values and y_full.values for TabNet
            final_tabnet_model.fit(
                X_train=X_full.values, y_train=y_full.values.astype(int),
                max_epochs=100,
                patience=10,
                batch_size=1024,
                virtual_batch_size=128,
                drop_last=False
            )
            print("TabNet model trained on full data.")
        except Exception as e:
            print(f"Error during TabNet full data training: {e}. TabNet will not be used in final ensemble.")
            can_proceed_tabnet_local = False
            final_tabnet_model = None


    # --- Prediction on Test Data ---
    print("Loading test data...")
    test_df_raw = load_data_from_dir(TEST_DATA_DIR)
    
    # Handle case where test_df_raw might be empty if directory not found
    if test_df_raw.empty:
        print("Test data could not be loaded. Generating empty submission.")
        submission_df = pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'])
        output_dir = "./final"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Empty predictions saved to {output_path}")
        return # Exit if no test data

    # Keep original test_df for output merge
    test_original_df = test_df_raw[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()

    test_processed_df = test_df_raw.copy()
    test_processed_df['TERM_CODE'] = pd.to_numeric(test_processed_df['TERM_CODE'])

    # Preprocess test data using training-derived transformations
    # Handle categorical features
    for i, col in enumerate(categorical_features):
        if col in test_processed_df.columns:
            test_processed_df[col] = test_processed_df[col].astype(str).fillna('nan_category')
            le = label_encoders[col]
            def transform_with_unseen(x, encoder, cat_dim_val):
                try:
                    return encoder.transform([x])[0]
                except ValueError:
                    # Assign to the last category index for unseen
                    return cat_dim_val - 1 
            
            # Use the correct `tabnet_cat_dims[i]` for the current column's dimensions
            test_processed_df[col] = test_processed_df[col].apply(lambda x: transform_with_unseen(x, le, tabnet_cat_dims[i]))
        else:
            # If categorical feature is entirely missing in test data, fill with an appropriate value.
            # For TabNet, this would be the 'unseen' code (tabnet_cat_dims[i] - 1).
            # If TabNet is not active, filling with 0 (first category) is a reasonable fallback.
            test_processed_df[col] = tabnet_cat_dims[i] - 1 if can_proceed_tabnet_local else 0

    # Handle numerical features
    for col in numerical_features:
        if col in test_processed_df.columns:
            if test_processed_df[col].isnull().any():
                mean_val = numerical_means.get(col, 0) # Use stored mean, default to 0 if not found
                test_processed_df[col] = test_processed_df[col].fillna(mean_val)
        else:
            # If numerical feature is entirely missing in test data, fill with its training mean
            test_processed_df[col] = numerical_means.get(col, 0) 
            
    # Ensure all features used in training are present in test_processed_df and in the correct order
    # Add missing columns with 0, remove extra columns
    for col in feature_columns:
        if col not in test_processed_df.columns:
            test_processed_df[col] = 0 # Default value for missing features in test

    # Drop columns in test_processed_df that are not in feature_columns (new features in test data)
    extra_cols_in_test = set(test_processed_df.columns) - set(feature_columns)
    if extra_cols_in_test:
        test_processed_df = test_processed_df.drop(columns=list(extra_cols_in_test))

    # Ensure the order of columns is the same as training features
    X_test_processed = test_processed_df[feature_columns]

    print("Making predictions on test data using models trained on full data...")
    rf_y_proba_test = final_rf_model.predict_proba(X_test_processed)[:, 1]
    
    tabnet_y_proba_test = np.zeros(len(X_test_processed), dtype=float) # Default to zeros
    if can_proceed_tabnet_local and final_tabnet_model is not None:
        tabnet_y_proba_test = final_tabnet_model.predict_proba(X_test_processed.values)[:, 1]
            
    # --- Ensemble Test Probabilities ---
    ensemble_y_proba_test = (weight_rf * rf_y_proba_test) + (weight_tabnet * tabnet_y_proba_test)

    # --- Apply Optimal Threshold to Test Predictions ---
    final_ensemble_y_pred_test = (ensemble_y_proba_test >= optimal_threshold).astype(int)

    # --- Prepare Submission ---
    submission_df = test_original_df.copy()
    submission_df['HIGH_ENROLLMENT'] = final_ensemble_y_pred_test

    # Create the final directory if it doesn't exist
    output_dir = "./final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output predictions to a CSV file (e.g., 'submission.csv')
    output_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
