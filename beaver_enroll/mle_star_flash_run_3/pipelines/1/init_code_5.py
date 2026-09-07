
import os
import sys
import subprocess

# Flag to control execution flow based on successful module import/installation
_can_proceed = True

# Check and install pytorch_tabnet if not present
try:
    import pytorch_tabnet
    import torch
except ImportError:
    print("pytorch_tabnet or torch not found. Attempting to install pytorch-tabnet and torch...")
    try:
        # Use check_call to run pip installation, capturing potential errors
        # The 'expected str, bytes or os.PathLike object, not NoneType' error often indicates
        # issues with the subprocess call itself, or environment, rather than the pip command structure.
        # This standard form is generally correct for programmatic installation.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "torch"])
        print("pytorch-tabnet and torch installed successfully.")
        # Attempt to import again after successful installation
        import pytorch_tabnet
        import torch
    except Exception as e:
        print(f"Failed to install pytorch-tabnet and torch: {e}")
        print("Final Validation Performance: 0.0")
        _can_proceed = False # Installation failed, cannot proceed
else:
    # Modules were already imported, no installation needed
    pass

# Only proceed with the main script if all necessary modules are available
if _can_proceed:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
    from pytorch_tabnet.tab_model import TabNetClassifier
    import warnings

    # Suppress all warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Configuration
    TRAIN_DATA_DIR = "./input/table_splits/train"
    TEST_DATA_DIR = "./input/table_splits/test"  # Corrected path for test data
    GOLD_LABELS_PATH = "./input/eval/gold_enrollment_train.csv"

    # --- Data Loading Function ---
    def load_data_from_dir(data_dir):
        all_files = os.listdir(data_dir)
        df_list = []
        
        # Define primary keys for merging
        primary_keys = ['TERM_CODE', 'SUBJECT_ID_SORT']

        for f in all_files:
            if f.endswith(".csv"):
                file_path = os.path.join(data_dir, f)
                try:
                    df = pd.read_csv(file_path)
                    
                    # Ensure primary keys are present
                    if not all(key in df.columns for key in primary_keys):
                        print(f"Skipping {f} as it does not contain all primary keys ({primary_keys}).")
                        continue
                    
                    # Ensure consistent data types for merging
                    df['TERM_CODE'] = df['TERM_CODE'].astype(int)
                    df['SUBJECT_ID_SORT'] = df['SUBJECT_ID_SORT'].astype(str)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        if not df_list:
            print(f"No valid CSV files found or processed in {data_dir}.")
            return pd.DataFrame()

        # Start merging with the first dataframe, using outer merge to keep all course offerings
        merged_df = df_list[0]
        for i in range(1, len(df_list)):
            merged_df = pd.merge(merged_df, df_list[i], on=primary_keys, how='outer', suffixes=('', f'_{i}'))
            
        return merged_df

    # --- Main Script Execution Block ---
    should_run_prediction = True
    final_validation_score = 0.0

    print("Loading training data...")
    train_data = load_data_from_dir(TRAIN_DATA_DIR)
    
    print("Loading gold labels...")
    try:
        gold_labels = pd.read_csv(GOLD_LABELS_PATH)
        gold_labels['TERM_CODE'] = gold_labels['TERM_CODE'].astype(int)
        gold_labels['SUBJECT_ID_SORT'] = gold_labels['SUBJECT_ID_SORT'].astype(str)
    except Exception as e:
        print(f"Error loading gold labels from {GOLD_LABELS_PATH}: {e}")
        should_run_prediction = False

    if should_run_prediction:
        # Merge training data with gold labels
        train_df = pd.merge(train_data, gold_labels, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='inner')
        
        if train_df.empty:
            print("Error: Training DataFrame is empty after merging with gold labels. Check data paths and keys.")
            should_run_prediction = False
        else:
            print(f"Train data shape: {train_df.shape}")

    if should_run_prediction:
        # --- Feature Engineering & Preprocessing ---
        target = 'HIGH_ENROLLMENT'
        
        # Columns to be dropped from features (identifiers or target itself)
        features_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', target]
        
        # Identify numerical and categorical features
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
        # Store dimensions for TabNet's categorical embeddings, adjusted for potential unseen categories in test
        tabnet_cat_dims = [] 

        # Process categorical features in training data
        for col in categorical_features:
            # Convert to string to handle potential mixed types and NaNs as 'nan_category' string
            train_df[col] = train_df[col].astype(str).fillna('nan_category') 
            
            le = LabelEncoder()
            le.fit(train_df[col].unique()) 
            train_df[col] = le.transform(train_df[col])
            label_encoders[col] = le
            
            # TabNet's cat_dims expects (max_encoded_value + 1).
            # If we map unseen categories in test to len(le.classes_), then we need to add 1 to the dimension.
            tabnet_cat_dims.append(len(le.classes_) + 1)

        # Process numerical features in training data (impute NaNs with mean)
        # Store means for later use in test data imputation
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
            print("Error: No features identified for training. Check data columns and feature selection logic.")
            should_run_prediction = False
        else:
            X_train_val_full = train_df[feature_columns].values
            y_train_val_full = train_df[target].values.astype(int)

            # --- Time-based Validation Split ---
            train_df_sorted = train_df.sort_values(by='TERM_CODE')
            
            unique_terms = train_df_sorted['TERM_CODE'].unique()
            if len(unique_terms) < 5: 
                print(f"Warning: Only {len(unique_terms)} unique terms found. Using random split instead of time-based.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val_full, y_train_val_full, test_size=0.2, random_state=42, stratify=y_train_val_full
                )
            else:
                split_idx = int(len(unique_terms) * 0.8) 
                train_terms = unique_terms[:split_idx]
                val_terms = unique_terms[split_idx:]
                
                train_mask = train_df_sorted['TERM_CODE'].isin(train_terms)
                val_mask = train_df_sorted['TERM_CODE'].isin(val_terms)

                X_train = train_df_sorted.loc[train_mask, feature_columns].values
                y_train = train_df_sorted.loc[train_mask, target].values.astype(int)
                X_val = train_df_sorted.loc[val_mask, feature_columns].values
                y_val = train_df_sorted.loc[val_mask, target].values.astype(int)

            print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
            
            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                print("Error: Empty training or validation set after split. Adjust split logic or data.")
                should_run_prediction = False

    if should_run_prediction:
        # TabNet requires categorical_idxs (indices of categorical features in the feature_columns list)
        cat_idxs = [i for i, col in enumerate(feature_columns) if col in categorical_features]

        # --- TabNet Model Training ---
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=tabnet_cat_dims, # Use the adjusted dimensions
            cat_emb_dim=1, # Embedding dimension for categorical features
            n_d=8, n_a=8, # Model capacity
            n_steps=3, # Number of decision steps
            gamma=1.3, # Sparsity regularization
            lambda_sparse=1e-3, # Sparsity regularization strength
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax', 
            verbose=0 
        )
        
        print("Training TabNet model...")
        try:
            clf.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_name=['train', 'valid'],
                eval_metric=['f1', 'accuracy'], 
                max_epochs=100, 
                patience=10, 
                batch_size=1024, 
                virtual_batch_size=128,
                drop_last=False 
            )

            # --- Validation Performance ---
            y_pred_val = clf.predict(X_val)
            final_validation_score = f1_score(y_val, y_pred_val, average='macro')
        except Exception as e:
            print(f"Error during model training or validation: {e}")
            should_run_prediction = False

    print(f"Final Validation Performance: {final_validation_score}")

    if should_run_prediction:
        # --- Prediction on Test Data ---
        print("Loading test data...")
        test_data = load_data_from_dir(TEST_DATA_DIR)
        
        if test_data.empty:
            print("Error: Test DataFrame is empty. Cannot make predictions.")
            should_run_prediction = False
        else:
            # Keep track of original test keys for submission
            test_keys = test_data[['TERM_CODE', 'SUBJECT_ID_SORT']].copy()

            test_processed_df = test_data.copy()

            # Handle categorical features in test_data using fitted LabelEncoders
            for col_idx, col in enumerate(categorical_features):
                if col in test_processed_df.columns:
                    test_processed_df[col] = test_processed_df[col].astype(str).fillna('nan_category')
                    if col in label_encoders:
                        le = label_encoders[col]
                        # Function to apply label encoding or assign len(le.classes_) for unseen
                        def transform_with_unseen(x, encoder):
                            try:
                                return encoder.transform([x])[0]
                            except ValueError:
                                # Assign an index beyond the fitted classes for unseen categories
                                return len(encoder.classes_) 
                        
                        test_processed_df[col] = test_processed_df[col].apply(lambda x: transform_with_unseen(x, le))
                    else:
                        # If a categorical feature was trained but missing in test, fill with its 'unseen' code
                        # This should map to the index we added to tabnet_cat_dims for unseen
                        test_processed_df[col] = tabnet_cat_dims[col_idx] - 1
                else:
                    # If categorical feature is entirely missing in test data, fill with its 'unseen' code
                    test_processed_df[col] = tabnet_cat_dims[col_idx] - 1

            # Handle numerical features in test_data (impute NaNs using training data means)
            for col in numerical_features:
                if col in test_processed_df.columns:
                    if test_processed_df[col].isnull().any():
                        mean_val = numerical_means.get(col, 0) # Use stored mean, default to 0 if not found
                        test_processed_df[col] = test_processed_df[col].fillna(mean_val)
                else:
                    # If numerical feature is entirely missing in test data, fill with its training mean
                    test_processed_df[col] = numerical_means.get(col, 0) 
            
            # Ensure all features used in training are present in test_processed_df and in the correct order
            for col in feature_columns:
                if col not in test_processed_df.columns:
                    test_processed_df[col] = 0 # Default for missing numerical feature, 0 for categorical might be ambiguous

            X_test = test_processed_df[feature_columns].values

            print("Making predictions on test data...")
            try:
                test_predictions = clf.predict(X_test)

                # --- Create Submission File ---
                submission_df = pd.DataFrame({
                    'TERM_CODE': test_keys['TERM_CODE'],
                    'SUBJECT_ID_SORT': test_keys['SUBJECT_ID_SORT'],
                    'HIGH_ENROLLMENT': test_predictions
                })
                
                # Save predictions
                submission_output_path = "./submission.csv" 
                submission_df.to_csv(submission_output_path, index=False)
                print(f"Predictions saved to {submission_output_path}")
            except Exception as e:
                print(f"Error during test prediction or submission file creation: {e}")
    else:
        # If should_run_prediction is False, create an empty submission file or log the failure.
        print("Skipping prediction due to previous errors or insufficient data.")
        empty_submission_df = pd.DataFrame(columns=['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT'])
        empty_submission_df.to_csv("./submission.csv", index=False)
        print("Empty submission.csv created.")
