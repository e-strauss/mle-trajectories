
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Suppress warnings from scikit-learn regarding feature names, which can be noisy
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Conditional import for XGBoost ---
# Per instructions, we assume libraries are installed or let the import fail naturally.
# No dynamic installation or try-except blocks to "ignore unintended behavior" for imports.
try:
    import xgboost
    from xgboost import XGBClassifier
    _has_xgboost = True
except ImportError:
    logging.warning("XGBoost library not found. XGBoost model will not be used.")
    _has_xgboost = False

# --- Define paths ---
INPUT_DIR = './input'
TRAIN_DATA_DIR_TABLE_SPLITS = os.path.join(INPUT_DIR, 'table_splits', 'train')
GOLD_ENROLLMENT_FILE_NAME = "gold_enrollment_train.csv"
GOLD_ENROLLMENT_TRAIN_FILE_PATH = os.path.join(INPUT_DIR, GOLD_ENROLLMENT_FILE_NAME) # Corrected path for gold file


# --- PyTorch MLP Model (from reference solution) ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) # Output a single logit for binary classification
        )

    def forward(self, x):
        return self.network(x)


# --- Function to load data and engineer features (integrated from both solutions) ---
def load_and_preprocess_data(data_dir_table_splits, gold_file_path):
    """
    Loads gold labels and merges with subject summary data, performs feature engineering,
    and identifies feature types for subsequent preprocessing.
    """
    try:
        gold_df = pd.read_csv(gold_file_path)
    except FileNotFoundError:
        logging.error(f"Gold enrollment file not found: {gold_file_path}")
        return pd.DataFrame() # Return empty DataFrame on critical error

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    gold_df['HIGH_ENROLLMENT'] = gold_df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    # Try to find a central table containing course offering details (prioritizing subject_summary)
    main_df = None
    subject_summary_path = os.path.join(data_dir_table_splits, 'subject_summary.csv')
    course_offerings_path = os.path.join(data_dir_table_splits, 'course_offerings.csv')

    if os.path.exists(subject_summary_path):
        main_df = pd.read_csv(subject_summary_path, low_memory=False)
        logging.info(f"Loaded subject_summary.csv with {len(main_df)} rows as main_table.")
    elif os.path.exists(course_offerings_path):
        main_df = pd.read_csv(course_offerings_path, low_memory=False)
        logging.info(f"Loaded course_offerings.csv with {len(main_df)} rows as main_table.")
    else:
        logging.warning("Neither subject_summary.csv nor course_offerings.csv found. Proceeding with minimal features from gold_df.")

    # Prepare the base dataframe, starting with gold_df
    df = gold_df.copy()

    # Identifier columns used for merging and to be excluded from direct feature selection
    identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
    
    # Merge gold_df with main_df if available
    if main_df is not None:
        # Ensure join keys are present in gold_df
        missing_gold_keys = [key for key in identifier_cols if key not in df.columns]
        if missing_gold_keys:
            logging.error(f"Required join keys {missing_gold_keys} not found in gold labels. Cannot merge with main data.")
            main_df = None # Abort merge
        
        if main_df is not None:
            # Ensure required join keys are present in main_df and attempt merge
            missing_main_keys = [key for key in identifier_cols if key not in main_df.columns]
            if missing_main_keys:
                logging.warning(f"Required join keys {missing_main_keys} not found in the main data frame. Attempting partial merge on common keys.")
                common_keys = [key for key in identifier_cols if key in main_df.columns]
                if common_keys:
                    df = pd.merge(df, main_df, on=common_keys, how='left')
                    logging.info(f"Merged data with main table on common keys {common_keys}. Resulting rows: {len(df)}")
                else:
                    logging.warning("No common keys to merge main table. Proceeding without main table features.")
            else:
                # Merge gold_labels with main_df
                df = pd.merge(df, main_df, on=identifier_cols, how='left')
                logging.info(f"Merged data has {len(df)} rows after joining with main table.")

    # Validate DataFrame state after merging
    if df.empty or 'HIGH_ENROLLMENT' not in df.columns:
        logging.error("Merged DataFrame is empty or target column 'HIGH_ENROLLMENT' is missing. Cannot proceed with feature engineering.")
        return pd.DataFrame()

    # --- Feature Engineering (from reference solution) ---
    processed_df = df.copy()

    # Ensure TERM_CODE is treated as string for slicing
    if 'TERM_CODE' in processed_df.columns:
        processed_df['TERM_CODE_STR'] = processed_df['TERM_CODE'].astype(str)
        # Convert year part to int; use errors='ignore' for robustness against non-numeric entries
        processed_df['TERM_YEAR'] = pd.to_numeric(processed_df['TERM_CODE_STR'].str[:4], errors='coerce').fillna(-1).astype(int)
        # TERM_SEMESTER can contain non-numeric values like 'FA', keep as object for categorical encoding
        processed_df['TERM_SEMESTER'] = processed_df['TERM_CODE_STR'].str[4:] 
        logging.info("Generated 'TERM_YEAR' and 'TERM_SEMESTER' features.")
    else:
        logging.warning("'TERM_CODE' column not found, cannot generate time-based features.")

    # Create 'FILL_RATE' feature
    if 'ENROLLMENT_COUNT' in processed_df.columns and 'CAPACITY' in processed_df.columns:
        processed_df['CAPACITY_ADJ'] = processed_df['CAPACITY'].replace(0, np.nan) 
        processed_df['FILL_RATE'] = processed_df['ENROLLMENT_COUNT'] / processed_df['CAPACITY_ADJ']
        processed_df['FILL_RATE'] = processed_df['FILL_RATE'].replace([np.inf, -np.inf], np.nan) # Handle potential inf values
        processed_df.drop(columns=['CAPACITY_ADJ'], inplace=True, errors='ignore') # Drop temporary column
        logging.info("Created 'FILL_RATE' feature.")
    else:
        logging.warning("Columns 'ENROLLMENT_COUNT' or 'CAPACITY' not found for 'FILL_RATE' feature generation.")

    # Convert object columns that contain mostly numeric data to numeric
    for col in processed_df.select_dtypes(include='object').columns:
        if processed_df[col].notna().sum() > 0:
            is_numeric_like_ratio = pd.to_numeric(processed_df[col], errors='coerce').notna().sum() / processed_df[col].notna().sum()
            if is_numeric_like_ratio > 0.8: # If >80% of non-null values are numeric, convert
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                logging.info(f"Converted '{col}' to numeric type.")
        
    df = processed_df # Update df with engineered features

    # Drop rows where HIGH_ENROLLMENT is missing (if any after merge/feature engineering)
    initial_rows = len(df)
    df.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    if len(df) < initial_rows:
        logging.warning(f"Dropped {initial_rows - len(df)} rows due to missing 'HIGH_ENROLLMENT' target.")
    
    # Exclude ID columns, target, and temporary helper columns from features
    id_cols_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', 'TERM_CODE_STR']
    target_col = 'HIGH_ENROLLMENT'
    
    # Identify candidate features for modeling
    candidate_features = [
        col for col in df.columns 
        if col not in id_cols_to_exclude + [target_col] 
        and not pd.api.types.is_datetime64_any_dtype(df[col])
        and df[col].nunique() > 1 # Exclude constant columns which provide no information
    ]
    
    numerical_features = df[candidate_features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[candidate_features].select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out high cardinality categorical features for OneHotEncoder (sklearn models)
    initial_categorical_features_count = len(categorical_features)
    categorical_features = [
        f for f in categorical_features 
        if df[f].nunique() <= 50 and df[f].nunique() < len(df) * 0.1 # Limit cardinality and proportion
    ]
    if len(categorical_features) < initial_categorical_features_count:
        logging.warning(f"Filtered out {initial_categorical_features_count - len(categorical_features)} high cardinality categorical features for sklearn pipeline.")

    # Fallback: If no features are identified, add a dummy feature.
    if not numerical_features and not categorical_features:
        logging.warning("No suitable features identified after engineering and filtering. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0
        numerical_features.append('DUMMY_FEATURE')
    
    # Store feature column names as attributes on the DataFrame for later use
    df._numerical_features = numerical_features
    df._categorical_features = categorical_features
    
    return df

# --- Main script ---
def run_training_and_validation():
    df = load_and_preprocess_data(TRAIN_DATA_DIR_TABLE_SPLITS, GOLD_ENROLLMENT_TRAIN_FILE_PATH)

    if df.empty or 'HIGH_ENROLLMENT' not in df.columns or df['HIGH_ENROLLMENT'].isnull().all():
        logging.error("Loaded DataFrame is empty or target column is missing/empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    train_df, val_df = pd.DataFrame(), pd.DataFrame()

    # Ensure there are enough terms for a time-based split, otherwise fall back to random split
    if len(unique_terms) < 2:
        logging.warning("Not enough unique terms for a time-based validation split (at least 2 required). Using a simple random split.")
        
        # Identify all features available for model training
        all_features = getattr(df, '_numerical_features', []) + getattr(df, '_categorical_features', [])
        
        # If no features identified, add a dummy feature to prevent errors during split
        if not all_features: 
            df['DUMMY_FEATURE'] = 0
            all_features = ['DUMMY_FEATURE']
            logging.warning("Using dummy feature for random split due to lack of other features.")

        X_full = df[all_features]
        y_full = df['HIGH_ENROLLMENT']

        # If target has only one class, stratification is not possible
        stratify_param = y_full if len(y_full.unique()) > 1 else None
        
        # Split indices to ensure the full dataframe structure is preserved for feature selection later
        train_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=stratify_param)
        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy()

    else:
        # Use the last term for validation to simulate a future term, as per problem description.
        validation_term = unique_terms[-1] 
        train_df = df[df['TERM_CODE'] != validation_term].copy()
        val_df = df[df['TERM_CODE'] == validation_term].copy()

    if train_df.empty or val_df.empty:
        logging.error("Training or validation set is empty after splitting. Cannot proceed.")
        print("Final Validation Performance: 0.0")
        return

    # Retrieve feature lists determined during data loading
    numerical_features = getattr(df, '_numerical_features', [])
    categorical_features = getattr(df, '_categorical_features', [])
    
    # Filter features to ensure they exist in the respective split dataframes
    numerical_features = [col for col in numerical_features if col in train_df.columns]
    categorical_features = [col for col in categorical_features if col in train_df.columns]

    # Re-check if features are available after filtering (important for robustness)
    if not numerical_features and not categorical_features:
        logging.error("No usable features remain after filtering for split dataframes. Cannot train models.")
        print("Final Validation Performance: 0.0")
        return

    # Prepare X and y for training/validation
    X_train = train_df[numerical_features + categorical_features]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[numerical_features + categorical_features]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for target variable issues in splits
    if len(y_train.unique()) < 2:
        logging.warning(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions or model errors.")
    if len(y_val.unique()) < 2:
        logging.warning(f"Validation set target 'HIGH_ENROLLMENT' has only one class: {y_val.unique()}. This might affect F1 score calculation.")

    # --- Create Preprocessing Pipeline (ColumnTransformer) for scikit-learn models and MLP ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for dense numpy array
    ])

    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
        logging.error("No features configured for ColumnTransformer. Cannot proceed with model training.")
        print("Final Validation Performance: 0.0")
        return

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # Drop columns not specified in transformers
    )

    # Apply preprocessing (fit on train, transform on train and val)
    # Ensure X_train and X_val are not empty before transformation
    if X_train.empty or X_val.empty:
        logging.error("X_train or X_val is empty before preprocessing. Cannot train scikit-learn compatible models.")
        X_train_transformed = np.empty((0, 0)) # Placeholder for empty transformed data
        X_val_transformed = np.empty((0, 0))
    else:
        try:
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_val_transformed = preprocessor.transform(X_val)
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}. Cannot train models requiring ColumnTransformer.")
            X_train_transformed = np.empty((0, 0))
            X_val_transformed = np.empty((0, 0))

    # --- Model Training ---
    logging.info("Starting ensemble model training.")

    rf_pred_proba = np.array([])
    xgb_pred_proba = np.array([])
    mlp_pred_proba = np.array([])

    # Check for valid transformed data shape before training any model
    if X_train_transformed.shape[0] == 0 or X_train_transformed.shape[1] == 0 or y_train.shape[0] == 0:
        logging.warning("Transformed training data is empty or ill-formed. Skipping all model training.")
    else:
        # Model 1: RandomForestClassifier (from base solution)
        logging.info("Training RandomForestClassifier...")
        model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
        model_rf.fit(X_train_transformed, y_train)
        if X_val_transformed.shape[0] > 0:
            rf_pred_proba = model_rf.predict_proba(X_val_transformed)[:, 1]
        else:
            logging.warning("Validation set is empty for RandomForest predictions.")

        # Model 2: XGBoost Classifier (additional model)
        if _has_xgboost:
            logging.info("Training XGBClassifier...")
            # use_label_encoder=False to suppress a common warning in newer XGBoost versions
            model_xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', 
                                      random_state=42, n_estimators=100, learning_rate=0.1, 
                                      use_label_encoder=False)
            model_xgb.fit(X_train_transformed, y_train)
            if X_val_transformed.shape[0] > 0:
                xgb_pred_proba = model_xgb.predict_proba(X_val_transformed)[:, 1]
            else:
                logging.warning("Validation set is empty for XGBoost predictions.")
        else:
            logging.warning("XGBoost is not available. Skipping XGBClassifier training.")

        # Model 3: PyTorch MLP (from reference solution)
        logging.info("Training PyTorch MLP...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device for MLP: {device}")

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_transformed.astype(np.float32)).to(device)
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1).to(device)

        # Create validation tensors only if validation data exists
        X_val_tensor_mlp = torch.empty((0, X_train_transformed.shape[1]), device=device) # Initialize as empty
        if X_val_transformed.shape[0] > 0:
            X_val_tensor_mlp = torch.tensor(X_val_transformed.astype(np.float32)).to(device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_dim = X_train_transformed.shape[1]
        model_mlp = MLP(input_dim).to(device)
        optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss() 

        num_epochs = 50
        for epoch in range(num_epochs):
            model_mlp.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model_mlp(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Predict with MLP if validation data exists
        if X_val_transformed.shape[0] > 0:
            model_mlp.eval()
            with torch.no_grad():
                outputs = model_mlp(X_val_tensor_mlp)
                mlp_pred_proba = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            logging.warning("Validation set is empty for MLP predictions.")


    # --- Ensemble Predictions ---
    logging.info("Ensembling predictions...")
    all_pred_probas = []
    
    # Only add prediction arrays if they are not empty
    if rf_pred_proba.size > 0:
        all_pred_probas.append(rf_pred_proba)
    if xgb_pred_proba.size > 0:
        all_pred_probas.append(xgb_pred_proba)
    if mlp_pred_proba.size > 0:
        all_pred_probas.append(mlp_pred_proba)

    if not all_pred_probas:
        logging.error("No models were successfully trained and made predictions. Cannot ensemble.")
        print("Final Validation Performance: 0.0")
        return

    # Simple average of predicted probabilities from all successfully trained models.
    ensemble_pred_proba = np.mean(all_pred_probas, axis=0)
    
    # Convert ensembled probabilities to binary predictions using a default threshold of 0.5.
    ensemble_y_pred = (ensemble_pred_proba >= 0.5).astype(int)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    if y_val.empty:
        logging.warning("Validation set is empty. F1 score cannot be calculated.")
    else:
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(ensemble_y_pred)

        # Handle cases where the validation set contains only one class.
        if len(unique_y_val) < 2:
            logging.warning(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0 # Perfect score if all predictions match the single class.
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set.
            # `zero_division=0` handles cases where a class has no true instances or no predicted instances,
            # assigning 0 to its F1 score contribution.
            final_validation_score = f1_score(y_val, ensemble_y_pred, average='macro', zero_division=0)

    print(f"Final Validation Performance: {final_validation_score}")

# Run the training and validation process
if __name__ == "__main__":
    run_training_and_validation()
