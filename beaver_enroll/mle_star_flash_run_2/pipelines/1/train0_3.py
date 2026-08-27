
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import subprocess
import sys

# Install necessary libraries if not already installed
try:
    import xgboost
except ImportError:
    print("xgboost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost
from xgboost import XGBClassifier

try:
    import catboost
except ImportError:
    print("catboost not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    import catboost
from catboost import CatBoostClassifier


# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
def load_and_preprocess_data(data_dir, gold_file):
    """
    Loads gold labels and merges with subject summary data, performs feature engineering,
    and prepares data for multiple models. This integrates elements from both base
    and reference solutions, preparing features for a scikit-learn pipeline
    and native CatBoost handling.
    """
    gold_df = pd.read_csv(gold_file)

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    gold_df['HIGH_ENROLLMENT'] = gold_df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    # Try to find a central table containing course offering details
    # Prioritize 'subject_summary.csv' as mentioned in problem description context,
    # falling back to 'course_offerings.csv' if not found.
    main_df = None
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    course_offerings_path = os.path.join(data_dir, 'course_offerings.csv')

    if os.path.exists(subject_summary_path):
        main_df = pd.read_csv(subject_summary_path, low_memory=False)
        print(f"Loaded subject_summary.csv with {len(main_df)} rows as main_table.")
    elif os.path.exists(course_offerings_path):
        main_df = pd.read_csv(course_offerings_path, low_memory=False)
        print(f"Loaded course_offerings.csv with {len(main_df)} rows as main_table.")
    else:
        warnings.warn("Neither subject_summary.csv nor course_offerings.csv found. Proceeding with minimal features from gold_df.")

    # Prepare the base dataframe, starting with gold_df
    df = gold_df.copy()

    # Identifier columns used for merging and to be excluded from direct feature selection
    identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
    
    # Merge gold_df (which now contains numerical 'HIGH_ENROLLMENT') with main_df if available
    if main_df is not None:
        # Ensure join keys are present in both dataframes
        for key in identifier_cols:
            if key not in main_df.columns:
                warnings.warn(f"Required join key '{key}' not found in the main data frame. Cannot merge with gold labels using this key.")
                main_df = None # Abort merge if keys are missing
                break
            if key not in df.columns:
                warnings.warn(f"Required join key '{key}' not found in gold labels. Cannot merge.")
                main_df = None # Abort merge if keys are missing
                break
        
        if main_df is not None:
            # Ensure gold_labels keys are unique for merging if there are duplicates
            df = df.drop_duplicates(subset=identifier_cols)
            df = pd.merge(df, main_df, on=identifier_cols, how='left')
            print(f"Merged data has {len(df)} rows after joining with main table.")

    if df.empty:
        warnings.warn("Merged DataFrame is empty. Cannot proceed with feature engineering.")
        # Add a dummy feature if df is empty to prevent further errors
        df['HIGH_ENROLLMENT'] = 0
        df['DUMMY_FEATURE'] = 0
        df._numerical_features_sklearn = ['DUMMY_FEATURE']
        df._categorical_features_sklearn = []
        df._catboost_numerical_features = ['DUMMY_FEATURE']
        df._catboost_categorical_features = []
        return df

    # --- Feature Engineering (from reference solution) ---
    processed_df = df.copy()

    # Ensure TERM_CODE is treated as string for slicing
    if 'TERM_CODE' in processed_df.columns:
        processed_df['TERM_CODE_STR'] = processed_df['TERM_CODE'].astype(str)
        processed_df['TERM_YEAR'] = processed_df['TERM_CODE_STR'].str[:4].astype(int)
        # FIX: The original code tried to convert TERM_SEMESTER to int, which fails for non-numeric semesters like 'FA'.
        # Keep TERM_SEMESTER as a string/object type for categorical encoding.
        processed_df['TERM_SEMESTER'] = processed_df['TERM_CODE_STR'].str[4:] 
        # print("Generated 'TERM_YEAR' and 'TERM_SEMESTER' features.")
    else:
        warnings.warn("'TERM_CODE' column not found, cannot generate time-based features.")

    if 'ENROLLMENT_COUNT' in processed_df.columns and 'CAPACITY' in processed_df.columns:
        processed_df['CAPACITY_ADJ'] = processed_df['CAPACITY'].replace(0, np.nan) 
        processed_df['FILL_RATE'] = processed_df['ENROLLMENT_COUNT'] / processed_df['CAPACITY_ADJ']
        processed_df['FILL_RATE'] = processed_df['FILL_RATE'].replace([np.inf, -np.inf], np.nan)
        processed_df.drop(columns=['CAPACITY_ADJ'], inplace=True, errors='ignore')
        # print("Created 'FILL_RATE' feature.")
    else:
        warnings.warn("Columns 'ENROLLMENT_COUNT' or 'CAPACITY' not found for 'FILL_RATE' feature generation.")

    # Convert object columns that contain mostly numeric data to numeric
    for col in processed_df.select_dtypes(include='object').columns:
        # Check if a significant portion of non-null values can be converted to numeric
        is_numeric_like = pd.to_numeric(processed_df[col], errors='coerce').notna().sum() / processed_df[col].count() > 0.8
        if is_numeric_like:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            # print(f"Converted '{col}' to numeric type.")
        
    df = processed_df # Update df with engineered features

    # --- Feature preparation for ColumnTransformer (RF/XGB) and native CatBoost ---
    
    # Drop rows where HIGH_ENROLLMENT is missing (shouldn't happen with initial gold_df but good practice)
    df.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    
    # Exclude ID columns, target, and temporary helper columns from features
    id_cols_to_exclude = ['TERM_CODE', 'SUBJECT_ID_SORT', 'TERM_CODE_STR'] # TERM_CODE_STR is temporary
    target_col = 'HIGH_ENROLLMENT'
    
    # Automatically identify features for the models
    # Exclude columns with too many unique values (potential IDs) or non-numeric/non-categorical types
    candidate_features = [
        col for col in df.columns 
        if col not in id_cols_to_exclude + [target_col] 
        and not pd.api.types.is_datetime64_any_dtype(df[col])
        and df[col].nunique() > 1 # Exclude columns with only one unique value (constant columns)
    ]
    
    # Refine numerical and categorical features for sklearn pipeline (RF/XGB)
    numerical_features_sklearn = df[candidate_features].select_dtypes(include=np.number).columns.tolist()
    categorical_features_sklearn = df[candidate_features].select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out high cardinality categorical features for OneHotEncoder
    initial_categorical_features_count = len(categorical_features_sklearn)
    categorical_features_sklearn = [
        f for f in categorical_features_sklearn 
        if df[f].nunique() <= 50 and df[f].nunique() < len(df) * 0.1
    ]
    if len(categorical_features_sklearn) < initial_categorical_features_count:
        warnings.warn(f"Filtered out {initial_categorical_features_count - len(categorical_features_sklearn)} high cardinality categorical features for sklearn pipeline.")

    # For CatBoost, numerical features are straightforward, but categorical can include high-cardinality ones.
    # We use original `identifier_cols` and other object/category columns as native categoricals.
    catboost_numerical_features = df[candidate_features].select_dtypes(include=np.number).columns.tolist()
    # For CatBoost, we can include more categorical features as it handles them natively.
    catboost_categorical_features = [
        col for col in df.columns
        if col not in id_cols_to_exclude + [target_col]
        and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]))
    ]
    # Also add original identifier columns to catboost_categorical_features
    catboost_categorical_features.extend(identifier_cols)
    catboost_categorical_features = list(set(catboost_categorical_features)) # Remove duplicates

    # Fill NaNs in categorical columns for CatBoost, as it expects non-NaN string/int categories
    for col in catboost_categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Missing_Category')
    
    # Fallback: If no features are identified, add a dummy feature.
    if not numerical_features_sklearn and not categorical_features_sklearn and not catboost_numerical_features and not catboost_categorical_features:
        warnings.warn("No suitable features identified. Adding a dummy feature.")
        df['DUMMY_FEATURE'] = 0
        numerical_features_sklearn.append('DUMMY_FEATURE')
        catboost_numerical_features.append('DUMMY_FEATURE')

    # Store feature column names as attributes on the DataFrame for later use, ensuring uniqueness.
    df._numerical_features_sklearn = list(set(numerical_features_sklearn))
    df._categorical_features_sklearn = list(set(categorical_features_sklearn))
    df._catboost_numerical_features = list(set(catboost_numerical_features))
    df._catboost_categorical_features = list(set(catboost_categorical_features))
    
    return df

# --- Main script ---
def run_training_and_validation():
    df = load_and_preprocess_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE)

    if df.empty or 'HIGH_ENROLLMENT' not in df.columns or df['HIGH_ENROLLMENT'].isnull().all():
        print("Loaded DataFrame is empty or target column is missing/empty. Cannot proceed with training.")
        print("Final Validation Performance: 0.0")
        return

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        warnings.warn("Not enough unique terms for a time-based validation split (at least 2 required). Using a simple random split.")
        # Fallback to random split
        y = df['HIGH_ENROLLMENT']
        # Identify features for random split, prioritize sklearn features
        all_sklearn_features = df._numerical_features_sklearn + df._categorical_features_sklearn
        if not all_sklearn_features: # If no sklearn features, use catboost features
            all_sklearn_features = df._catboost_numerical_features + df._catboost_categorical_features
        if not all_sklearn_features: # If still no features, add dummy
            df['DUMMY_FEATURE'] = 0
            all_sklearn_features = ['DUMMY_FEATURE']

        # If target has only one class, cannot stratify
        stratify_param = y if len(y.unique()) > 1 else None
        
        # We need the full dataframe for further processing, so split the indices or use masks
        train_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=stratify_param)
        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy()

    else:
        # Use the last term for validation to simulate a future term.
        validation_term = unique_terms[-1] 
        train_df = df[df['TERM_CODE'] != validation_term].copy()
        val_df = df[df['TERM_CODE'] == validation_term].copy()

    if train_df.empty or val_df.empty:
        print("Training or validation set is empty after splitting. Cannot proceed.")
        print("Final Validation Performance: 0.0")
        return

    # Retrieve feature lists determined during data loading
    numerical_features_sklearn = getattr(df, '_numerical_features_sklearn', [])
    categorical_features_sklearn = getattr(df, '_categorical_features_sklearn', [])
    catboost_numerical_features = getattr(df, '_catboost_numerical_features', [])
    catboost_categorical_features = getattr(df, '_catboost_categorical_features', [])
    
    # Filter features to ensure they exist in the respective dataframes (X_train/X_val)
    # This helps if a column was dropped during split or something unexpected
    numerical_features_sklearn = [col for col in numerical_features_sklearn if col in train_df.columns]
    categorical_features_sklearn = [col for col in categorical_features_sklearn if col in train_df.columns]
    catboost_numerical_features = [col for col in catboost_numerical_features if col in train_df.columns]
    catboost_categorical_features = [col for col in catboost_categorical_features if col in train_df.columns]

    # Prepare X and y for training
    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']

    # For RandomForest/XGBoost (Scikit-learn pipeline)
    X_train_sklearn = train_df[numerical_features_sklearn + categorical_features_sklearn]
    X_val_sklearn = val_df[numerical_features_sklearn + categorical_features_sklearn]

    # For CatBoost (native handling)
    X_train_catboost = train_df[catboost_numerical_features + catboost_categorical_features]
    X_val_catboost = val_df[catboost_numerical_features + catboost_categorical_features]
    
    if X_train_sklearn.empty and X_train_catboost.empty:
        print("No usable features available for any model. Cannot proceed.")
        print("Final Validation Performance: 0.0")
        return

    # Check for target variable issues
    if len(y_train.unique()) < 2:
        warnings.warn(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    if len(y_val.unique()) < 2:
        warnings.warn(f"Validation set target 'HIGH_ENROLLMENT' has only one class: {y_val.unique()}. This might affect F1 score calculation.")


    # --- Create Preprocessing Pipeline for Scikit-learn models (RF/XGB) ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' is important for unseen categories
    ])

    transformers = []
    if numerical_features_sklearn:
        transformers.append(('num', numerical_transformer, numerical_features_sklearn))
    if categorical_features_sklearn:
        transformers.append(('cat', categorical_transformer, categorical_features_sklearn))

    if not transformers:
        warnings.warn("No features for ColumnTransformer. Scikit-learn models might fail or rely on dummy features.")
        # Add a dummy transformer if no features are available
        X_train_sklearn['DUMMY_FEATURE'] = 0
        X_val_sklearn['DUMMY_FEATURE'] = 0
        numerical_features_sklearn.append('DUMMY_FEATURE')
        transformers.append(('num', numerical_transformer, ['DUMMY_FEATURE']))


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' 
    )

    # --- Model Training ---
    print("Starting model training for ensemble.")

    # Model 1: RandomForestClassifier (from base solution + pipeline)
    rf_pred_proba = np.array([])
    if not X_train_sklearn.empty and not y_train.empty:
        print("Training RandomForestClassifier...")
        rf_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])
        rf_model_pipeline.fit(X_train_sklearn, y_train)
        rf_pred_proba = rf_model_pipeline.predict_proba(X_val_sklearn)[:, 1]
    else:
        warnings.warn("Skipping RandomForestClassifier training due to empty feature set or target.")

    # Model 2: XGBoost Classifier (additional model)
    xgb_pred_proba = np.array([])
    if not X_train_sklearn.empty and not y_train.empty:
        print("Training XGBClassifier...")
        xgb_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                              ('classifier', XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, 
                                                                           n_estimators=500, learning_rate=0.05, use_label_encoder=False))])
        xgb_model_pipeline.fit(X_train_sklearn, y_train)
        xgb_pred_proba = xgb_model_pipeline.predict_proba(X_val_sklearn)[:, 1]
    else:
        warnings.warn("Skipping XGBClassifier training due to empty feature set or target.")

    # Model 3: CatBoost Classifier (with native categorical handling)
    cat_pred_proba = np.array([])
    if not X_train_catboost.empty and not y_train.empty:
        print("Training CatBoostClassifier...")
        # CatBoost needs to know which columns are categorical from X_train_catboost.
        cat_features_indices = [
            X_train_catboost.columns.get_loc(col) 
            for col in catboost_categorical_features 
            if col in X_train_catboost.columns
        ]

        cat_model = CatBoostClassifier(
            iterations=500, learning_rate=0.05, random_seed=42, 
            loss_function='Logloss', eval_metric='F1', verbose=0, early_stopping_rounds=50,
            # Handle class imbalance if target has only one class in train, will cause error.
            # Only set auto_class_weights if both classes are present.
            auto_class_weights='Balanced' if len(y_train.unique()) > 1 else None
        )
        try:
            cat_model.fit(
                X_train_catboost, y_train,
                cat_features=cat_features_indices,
                eval_set=(X_val_catboost, y_val),
                early_stopping_rounds=50
            )
            cat_pred_proba = cat_model.predict_proba(X_val_catboost)[:, 1]
        except Exception as e:
            warnings.warn(f"CatBoost training failed: {e}. Skipping CatBoost predictions.")
            cat_pred_proba = np.array([]) # Ensure it's empty if training fails
    else:
        warnings.warn("Skipping CatBoostClassifier training due to empty feature set or target.")

    # --- Ensemble Predictions ---
    print("Ensembling predictions...")
    all_pred_probas = []
    if rf_pred_proba.size > 0:
        all_pred_probas.append(rf_pred_proba)
    if xgb_pred_proba.size > 0:
        all_pred_probas.append(xgb_pred_proba)
    if cat_pred_proba.size > 0:
        all_pred_probas.append(cat_pred_proba)

    if not all_pred_probas:
        print("No models were successfully trained, cannot ensemble.")
        print("Final Validation Performance: 0.0")
        return

    # Simple average of predicted probabilities from all successfully trained models.
    ensemble_pred_proba = np.mean(all_pred_probas, axis=0)
    
    # Convert ensembled probabilities to binary predictions using a default threshold of 0.5.
    ensemble_y_pred = (ensemble_pred_proba >= 0.5).astype(int)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    if y_val.empty:
        print("Validation set is empty. F1 score cannot be calculated.")
    else:
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(ensemble_y_pred)

        # Handle cases where the validation set contains only one class.
        if len(unique_y_val) < 2:
            warnings.warn(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
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
