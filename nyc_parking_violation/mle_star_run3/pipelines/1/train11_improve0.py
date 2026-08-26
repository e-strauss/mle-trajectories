
import argparse
import os
import pandas as pd
import numpy as np
import logging
import sys
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Try to import heavy ML libraries and provide guidance if they are missing
try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please install the required libraries: pip install lightgbm xgboost catboost scikit-learn pandas numpy")
    sys.exit(1)

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Functions ---

def load_data(file_path):
    """Loads a CSV file with robust error handling."""
    if not os.path.exists(file_path):
        logging.error(f"File not found at path: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def augment_features(df, input_dir):
    """
    Augments the main dataframe with features from auxiliary datasets.
    Handles optional auxiliary files gracefully.
    """
    df_augmented = df.copy()
    
    # Feature 1: Camera counts per street
    camera_path = os.path.join(input_dir, 'physical_camera_locations.csv')
    if os.path.exists(camera_path):
        try:
            df_camera = pd.read_csv(camera_path)
            # Standardize street names for joining
            df_camera['Street Name'] = df_camera['Street Name'].str.upper().str.strip()
            camera_counts = df_camera.groupby('Street Name').size().reset_index(name='camera_count')
            
            # Merge with main data
            df_augmented = pd.merge(df_augmented, camera_counts, on='Street Name', how='left')
            df_augmented['camera_count'] = df_augmented['camera_count'].fillna(0)
            logging.info("Augmented with camera location data.")
        except Exception as e:
            logging.warning(f"Could not process camera data from {camera_path}: {e}. Skipping.")
            df_augmented['camera_count'] = 0
    else:
        logging.warning(f"Camera data file not found at {camera_path}. Skipping augmentation.")
        df_augmented['camera_count'] = 0

    # Feature 2: Parking regulation counts per street
    regulations_path = os.path.join(input_dir, 'dot_parking_regulations.csv')
    if os.path.exists(regulations_path):
        try:
            df_regs = pd.read_csv(regulations_path)
            # Standardize street names
            df_regs['main_street'] = df_regs['main_street'].str.upper().str.strip()
            reg_counts = df_regs.groupby('main_street').size().reset_index(name='regulation_count')
            
            # Merge with main data
            df_augmented = pd.merge(df_augmented, reg_counts, left_on='Street Name', right_on='main_street', how='left')
            df_augmented['regulation_count'] = df_augmented['regulation_count'].fillna(0)
            df_augmented.drop(columns=['main_street'], inplace=True, errors='ignore')
            logging.info("Augmented with parking regulation data.")
        except Exception as e:
            logging.warning(f"Could not process regulation data from {regulations_path}: {e}. Skipping.")
            df_augmented['regulation_count'] = 0
    else:
        logging.warning(f"Regulation data file not found at {regulations_path}. Skipping augmentation.")
        df_augmented['regulation_count'] = 0
        
    return df_augmented

def preprocess(df, encoders=None, is_train=True):
    """
    Preprocesses the data: handles categorical features and ensures consistency.
    """
    df_processed = df.copy()
    
    # Standardize column names
    df_processed.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df_processed.columns]
    
    # Standardize text data
    df_processed['street_name'] = df_processed['street_name'].str.upper().str.strip()
    df_processed['violation_description'] = df_processed['violation_description'].str.upper().str.strip()

    # Handle categorical features
    cat_features = ['street_name', 'violation_description']
    
    if is_train:
        encoders = {col: LabelEncoder() for col in cat_features}
        for col in cat_features:
            df_processed[col] = encoders[col].fit_transform(df_processed[col])
    else:
        for col in cat_features:
            # Handle unseen labels in test data
            le = encoders[col]
            unseen_mask = ~df_processed[col].isin(le.classes_)
            logging.info(f"Found {unseen_mask.sum()} unseen values in '{col}'.")
            
            # Add unseen labels to the encoder
            if unseen_mask.sum() > 0:
                le.classes_ = np.append(le.classes_, df_processed.loc[unseen_mask, col].unique())

            df_processed[col] = le.transform(df_processed[col])

    return df_processed, encoders

def train_and_evaluate(args):
    """Main training and evaluation workflow."""
    input_dir = os.path.dirname(args.train_path)

    # 1. Load Training Data
    df_train_raw = load_data(args.train_path)
    if df_train_raw is None:
        return

    # Subsampling if specified
    if args.subsample < 1.0:
        logging.info(f"Subsampling training data to {args.subsample * 100}%.")
        df_train_raw = df_train_raw.sample(frac=args.subsample, random_state=42)

    # 2. Preprocess and Augment
    df_train_processed, encoders = preprocess(df_train_raw, is_train=True)
    df_train_augmented = augment_features(df_train_processed, input_dir)
    
    # Define features and target
    features = [col for col in df_train_augmented.columns if col not in ['violation_count', 'street_name_orig', 'violation_description_orig']]
    categorical_features_indices = [features.index(c) for c in ['street_name', 'violation_description'] if c in features]
    
    X = df_train_augmented[features]
    y = df_train_augmented['violation_count']
    groups = df_train_augmented['street_name'] # Use street for grouped validation

    # 3. Model Training
    logging.info("Starting model training...")

    # Define base models
    base_models = [
        ('lgbm', lgb.LGBMRegressor(random_state=42)),
        ('xgb', xgb.XGBRegressor(random_state=42, objective='reg:squarederror')),
        ('catboost', cb.CatBoostRegressor(random_state=42, verbose=0, cat_features=categorical_features_indices, allow_writing_files=False))
    ]

    # Define stacking ensemble
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(),
        cv=GroupKFold(n_splits=args.folds) # Use GroupKFold within the stacker
    )

    # 4. Validation
    gkf = GroupKFold(n_splits=args.folds)
    oof_preds = np.zeros(len(X))
    
    logging.info(f"Performing {args.folds}-fold GroupKFold validation...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logging.info(f"--- Fold {fold+1}/{args.folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        stacking_model.fit(X_train, y_train)
        preds = stacking_model.predict(X_val)
        oof_preds[val_idx] = preds
    
    # Clip predictions to be non-negative
    oof_preds[oof_preds < 0] = 0
    final_validation_score = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"Final Validation Performance: {final_validation_score}")
    logging.info(f"Cross-validation complete. Overall RMSE: {final_validation_score}")

    # 5. Full Retraining & Prediction (if test_path is provided)
    if args.test_path:
        logging.info("Retraining on full dataset and generating predictions...")
        
        # Retrain the model on all 2022 data
        stacking_model.fit(X, y)
        
        # Load and process test data
        df_test_raw = load_data(args.test_path)
        if df_test_raw is None:
            return

        df_test_processed, _ = preprocess(df_test_raw, encoders=encoders, is_train=False)
        df_test_augmented = augment_features(df_test_processed, input_dir)
        
        X_test = df_test_augmented[features]
        
        # Predict
        predictions = stacking_model.predict(X_test)
        predictions[predictions < 0] = 0
        
        # Create submission file
        submission_df = pd.DataFrame({
            'street_name': df_test_raw['Street Name'],
            'violation_type': df_test_raw['Violation Description'],
            'predicted_count': predictions.round().astype(int)
        })
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        logging.info(f"Submission file created at {submission_path}")

        # Score if ground truth is available in the test file
        if 'violation_count' in df_test_augmented.columns:
            test_rmse = np.sqrt(mean_squared_error(df_test_augmented['violation_count'], predictions))
            logging.info(f"Test Set RMSE: {test_rmse}")

def main():
    """Main function to parse arguments and orchestrate the pipeline."""
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction Pipeline")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional: Path to the test/evaluation data CSV file.')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds for GroupKFold cross-validation.')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Fraction of training data to use (e.g., 0.1 for 10%).')
    
    args = parser.parse_args()

    try:
        train_and_evaluate(args)
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        # No sys.exit(1) as per instructions

if __name__ == '__main__':
    main()
