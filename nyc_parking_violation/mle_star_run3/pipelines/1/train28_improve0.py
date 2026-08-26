
import pandas as pd
import numpy as np
import os
import argparse
import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import catboost as ctb
import xgboost as xgb
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads data from a CSV file, handling potential file not found errors."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    logging.info(f"Loading data from {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def create_and_apply_features(df_train, df_val, df_test=None):
    """
    Creates features on the training data and applies them consistently to validation and test sets.
    This function is designed to be called inside a cross-validation loop to prevent data leakage.
    """
    # --- 1. Target-based Aggregations (Calculated ONLY on df_train) ---
    street_agg = df_train.groupby('Street Name')['violation_count'].agg(['mean', 'std', 'sum']).rename(columns=lambda x: f'street_{x}')
    violation_agg = df_train.groupby('Violation Description')['violation_count'].agg(['mean', 'std', 'sum']).rename(columns=lambda x: f'violation_{x}')

    # Global mean for filling NaNs for unseen groups
    global_mean = df_train['violation_count'].mean()
    global_street_mean = street_agg['street_mean'].mean()
    global_street_std = street_agg['street_std'].mean()
    global_street_sum = street_agg['street_sum'].mean()
    global_violation_mean = violation_agg['violation_mean'].mean()
    global_violation_std = violation_agg['violation_std'].mean()
    global_violation_sum = violation_agg['violation_sum'].mean()


    # --- 2. Apply to all dataframes ---
    dfs_to_process = [df_train, df_val]
    if df_test is not None:
        dfs_to_process.append(df_test)

    processed_dfs = []
    for i, df in enumerate(dfs_to_process):
        is_train = (i == 0)
        # Make copies to avoid modifying original dataframes
        df_featured = df.copy()

        # --- Label Encoding (fit on train, transform others) ---
        if is_train:
            # For training data, we fit and transform
            street_encoder = LabelEncoder().fit(df_featured['Street Name'])
            violation_encoder = LabelEncoder().fit(df_featured['Violation Description'])
        
        # Transform using the encoders fitted on the training fold
        df_featured['Street Name_encoded'] = df_featured['Street Name'].map(lambda s: '-1' if s not in street_encoder.classes_ else s).map(dict(zip(street_encoder.classes_, street_encoder.transform(street_encoder.classes_))))
        df_featured['Violation Description_encoded'] = df_featured['Violation Description'].map(lambda s: '-1' if s not in violation_encoder.classes_ else s).map(dict(zip(violation_encoder.classes_, violation_encoder.transform(violation_encoder.classes_))))
        
        # --- Merge Aggregated Features ---
        df_featured = pd.merge(df_featured, street_agg, on='Street Name', how='left')
        df_featured = pd.merge(df_featured, violation_agg, on='Violation Description', how='left')
        
        # Fill NaNs for groups not seen in the training fold
        df_featured['street_mean'].fillna(global_street_mean, inplace=True)
        df_featured['street_std'].fillna(global_street_std, inplace=True)
        df_featured['street_sum'].fillna(global_street_sum, inplace=True)
        df_featured['violation_mean'].fillna(global_violation_mean, inplace=True)
        df_featured['violation_std'].fillna(global_violation_std, inplace=True)
        df_featured['violation_sum'].fillna(global_violation_sum, inplace=True)

        # --- Simple Interaction & Frequency Features ---
        df_featured['street_freq'] = df_featured['Street Name'].map(df_train['Street Name'].value_counts(normalize=True)).fillna(0)
        df_featured['violation_freq'] = df_featured['Violation Description'].map(df_train['Violation Description'].value_counts(normalize=True)).fillna(0)
        
        processed_dfs.append(df_featured)
        
    if df_test is not None:
        return processed_dfs[0], processed_dfs[1], processed_dfs[2]
    return processed_dfs[0], processed_dfs[1]


def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations.")
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    default_train_path = os.path.join(script_dir, 'input', 'violations_per_street_2022.csv')

    parser.add_argument('--train-path', type=str, default=default_train_path, help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None, help='Optional path to the test data CSV file.')
    args = parser.parse_args()

    df_train_full = load_data(args.train_path)
    if df_train_full is None:
        return

    df_train_full.rename(columns=lambda x: x.strip(), inplace=True)
    
    # --- Cross-Validation Setup ---
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    groups = pd.factorize(df_train_full['Street Name'])[0]
    
    oof_preds_cb = np.zeros(len(df_train_full))
    oof_preds_xgb = np.zeros(len(df_train_full))
    
    models_cb = []
    models_xgb = []

    logging.info("Starting cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train_full, groups=groups)):
        logging.info(f"--- Fold {fold+1}/{n_splits} ---")
        train_fold_df = df_train_full.iloc[train_idx]
        val_fold_df = df_train_full.iloc[val_idx]

        # Create features in a leak-proof way
        train_fold_featured, val_fold_featured = create_and_apply_features(train_fold_df, val_fold_df)

        features = [col for col in train_fold_featured.columns if col not in ['Street Name', 'Violation Description', 'violation_count']]
        categorical_features = ['Street Name_encoded', 'Violation Description_encoded']
        target = 'violation_count'
        
        X_train, y_train = train_fold_featured[features], train_fold_featured[target]
        X_val, y_val = val_fold_featured[features], val_fold_featured[target]

        # --- CatBoost Model ---
        cb_model = ctb.CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=8, loss_function='RMSE',
                                         eval_metric='RMSE', random_seed=42, verbose=0, subsample=0.8,
                                         allow_writing_files=False)
        cb_model.fit(X_train, y_train, cat_features=categorical_features,
                     eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True)
        oof_preds_cb[val_idx] = cb_model.predict(X_val)
        models_cb.append(cb_model)

        # --- XGBoost Model ---
        X_train_xgb, X_val_xgb = X_train.copy(), X_val.copy()
        for col in categorical_features:
            X_train_xgb[col] = X_train_xgb[col].astype("category")
            X_val_xgb[col] = X_val_xgb[col].astype("category")
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1500, learning_rate=0.03,
                                     max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     n_jobs=-1, tree_method='hist', enable_categorical=True)
        xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], early_stopping_rounds=50, verbose=False)
        oof_preds_xgb[val_idx] = xgb_model.predict(X_val_xgb)
        models_xgb.append(xgb_model)

    # --- Validation Performance ---
    oof_preds_ensemble = (oof_preds_cb + oof_preds_xgb) / 2
    oof_preds_ensemble[oof_preds_ensemble < 0] = 0
    final_validation_score = np.sqrt(mean_squared_error(df_train_full[target], oof_preds_ensemble))
    logging.info(f"OOF Ensemble RMSE: {final_validation_score}")
    print(f"Final Validation Performance: {final_validation_score}")

    # --- Inference on Test Set ---
    if args.test_path:
        logging.info("--- Starting Inference on Test Set ---")
        df_test_orig = load_data(args.test_path)
        if df_test_orig is None:
            return

        df_test_orig.rename(columns=lambda x: x.strip(), inplace=True)
        has_target = target in df_test_orig.columns
        if has_target:
            y_test_true = df_test_orig[target]

        train_keys = set(zip(df_train_full['Street Name'], df_train_full['Violation Description']))
        test_keys = set(zip(df_test_orig['Street Name'], df_test_orig['Violation Description']))
        unseen_keys_count = len(test_keys - train_keys)
        logging.info(f"Number of unseen (street, violation) pairs in test set: {unseen_keys_count}")
        
        # For inference, create features using the FULL training data
        _, _, test_featured = create_and_apply_features(df_train_full, df_train_full.head(1), df_test_orig)
        X_test = test_featured[features]

        # --- Predict ---
        test_preds_cb = np.zeros(len(X_test))
        test_preds_xgb = np.zeros(len(X_test))

        for model in models_cb:
            test_preds_cb += model.predict(X_test) / len(models_cb)
        
        X_test_xgb = X_test.copy()
        for col in categorical_features:
            X_test_xgb[col] = X_test_xgb[col].astype("category")

        for model in models_xgb:
            test_preds_xgb += model.predict(X_test_xgb) / len(models_xgb)

        test_preds_ensemble = (test_preds_cb + test_preds_xgb) / 2
        test_preds_ensemble[test_preds_ensemble < 0] = 0

        if has_target:
            test_rmse = np.sqrt(mean_squared_error(y_test_true, test_preds_ensemble))
            logging.info(f"Test Set RMSE: {test_rmse}")

        submission = df_test_orig[['Street Name', 'Violation Description']].copy()
        submission.rename(columns={'Street Name': 'street_name', 'Violation Description': 'violation_type'}, inplace=True)
        submission['predicted_count'] = test_preds_ensemble.round()
        
        submission_path = 'submission.csv'
        submission.to_csv(submission_path, index=False)
        logging.info(f"Submission file created at {submission_path}")

if __name__ == '__main__':
    main()
