
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, test_path=None):
    """Loads, preprocesses, and prepares data for all models."""
    # --- 1. Load Data ---
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    # Load augmentation data
    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError as e:
        print(f"Warning: Augmentation data not found, proceeding without it. {e}")
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])

    # --- 2. Feature Engineering & Merging ---
    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    # Impute missing values for the 'borough' column first
    full_df['borough'].fillna('Unknown', inplace=True)

    # Identify potential features from physical_df and handle their imputation
    potential_physical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    imputation_medians = {}
    newly_discovered_cat_features = []

    for col in potential_physical_cols:
        if col in full_df.columns:
            # FIX: Check if the column is numeric before trying to calculate median
            if pd.api.types.is_numeric_dtype(full_df[col]):
                median_val = full_df[col].median()
                imputation_medians[col] = median_val
                full_df[col].fillna(median_val, inplace=True)
            else:
                # If not numeric, treat it as categorical. This resolves the TypeError.
                full_df[col].fillna('Unknown', inplace=True)
                if col not in newly_discovered_cat_features:
                    newly_discovered_cat_features.append(col)

    # Define the complete list of categorical features
    cat_features = ['street_name', 'violation_type', 'borough'] + newly_discovered_cat_features

    # Cast all identified categorical features to 'category' dtype
    for col in cat_features:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    # --- 3. Handle Test Set (if provided) ---
    test_df_processed = None
    unseen_keys_count = 0
    test_ground_truth = None

    if test_path:
        test_df = pd.read_csv(test_path)
        test_df = clean_col_names(test_df)
        if 'violation_count' in test_df.columns:
            test_ground_truth = test_df['violation_count'].copy()

        # Identify keys in test but not in train
        train_keys = set(full_df.apply(lambda row: f"{row['street_name']}_{row['violation_type']}", axis=1))
        test_keys = set(test_df.apply(lambda row: f"{row['street_name']}_{row['violation_type']}", axis=1))
        new_keys = test_keys - train_keys
        unseen_keys_count = len(new_keys)

        # Prepare test data for prediction
        test_df_processed = test_df.copy()
        
        # Merge with augmentation features
        test_df_processed = pd.merge(test_df_processed, boroughs_df, on='street_name', how='left')
        test_df_processed = pd.merge(test_df_processed, physical_df, on='street_name', how='left')

        # Impute missing values using medians and modes from training data
        test_df_processed['borough'].fillna('Unknown', inplace=True)
        
        # Impute numeric columns using stored medians
        for col, median_val in imputation_medians.items():
            if col in test_df_processed.columns:
                test_df_processed[col].fillna(median_val, inplace=True)
                # Coerce to numeric in case of mixed types, filling any new NaNs with the median
                test_df_processed[col] = pd.to_numeric(test_df_processed[col], errors='coerce').fillna(median_val)
            else:
                test_df_processed[col] = median_val

        # Impute newly discovered categorical columns
        for col in newly_discovered_cat_features:
            if col in test_df_processed.columns:
                test_df_processed[col].fillna('Unknown', inplace=True)
            else:
                test_df_processed[col] = 'Unknown'
        
        # Cast all categorical features to 'category' dtype for the test set
        for col in cat_features:
            if col in test_df_processed.columns:
                 test_df_processed[col] = test_df_processed[col].astype('category')

    return full_df, test_df_processed, unseen_keys_count, test_ground_truth, cat_features

def main():
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction with Ensemble Model")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv')
    parser.add_argument('--test-path', type=str, default=None)
    args = parser.parse_args()

    # --- 1. Data Preparation ---
    print("Loading and preparing data...")
    train_data, test_data, unseen_keys_count, test_ground_truth, cat_features = \
        load_and_prepare_data(args.train_path, args.test_path)

    # Create log-transformed target
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- 3. Model Training ---
    # Model 1 (CatBoost Base): Predicts original target
    print("Training CatBoost Model (predicting original count)...")
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 200,
        'cat_features': cat_features,
        'early_stopping_rounds': 50,
        'task_type': 'CPU',
        'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    # Model 2 (CatBoost Log): Predicts log-transformed target
    print("\nTraining CatBoost Model (predicting log-transformed count)...")
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Model 3 (XGBoost Log): Predicts log-transformed target
    print("\nTraining XGBoost Model (predicting log-transformed count)...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    # Ensure categorical features are correctly handled by XGBoost
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for col in cat_features:
        if col in X_train_xgb.columns:
            X_train_xgb[col] = X_train_xgb[col].astype("category")
            X_val_xgb[col] = X_val_xgb[col].astype("category")

    xgb_model.fit(
        X_train_xgb, y_train_log,
        eval_set=[(X_val_xgb, y_val_log)],
        verbose=False
    )

    # --- 4. Validation Performance & Ensembling ---
    print("\nEnsembling models and evaluating on validation set...")
    from scipy.optimize import minimize

    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))

    # Stack validation predictions for optimization
    val_predictions_stacked = np.column_stack([
        val_preds_cat_base,
        val_preds_cat_log_transformed,
        val_preds_xgb_log_transformed
    ])

    # Define the objective function to minimize (RMSE)
    def rmse_objective(weights, predictions, true_values):
        """Calculates RMSE for a weighted ensemble."""
        weighted_preds = np.dot(predictions, weights)
        weighted_preds = np.maximum(0, weighted_preds) # Ensure non-negativity
        return np.sqrt(mean_squared_error(true_values, weighted_preds))

    # Initial guess for weights
    initial_weights = np.array([1/3, 1/3, 1/3])
    
    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: each weight must be between 0 and 1
    bounds = [(0, 1)] * 3
    
    # Find the optimal weights
    print("Optimizing ensemble weights...")
    result = minimize(
        rmse_objective,
        initial_weights,
        args=(val_predictions_stacked, y_val_base),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x
    print(f"Optimal Weights: {optimal_weights}")

    # Ensemble with optimized weights
    ensemble_predictions = np.dot(val_predictions_stacked, optimal_weights)
    ensemble_predictions = np.maximum(0, ensemble_predictions) # Ensure non-negativity

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Final Validation Performance: {val_rmse:.4f}")

    # --- 5. Test Set Prediction ---
    if args.test_path and test_data is not None:
        print(f"\nProcessing test file: {args.test_path}")
        print(f"Found {unseen_keys_count} (street_name, violation_type) pairs in test set not present in training set.")
        
        # Ensure test_data has the same columns in the same order as X_train
        test_data_aligned = test_data[features].copy()
        
        # Prepare data for XGBoost (ensure category types)
        test_data_xgb = test_data_aligned.copy()
        for col in cat_features:
            if col in test_data_xgb.columns:
                test_data_xgb[col] = test_data_xgb[col].astype("category")

        # Predict with all three models
        test_preds_cat_base = model_cat_base.predict(test_data_aligned)
        test_preds_cat_log = model_cat_log.predict(test_data_aligned)
        test_preds_xgb_log = xgb_model.predict(test_data_xgb)
        
        # Inverse transform
        test_preds_cat_log_transformed = np.expm1(test_preds_cat_log)
        test_preds_xgb_log_transformed = np.expm1(test_preds_xgb_log)
        
        # Stack test predictions
        test_predictions_stacked = np.column_stack([
            test_preds_cat_base,
            test_preds_cat_log_transformed,
            test_preds_xgb_log_transformed
        ])
        
        # Ensemble using the optimal weights found on the validation set
        ensemble_test_preds = np.dot(test_predictions_stacked, optimal_weights)
        
        # Create submission file
        submission_df = test_data[['street_name', 'violation_type']].copy()
        
        # Post-process predictions
        submission_df['predicted_count'] = np.maximum(0, ensemble_test_preds) # Clip at 0
        submission_df['predicted_count'] = submission_df['predicted_count'].round()

        submission_df.to_csv("submission.csv", index=False)
        print("Generated submission.csv")
        
        # Score if ground truth is available
        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, submission_df['predicted_count']))
            print(f"Test Set RMSE: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
