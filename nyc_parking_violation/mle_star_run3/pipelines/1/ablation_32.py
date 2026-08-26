
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_dummy_data():
    """Creates dummy data files for a self-contained, runnable script."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # Main training data
    train_data = {
        'Street Name': ['A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St', 'A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St'],
        'Violation Description': ['Parking', 'Speeding', 'Parking', 'Speeding', 'Parking', 'Speeding', 'Parking', 'Jaywalking', 'Jaywalking', 'Jaywalking', 'Jaywalking', 'Jaywalking', 'Jaywalking', 'Jaywalking'],
        'Violation Count': [10, 5, 20, 8, 15, 3, 25, 2, 3, 4, 1, 2, 5, 3]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    # Augmentation data 1
    borough_data = {
        'Street Name': ['A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St'],
        'Borough': ['Brooklyn', 'Manhattan', 'Brooklyn', 'Queens', 'Manhattan', 'Bronx', 'Queens']
    }
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    # Augmentation data 2
    physical_data = {
        'Street Name': ['A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St'],
        'Street Width': [30, 50, 35, 40, 55, 25, 45],
        'Pavement Quality': [8, 5, 7, 9, 4, 8, 6],
        'Num Lanes': [2, 4, 2, 3, 4, 1, 3]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)


def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df


def load_and_prepare_data(train_path):
    """Loads, preprocesses, and prepares data for all models."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features


def run_experiment(use_cat_base_in_stack=True, use_cat_log_in_stack=True):
    """
    Runs a single experiment variation based on the provided flags.
    Args:
        use_cat_base_in_stack (bool): Whether to include the base CatBoost model in the stack.
        use_cat_log_in_stack (bool): Whether to include the log CatBoost model in the stack.
    Returns:
        float: The validation RMSE of the experiment.
    """
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]

    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.3, random_state=42
    )

    # --- 3. Model Training (All base models are always trained) ---
    cat_params = {
        'iterations': 500, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'random_seed': 42, 'verbose': 0, 'cat_features': cat_features,
        'early_stopping_rounds': 20, 'depth': 8
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=20, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Stacking & Evaluation ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))

    # Build meta-features based on ablation flags
    meta_features_list = []
    if use_cat_base_in_stack:
        meta_features_list.append(val_preds_cat_base)
    if use_cat_log_in_stack:
        meta_features_list.append(val_preds_cat_log_transformed)
    
    # XGBoost is always in the stack for this study
    meta_features_list.append(val_preds_xgb_log_transformed)
    
    meta_features_val = np.column_stack(meta_features_list)

    # Train Ridge meta-learner
    meta_learner = Ridge(alpha=1.0, random_state=42)
    meta_learner.fit(meta_features_val, y_val_base)
    ensemble_predictions = meta_learner.predict(meta_features_val)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse


if __name__ == '__main__':
    # Create dummy data for reproducibility
    create_dummy_data()

    print("Running ablation study on Stacking Ensemble Composition...\n")
    results = {}

    # --- Baseline ---
    # Full stack: [CatBoost_Base, CatBoost_Log, XGBoost_Log]
    baseline_rmse = run_experiment(
        use_cat_base_in_stack=True,
        use_cat_log_in_stack=True
    )
    results['Baseline'] = baseline_rmse
    print(f"Baseline (Full Stack) RMSE: {baseline_rmse:.4f}")

    # --- Ablation 1: No CatBoost Base Model ---
    # Stack: [CatBoost_Log, XGBoost_Log]
    no_cat_base_rmse = run_experiment(
        use_cat_base_in_stack=False,
        use_cat_log_in_stack=True
    )
    results['No CatBoost Base Model'] = no_cat_base_rmse
    impact_no_cat_base = no_cat_base_rmse - baseline_rmse
    print(f"Ablation 'No CatBoost Base Model' RMSE: {no_cat_base_rmse:.4f} (Impact: {impact_no_cat_base:+.4f})")

    # --- Ablation 2: No CatBoost Log Model ---
    # Stack: [CatBoost_Base, XGBoost_Log]
    no_cat_log_rmse = run_experiment(
        use_cat_base_in_stack=True,
        use_cat_log_in_stack=False
    )
    results['No CatBoost Log Model'] = no_cat_log_rmse
    impact_no_cat_log = no_cat_log_rmse - baseline_rmse
    print(f"Ablation 'No CatBoost Log Model' RMSE: {no_cat_log_rmse:.4f} (Impact: {impact_no_cat_log:+.4f})")

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    impacts = {
        'CatBoost Base Model': abs(impact_no_cat_base),
        'CatBoost Log Model': abs(impact_no_cat_log)
    }

    most_impactful_component = max(impacts, key=impacts.get)
    print(f"The most impactful component is the '{most_impactful_component}'.")
    print("Removing it caused the largest change in validation RMSE, demonstrating its importance to the stacking ensemble's performance.")

