
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GroupKFold

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_dummy_data():
    """Creates dummy data files for the script to run."""
    print("Setting up dummy data for ablation study...")
    os.makedirs('./input', exist_ok=True)
    
    train_data_dummy = {
        'street_name': ['STREET A', 'STREET B', 'STREET A', 'STREET C', 'STREET B', 'STREET D', 'STREET A', 'STREET E', 'STREET E', 'STREET F'],
        'violation_description': ['V-TYPE 1', 'V-TYPE 1', 'V-TYPE 2', 'V-TYPE 1', 'V-TYPE 2', 'V-TYPE 1', 'V-TYPE 1', 'V-TYPE 1', 'V-TYPE 2', 'V-TYPE 1'],
        'violation_count': [100, 50, 20, 75, 30, 120, 110, 5, 15, 200]
    }
    pd.DataFrame(train_data_dummy).to_csv('./input/violations_per_street_2022.csv', index=False)

    boroughs_data_dummy = {
        'street_name': ['STREET A', 'STREET B', 'STREET C', 'STREET D', 'STREET E', 'STREET F'],
        'borough': ['Brooklyn', 'Manhattan', 'Queens', 'Brooklyn', 'Manhattan', 'Staten Island']
    }
    pd.DataFrame(boroughs_data_dummy).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data_dummy = {
        'street_name': ['STREET A', 'STREET B', 'STREET C', 'STREET F'],
        'pavement_quality': [0.5, 0.8, 0.2, 0.9],
        'street_width': [10, 20, np.nan, 15] # Add a NaN to test imputation
    }
    pd.DataFrame(physical_data_dummy).to_csv('./input/physical_features_per_street.csv', index=False)
    print("Dummy data created.\n")

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_physical_features=True):
    """Loads and prepares data based on ablation settings."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    
    physical_df = pd.DataFrame(columns=['street_name'])
    if use_physical_features:
        try:
            physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
        except FileNotFoundError:
            pass # Silently fail if not found

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    if use_physical_features:
        full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    if use_physical_features:
        numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
        for col in numerical_cols:
            if col in full_df.columns:
                median_val = full_df[col].median()
                full_df[col].fillna(median_val, inplace=True)
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(use_group_kfold, use_physical_features, xgb_n_estimators):
    """
    Runs a single experiment with a specific configuration.
    
    Ablation Components:
    1. use_group_kfold (bool): If True, uses GroupKFold for validation to prevent data leakage.
       If False, uses a standard random train_test_split.
    2. use_physical_features (bool): If True, merges and uses the physical features dataset.
    3. xgb_n_estimators (int): Sets the number of estimators for the XGBoost model.
    """
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        use_physical_features=use_physical_features
    )
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features_to_use = [c for c in train_data.columns if c not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features_to_use]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    if use_group_kfold:
        groups = train_data['street_name']
        gkf = GroupKFold(n_splits=3)
        train_idx, val_idx = next(gkf.split(X, y_base, groups=groups))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_base, y_val_base = y_base.iloc[train_idx], y_base.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]
    else:
        # Standard random split
        X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
            X, y_base, y_log, test_size=0.33, random_state=42
        )

    # --- 3. Model Training ---
    cat_params = {
        'iterations': 500, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'random_seed': 42, 'verbose': 0, 'cat_features': cat_features,
        'early_stopping_rounds': 20, 'depth': 8
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base))
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log))

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=xgb_n_estimators, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=20, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

if __name__ == '__main__':
    setup_dummy_data()
    results = {}

    # --- Baseline Experiment ---
    # The baseline uses the most robust configuration: GroupKFold split, all features, and a high number of estimators.
    print("--- Running Baseline (GroupKFold, Physical Features, High Estimators) ---")
    results['Baseline'] = run_experiment(
        use_group_kfold=True,
        use_physical_features=True,
        xgb_n_estimators=1000
    )
    print(f"Baseline Validation RMSE: {results['Baseline']:.4f}\n")

    # --- Ablation 1: Using Random Split instead of GroupKFold ---
    print("--- Running Ablation 1: No GroupKFold (uses Random Split) ---")
    results['No GroupKFold'] = run_experiment(
        use_group_kfold=False,
        use_physical_features=True,
        xgb_n_estimators=1000
    )
    print(f"Validation RMSE with Random Split: {results['No GroupKFold']:.4f}\n")

    # --- Ablation 2: Removing Physical Features ---
    print("--- Running Ablation 2: No Physical Features ---")
    results['No Physical Features'] = run_experiment(
        use_group_kfold=True,
        use_physical_features=False,
        xgb_n_estimators=1000
    )
    print(f"Validation RMSE without Physical Features: {results['No Physical Features']:.4f}\n")

    # --- Ablation 3: Reducing XGBoost Estimators ---
    print("--- Running Ablation 3: Reduced XGBoost Estimators ---")
    results['Reduced XGBoost Estimators'] = run_experiment(
        use_group_kfold=True,
        use_physical_features=True,
        xgb_n_estimators=100
    )
    print(f"Validation RMSE with Reduced XGBoost Estimators: {results['Reduced XGBoost Estimators']:.4f}\n")

    # --- Final Conclusion ---
    baseline_rmse = results['Baseline']
    impacts = {
        # A higher RMSE when using random split indicates GroupKFold was beneficial
        'Validation Split Strategy (GroupKFold)': results['No GroupKFold'] - baseline_rmse,
        # A higher RMSE when removing features indicates they were beneficial
        'Physical Features': results['No Physical Features'] - baseline_rmse,
        # A different RMSE indicates this hyperparameter matters
        'XGBoost Estimator Count': results['Reduced XGBoost Estimators'] - baseline_rmse
    }

    # Calculate absolute impact for ranking
    abs_impacts = {k: abs(v) for k, v in impacts.items()}
    most_impactful_component = max(abs_impacts, key=abs_impacts.get)
    
    print("--- Ablation Study Conclusion ---")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    for component, impact_val in impacts.items():
        print(f"Impact of '{component}': {impact_val:+.4f} RMSE")
        
    print(f"\nThe most impactful component was: '{most_impactful_component}' with an absolute change of {abs_impacts[most_impactful_component]:.4f} RMSE.")
