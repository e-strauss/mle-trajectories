
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def create_dummy_data():
    """Creates dummy data files for a self-contained, runnable example."""
    os.makedirs('./input', exist_ok=True)
    train_data_content = """street_name,violation_type,violation_count
Main St,Double Parking,150
Main St,No Standing,50
Oak Ave,Fire Hydrant,30
Oak Ave,Double Parking,20
Pine St,No Standing,80
Elm St,Bus Lane,250
Elm St,Double Parking,100
Broadway,No Standing,600
Broadway,Bus Lane,400
Fifth Ave,Fire Hydrant,120
Park Ave,Bus Lane,350
Park Ave,No Standing,220
Lexington Ave,Double Parking,180
"""
    borough_data_content = """street_name,borough
Main St,Queens
Oak Ave,Brooklyn
Pine St,Manhattan
Elm St,Bronx
Broadway,Manhattan
Fifth Ave,Manhattan
Park Ave,Manhattan
"""
    physical_data_content = """street_name,street_width,street_length
Main St,30,500
Oak Ave,25,300
Pine St,20,200
Elm St,35,800
Broadway,40,1500
Fifth Ave,38,1200
Lexington Ave,28,900
"""
    with open('./input/violations_per_street_2022.csv', 'w') as f:
        f.write(train_data_content)
    with open('./input/street_names_and_boroughs.csv', 'w') as f:
        f.write(borough_data_content)
    with open('./input/physical_features_per_street.csv', 'w') as f:
        f.write(physical_data_content)

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def run_experiment(split_strategy='stratified_group_kfold', cat_iterations=1000, cat_depth=10):
    """
    Runs a training and evaluation experiment with configurable components.
    
    Args:
        split_strategy (str): 'stratified_group_kfold' or 'group_shuffle_split'.
        cat_iterations (int): Number of iterations for CatBoost models.
        cat_depth (int): Tree depth for CatBoost models.

    Returns:
        float: The validation RMSE of the ensemble.
    """
    # --- 1. Load and Prepare Data ---
    train_df = clean_col_names(pd.read_csv('./input/violations_per_street_2022.csv'))
    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        median_val = full_df[col].median()
        full_df[col].fillna(median_val, inplace=True)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    full_df['log_violation_count'] = np.log1p(full_df['violation_count'])
    features = [col for col in full_df.columns if col not in ['violation_count', 'log_violation_count', 'stratify_group']]

    X = full_df[features]
    y_base = full_df['violation_count']
    y_log = full_df['log_violation_count']
    groups = full_df['street_name']

    # --- 2. Validation Split ---
    # Decide whether to attempt stratified splitting, with a fallback mechanism
    use_stratified_split = (split_strategy == 'stratified_group_kfold')

    if use_stratified_split:
        street_avg_violations = full_df.groupby('street_name')['violation_count'].mean()
        q = min(5, len(street_avg_violations) - 1 if len(street_avg_violations) > 1 else 1)
        
        # Check if stratification is possible. It requires being able to create bins (q>=1)
        # and having at least 2 groups in every resulting stratum.
        if q < 1:
            print("Warning: Not enough unique groups for stratification. Falling back to GroupShuffleSplit.")
            use_stratified_split = False
        else:
            stratify_bins = pd.qcut(street_avg_violations, q=q, labels=False, duplicates='drop')
            full_df['stratify_group'] = full_df['street_name'].map(stratify_bins)
            # Handle cases where a street was not in the average violations calculation (e.g., NaN)
            if full_df['stratify_group'].isnull().any():
                full_df['stratify_group'].fillna(full_df['stratify_group'].median(), inplace=True)

            min_groups_per_stratum = full_df.groupby('stratify_group')['street_name'].nunique().min()
            if min_groups_per_stratum < 2:
                print(f"Warning: Stratification resulted in a stratum with {min_groups_per_stratum} group(s). Need at least 2. Falling back to GroupShuffleSplit.")
                use_stratified_split = False
    
    if use_stratified_split:
        # If checks passed, perform the stratified split
        n_splits = min(5, full_df.groupby('stratify_group')['street_name'].nunique().min())
        stratify_col = full_df['stratify_group']
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_idx, val_idx = next(sgkf.split(X, stratify_col, groups))
    else: 
        # This block is used for 'group_shuffle_split' or as a fallback
        if split_strategy not in ['group_shuffle_split', 'stratified_group_kfold']:
             raise ValueError(f"Unknown split strategy: {split_strategy}")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y_base, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_base, y_val_base = y_base.iloc[train_idx], y_base.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # --- 3. Model Training ---
    cat_params = {
        'iterations': cat_iterations, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0, 'cat_features': cat_features,
        'early_stopping_rounds': 20, 'depth': cat_depth
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5,
        early_stopping_rounds=50, enable_categorical=True, tree_method='hist',
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    return np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))

if __name__ == '__main__':
    create_dummy_data()
    
    print("--- Running Ablation Study ---")

    # Baseline: Stratified splitting and full-complexity CatBoost models
    baseline_rmse = run_experiment(split_strategy='stratified_group_kfold', cat_iterations=1000, cat_depth=10)
    print(f"Baseline RMSE (Stratified Split, High Complexity): {baseline_rmse:.4f}")
    final_validation_score = baseline_rmse
    print(f"Final Validation Performance: {final_validation_score}")

    # Ablation 1: Using a simpler group split without stratification
    no_strat_rmse = run_experiment(split_strategy='group_shuffle_split', cat_iterations=1000, cat_depth=10)
    print(f"Ablation 'No Stratification' RMSE: {no_strat_rmse:.4f} (Change: {no_strat_rmse - baseline_rmse:+.4f})")
    
    # Ablation 2: Reducing the complexity of the CatBoost models
    low_complexity_rmse = run_experiment(split_strategy='stratified_group_kfold', cat_iterations=200, cat_depth=6)
    print(f"Ablation 'Reduced CatBoost Complexity' RMSE: {low_complexity_rmse:.4f} (Change: {low_complexity_rmse - baseline_rmse:+.4f})")

    print("\n--- Conclusion ---")
    
    impact_stratification = abs(no_strat_rmse - baseline_rmse)
    impact_complexity = abs(low_complexity_rmse - baseline_rmse)

    if impact_stratification > impact_complexity:
        print("The most impactful component is the Validation Split Strategy.")
        print("Using a stratified group split provides a more stable and representative validation set, significantly affecting the measured performance.")
    else:
        print("The most impactful component is the CatBoost Model Complexity.")
        print("Reducing model iterations and depth has a larger effect on performance than the choice of splitting strategy, suggesting model complexity is a key factor.")
