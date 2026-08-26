
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_dummy_data():
    """Creates dummy data files for a runnable example."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    train_data = {
        'Street Name': ['BROADWAY', 'MAIN ST', 'PARK AVE', 'WALL ST', '5TH AVE', 'MAIN ST', 'BROADWAY', 'WALL ST', 'PARK AVE'],
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'NO PARKING', 'DOUBLE PARKING', 'FIRE HYDRANT', 'DOUBLE PARKING'],
        'Violation Count': [150, 45, 120, 88, 200, 55, 95, 30, 70]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    borough_data = {
        'Street Name': ['BROADWAY', 'MAIN ST', 'PARK AVE', 'WALL ST', '5TH AVE'],
        'Borough': ['Manhattan', 'Queens', 'Manhattan', 'Manhattan', 'Manhattan']
    }
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data = {
        'Street Name': ['BROADWAY', 'MAIN ST', 'PARK AVE', 'WALL ST', '5TH AVE'],
        'street_width': [15.2, 12.1, 14.5, 10.0, 16.0],
        'num_lanes': [4, 2, 'four', 2, np.nan] # Mix of int, str, nan
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)
    print("Dummy data created in './input/' directory.")


def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df


def load_and_prepare_data(train_path, fillna_after_coerce):
    """Loads, preprocesses, and prepares data for models."""
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
            # FIX: Calculate median on a coerced version of the column to avoid TypeError.
            # This handles cases where the column contains non-numeric strings like 'four'.
            median_val = pd.to_numeric(full_df[col], errors='coerce').median()
            
            # This part of the original code fills only existing np.nan values.
            # It does not affect string values like 'four'.
            full_df[col].fillna(median_val, inplace=True)
            
            # --- Ablation point: `fillna_after_coerce` ---
            # This block handles the conversion of strings and subsequent NaN handling
            # based on the experiment's parameters.
            if fillna_after_coerce:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)
            else:
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce') # Let models handle NaNs

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')
    
    return full_df, cat_features


def run_experiment(name, meta_model_type, fillna_after_coerce, catboost_loss):
    """Runs a single training and validation experiment with a given configuration."""
    print(f"\n--- Running Experiment: {name} ---")
    start_time = time.time()
    
    # --- 1. Data Preparation ---
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        fillna_after_coerce=fillna_after_coerce
    )
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    # --- 2. Validation Split ---
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.3, random_state=42 # Using 30% for stability on small data
    )

    # --- 3. Model Training ---
    # Model 1 (CatBoost Base)
    cat_params = {
        'iterations': 200, 'learning_rate': 0.05, 'loss_function': catboost_loss,
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0, 'cat_features': cat_features,
        'early_stopping_rounds': 20, 'depth': 6
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    # Model 2 (CatBoost Log)
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Model 3 (XGBoost Log)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=200, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=20, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    X_meta_train = np.c_[val_preds_cat_base, val_preds_cat_log_transformed, val_preds_xgb_log_transformed]

    # --- Ablation point: `meta_model_type` ---
    if meta_model_type == 'ridge':
        meta_model = Ridge(random_state=42)
    elif meta_model_type == 'linear':
        meta_model = LinearRegression()
    else:
        raise ValueError("Invalid meta_model_type")

    meta_model.fit(X_meta_train, y_val_base)
    ensemble_predictions = meta_model.predict(X_meta_train)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")
    return val_rmse


def main():
    setup_dummy_data()

    results = {}

    # Experiment 1: Baseline
    # Ridge Stacker, Coerce-FillNA enabled, CatBoost loss is RMSE
    baseline_rmse = run_experiment(
        name="Baseline",
        meta_model_type='ridge',
        fillna_after_coerce=True,
        catboost_loss='RMSE'
    )
    results["Baseline"] = baseline_rmse
    print(f'Final Validation Performance: {baseline_rmse}')


    # Experiment 2: Ablation on Stacking Regularization
    # Uses LinearRegression (Ridge without regularization) as meta-model
    ablation_1_rmse = run_experiment(
        name="No Stacking Regularization (LinearRegression)",
        meta_model_type='linear',
        fillna_after_coerce=True,
        catboost_loss='RMSE'
    )
    results["No Stacking Regularization"] = ablation_1_rmse

    # Experiment 3: Ablation on Coercion Safeguard
    # Lets models handle NaNs from non-numeric strings
    ablation_2_rmse = run_experiment(
        name="No Coercion FillNA",
        meta_model_type='ridge',
        fillna_after_coerce=False,
        catboost_loss='RMSE'
    )
    results["No Coercion FillNA"] = ablation_2_rmse

    # Experiment 4: Ablation on CatBoost Loss Function
    # Uses MAE instead of RMSE loss for CatBoost models
    ablation_3_rmse = run_experiment(
        name="CatBoost Loss = MAE",
        meta_model_type='ridge',
        fillna_after_coerce=True,
        catboost_loss='MAE'
    )
    results["CatBoost Loss = MAE"] = ablation_3_rmse


    # --- 5. Summary and Conclusion ---
    print("\n\n--- Ablation Study Summary ---")
    print(f"{'Configuration':<40} | {'Validation RMSE':<20} | {'Change from Baseline':<20}")
    print("-" * 85)
    
    baseline_val = results["Baseline"]
    print(f"{'Baseline':<40} | {baseline_val:<20.4f} | {'N/A':<20}")

    impacts = {}
    
    # Impact of Stacking Regularization
    change_1 = results["No Stacking Regularization"] - baseline_val
    impacts["Stacking Regularization"] = abs(change_1)
    print(f"{'No Stacking Regularization':<40} | {results['No Stacking Regularization']:<20.4f} | {f'{change_1:+,.4f}':<20}")

    # Impact of Coercion FillNA
    change_2 = results["No Coercion FillNA"] - baseline_val
    impacts["Coercion FillNA"] = abs(change_2)
    print(f"{'No Coercion FillNA':<40} | {results['No Coercion FillNA']:<20.4f} | {f'{change_2:+,.4f}':<20}")

    # Impact of CatBoost Loss
    change_3 = results["CatBoost Loss = MAE"] - baseline_val
    impacts["CatBoost Loss Function (RMSE vs MAE)"] = abs(change_3)
    print(f"{'CatBoost Loss = MAE':<40} | {results['CatBoost Loss = MAE']:<20.4f} | {f'{change_3:+,.4f}':<20}")

    # Determine the most impactful component
    most_impactful_component = max(impacts, key=impacts.get)
    
    print("\n--- Conclusion ---")
    print(f"The most impactful component is: **{most_impactful_component}**")
    print("This is determined by the largest absolute change in RMSE when the component was modified from the baseline.")


if __name__ == '__main__':
    main()
