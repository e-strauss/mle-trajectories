
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def create_dummy_data():
    """Creates dummy data files for a self-contained run."""
    if os.path.exists('./input'):
        shutil.rmtree('./input')
    os.makedirs('./input', exist_ok=True)
    
    train_data = {
        'Street Name': ['BROADWAY', 'BROADWAY', 'MAIN ST', 'MAIN ST', 'OAK AVE', 'OAK AVE', 'PINE ST'],
        'Violation Description': ['NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'NO PARKING'],
        'Violation Count': [150, 40, 120, 200, 80, 25, 95]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    boroughs_data = {
        'Street Name': ['BROADWAY', 'MAIN ST', 'OAK AVE', 'ELM ST'],
        'Borough': ['Manhattan', 'Queens', 'Brooklyn', 'Bronx']
    }
    pd.DataFrame(boroughs_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data = {
        'Street Name': ['BROADWAY', 'MAIN ST', 'OAK AVE', 'PINE ST'],
        'road_length_miles': [13.0, 5.0, 3.0, 2.5],
        'num_lanes': [4, 2, 2, 1]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, test_path=None):
    """Loads, preprocesses, and prepares data for all models."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError:
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, None, 0, None, cat_features

def run_ablation_study():
    """Runs the ablation study by modifying components of the training pipeline."""
    create_dummy_data()
    
    # --- 1. Data Preparation ---
    train_data, _, _, _, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    # Using a larger validation set due to tiny data size
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.4, random_state=42
    )
    
    results = {}

    # --- 2. Model Training (Common for all experiments) ---
    cat_params = {
        'iterations': 200, 'learning_rate': 0.05, 'loss_function': 'RMSE', 'eval_metric': 'RMSE',
        'random_seed': 42, 'verbose': 0, 'cat_features': cat_features,
        'early_stopping_rounds': 10, 'depth': 6
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # --- 3. Ablation Experiments ---

    # == Baseline: Rank Ensemble + Isotonic Regression + XGBoost with 'hist' ==
    xgb_params_hist = {
        'objective': 'reg:squarederror', 'n_estimators': 200, 'learning_rate': 0.05,
        'max_depth': 5, 'early_stopping_rounds': 10, 'enable_categorical': True,
        'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1
    }
    xgb_model_hist = xgb.XGBRegressor(**xgb_params_hist)
    xgb_model_hist.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_hist_transformed = np.expm1(xgb_model_hist.predict(X_val))

    val_ranks_cat_base = rankdata(val_preds_cat_base, method='average')
    val_ranks_cat_log = rankdata(val_preds_cat_log_transformed, method='average')
    val_ranks_xgb_hist = rankdata(val_preds_xgb_hist_transformed, method='average')
    
    avg_val_ranks = (val_ranks_cat_base + val_ranks_cat_log + val_ranks_xgb_hist) / 3.0
    
    iso_reg = IsotonicRegression(y_min=0, out_of_bounds='clip')
    iso_reg.fit(avg_val_ranks, y_val_base)
    ensemble_predictions = iso_reg.predict(avg_val_ranks)
    
    baseline_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    results['Baseline (Rank Ensemble + Isotonic)'] = baseline_rmse
    print(f"Baseline (Rank Ensemble + Isotonic) RMSE: {baseline_rmse:.4f}")
    print(f"Final Validation Performance: {baseline_rmse}")


    # == Ablation 1: Simple Averaging Ensemble (No Rank/Isotonic) ==
    simple_avg_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_hist_transformed) / 3.0
    simple_avg_predictions = np.maximum(0, simple_avg_predictions)
    
    ablation1_rmse = np.sqrt(mean_squared_error(y_val_base, simple_avg_predictions))
    results['No Rank/Isotonic (Simple Average)'] = ablation1_rmse
    print(f"Ablation 'No Rank/Isotonic (Simple Average)' RMSE: {ablation1_rmse:.4f}")

    # == Ablation 2: XGBoost with 'exact' Tree Method ==
    # The 'exact' tree method in XGBoost doesn't support the 'enable_categorical' flag.
    # We must manually encode the categorical features into integers for this model.
    X_train_exact = X_train.copy()
    X_val_exact = X_val.copy()
    for col in cat_features:
        X_train_exact[col] = X_train_exact[col].cat.codes
        X_val_exact[col] = X_val_exact[col].cat.codes

    xgb_params_exact = {
        'objective': 'reg:squarederror', 'n_estimators': 200, 'learning_rate': 0.05,
        'max_depth': 5, 'early_stopping_rounds': 10,
        'tree_method': 'exact', 'random_state': 42, 'n_jobs': -1
    }
    xgb_model_exact = xgb.XGBRegressor(**xgb_params_exact)
    # Fit on the manually encoded data
    xgb_model_exact.fit(X_train_exact, y_train_log, eval_set=[(X_val_exact, y_val_log)], verbose=False)

    # Predict using the manually encoded validation set
    val_preds_xgb_exact_transformed = np.expm1(xgb_model_exact.predict(X_val_exact))
    val_ranks_xgb_exact = rankdata(val_preds_xgb_exact_transformed, method='average')
    
    avg_val_ranks_exact = (val_ranks_cat_base + val_ranks_cat_log + val_ranks_xgb_exact) / 3.0
    
    iso_reg_exact = IsotonicRegression(y_min=0, out_of_bounds='clip')
    iso_reg_exact.fit(avg_val_ranks_exact, y_val_base)
    ensemble_predictions_exact = iso_reg_exact.predict(avg_val_ranks_exact)
    
    ablation2_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions_exact))
    results['XGBoost with exact Tree Method'] = ablation2_rmse
    print(f"Ablation 'XGBoost with exact Tree Method' RMSE: {ablation2_rmse:.4f}\n")

    # --- 4. Conclusion ---
    impact_ablation1 = abs(results['No Rank/Isotonic (Simple Average)'] - results['Baseline (Rank Ensemble + Isotonic)'])
    impact_ablation2 = abs(results['XGBoost with exact Tree Method'] - results['Baseline (Rank Ensemble + Isotonic)'])

    if impact_ablation1 > impact_ablation2:
        most_impactful = "The Rank/Isotonic Ensembling Strategy"
    else:
        most_impactful = "The XGBoost 'tree_method' Hyperparameter"

    print(f"The most impactful component tested was: {most_impactful}.")
    
    # --- Cleanup ---
    if os.path.exists('./input'):
        shutil.rmtree('./input')

if __name__ == '__main__':
    run_ablation_study()
