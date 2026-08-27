
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import collections

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Create dummy data files if they don't exist
def create_dummy_data():
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # Main violations data
    train_data = {
        'street_name': ['Street A', 'Street B', 'Street A', 'Street C', 'Street B', 'Street D', 'Street E', 'Street C', 'Street D', 'Street A'],
        'violation_type': ['Parking', 'Parking', 'Speeding', 'Jaywalking', 'Speeding', 'Parking', 'Speeding', 'Parking', 'Speeding', 'Jaywalking'],
        'violation_count': [10, 5, 2, 1, 8, 12, 3, 6, 7, 4]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    # Augmentation data 1
    borough_data = {
        'street_name': ['Street A', 'Street B', 'Street C', 'Street D', 'Street E'],
        'borough': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn']
    }
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    # Augmentation data 2
    physical_data = {
        'street_name': ['Street A', 'Street B', 'Street C', 'Street D', 'Street E'],
        'street_width': [10.5, 8.2, 12.0, 9.0, 7.5],
        'pavement_quality': [8, 7, 9, 6, 8]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path):
    """Loads and preprocesses data."""
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
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(ensemble_strategy="weighted", uniform_depth=None):
    """
    Runs a single training and evaluation experiment with specified configurations.

    Args:
        ensemble_strategy (str): 'weighted' for inverse-MSE weights or 'simple' for average.
        uniform_depth (int, optional): If set, all models will use this tree depth.
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
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- 3. Model Training ---
    # Define depths based on the experiment config
    cat_base_depth = uniform_depth if uniform_depth is not None else 10
    cat_log_depth = uniform_depth if uniform_depth is not None else 10
    xgb_log_depth = uniform_depth if uniform_depth is not None else 5
    
    # Model 1 (CatBoost Base)
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50, 'depth': cat_base_depth
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    # Model 2 (CatBoost Log)
    cat_params['depth'] = cat_log_depth
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # Model 3 (XGBoost Log)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=xgb_log_depth, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- 4. Validation Performance & Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = None
    if ensemble_strategy == "weighted":
        mses = [
            mean_squared_error(y_val_base, val_preds_cat_base),
            mean_squared_error(y_val_base, val_preds_cat_log_transformed),
            mean_squared_error(y_val_base, val_preds_xgb_log_transformed)
        ]
        inverse_mses = [1/mse if mse > 1e-9 else 0 for mse in mses]
        total_inverse_mse = sum(inverse_mses)
        weights = [inv_mse / total_inverse_mse if total_inverse_mse > 0 else 1/3 for inv_mse in inverse_mses]
        ensemble_predictions = (weights[0] * val_preds_cat_base +
                                weights[1] * val_preds_cat_log_transformed +
                                weights[2] * val_preds_xgb_log_transformed)
    elif ensemble_strategy == "simple":
        ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0

    ensemble_predictions = np.maximum(0, ensemble_predictions)
    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

def ablation_study():
    """Performs an ablation study on ensemble strategy and model complexity."""
    create_dummy_data()
    results = collections.OrderedDict()

    print("Running Ablation Study...")

    # Baseline: Weighted ensemble with diverse tree depths
    baseline_rmse = run_experiment(ensemble_strategy="weighted", uniform_depth=None)
    results['Baseline (Weighted Ensemble, Diverse Depths)'] = baseline_rmse

    # Ablation 1: Simple average ensemble
    ablation_simple_avg_rmse = run_experiment(ensemble_strategy="simple", uniform_depth=None)
    results['Ablation: No Weighted Ensemble (Simple Average)'] = ablation_simple_avg_rmse

    # Ablation 2: Uniform tree depths for all models
    ablation_uniform_depth_rmse = run_experiment(ensemble_strategy="weighted", uniform_depth=7)
    results['Ablation: No Diverse Depths (Uniform Depth=7)'] = ablation_uniform_depth_rmse

    print("\n--- Ablation Study Results (Validation RMSE) ---")
    
    impacts = {}
    for name, score in results.items():
        change = score - baseline_rmse
        print(f"- {name}: {score:.4f} (Impact: {change:+.4f})")
        if "Ablation" in name:
            # Use absolute change to measure magnitude of impact
            impacts[name] = abs(change)

    # Determine the most impactful component
    if not impacts:
        most_impactful = "No ablations were performed."
    else:
        most_impactful_name = max(impacts, key=impacts.get)
        # Extract the component name from the experiment name
        most_impactful_component = most_impactful_name.split("(")[0].replace("Ablation: No ", "").strip()

    print("\n--- Conclusion ---")
    print(f"The most impactful component was '{most_impactful_component}'.")
    print("This is because removing or altering it caused the largest change in the validation RMSE.")


if __name__ == '__main__':
    ablation_study()
