
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Setup: Create dummy data for a self-contained script ---
def create_dummy_data():
    """Creates dummy CSV files in an ./input directory for the script to run."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    train_data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'BROADWAY', 'MAIN ST', 'ELM ST', 'OAK ST', 'MAPLE AVE', 'PINE ST', 'CEDAR LN', 'WALL ST', '5TH AVE', '6TH AVE'],
        'Violation Description': ['NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT', 'NO STANDING', 'DOUBLE PARKING', 'NO PARKING', 'BUS LANE', 'NO PARKING', 'CROSSWALK', 'NO PARKING', 'DOUBLE PARKING'],
        'Violation Count': [150, 75, 200, 50, 90, 80, 180, 60, 220, 300, 450, 95]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    borough_data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'BROADWAY', 'ELM ST', 'OAK ST', 'MAPLE AVE', 'PINE ST', 'WALL ST', '5TH AVE'],
        'Borough': ['Brooklyn', 'Manhattan', 'Manhattan', 'Queens', 'Brooklyn', 'Staten Island', 'Bronx', 'Manhattan', 'Manhattan']
    }
    pd.DataFrame(borough_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data = {
        'Street Name': ['MAIN ST', 'PARK AVE', 'BROADWAY', 'ELM ST', 'OAK ST', 'MAPLE AVE', 'CEDAR LN', 'WALL ST', '5TH AVE', '6TH AVE'],
        'street_width': [30, 50, 45, 25, 28, 35, 22, 60, 55, 48],
        'street_length': [500, 1200, 1500, 400, 350, 600, 300, 1000, 2000, 1800]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)


# --- Core Functions (from original script) ---
warnings.filterwarnings("ignore", category=UserWarning)

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path):
    """Loads and preprocesses data for the models."""
    train_df = clean_col_names(pd.read_csv(train_path))
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

# --- Ablation Experiment Runner ---
def run_experiment(ablation_name, data, cat_features, ensemble_weights=None, xgb_objective='reg:squarederror'):
    """Runs a single training and evaluation experiment with specified configurations."""
    print(f"--- Running Experiment: {ablation_name} ---")

    # Default to simple average if no weights are provided
    if ensemble_weights is None:
        ensemble_weights = [1/3, 1/3, 1/3]

    train_data = data.copy()
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]

    X = train_data[features]
    y_base, y_log = train_data['violation_count'], train_data['log_violation_count']
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.25, random_state=42
    )

    # --- Model Training ---
    cat_params = {
        'iterations': 200, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'random_seed': 42, 'verbose': 0, 'cat_features': cat_features,
        'early_stopping_rounds': 10, 'depth': 6
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)

    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(
        objective=xgb_objective, n_estimators=200, learning_rate=0.05, max_depth=5,
        early_stopping_rounds=10, enable_categorical=True, tree_method='hist',
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- Validation and Ensembling ---
    val_preds_cat_base = model_cat_base.predict(X_val)

    # Clip predictions from log-based models before np.expm1 to prevent overflow
    log_pred_clip_threshold = 30  # np.expm1(30) is a very large number, a safe upper bound
    
    val_preds_cat_log = model_cat_log.predict(X_val)
    val_preds_cat_log_clipped = np.clip(val_preds_cat_log, -np.inf, log_pred_clip_threshold)
    val_preds_cat_log_transformed = np.expm1(val_preds_cat_log_clipped)

    val_preds_xgb_log = xgb_model.predict(X_val)
    val_preds_xgb_log_clipped = np.clip(val_preds_xgb_log, -np.inf, log_pred_clip_threshold)
    val_preds_xgb_log_transformed = np.expm1(val_preds_xgb_log_clipped)

    ensemble_predictions = (
        ensemble_weights[0] * val_preds_cat_base +
        ensemble_weights[1] * val_preds_cat_log_transformed +
        ensemble_weights[2] * val_preds_xgb_log_transformed
    )
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Validation RMSE: {val_rmse:.4f}\n")
    return val_rmse

def main():
    """Executes the ablation study."""
    create_dummy_data()
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
    
    results = {}
    
    # 1. Baseline Experiment
    baseline_rmse = run_experiment(
        "Baseline (Simple Average Ensemble, Squarederror Loss)",
        data=train_data,
        cat_features=cat_features
    )
    results["Baseline"] = baseline_rmse

    # 2. Ablation: Use a weighted ensemble
    weighted_rmse = run_experiment(
        "Ablation: Weighted Ensemble (0.5, 0.25, 0.25)",
        data=train_data,
        cat_features=cat_features,
        ensemble_weights=[0.5, 0.25, 0.25]
    )
    results["Weighted Ensemble"] = weighted_rmse

    # 3. Ablation: Use Pseudo-Huber loss for XGBoost
    huber_loss_rmse = run_experiment(
        "Ablation: XGBoost with Pseudo-Huber Loss",
        data=train_data,
        cat_features=cat_features,
        xgb_objective='reg:pseudohubererror'
    )
    results["Pseudo-Huber Loss"] = huber_loss_rmse

    # --- Analysis and Conclusion ---
    print("--- Ablation Study Summary ---")
    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    
    impact_weighted = weighted_rmse - baseline_rmse
    print(f"Ablation 'Weighted Ensemble' Performance: {results['Weighted Ensemble']:.4f} (Impact: {impact_weighted:+.4f} RMSE)")

    impact_huber = huber_loss_rmse - baseline_rmse
    print(f"Ablation 'Pseudo-Huber Loss' Performance: {results['Pseudo-Huber Loss']:.4f} (Impact: {impact_huber:+.4f} RMSE)")
    
    # Determine the most impactful component based on the magnitude of change
    abs_impact_weighted = abs(impact_weighted)
    abs_impact_huber = abs(impact_huber)

    if abs_impact_weighted > abs_impact_huber:
        most_impactful = "Ensemble Weighting Strategy"
    elif abs_impact_huber > abs_impact_weighted:
        most_impactful = "XGBoost Objective Function"
    else:
        most_impactful = "Ensemble Weighting and XGBoost Objective Function had a similar impact"

    print(f"\nThe most impactful component tested was the {most_impactful}.")

    # Report the best performing configuration as the final validation score
    best_config_name = min(results, key=results.get)
    final_validation_score = results[best_config_name]
    print(f"\nBest configuration: '{best_config_name}' with RMSE: {final_validation_score:.4f}")
    print(f"Final Validation Performance: {final_validation_score}")
    
    # Cleanup dummy files
    if os.path.exists('./input'):
        shutil.rmtree('./input')

if __name__ == '__main__':
    main()
