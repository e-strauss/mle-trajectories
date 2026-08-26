
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
        df = df.rename(columns={'violation_description': 'violation_type'})
    return df

def load_and_prepare_data(train_path):
    """Loads and prepares data for the ablation study, focusing only on training."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    try:
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
    except FileNotFoundError:
        print("Warning: Augmentation data not found. Proceeding with base data only.")
        full_df = train_df
        # Create a dummy 'borough' column if augmentation fails, to ensure consistency
        if 'borough' not in full_df.columns:
            full_df['borough'] = 'Unknown'

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        if col in full_df.columns:
            full_df[col] = full_df[col].astype('category')

    full_df['log_violation_count'] = np.log1p(full_df['violation_count'])
    return full_df, cat_features

def run_experiment(X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
                   experiment_name, use_early_stopping=True, xgb_objective='reg:tweedie', clip_at_zero=True):
    """
    Runs a single training and evaluation experiment with a specific configuration.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")

    # --- Model Training ---
    # CatBoost Models
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'task_type': 'CPU', 'depth': 10,
        'early_stopping_rounds': 50 if use_early_stopping else None
    }
    
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=use_early_stopping)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=use_early_stopping)

    # XGBoost Model
    xgb_params = {
        'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 5,
        'early_stopping_rounds': 50 if use_early_stopping else None,
        'enable_categorical': True, 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1
    }
    xgb_model = xgb.XGBRegressor(objective=xgb_objective, **xgb_params)
    
    # Adjust training target based on objective
    if xgb_objective == 'reg:tweedie':
        xgb_model.fit(X_train, y_train_base, eval_set=[(X_val, y_val_base)], verbose=False)
    else: # e.g., 'reg:squarederror'
        xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- Ensembling and Evaluation ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    
    if xgb_objective == 'reg:tweedie':
        val_preds_xgb = xgb_model.predict(X_val)
    else: # The prediction is in log scale, so transform it back
        val_preds_xgb = np.expm1(xgb_model.predict(X_val))

    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb) / 3.0
    
    if clip_at_zero:
        ensemble_predictions = np.maximum(0, ensemble_predictions)

    rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    # --- 1. Data Preparation ---
    print("Loading and preparing data for ablation study...")
    try:
        train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
    except FileNotFoundError:
        print("Error: Training data not found at './input/violations_per_street_2022.csv'. Aborting.")
        return
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- 2. Run Experiments ---
    results = {}
    
    # Baseline: Full model with all features enabled
    results['Baseline'] = run_experiment(
        X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
        "Baseline (Tweedie XGB, Early Stopping, Clipping)"
    )

    # Ablation 1: Disable Early Stopping to check for overfitting
    results['No Early Stopping'] = run_experiment(
        X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
        "Ablation: No Early Stopping", use_early_stopping=False
    )

    # Ablation 2: Use the standard 'squarederror' objective for XGBoost
    results['XGBoost with Squarederror'] = run_experiment(
        X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log, cat_features,
        "Ablation: XGBoost with Squarederror", xgb_objective='reg:squarederror'
    )
    
    # --- 3. Summarize and Conclude ---
    print("\n\n--- Ablation Study Summary ---")
    baseline_rmse = results['Baseline']
    print(f"Baseline RMSE: {baseline_rmse:.4f}\n")
    
    impacts = {}
    for name, rmse in results.items():
        if name != 'Baseline':
            impact = rmse - baseline_rmse
            impacts[name] = impact
            print(f"Ablation '{name}':")
            print(f"  RMSE: {rmse:.4f} (Performance Change: {impact:+.4f})")

    if not impacts:
        print("No ablation studies were run.")
        return

    # Find the component with the largest impact (absolute value)
    most_impactful_component = max(impacts, key=lambda k: abs(impacts[k]))
    largest_impact_value = impacts[most_impactful_component]

    print("\n--- Conclusion ---")
    if largest_impact_value > 0:
        conclusion = (f"The most impactful component was '{most_impactful_component}'. "
                      f"Removing or changing it worsened the RMSE by {abs(largest_impact_value):.4f}, "
                      "proving it is the most beneficial component tested.")
    else:
        conclusion = (f"The most impactful component was '{most_impactful_component}'. "
                      f"Removing or changing it improved the RMSE by {abs(largest_impact_value):.4f}, "
                      "proving it was the most detrimental component tested.")
    
    print(conclusion)

if __name__ == '__main__':
    main()
