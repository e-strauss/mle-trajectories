
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Data Preparation Functions (Adapted for Ablation) ---

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_ordinal_mapping=True):
    """Loads and preprocesses data, with an ablation flag for ordinal mapping."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    # Load augmentation data, creating dummy frames if not found
    try:
        boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
        physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))
    except FileNotFoundError:
        boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
        physical_df = pd.DataFrame(columns=['street_name'])

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)

    # Ordinal mapping for quality-related string columns
    ordinal_mapping = {'Excellent': 5, 'Good': 4, 'Fair': 3, 'Poor': 2}
    
    physical_cols_to_process = [col for col in physical_df.columns if col != 'street_name']
    
    for col in physical_cols_to_process:
        if col in full_df.columns:
            # ABLATION POINT: Conditionally apply the ordinal mapping
            if full_df[col].dtype == 'object' and use_ordinal_mapping:
                full_df[col] = full_df[col].map(ordinal_mapping).fillna(0)

            if pd.api.types.is_numeric_dtype(full_df[col]):
                median_val = full_df[col].median()
                full_df[col].fillna(median_val, inplace=True)
            
            # This line ensures non-numeric values become 0, which is the fallback for the ablation
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

# --- Model Training and Evaluation Function ---

def train_and_evaluate(
    use_ordinal_mapping=True,
    use_xgboost_in_stack=True
):
    """
    Main function to run a single configuration of the model pipeline.
    
    Args:
        use_ordinal_mapping (bool): If True, applies ordinal mapping to physical features.
        use_xgboost_in_stack (bool): If True, includes XGBoost predictions in the stacking layer.
    
    Returns:
        float: The validation RMSE for this configuration.
    """
    # 1. Data Preparation
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        use_ordinal_mapping=use_ordinal_mapping
    )

    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]

    # 2. Validation Split
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # 3. Model Training
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50, 'depth': 10
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

    # 4. Ensembling and Evaluation
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))

    # ABLATION POINT: Conditionally build the features for the meta-model
    if use_xgboost_in_stack:
        meta_features_val = np.column_stack((
            val_preds_cat_base,
            val_preds_cat_log_transformed,
            val_preds_xgb_log_transformed
        ))
    else:
        meta_features_val = np.column_stack((
            val_preds_cat_base,
            val_preds_cat_log_transformed
        ))

    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    meta_model.fit(meta_features_val, y_val_base)

    ensemble_predictions = meta_model.predict(meta_features_val)
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    return np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))

# --- Ablation Study Execution ---

if __name__ == '__main__':
    print("Running ablation study...")
    results = {}

    # Baseline
    print("1. Running Baseline (Full Model)...")
    baseline_rmse = train_and_evaluate(
        use_ordinal_mapping=True,
        use_xgboost_in_stack=True
    )
    results['Baseline'] = {'rmse': baseline_rmse, 'impact': 0.0}

    # Ablation 1: No Ordinal Mapping
    print("2. Running Ablation: No Ordinal Mapping for Physical Features...")
    ablation1_rmse = train_and_evaluate(
        use_ordinal_mapping=False,
        use_xgboost_in_stack=True
    )
    results['No Ordinal Mapping'] = {
        'rmse': ablation1_rmse,
        'impact': ablation1_rmse - baseline_rmse
    }

    # Ablation 2: No XGBoost in Stacking
    print("3. Running Ablation: No XGBoost in Stacking Ensemble...")
    ablation2_rmse = train_and_evaluate(
        use_ordinal_mapping=True,
        use_xgboost_in_stack=False
    )
    results['No XGBoost in Stacking'] = {
        'rmse': ablation2_rmse,
        'impact': ablation2_rmse - baseline_rmse
    }

    print("\n--- Ablation Study Results ---")
    print(f"{'Configuration':<30} | {'Validation RMSE':<20} | {'Impact (vs Baseline)':<25}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<30} | {data['rmse']:<20.4f} | {data['impact']:<+25.4f}")

    # Determine the most impactful component
    max_impact_component = max(results.keys() - {'Baseline'}, 
                               key=lambda k: abs(results[k]['impact']))

    print("\n--- Conclusion ---")
    print(f"The most impactful component tested is: \"{max_impact_component}\"")
    print(f"Altering this component changed the validation RMSE by {results[max_impact_component]['impact']:.4f}.")

