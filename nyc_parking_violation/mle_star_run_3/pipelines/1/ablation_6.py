
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

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_target_encoding=True, smoothing_factor=100):
    """Loads and preprocesses data with configurable feature engineering."""
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

    cat_features = ['street_name', 'violation_type', 'borough']

    if use_target_encoding:
        full_df['street_violation_interaction'] = full_df['street_name'].astype(str) + '_' + full_df['violation_type'].astype(str)

        global_mean = full_df['violation_count'].mean()
        agg = full_df.groupby('street_violation_interaction')['violation_count'].agg(['mean', 'count'])

        # Apply smoothing; if smoothing_factor is 0, this becomes a raw mean
        agg['smoothed_mean'] = (agg['mean'] * agg['count'] + global_mean * smoothing_factor) / (agg['count'] + smoothing_factor)

        encoding_map = agg['smoothed_mean'].to_dict()
        full_df['interaction_encoded'] = full_df['street_violation_interaction'].map(encoding_map)
        cat_features.append('street_violation_interaction')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            median_val = full_df[col].median()
            full_df[col].fillna(median_val, inplace=True)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def train_and_evaluate(train_data, cat_features, clip_negatives=True):
    """Trains the ensemble and evaluates it on a validation set."""
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- Model Training ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 50,
        'task_type': 'CPU', 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # --- Ensembling and Evaluation ---
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    
    if clip_negatives:
        ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

def main():
    """Performs an ablation study on key feature engineering and post-processing steps."""
    train_path = './input/violations_per_street_2022.csv'
    results = {}

    print("--- Running Ablation Study ---")

    # --- Baseline ---
    print("\n1. Running Baseline: Full model with Smoothed Target Encoding and Negative Clipping...")
    train_data_base, cat_features_base = load_and_prepare_data(train_path, use_target_encoding=True, smoothing_factor=100)
    baseline_rmse = train_and_evaluate(train_data_base, cat_features_base, clip_negatives=True)
    results['Baseline'] = baseline_rmse
    print(f"   > Baseline RMSE: {baseline_rmse:.4f}")

    # --- Ablation 1: No Smoothed Target Encoding ---
    print("\n2. Running Ablation: No Smoothed Target Encoding...")
    train_data_no_enc, cat_features_no_enc = load_and_prepare_data(train_path, use_target_encoding=False)
    no_enc_rmse = train_and_evaluate(train_data_no_enc, cat_features_no_enc, clip_negatives=True)
    results['No Target Encoding'] = no_enc_rmse
    print(f"   > RMSE without Target Encoding: {no_enc_rmse:.4f}")

    # --- Ablation 2: Target Encoding without Smoothing ---
    print("\n3. Running Ablation: Target Encoding without Smoothing (raw mean)...")
    train_data_no_smooth, cat_features_no_smooth = load_and_prepare_data(train_path, use_target_encoding=True, smoothing_factor=0)
    no_smooth_rmse = train_and_evaluate(train_data_no_smooth, cat_features_no_smooth, clip_negatives=True)
    results['No Smoothing in Encoding'] = no_smooth_rmse
    print(f"   > RMSE with Unsmoothed Target Encoding: {no_smooth_rmse:.4f}")

    # --- Ablation 3: No Negative Prediction Clipping ---
    print("\n4. Running Ablation: No Negative Prediction Clipping...")
    no_clip_rmse = train_and_evaluate(train_data_base, cat_features_base, clip_negatives=False)
    results['No Negative Clipping'] = no_clip_rmse
    print(f"   > RMSE without Negative Clipping: {no_clip_rmse:.4f}")
    
    # --- Conclusion ---
    print("\n--- Ablation Study Results Summary ---")
    
    impact = {}
    # Higher score is worse, so a positive impact means the component was helpful
    impact['Smoothed Target Encoding'] = results['No Target Encoding'] - results['Baseline']
    impact['Smoothing Regularization in Encoding'] = results['No Smoothing in Encoding'] - results['Baseline']
    impact['Negative Prediction Clipping'] = results['No Negative Clipping'] - results['Baseline']
    
    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    print(f"Impact of removing 'Smoothed Target Encoding': +{impact['Smoothed Target Encoding']:.4f} RMSE")
    print(f"Impact of removing 'Smoothing Regularization': +{impact['Smoothing Regularization in Encoding']:.4f} RMSE")
    print(f"Impact of removing 'Negative Prediction Clipping': +{impact['Negative Prediction Clipping']:.4f} RMSE")
    print("--------------------------------------")
    
    # Find the most impactful component (largest positive impact on RMSE when removed)
    if all(v <= 0 for v in impact.values()):
        most_impactful_component = "None (all changes improved performance)"
    else:
        most_impactful_component = max(impact, key=impact.get)

    print(f"\nConclusion: The most impactful component is '{most_impactful_component}', as its removal caused the largest degradation in performance (increase in RMSE).")

if __name__ == '__main__':
    main()
