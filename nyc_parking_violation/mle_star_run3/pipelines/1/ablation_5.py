
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_base_data(train_path):
    """Loads and performs initial merging of datasets."""
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
    
    return full_df

def apply_oof_encoding(df, agg_funcs):
    """Applies Out-of-Fold target encoding to the dataframe."""
    encoded_df = df.copy()
    target = 'violation_count'
    cat_features_for_encoding = ['street_name', 'borough', 'violation_type']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in cat_features_for_encoding:
        for func in agg_funcs:
            encoded_df[f'oof_{col}_{target}_{func}'] = np.nan

    for train_index, val_index in kf.split(encoded_df):
        train_fold = encoded_df.iloc[train_index]
        for col in cat_features_for_encoding:
            mapping = train_fold.groupby(col)[target].agg(agg_funcs)
            mapping.columns = [f'oof_{col}_{target}_{func}' for func in agg_funcs]
            val_encoded = encoded_df.iloc[val_index][[col]].merge(mapping, on=col, how='left')
            for agg_col in mapping.columns:
                encoded_df.loc[val_index, agg_col] = val_encoded[agg_col].values

    # Fill any remaining NaNs
    for col in cat_features_for_encoding:
        for func in agg_funcs:
            oof_col_name = f'oof_{col}_{target}_{func}'
            encoded_df[oof_col_name].fillna(encoded_df[oof_col_name].mean(), inplace=True)
            
    return encoded_df

def train_and_evaluate(train_data):
    """Trains the 3-model ensemble and evaluates it on a validation set."""
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('category')

    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # CatBoost Base Model
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': [c for c in cat_features if c in X_train.columns],
        'early_stopping_rounds': 50, 'depth': 10
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    # CatBoost Log Model
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # XGBoost Log Model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    # Ensemble and Evaluate
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    return np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))


def main():
    """Main function to run the ablation study."""
    train_path = './input/violations_per_street_2022.csv'
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}")
        print("Please ensure the input data is available to run the study.")
        return

    base_df = load_base_data(train_path)
    results = {}

    # --- Baseline: Full Out-of-Fold Encoding ---
    print("Running Baseline: Full OOF Encoding (mean, std, count)...")
    baseline_df = apply_oof_encoding(base_df, agg_funcs=['mean', 'std', 'count'])
    baseline_rmse = train_and_evaluate(baseline_df)
    results['Baseline (Full OOF)'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}\n")

    # --- Ablation 1: No Out-of-Fold Encoding ---
    print("Running Ablation 1: No OOF Encoding...")
    # Using the original `base_df` which has no OOF features
    ablation_1_rmse = train_and_evaluate(base_df.copy())
    results['Ablation (No OOF)'] = ablation_1_rmse
    print(f"RMSE without OOF Encoding: {ablation_1_rmse:.4f}\n")

    # --- Ablation 2: Simplified OOF Encoding (Mean only) ---
    print("Running Ablation 2: Simplified OOF Encoding (mean only)...")
    ablation_2_df = apply_oof_encoding(base_df, agg_funcs=['mean'])
    ablation_2_rmse = train_and_evaluate(ablation_2_df)
    results['Ablation (OOF with mean only)'] = ablation_2_rmse
    print(f"RMSE with Mean-Only OOF: {ablation_2_rmse:.4f}\n")
    
    # --- Conclusion ---
    print("--- Ablation Study Results ---")
    impact = {}
    baseline_score = results['Baseline (Full OOF)']
    
    impact_no_oof = results['Ablation (No OOF)'] - baseline_score
    impact['Full Out-of-Fold Encoding vs. None'] = impact_no_oof
    print(f"Removing Full OOF Encoding changed RMSE by: {impact_no_oof:+.4f}")
    
    impact_simplified_oof = results['Ablation (OOF with mean only)'] - baseline_score
    impact['Adding std/count to mean OOF'] = -impact_simplified_oof # Inverting to show contribution
    print(f"Removing 'std' and 'count' from OOF changed RMSE by: {impact_simplified_oof:+.4f}")

    print("\n--- Conclusion ---")
    if impact_no_oof > 0:
        print("The most impactful component is the Out-of-Fold (OOF) target encoding itself.")
        print(f"Adding it improved the model's RMSE by approximately {impact_no_oof:.4f}, demonstrating its power in creating predictive features from categorical variables without data leakage.")
    else:
        print("Surprisingly, the Out-of-Fold (OOF) target encoding did not improve performance in this configuration.")

if __name__ == '__main__':
    main()
