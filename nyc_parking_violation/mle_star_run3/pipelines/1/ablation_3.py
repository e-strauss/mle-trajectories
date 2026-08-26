
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import copy

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_categorical_casting=True):
    """Loads, preprocesses, and prepares data."""
    train_df = pd.read_csv(train_path)
    train_df = clean_col_names(train_df)

    # Mock augmentation data as it is not the focus of this study
    boroughs_df = pd.DataFrame(columns=['street_name', 'borough'])
    physical_df = pd.DataFrame(columns=['street_name'])

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')
    full_df['borough'].fillna('Unknown', inplace=True)
    
    # Cast categorical features to 'category' dtype
    cat_features = ['street_name', 'violation_type', 'borough']
    if use_categorical_casting:
        for col in cat_features:
            if col in full_df.columns:
                full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def main():
    """Performs an ablation study on model components."""
    ablation_results = {}
    # Use the input directory specified in the prompt
    input_dir = './input'
    train_path = os.path.join(input_dir, 'violations_per_street_2022.csv')

    # --- Data Preparation (done once for most experiments) ---
    train_data_base, cat_features_base = load_and_prepare_data(train_path, use_categorical_casting=True)
    train_data_base['log_violation_count'] = np.log1p(train_data_base['violation_count'])
    features = [col for col in train_data_base.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data_base[features]
    y_base = train_data_base['violation_count']
    y_log = train_data_base['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.2, random_state=42
    )

    # --- Common Model Parameters ---
    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features_base, 'early_stopping_rounds': 50, 'depth': 10
    }
    
    xgb_params_base = {
        'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 5, 'early_stopping_rounds': 50, 'enable_categorical': True,
        'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_lambda': 1
    }

    # --- 1. Baseline Experiment ---
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    xgb_model_base = xgb.XGBRegressor(**xgb_params_base)
    xgb_model_base.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log = np.expm1(xgb_model_base.predict(X_val))
    
    ensemble_preds = (val_preds_cat_base + val_preds_cat_log + val_preds_xgb_log) / 3.0
    baseline_rmse = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds)))
    ablation_results['Baseline'] = baseline_rmse
    print(f"Baseline Validation RMSE: {baseline_rmse:.4f}")
    print(f"Final Validation Performance: {baseline_rmse}")


    # --- 2. Ablation: No Log Transformation ---
    # Train all models on the base target
    model_cat_base_abl1 = CatBoostRegressor(**cat_params).fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    model_cat_log_abl1 = CatBoostRegressor(**cat_params).fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    xgb_params_abl1 = copy.deepcopy(xgb_params_base)
    xgb_model_abl1 = xgb.XGBRegressor(**xgb_params_abl1).fit(X_train, y_train_base, eval_set=[(X_val, y_val_base)], verbose=False)
    
    preds1 = model_cat_base_abl1.predict(X_val)
    preds2 = model_cat_log_abl1.predict(X_val)
    preds3 = xgb_model_abl1.predict(X_val)
    
    ensemble_preds_abl1 = (preds1 + preds2 + preds3) / 3.0
    rmse_abl1 = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds_abl1)))
    ablation_results['No Log Transformation'] = rmse_abl1
    print(f"Ablation (No Log Transformation) RMSE: {rmse_abl1:.4f}")

    # --- 3. Ablation: No XGBoost Regularization ---
    # Remove subsample, colsample_bytree, reg_lambda
    xgb_params_abl2 = {
        'objective': 'reg:squarederror', 'n_estimators': 1000, 'learning_rate': 0.05,
        'max_depth': 5, 'early_stopping_rounds': 50, 'enable_categorical': True,
        'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1
    }
    xgb_model_abl2 = xgb.XGBRegressor(**xgb_params_abl2)
    xgb_model_abl2.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    val_preds_xgb_log_abl2 = np.expm1(xgb_model_abl2.predict(X_val))
    ensemble_preds_abl2 = (val_preds_cat_base + val_preds_cat_log + val_preds_xgb_log_abl2) / 3.0
    rmse_abl2 = np.sqrt(mean_squared_error(y_val_base, np.maximum(0, ensemble_preds_abl2)))
    ablation_results['No XGBoost Regularization'] = rmse_abl2
    print(f"Ablation (No XGBoost Regularization) RMSE: {rmse_abl2:.4f}")

    # --- 4. Ablation: No Categorical Dtype Casting ---
    # Reload data without casting to 'category'
    train_data_abl3, cat_features_abl3 = load_and_prepare_data(train_path, use_categorical_casting=False)
    train_data_abl3['log_violation_count'] = np.log1p(train_data_abl3['violation_count'])
    
    X_abl3 = train_data_abl3[features]
    y_base_abl3 = train_data_abl3['violation_count']
    y_log_abl3 = train_data_abl3['log_violation_count']
    
    X_train_abl3, X_val_abl3, y_train_base_abl3, y_val_base_abl3, y_train_log_abl3, y_val_log_abl3 = train_test_split(
        X_abl3, y_base_abl3, y_log_abl3, test_size=0.2, random_state=42
    )

    cat_params_abl3 = copy.deepcopy(cat_params)
    cat_params_abl3['cat_features'] = cat_features_abl3 # These are now just string columns
    
    model_cat_base_abl3 = CatBoostRegressor(**cat_params_abl3).fit(X_train_abl3, y_train_base_abl3, eval_set=(X_val_abl3, y_val_base_abl3), use_best_model=True)
    model_cat_log_abl3 = CatBoostRegressor(**cat_params_abl3).fit(X_train_abl3, y_train_log_abl3, eval_set=(X_val_abl3, y_val_log_abl3), use_best_model=True)
    
    # XGBoost can't use enable_categorical=True without category dtype, so we disable it
    xgb_params_abl3 = copy.deepcopy(xgb_params_base)
    xgb_params_abl3['enable_categorical'] = False
    
    # XGBoost also can't handle string columns, so we must manually encode them.
    X_train_abl3_xgb = X_train_abl3.copy()
    X_val_abl3_xgb = X_val_abl3.copy()
    for col in cat_features_abl3:
        # Use pandas factorize to convert strings to numerical codes
        codes, uniques = pd.factorize(X_train_abl3[col])
        X_train_abl3_xgb[col] = codes
        
        # Apply the same mapping to the validation set
        # Create a map from unique values to codes
        unique_map = {val: i for i, val in enumerate(uniques)}
        # Map validation data, using -1 for unseen values
        X_val_abl3_xgb[col] = X_val_abl3[col].map(unique_map).fillna(-1).astype(int)

    xgb_model_abl3 = xgb.XGBRegressor(**xgb_params_abl3).fit(X_train_abl3_xgb, y_train_log_abl3, eval_set=[(X_val_abl3_xgb, y_val_log_abl3)], verbose=False)

    preds1_abl3 = model_cat_base_abl3.predict(X_val_abl3)
    preds2_abl3 = np.expm1(model_cat_log_abl3.predict(X_val_abl3))
    preds3_abl3 = np.expm1(xgb_model_abl3.predict(X_val_abl3_xgb))
    
    ensemble_preds_abl3 = (preds1_abl3 + preds2_abl3 + preds3_abl3) / 3.0
    rmse_abl3 = np.sqrt(mean_squared_error(y_val_base_abl3, np.maximum(0, ensemble_preds_abl3)))
    ablation_results['No Categorical Dtype Casting'] = rmse_abl3
    print(f"Ablation (No Categorical Dtype Casting) RMSE: {rmse_abl3:.4f}")
    
    # --- Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    baseline_score = ablation_results['Baseline']
    impact = {}
    for name, score in ablation_results.items():
        if name != 'Baseline':
            impact[name] = score - baseline_score

    if impact:
        most_impactful = max(impact, key=impact.get)
        print(f"The most impactful component was '{most_impactful}', as its removal increased RMSE by {impact[most_impactful]:.4f}.")
    else:
        print("Could not determine the most impactful component from the ablation study.")


if __name__ == '__main__':
    # Create dummy files for the script to run
    input_dir = './input'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    train_file_path = os.path.join(input_dir, 'violations_per_street_2022.csv')
    if not os.path.exists(train_file_path):
        street_names = [f'street_{i}' for i in range(50)] + [f'street_{i}' for i in range(50, 70, 2)]
        violation_types = [f'type_{i%5}' for i in range(60)]
        dummy_df = pd.DataFrame({
            'Street Name': np.random.choice(street_names, 200),
            'Violation Description': np.random.choice(violation_types, 200),
            'violation_count': np.random.randint(10, 1000, 200)
        })
        # Aggregate to match expected format
        dummy_df = dummy_df.groupby(['Street Name', 'Violation Description']).sum().reset_index()
        dummy_df.to_csv(train_file_path, index=False)
    
    main()
