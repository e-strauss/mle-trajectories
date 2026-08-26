
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def clean_col_names(df):
    """Standardizes column names to be Python-friendly."""
    cols = df.columns
    new_cols = [col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').lower() for col in cols]
    df.columns = new_cols
    return df

def feature_engineer(df, is_train, train_artifacts=None, use_aggregates=True, use_external_data=True):
    """
    Applies feature engineering.
    Flags `use_aggregates` and `use_external_data` control which features are created for the ablation study.
    """
    input_dir = './input'
    df = clean_col_names(df.copy())

    if use_external_data:
        try:
            codes_df = pd.read_csv(os.path.join(input_dir, 'dof_parking_violation_codes.csv'))
            geo_df = pd.read_csv(os.path.join(input_dir, 'physical_id_to_address_name.csv'))
        except FileNotFoundError as e:
            print(f"Error: Augmentation file not found. Ensure '{e.filename}' is in the './input' directory.", file=sys.stderr)
            return None, None

        codes_df = clean_col_names(codes_df)
        geo_df = clean_col_names(geo_df)

        # 1. Merge violation code information (fine amount)
        df = pd.merge(df, codes_df[['definition', 'all_other_areas']],
                      left_on='violation_description', right_on='definition', how='left')
        df.drop('definition', axis=1, inplace=True)
        df = df.rename(columns={'all_other_areas': 'fine_amt'})

        # 2. Merge geographical information (Borough)
        boro_map = geo_df.groupby('st_name')['borocode'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1).reset_index()
        df = df.merge(boro_map, left_on='street_name', right_on='st_name', how='left')
        df.drop('st_name', axis=1, inplace=True)
        df['borocode'].fillna(-1, inplace=True)
        df['borocode'] = df['borocode'].astype(str)

    categorical_features = ['street_name', 'violation_description']
    if use_external_data:
        categorical_features.append('borocode')


    if is_train:
        train_artifacts = {}
        # a) Calculate and store median for imputation
        if use_external_data:
            fine_amt_median = df['fine_amt'].median()
            train_artifacts['fine_amt_median'] = fine_amt_median

        # b) Calculate and store aggregate features if enabled
        if use_aggregates:
            street_aggs = df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
            street_aggs.columns = [f'street_{agg}' for agg in street_aggs.columns]
            train_artifacts['street_aggs'] = street_aggs.reset_index()

            violation_aggs = df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
            violation_aggs.columns = [f'violation_{agg}' for agg in violation_aggs.columns]
            train_artifacts['violation_aggs'] = violation_aggs.reset_index()

        # c) Create and fit the OrdinalEncoder
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
        encoder.fit(df[categorical_features])
        train_artifacts['encoder'] = encoder

    # Apply transformations using artifacts
    if use_external_data:
        df['fine_amt'].fillna(train_artifacts['fine_amt_median'], inplace=True)

    if use_aggregates:
        # Merge aggregates
        df = df.merge(train_artifacts['street_aggs'], on='street_name', how='left')
        df = df.merge(train_artifacts['violation_aggs'], on='violation_description', how='left')

        # Fill NaNs that result from merges
        agg_cols = [col for col in df.columns if 'street_' in col or 'violation_' in col]
        for col in agg_cols:
            df[col].fillna(0, inplace=True)

    # Apply Ordinal Encoding to categorical features
    df[categorical_features] = train_artifacts['encoder'].transform(df[categorical_features])

    return df, train_artifacts

def run_experiment(train_df_raw, use_aggregates=True, use_ensemble=True):
    """
    Runs a single experiment variation for the ablation study.
    - Preprocesses data with specified features.
    - Trains either a single model or an ensemble.
    - Evaluates and returns the validation RMSE.
    """
    # 1. Split Data
    train_data_raw, val_data_raw = train_test_split(train_df_raw, test_size=0.2, random_state=42)

    # 2. Feature Engineering
    train_df_processed, train_artifacts = feature_engineer(train_data_raw, is_train=True, use_aggregates=use_aggregates, use_external_data=True)
    
    # FIX: Check if feature engineering failed due to missing files.
    if train_df_processed is None:
        return None # Propagate the failure signal to the main loop.

    val_df_processed, _ = feature_engineer(val_data_raw, is_train=False, train_artifacts=train_artifacts, use_aggregates=use_aggregates, use_external_data=True)
    
    # This second check is for robustness, though if the first passed, this should too.
    if val_df_processed is None:
        return None

    # 3. Prepare Data for Models
    target = 'violation_count'
    features = [col for col in train_df_processed.columns if col != target]
    X_train, y_train = train_df_processed[features], train_df_processed[target]
    X_val, y_val = val_df_processed[features], val_df_processed[target]

    categorical_feature_names = ['street_name', 'violation_description', 'borocode']
    categorical_feature_indices = [features.index(col) for col in categorical_feature_names if col in features]

    # 4. Model Training & Validation
    y_train_log = np.log1p(y_train)

    # --- LightGBM ---
    lgbm = lgb.LGBMRegressor(random_state=42, force_col_wise=True)
    lgbm.fit(X_train, y_train_log, categorical_feature=categorical_feature_indices, feature_name=features)
    val_preds_lgbm = np.maximum(0, np.expm1(lgbm.predict(X_val)))

    if not use_ensemble:
        rmse = np.sqrt(mean_squared_error(y_val, val_preds_lgbm))
        return rmse

    # --- Train other models only if ensembling ---
    xgbr = xgb.XGBRegressor(objective='count:poisson', random_state=42, n_estimators=100)
    xgbr.fit(X_train, y_train)
    val_preds_xgb = np.maximum(0, xgbr.predict(X_val))

    cat = cb.CatBoostRegressor(random_state=42, verbose=0, cat_features=categorical_feature_indices,
                               loss_function='RMSE', iterations=500, early_stopping_rounds=50)
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    val_preds_cat = np.maximum(0, cat.predict(X_val))

    hgb = HistGradientBoostingRegressor(loss='poisson', random_state=42, categorical_features=categorical_feature_indices)
    hgb.fit(X_train, y_train)
    val_preds_hgb = np.maximum(0, hgb.predict(X_val))

    rf_poisson = RandomForestRegressor(n_estimators=100, criterion='poisson', random_state=42, n_jobs=-1)
    rf_poisson.fit(X_train, y_train)
    val_preds_rf = np.maximum(0, rf_poisson.predict(X_val))

    # 5. Ensemble predictions and calculate final RMSE
    ensemble_val_preds = (val_preds_lgbm + val_preds_xgb + val_preds_cat + val_preds_hgb + val_preds_rf) / 5.0
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
    return rmse

def main_ablation():
    """Main function to run the ablation study."""
    train_file_path = os.path.join('./input', 'violations_per_street_2022.csv')
    try:
        train_df_raw_full = pd.read_csv(train_file_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at '{train_file_path}'", file=sys.stderr)
        return

    # Clean names once on the raw data
    train_df_raw_clean = clean_col_names(train_df_raw_full.copy())
    results = {}

    # --- Experiment 1: Baseline (Full Features, Ensemble) ---
    print("Running: Baseline (Full Features, 5-Model Ensemble)")
    baseline_rmse = run_experiment(train_df_raw_clean, use_aggregates=True, use_ensemble=True)
    results['Baseline'] = baseline_rmse
    if baseline_rmse is not None:
        print(f"Validation RMSE: {baseline_rmse:.4f}\n")
    else:
        print("Experiment failed, likely due to missing augmentation files.\n")

    # --- Experiment 2: Ablation on Aggregate Features ---
    print("Running: Ablation 1 (No Aggregate Features, 5-Model Ensemble)")
    no_agg_rmse = run_experiment(train_df_raw_clean, use_aggregates=False, use_ensemble=True)
    results['No Aggregate Features'] = no_agg_rmse
    if no_agg_rmse is not None:
        print(f"Validation RMSE: {no_agg_rmse:.4f}\n")
    else:
        print("Experiment failed, likely due to missing augmentation files.\n")

    # --- Experiment 3: Ablation on Ensemble ---
    print("Running: Ablation 2 (Full Features, Single Model - LightGBM)")
    single_model_rmse = run_experiment(train_df_raw_clean, use_aggregates=True, use_ensemble=False)
    results['Single Model (LGBM)'] = single_model_rmse
    if single_model_rmse is not None:
        print(f"Validation RMSE: {single_model_rmse:.4f}\n")
    else:
        print("Experiment failed, likely due to missing augmentation files.\n")

    # --- Print Final Performance Metric ---
    if baseline_rmse is not None:
        print(f"Final Validation Performance: {baseline_rmse:.4f}")

    # --- Conclusion ---
    print("\n--- Ablation Study Summary ---")
    for name, score in results.items():
        if score is not None:
            print(f"{name}: {score:.4f}")
        else:
            print(f"{name}: FAILED")
    
    # Safely calculate impacts
    baseline_score = results.get('Baseline')
    impact_of_aggregates = np.nan
    impact_of_ensemble = np.nan

    if baseline_score is not None:
        no_agg_score = results.get('No Aggregate Features')
        if no_agg_score is not None:
            impact_of_aggregates = no_agg_score - baseline_score

        single_model_score = results.get('Single Model (LGBM)')
        if single_model_score is not None:
            impact_of_ensemble = single_model_score - baseline_score

    print("\n--- Conclusion ---")
    
    # Compare impacts, handling cases where experiments may have failed (resulting in NaN)
    if np.isnan(impact_of_aggregates) and np.isnan(impact_of_ensemble):
        print("Could not determine feature importance due to failed experiments.")
    elif np.isnan(impact_of_ensemble) or (~np.isnan(impact_of_aggregates) and impact_of_aggregates > impact_of_ensemble):
        print(f"Removing aggregate features caused the largest drop in performance (RMSE increase of {impact_of_aggregates:.4f}).")
        print("Therefore, the aggregate features (street_aggs, violation_aggs) contribute the most to the model's performance.")
    elif np.isnan(impact_of_aggregates) or (~np.isnan(impact_of_ensemble) and impact_of_ensemble >= impact_of_aggregates):
        print(f"Removing the model ensemble caused the largest drop in performance (RMSE increase of {impact_of_ensemble:.4f}).")
        print("Therefore, the 5-model ensemble contributes the most to the model's performance.")
    else:
        # This case handles if both impacts are calculated and are exactly equal, or if both are NaN.
        print("Could not conclusively determine the most impactful component from the successful experiments.")


if __name__ == '__main__':
    if not os.path.exists('./input'):
        print("Error: './input' directory not found. Please ensure it exists and contains the required data files.", file=sys.stderr)
    else:
        main_ablation()
