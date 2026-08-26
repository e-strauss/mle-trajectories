
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
warnings.filterwarnings("ignore", category=FutureWarning)


def create_dummy_data():
    """Creates dummy data files for the script to run."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    train_data = {
        'Street Name': ['A St', 'B St', 'C St', 'A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St', 'H St', 'I St', 'J St', 'K St', 'L St'],
        'Violation Type': ['Parking', 'Speeding', 'Parking', 'Speeding', 'Double Park', 'Double Park', 'Parking', 'Speeding', 'Parking', 'Speeding', 'Double Park', 'Parking', 'Speeding', 'Parking', 'Double Park'],
        'Violation Count': [100, 50, 200, 60, 20, 25, 110, 55, 220, 70, 15, 95, 45, 210, 30]
    }
    pd.DataFrame(train_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    boroughs_data = {
        'Street Name': ['A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St', 'H St', 'I St', 'J St', 'K St', 'L St'],
        'Borough': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn', 'Queens', 'Bronx', 'Bronx', 'Manhattan', 'Brooklyn', 'Staten Island', 'Staten Island']
    }
    pd.DataFrame(boroughs_data).to_csv('./input/street_names_and_boroughs.csv', index=False)

    physical_data = {
        'Street Name': ['A St', 'B St', 'C St', 'D St', 'E St', 'F St', 'G St', 'I St', 'J St', 'K St'],
        'Street Width': [30, 25, 35, 30, 25, 35, 40, 28, 26, 33],
        'Pavement Quality': [8, 7, 9, 8, 6, 9, 5, 8, 7, 9]
    }
    pd.DataFrame(physical_data).to_csv('./input/physical_features_per_street.csv', index=False)


def clean_col_names(df):
    """Standardizes column names by lowercasing and replacing spaces with underscores."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path, use_target_encoding=True, smoothing_factor=30.0, n_splits_te=5):
    """Loads, preprocesses, and prepares data, with configurable target encoding."""
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

    features_to_encode = ['street_name', 'borough']
    if use_target_encoding:
        target = 'violation_count'
        global_mean = full_df[target].mean()

        for col in features_to_encode:
            new_col_name = f'{col}_te'
            full_df[new_col_name] = 0.0
            kf = KFold(n_splits=n_splits_te, shuffle=True, random_state=42)

            for train_index, val_index in kf.split(full_df):
                train_fold, val_fold = full_df.iloc[train_index], full_df.iloc[val_index]
                agg = train_fold.groupby(col)[target].agg(['mean', 'count'])
                counts = agg['count']
                means = agg['mean']
                smoothed_means = (counts * means + smoothing_factor * global_mean) / (counts + smoothing_factor)
                full_df.loc[full_df.index[val_index], new_col_name] = val_fold[col].map(smoothed_means).fillna(global_mean)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features


def run_experiment(ablation_name, train_path, use_target_encoding, smoothing_factor, n_splits_te):
    """Runs a single training and validation experiment with given settings."""
    print(f"\n--- Running Experiment: {ablation_name} ---")
    
    train_data, cat_features = load_and_prepare_data(
        train_path,
        use_target_encoding=use_target_encoding,
        smoothing_factor=smoothing_factor,
        n_splits_te=n_splits_te
    )

    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    
    base_features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    features = base_features
    if not use_target_encoding:
        # If TE is off, remove the placeholder columns if they somehow exist, though they shouldn't
        features = [f for f in features if not f.endswith('_te')]

    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.3, random_state=42
    )

    cat_params = {
        'iterations': 1000, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features if use_target_encoding else [f for f in cat_features if f in X_train.columns],
        'early_stopping_rounds': 50
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base))
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log))

    # XGBoost requires manual handling if native support is not used with TE off
    X_train_xgb, X_val_xgb = X_train.copy(), X_val.copy()
    if not use_target_encoding:
        for col in cat_features:
            if col in X_train_xgb.columns:
                X_train_xgb[col] = X_train_xgb[col].cat.codes
                X_val_xgb[col] = X_val_xgb[col].cat.codes
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=50, enable_categorical=use_target_encoding,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)

    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    print(f"Validation RMSE: {val_rmse:.4f}")
    return val_rmse

if __name__ == '__main__':
    create_dummy_data()
    train_file_path = './input/violations_per_street_2022.csv'
    results = {}

    baseline_rmse = run_experiment(
        ablation_name="Baseline (Full Model with K-Fold TE)",
        train_path=train_file_path,
        use_target_encoding=True,
        smoothing_factor=30.0,
        n_splits_te=5
    )
    results["Baseline"] = baseline_rmse

    no_te_rmse = run_experiment(
        ablation_name="Ablation: No K-Fold Target Encoding",
        train_path=train_file_path,
        use_target_encoding=False,
        smoothing_factor=30.0,
        n_splits_te=5
    )
    results["No Target Encoding"] = no_te_rmse

    reduced_smoothing_rmse = run_experiment(
        ablation_name="Ablation: Reduced Smoothing in TE (factor=1.0)",
        train_path=train_file_path,
        use_target_encoding=True,
        smoothing_factor=1.0,
        n_splits_te=5
    )
    results["Reduced Smoothing"] = reduced_smoothing_rmse
    
    fewer_folds_rmse = run_experiment(
        ablation_name="Ablation: Fewer Folds for TE (n_splits=3)",
        train_path=train_file_path,
        use_target_encoding=True,
        smoothing_factor=30.0,
        n_splits_te=3
    )
    results["Fewer Folds for TE"] = fewer_folds_rmse

    print("\n\n--- Ablation Study Summary ---")
    impacts = {}
    print(f"Baseline RMSE: {results['Baseline']:.4f}")

    impact_no_te = results["No Target Encoding"] - results["Baseline"]
    impacts["K-Fold Target Encoding"] = abs(impact_no_te)
    print(f"Removing K-Fold Target Encoding      | New RMSE: {results['No Target Encoding']:.4f} | Impact: {impact_no_te:+.4f}")

    impact_smoothing = results["Reduced Smoothing"] - results["Baseline"]
    impacts["Smoothing Factor in TE"] = abs(impact_smoothing)
    print(f"Reducing Smoothing Factor in TE      | New RMSE: {results['Reduced Smoothing']:.4f} | Impact: {impact_smoothing:+.4f}")
    
    impact_folds = results["Fewer Folds for TE"] - results["Baseline"]
    impacts["Number of Folds in TE"] = abs(impact_folds)
    print(f"Reducing Number of Folds in TE       | New RMSE: {results['Fewer Folds for TE']:.4f} | Impact: {impact_folds:+.4f}")

    most_impactful_component = max(impacts, key=impacts.get)
    print(f"\nConclusion: The {most_impactful_component} is the most impactful component.")
