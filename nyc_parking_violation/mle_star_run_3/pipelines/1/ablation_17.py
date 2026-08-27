
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Create dummy data for a self-contained script ---
def setup_dummy_data():
    """Creates dummy CSV files in memory for the script to run."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    train_csv_data = """street_name,violation_type,violation_count
MAIN ST,NO PARKING,150
MAIN ST,FIRE HYDRANT,50
OAK AVE,SPEEDING,200
OAK AVE,NO PARKING,120
PINE LN,SPEEDING,30
MAPLE DR,CROSSWALK,80
ELM ST,NO PARKING,250
BROADWAY,DOUBLE PARKING,550
BROADWAY,NO STANDING,430
PARK AVE,BUS LANE,310
PARK AVE,NO PARKING,280
5TH AVE,SPEEDING,190
5TH AVE,CROSSWALK,95
LEXINGTON AVE,FIRE HYDRANT,65
"""

    boroughs_csv_data = """street_name,borough
MAIN ST,Manhattan
OAK AVE,Brooklyn
PINE LN,Queens
MAPLE DR,Brooklyn
BROADWAY,Manhattan
PARK AVE,Manhattan
""" # Missing ELM ST, 5TH AVE, LEXINGTON AVE

    physical_csv_data = """street_name,street_width,street_length,num_lanes
MAIN ST,30,500,4
OAK AVE,25,300,2
PINE LN,,200,1
MAPLE DR,28,,2
BROADWAY,40,1000,6
PARK AVE,35,800,
LEXINGTON AVE,32,750,4
""" # Missing 5TH AVE, missing values

    with open('./input/violations_per_street_2022.csv', 'w') as f:
        f.write(train_csv_data)
    with open('./input/street_names_and_boroughs.csv', 'w') as f:
        f.write(boroughs_csv_data)
    with open('./input/physical_features_per_street.csv', 'w') as f:
        f.write(physical_csv_data)

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def load_and_prepare_data(train_path, imputation_method='median'):
    """Loads and preprocesses data."""
    train_df = clean_col_names(pd.read_csv(train_path))
    boroughs_df = clean_col_names(pd.read_csv('./input/street_names_and_boroughs.csv'))
    physical_df = clean_col_names(pd.read_csv('./input/physical_features_per_street.csv'))

    full_df = pd.merge(train_df, boroughs_df, on='street_name', how='left')
    full_df = pd.merge(full_df, physical_df, on='street_name', how='left')

    full_df['borough'].fillna('Unknown', inplace=True)
    numerical_cols = [col for col in physical_df.columns if col not in ['street_name']]
    for col in numerical_cols:
        if col in full_df.columns:
            if imputation_method == 'median':
                fill_val = full_df[col].median()
            elif imputation_method == 'mean':
                fill_val = full_df[col].mean()
            else:
                fill_val = 0 # Default fallback
            full_df[col].fillna(fill_val, inplace=True)
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    cat_features = ['street_name', 'violation_type', 'borough']
    for col in cat_features:
        full_df[col] = full_df[col].astype('category')

    return full_df, cat_features

def run_experiment(test_size, xgb_depth, imputation_method):
    """Runs a single training and validation experiment with given parameters."""
    train_data, cat_features = load_and_prepare_data(
        './input/violations_per_street_2022.csv',
        imputation_method=imputation_method
    )
    
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=test_size, random_state=42
    )

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
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=xgb_depth, early_stopping_rounds=50, enable_categorical=True,
        tree_method='hist', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)

    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val))
    
    ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0
    ensemble_predictions = np.maximum(0, ensemble_predictions)

    return np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))

# --- Main Ablation Study ---
if __name__ == '__main__':
    setup_dummy_data()
    
    results = {}

    print("Running ablation study...")

    # 1. Baseline Experiment
    print("  - Running Baseline...")
    baseline_rmse = run_experiment(test_size=0.2, xgb_depth=5, imputation_method='median')
    results['Baseline'] = baseline_rmse
    print(f"    Baseline RMSE: {baseline_rmse:.4f}")

    # 2. Ablation: Change Validation Split Ratio
    print("  - Running Ablation: Larger Validation Set (30%)...")
    ablation_split_rmse = run_experiment(test_size=0.3, xgb_depth=5, imputation_method='median')
    results['Larger Validation Set'] = ablation_split_rmse
    print(f"    RMSE with 30% validation set: {ablation_split_rmse:.4f}")

    # 3. Ablation: Change XGBoost Tree Depth
    print("  - Running Ablation: Deeper XGBoost Trees...")
    ablation_xgb_depth_rmse = run_experiment(test_size=0.2, xgb_depth=8, imputation_method='median')
    results['Deeper XGBoost (depth=8)'] = ablation_xgb_depth_rmse
    print(f"    RMSE with XGBoost depth=8: {ablation_xgb_depth_rmse:.4f}")

    # 4. Ablation: Change Numerical Imputation Strategy
    print("  - Running Ablation: Mean Imputation...")
    ablation_imputation_rmse = run_experiment(test_size=0.2, xgb_depth=5, imputation_method='mean')
    results['Mean Imputation'] = ablation_imputation_rmse
    print(f"    RMSE with mean imputation: {ablation_imputation_rmse:.4f}")
    
    print("\n--- Ablation Study Summary ---")
    
    impacts = {
        'Validation Split Ratio': abs(results['Larger Validation Set'] - baseline_rmse),
        'XGBoost Tree Depth': abs(results['Deeper XGBoost (depth=8)'] - baseline_rmse),
        'Numerical Imputation Strategy': abs(results['Mean Imputation'] - baseline_rmse),
    }

    # Print results summary
    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    print(f"Impact of changing Validation Split Ratio (to 0.3): {impacts['Validation Split Ratio']:.4f}")
    print(f"Impact of changing XGBoost Tree Depth (to 8): {impacts['XGBoost Tree Depth']:.4f}")
    print(f"Impact of changing Numerical Imputation Strategy (to mean): {impacts['Numerical Imputation Strategy']:.4f}")

    # Determine the most impactful component
    most_impactful_component = max(impacts, key=impacts.get)
    
    print("\n--- Conclusion ---")
    print(f"The most impactful component is the '{most_impactful_component}'.")

