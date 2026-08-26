
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import shutil
import atexit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Setup for a self-contained script ---
# Create a dummy environment for the script to run
if not os.path.exists('./input'):
    os.makedirs('./input')

# Dummy training data
train_data_csv = """Street Name,Violation Description,Violation Count
MAIN ST,DOUBLE PARKING,150
OAK AVE,NO PARKING,200
PINE LN,BUS LANE,50
MAPLE DR,DOUBLE PARKING,300
ELM ST,NO STANDING,220
BROADWAY,FIRE HYDRANT,80
WALL ST,NO PARKING,450
LINCOLN AVE,BUS LANE,120
JEFFERSON ST,NO PARKING,250
WASHINGTON BLVD,DOUBLE PARKING,180
MADISON AVE,NO STANDING,280
"""

# Dummy augmentation data
boroughs_csv = """Street Name,Borough
MAIN ST,Queens
OAK AVE,Queens
PINE LN,Brooklyn
MAPLE DR,Manhattan
ELM ST,Bronx
BROADWAY,Manhattan
WALL ST,Manhattan
LINCOLN AVE,Brooklyn
JEFFERSON ST,Staten Island
WASHINGTON BLVD,Manhattan
MADISON AVE,Manhattan
"""

physical_features_csv = """Street Name,Length (m),Lanes,Has Bike Lane
MAIN ST,1200,4,1
OAK AVE,800,2,0
PINE LN,500,2,1
MAPLE DR,1500,6,0
ELM ST,750,2,0
BROADWAY,2000,4,1
WALL ST,600,2,0
LINCOLN AVE,900,2,1
JEFFERSON ST,400,2,0
WASHINGTON BLVD,1800,4,0
MADISON AVE,1600,4,1
"""

with open('./input/violations_per_street_2022.csv', 'w') as f:
    f.write(train_data_csv)
with open('./input/street_names_and_boroughs.csv', 'w') as f:
    f.write(boroughs_csv)
with open('./input/physical_features_per_street.csv', 'w') as f:
    f.write(physical_features_csv)

# Cleanup function to remove dummy files and directories
def cleanup():
    if os.path.exists('./input'):
        shutil.rmtree('./input')
atexit.register(cleanup)

# --- Core Logic from the Original Script ---

def clean_col_names(df):
    """Standardizes column names."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    if 'violation_description' in df.columns:
        df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
    return df

def load_and_prepare_data(train_path):
    """Loads and preprocesses training data."""
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

def run_experiment(config):
    """
    Runs a single training and validation experiment based on a configuration.
    """
    # Unpack configuration
    use_stacking = config.get('use_stacking', True)
    use_xgb_native_cat = config.get('use_xgb_native_cat', False)
    clip_negatives = config.get('clip_negatives', True)

    # 1. Data Preparation
    train_data, cat_features = load_and_prepare_data('./input/violations_per_street_2022.csv')
    train_data['log_violation_count'] = np.log1p(train_data['violation_count'])
    features = [col for col in train_data.columns if col not in ['violation_count', 'log_violation_count']]
    
    X = train_data[features]
    y_base = train_data['violation_count']
    y_log = train_data['log_violation_count']
    
    X_train, X_val, y_train_base, y_val_base, y_train_log, y_val_log = train_test_split(
        X, y_base, y_log, test_size=0.3, random_state=42
    )

    # 2. Model Training
    cat_params = {
        'iterations': 500, 'learning_rate': 0.05, 'loss_function': 'RMSE',
        'eval_metric': 'RMSE', 'random_seed': 42, 'verbose': 0,
        'cat_features': cat_features, 'early_stopping_rounds': 20, 'depth': 6
    }
    model_cat_base = CatBoostRegressor(**cat_params)
    model_cat_base.fit(X_train, y_train_base, eval_set=(X_val, y_val_base), use_best_model=True)
    
    model_cat_log = CatBoostRegressor(**cat_params)
    model_cat_log.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), use_best_model=True)

    # XGBoost Training with configuration
    if use_xgb_native_cat:
        X_train_xgb, X_val_xgb = X_train, X_val
        xgb_params = {'enable_categorical': True, 'tree_method': 'hist'}
    else:
        X_train_xgb, X_val_xgb = X_train.copy(), X_val.copy()
        for col in cat_features:
            X_train_xgb[col] = X_train_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
        xgb_params = {}

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=500, learning_rate=0.05,
        max_depth=5, early_stopping_rounds=20, random_state=42, n_jobs=-1, **xgb_params
    )
    xgb_model.fit(X_train_xgb, y_train_log, eval_set=[(X_val_xgb, y_val_log)], verbose=False)

    # 3. Validation and Ensembling
    val_preds_cat_base = model_cat_base.predict(X_val)
    val_preds_cat_log_transformed = np.expm1(model_cat_log.predict(X_val))
    val_preds_xgb_log_transformed = np.expm1(xgb_model.predict(X_val_xgb))

    if use_stacking:
        X_val_meta = np.column_stack((val_preds_cat_base, val_preds_cat_log_transformed, val_preds_xgb_log_transformed))
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(X_val_meta[:-1], y_val_base[:-1]) # Simulate fitting on a subset
        ensemble_predictions = meta_model.predict(X_val_meta)
    else: # Simple average
        ensemble_predictions = (val_preds_cat_base + val_preds_cat_log_transformed + val_preds_xgb_log_transformed) / 3.0

    if clip_negatives:
        ensemble_predictions = np.maximum(0, ensemble_predictions)

    val_rmse = np.sqrt(mean_squared_error(y_val_base, ensemble_predictions))
    return val_rmse

# --- Ablation Study Execution ---

if __name__ == '__main__':
    results = {}

    # 1. Baseline: Stacking + Manual XGBoost Encoding + Clipping
    baseline_config = {'use_stacking': True, 'use_xgb_native_cat': False, 'clip_negatives': True}
    baseline_rmse = run_experiment(baseline_config)
    results['Baseline'] = baseline_rmse

    # 2. Ablation: No Stacking (Simple Average Ensemble)
    no_stacking_config = {'use_stacking': False, 'use_xgb_native_cat': False, 'clip_negatives': True}
    results['No Stacking Ensemble'] = run_experiment(no_stacking_config)
    
    # 3. Ablation: Use XGBoost Native Categorical Support
    native_cat_config = {'use_stacking': True, 'use_xgb_native_cat': True, 'clip_negatives': True}
    results['XGBoost Native Categorical'] = run_experiment(native_cat_config)

    # 4. Ablation: No Negative Prediction Clipping
    no_clip_config = {'use_stacking': True, 'use_xgb_native_cat': False, 'clip_negatives': False}
    results['No Negative Clipping'] = run_experiment(no_clip_config)

    # --- Print and Analyze Results ---
    
    print("--- Ablation Study Results (RMSE on Validation Set) ---")
    print(f"{'Configuration':<30} | {'Validation RMSE':<20} | {'Change from Baseline':<20}")
    print("-" * 75)
    
    impacts = {}
    for name, rmse in results.items():
        change = rmse - baseline_rmse
        impacts[name] = change
        print(f"{name:<30} | {rmse:<20.4f} | {change:<+20.4f}")

    # Find the most impactful component (largest absolute change)
    impacts.pop('Baseline')
    most_impactful_component = max(impacts, key=lambda k: abs(impacts[k]))
    
    print("-" * 75)
    print(f"\nConclusion: The most impactful component is the '{most_impactful_component}'.")
    print("Its removal or modification caused the largest change in performance, highlighting its critical role in the model's architecture.")

