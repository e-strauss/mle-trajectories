
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import warnings

# Suppress LightGBM warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_preprocess_data():
    """
    Generates a reproducible dummy dataset and performs initial cleaning.
    The data is designed to have rare categories to test NaN handling.
    """
    # Create a synthetic dataset
    # The original code had a mismatch in array lengths.
    # 'Street Name' had 55 elements, while the others had 56.
    # Added one more street name ('HICKORY ST') to make all arrays of length 56.
    data = {
        'Street Name': ['MAIN ST'] * 20 + ['BROADWAY'] * 15 + ['ELM ST'] * 10 + ['OAK AVE'] * 5 + ['PINE LN'] * 3 + ['CEDAR CT', 'MAPLE DR', 'HICKORY ST'],
        'Violation Description': ['NO PARKING'] * 30 + ['EXPIRED METER'] * 20 + ['FIRE HYDRANT'] * 5 + ['CROSSWALK'],
        'Violation Count': np.random.randint(1, 100, size=56)
    }
    df = pd.DataFrame(data)
    
    # Introduce some pattern
    street_map = {name: i*10 for i, name in enumerate(df['Street Name'].unique())}
    desc_map = {name: i*5 for i, name in enumerate(df['Violation Description'].unique())}
    df['Violation Count'] = df['Violation Count'] + df['Street Name'].map(street_map) + df['Violation Description'].map(desc_map)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def run_ablation_study():
    """
    Runs an ablation study on a LightGBM model pipeline, testing the impact
    of the model type and the NaN filling strategy for mean-encoded features.
    """
    results = {}
    base_df = load_and_preprocess_data()

    # --- Feature Engineering (Common for all experiments) ---
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    base_df['street_name_encoded'] = le_street.fit_transform(base_df['street_name'])
    base_df['violation_description_encoded'] = le_desc.fit_transform(base_df['violation_description'])
    base_df['log_target'] = np.log1p(base_df['violation_count'])

    # --- 1. Baseline Experiment ---
    # Model: LightGBM
    # NaN Fill Strategy: Global Mean of Log Target
    df = base_df.copy()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    desc_mean_map = train_df.groupby('violation_description_encoded')['log_target'].mean()
    street_mean_map = train_df.groupby('street_name_encoded')['log_target'].mean()

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df['description_mean_count'] = train_df['violation_description_encoded'].map(desc_mean_map)
    train_df['street_mean_count'] = train_df['street_name_encoded'].map(street_mean_map)
    val_df['description_mean_count'] = val_df['violation_description_encoded'].map(desc_mean_map)
    val_df['street_mean_count'] = val_df['street_name_encoded'].map(street_mean_map)

    global_mean_log_target = train_df['log_target'].mean()
    val_df['description_mean_count'].fillna(global_mean_log_target, inplace=True)
    val_df['street_mean_count'].fillna(global_mean_log_target, inplace=True)

    features = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    X_train, y_train = train_df[features], train_df['log_target']
    X_val, y_val = val_df[features], val_df['log_target']
    
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    
    baseline_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), preds))
    results['Baseline (LGBM, Fill with Mean)'] = baseline_rmse

    # --- 2. Ablation 1: Change NaN Fill Strategy ---
    # Model: LightGBM
    # NaN Fill Strategy: Zero
    df = base_df.copy()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    desc_mean_map = train_df.groupby('violation_description_encoded')['log_target'].mean()
    street_mean_map = train_df.groupby('street_name_encoded')['log_target'].mean()

    train_df = train_df.copy(); val_df = val_df.copy()
    train_df['description_mean_count'] = train_df['violation_description_encoded'].map(desc_mean_map)
    train_df['street_mean_count'] = train_df['street_name_encoded'].map(street_mean_map)
    val_df['description_mean_count'] = val_df['violation_description_encoded'].map(desc_mean_map)
    val_df['street_mean_count'] = val_df['street_name_encoded'].map(street_mean_map)

    val_df.fillna(0, inplace=True) # Changed strategy

    X_train, y_train = train_df[features], train_df['log_target']
    X_val, y_val = val_df[features], val_df['log_target']
    
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    
    ablation1_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), preds))
    results['Ablation (LGBM, Fill with Zero)'] = ablation1_rmse

    # --- 3. Ablation 2: Change Model Type ---
    # Model: Linear Regression
    # NaN Fill Strategy: Global Mean of Log Target
    df = base_df.copy()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    desc_mean_map = train_df.groupby('violation_description_encoded')['log_target'].mean()
    street_mean_map = train_df.groupby('street_name_encoded')['log_target'].mean()

    train_df = train_df.copy(); val_df = val_df.copy()
    train_df['description_mean_count'] = train_df['violation_description_encoded'].map(desc_mean_map)
    train_df['street_mean_count'] = train_df['street_name_encoded'].map(street_mean_map)
    val_df['description_mean_count'] = val_df['violation_description_encoded'].map(desc_mean_map)
    val_df['street_mean_count'] = val_df['street_name_encoded'].map(street_mean_map)

    global_mean_log_target = train_df['log_target'].mean()
    val_df['description_mean_count'].fillna(global_mean_log_target, inplace=True)
    val_df['street_mean_count'].fillna(global_mean_log_target, inplace=True)

    X_train, y_train = train_df[features], train_df['log_target']
    X_val, y_val = val_df[features], val_df['log_target']

    model = LinearRegression() # Changed model
    model.fit(X_train, y_train)

    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0

    ablation2_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), preds))
    results['Ablation (Linear Model, Fill with Mean)'] = ablation2_rmse

    # --- Print and Analyze Results ---
    print("--- Ablation Study Results (RMSE) ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    impact = {
        'NaN Fill Strategy (Zero vs Mean)': ablation1_rmse - baseline_rmse,
        'Model Type (Linear vs LGBM)': ablation2_rmse - baseline_rmse,
    }

    print("\n--- Performance Impact (Degradation in RMSE) ---")
    for name, effect in impact.items():
        print(f"Impact of changing '{name}': {effect:+.4f}")
    
    if not impact:
        most_impactful = "N/A"
    else:
        most_impactful = max(impact, key=impact.get)

    print(f"\nConclusion: The component that contributes most to the overall performance is the '{most_impactful}'.")
    
    # Print the final validation performance as required
    final_validation_score = baseline_rmse
    print(f'Final Validation Performance: {final_validation_score}')


if __name__ == '__main__':
    run_ablation_study()
