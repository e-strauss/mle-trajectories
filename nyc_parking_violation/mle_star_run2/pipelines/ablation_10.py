
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import copy

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- Original Feature Engineering Function (unchanged) ---
def feature_engineer(df, train_stats=None):
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        try:
            cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
            cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
            cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
            df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
            df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
            df_engineered['boroname'].fillna('Unknown', inplace=True)
            df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
        except Exception:
            df_engineered['boroname'] = 'Unknown'
    else:
        df_engineered['boroname'] = 'Unknown'

    has_target = 'violation_count' in df_engineered.columns
    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        stats = {
            'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(df_original, experiment_name, seed=42, test_size=0.2, use_sparse_ohe=False):
    """
    Runs a single training and validation experiment with configurable parameters.
    """
    # --- 1. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- 2. Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    all_features = numerical_features + categorical_features
    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- 3. Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=use_sparse_ohe), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Use seed=None for the 'No Seed' ablation
    lgbm_random_state = None if seed is None else seed
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=lgbm_random_state, verbosity=-1))
    ])

    # --- 4. Training & Validation ---
    model_pipeline.fit(X_train, y_train)
    val_predictions = model_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"{experiment_name}: Validation RMSE = {rmse:.4f}")
    return rmse


if __name__ == '__main__':
    # Load data
    try:
        df_main = pd.read_csv('./input/violations_per_street_2022.csv')
    except FileNotFoundError:
        print("Error: Training file not found at ./input/violations_per_street_2022.csv. Exiting.")
        exit()

    results = {}

    # --- Baseline Experiment ---
    # Reproducible seed, 20% test size, dense OHE output
    np.random.seed(42)
    results['Baseline'] = run_experiment(df_main, "Baseline (Reproducible Seed, 20% Test Size, Dense OHE)", seed=42, test_size=0.2, use_sparse_ohe=False)
    
    # --- Ablation 1: Remove Reproducibility (Seed) ---
    # Using seed=None makes the split and model initialization random
    results['No Seed'] = run_experiment(df_main, "Ablation 1 (No Seed)", seed=None, test_size=0.2, use_sparse_ohe=False)

    # --- Ablation 2: Change Validation Split Size ---
    # Change test_size from 0.2 to 0.3
    np.random.seed(42)
    results['Larger Test Set'] = run_experiment(df_main, "Ablation 2 (Larger Test Set - 30%)", seed=42, test_size=0.3, use_sparse_ohe=False)
    
    # --- Ablation 3: Change OHE Sparsity ---
    # Change OneHotEncoder to produce a sparse matrix
    np.random.seed(42)
    results['Sparse OHE'] = run_experiment(df_main, "Ablation 3 (Sparse OHE Output)", seed=42, test_size=0.2, use_sparse_ohe=True)

    print("-" * 50)

    # --- Conclusion ---
    baseline_rmse = results['Baseline']
    
    impacts = {
        'Reproducibility (Seed)': abs(results['No Seed'] - baseline_rmse),
        'Validation Split Size (20% vs 30%)': abs(results['Larger Test Set'] - baseline_rmse),
        'OHE Sparsity (Dense vs Sparse)': abs(results['Sparse OHE'] - baseline_rmse)
    }

    if not impacts:
        most_impactful_component = "N/A"
    else:
        most_impactful_component = max(impacts, key=impacts.get)
        
    print(f"The ablation study shows that '{most_impactful_component}' has the most significant impact on model performance.")

