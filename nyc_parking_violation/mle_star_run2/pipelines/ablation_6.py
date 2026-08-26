
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def feature_engineer_baseline(df, train_stats=None):
    """
    Baseline feature engineering with regularized target encoding and smart NaN filling.
    """
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
        
        smoothing_factor = 20
        global_mean = df_engineered['violation_count'].mean()

        # Regularized encoding for street
        street_agg_raw = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'count', 'sum', 'std'])
        street_agg_raw['street_mean'] = ((street_agg_raw['mean'] * street_agg_raw['count'] + global_mean * smoothing_factor) / (street_agg_raw['count'] + smoothing_factor))
        street_agg = street_agg_raw.rename(columns={'sum': 'street_sum', 'std': 'street_std', 'count': 'street_key_count'})[['street_mean', 'street_sum', 'street_std', 'street_key_count']]

        # Regularized encoding for violation
        violation_agg_raw = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'count', 'sum', 'std'])
        violation_agg_raw['violation_mean'] = ((violation_agg_raw['mean'] * violation_agg_raw['count'] + global_mean * smoothing_factor) / (violation_agg_raw['count'] + smoothing_factor))
        violation_agg = violation_agg_raw.rename(columns={'sum': 'violation_sum', 'std': 'violation_std'})[['violation_mean', 'violation_sum', 'violation_std']]

        # Regularized encoding for borough
        boro_agg_raw = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'count', 'sum', 'std'])
        boro_agg_raw['boro_mean'] = ((boro_agg_raw['mean'] * boro_agg_raw['count'] + global_mean * smoothing_factor) / (boro_agg_raw['count'] + smoothing_factor))
        boro_agg = boro_agg_raw.rename(columns={'sum': 'boro_sum', 'std': 'boro_std'})[['boro_mean', 'boro_sum', 'boro_std']]
        
        stats = {'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg, 'global_mean': global_mean}
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Smart NaN filling
    global_fill_value = stats.get('global_mean', df_engineered['violation_count'].mean() if has_target else 0)
    for col in ['street_mean', 'violation_mean', 'boro_mean']:
        if col in df_engineered.columns:
            df_engineered[col].fillna(global_fill_value, inplace=True)
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def feature_engineer_no_regularization(df, train_stats=None):
    """
    Ablation version: Removes regularized target encoding, using simple mean instead.
    """
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
        
        # Simple mean aggregation (NO REGULARIZATION)
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        global_mean = df_engineered['violation_count'].mean()
        stats = {'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg, 'global_mean': global_mean}
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Keep smart NaN filling to isolate the effect of regularization
    global_fill_value = stats.get('global_mean', df_engineered['violation_count'].mean() if has_target else 0)
    for col in ['street_mean', 'violation_mean', 'boro_mean']:
        if col in df_engineered.columns:
            df_engineered[col].fillna(global_fill_value, inplace=True)
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def feature_engineer_simple_nan_fill(df, train_stats=None):
    """
    Ablation version: Uses simple fillna(0) instead of smart filling with global mean.
    """
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

    # Keep regularization from baseline
    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        smoothing_factor = 20
        global_mean = df_engineered['violation_count'].mean()

        street_agg_raw = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'count', 'sum', 'std'])
        street_agg_raw['street_mean'] = ((street_agg_raw['mean'] * street_agg_raw['count'] + global_mean * smoothing_factor) / (street_agg_raw['count'] + smoothing_factor))
        street_agg = street_agg_raw.rename(columns={'sum': 'street_sum', 'std': 'street_std', 'count': 'street_key_count'})[['street_mean', 'street_sum', 'street_std', 'street_key_count']]

        violation_agg_raw = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'count', 'sum', 'std'])
        violation_agg_raw['violation_mean'] = ((violation_agg_raw['mean'] * violation_agg_raw['count'] + global_mean * smoothing_factor) / (violation_agg_raw['count'] + smoothing_factor))
        violation_agg = violation_agg_raw.rename(columns={'sum': 'violation_sum', 'std': 'violation_std'})[['violation_mean', 'violation_sum', 'violation_std']]
        
        boro_agg_raw = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'count', 'sum', 'std'])
        boro_agg_raw['boro_mean'] = ((boro_agg_raw['mean'] * boro_agg_raw['count'] + global_mean * smoothing_factor) / (boro_agg_raw['count'] + smoothing_factor))
        boro_agg = boro_agg_raw.rename(columns={'sum': 'boro_sum', 'std': 'boro_std'})[['boro_mean', 'boro_sum', 'boro_std']]
        
        stats = {'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg, 'global_mean': global_mean}
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # SIMPLE NaN FILLING
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(feature_engineer_func, df_original):
    """
    Runs a single experiment with a given feature engineering function.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    train_featured, train_stats = feature_engineer_func(train_df)
    val_featured, _ = feature_engineer_func(val_df, train_stats=train_stats)

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    print(f"Loading data from {train_path}...")
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Exiting.")
        return

    results = {}

    print("\nRunning Baseline (Regularized Encoding + Smart NaN Fill)...")
    results['Baseline'] = run_experiment(feature_engineer_baseline, df_original)

    print("Running Ablation 1 (No Regularized Encoding)...")
    results['Ablation: No Regularization'] = run_experiment(feature_engineer_no_regularization, df_original)
    
    print("Running Ablation 2 (Simple NaN Fill)...")
    results['Ablation: Simple NaN Fill'] = run_experiment(feature_engineer_simple_nan_fill, df_original)

    print("\n--- Ablation Study Results ---")
    baseline_rmse = results['Baseline']
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    performance_impact = {}

    rmse_no_reg = results['Ablation: No Regularization']
    impact_no_reg = rmse_no_reg - baseline_rmse
    performance_impact['Regularized Encoding'] = impact_no_reg
    print(f"Impact of Removing 'Regularized Encoding': RMSE changed by {impact_no_reg:+.4f} (New RMSE: {rmse_no_reg:.4f})")

    rmse_simple_nan = results['Ablation: Simple NaN Fill']
    impact_simple_nan = rmse_simple_nan - baseline_rmse
    performance_impact['Smart NaN Fill'] = impact_simple_nan
    print(f"Impact of Removing 'Smart NaN Fill': RMSE changed by {impact_simple_nan:+.4f} (New RMSE: {rmse_simple_nan:.4f})")

    print("\n--- Conclusion ---")
    # A larger positive change indicates that removing the component hurt performance more,
    # meaning the component was more beneficial.
    most_impactful_component = max(performance_impact, key=performance_impact.get)
    print(f"The component that contributes most to the model's performance is: '{most_impactful_component}'")

if __name__ == '__main__':
    main()
