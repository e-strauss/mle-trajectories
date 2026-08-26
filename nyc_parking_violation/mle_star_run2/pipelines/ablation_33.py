
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
from collections import OrderedDict

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def feature_engineer(df, train_stats=None, use_borough_data=True):
    """
    Engineers features for the model, with an option to ablate borough data.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data (Ablation Point) ---
    cscl_path = './input/nyc_cscl.csv'
    if use_borough_data and os.path.exists(cscl_path):
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

    # --- Create Aggregate Features ---
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

def run_experiment(train_df, val_df, exp_config):
    """
    Runs a single experiment based on the provided configuration.
    """
    print(f"--- Running: {exp_config['name']} ---")
    
    # --- 1. Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df, use_borough_data=exp_config['use_borough_data'])
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, use_borough_data=exp_config['use_borough_data'])

    # --- 2. Define Features and Target ---
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
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 4. Select Model (Ablation Point) ---
    model_name = exp_config['model']
    if model_name == 'RidgeCV':
        model = RidgeCV(
            alphas=np.logspace(-2, 2, 5), 
            cv=exp_config['ridge_cv'],
            scoring=exp_config.get('ridge_scoring') # Use .get for safety
        )
    elif model_name == 'LinearRegression':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

    # --- 5. Training and Validation ---
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Validation RMSE: {rmse:.4f}\n")
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    print(f"Loading training data from {train_path}...")
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Exiting.")
        return

    print("Splitting data into train and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Define Experiments ---
    experiments = OrderedDict([
        ('Baseline', {
            'name': 'Baseline (RidgeCV)',
            'model': 'RidgeCV',
            'ridge_cv': 5,
            'use_borough_data': True
        }),
        ('No Regularization', {
            'name': 'Ablation: No Regularization (LinearRegression)',
            'model': 'LinearRegression',
            'ridge_cv': None,
            'use_borough_data': True
        }),
        ('No Borough Data', {
            'name': 'Ablation: No Borough Data Augmentation',
            'model': 'RidgeCV',
            'ridge_cv': 5,
            'use_borough_data': False
        }),
        ('3-Fold CV', {
            'name': 'Ablation: 3-Fold CV in Ridge',
            'model': 'RidgeCV',
            'ridge_cv': 3,
            'use_borough_data': True
        })
    ])

    results = {}
    for name, config in experiments.items():
        results[name] = run_experiment(train_df.copy(), val_df.copy(), config)

    # --- Print Summary and Conclusion ---
    baseline_rmse = results['Baseline']
    print("--- Ablation Study Results ---")
    print(f"{'Experiment':<45} | {'Validation RMSE':<20} | {'Change from Baseline':<20}")
    print("-" * 90)

    performance_changes = {}
    for name, rmse in results.items():
        change = rmse - baseline_rmse
        print(f"{experiments[name]['name']:<45} | {rmse:<20.4f} | {change:<+20.4f}")
        if name != 'Baseline':
            performance_changes[name] = abs(change)

    most_impactful_component = max(performance_changes, key=performance_changes.get)
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

if __name__ == '__main__':
    main()
