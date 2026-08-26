
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import os
import warnings
import copy

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def feature_engineer(df, train_stats=None, fill_na_strategy='zero'):
    """
    Engineers features for the model.
    This version is adapted for the ablation study.
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
        
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Ablation point: `fillna` strategy
    if fill_na_strategy == 'zero':
        df_engineered.fillna(0, inplace=True)
    elif fill_na_strategy == 'global_mean':
        fill_value = stats.get('global_mean', 0)
        df_engineered.fillna(fill_value, inplace=True)
    else:
        raise ValueError(f"Unknown fill_na_strategy: {fill_na_strategy}")

    return df_engineered, stats


def run_experiment(train_df, val_df, experiment_name, model_params=None, fill_na_strategy='zero', use_boroname_ohe=True):
    """
    Runs a single training and evaluation experiment with specified configurations.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")

    # Feature Engineering
    train_featured, train_stats = feature_engineer(train_df, fill_na_strategy=fill_na_strategy)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, fill_na_strategy=fill_na_strategy)

    # Feature definitions
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    categorical_features = ['violation_description']
    # Ablation point: Include 'boroname' in one-hot encoding
    if use_boroname_ohe:
        categorical_features.append('boroname')

    all_features = numerical_features + categorical_features
    
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Model definition
    # Ablation point: LGBM hyperparameters
    if model_params is None:
        model_params = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 64, 'random_state': SEED}
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(**model_params))
    ])

    # Training and Evaluation
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    
    print(f"Loading and splitting data from {train_path}...")
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Exiting.")
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    # Experiment 1: Baseline (Full Model)
    # This uses the tuned hyperparameters, zero-filling for NaNs, and includes boroname in OHE.
    baseline_rmse = run_experiment(train_df, val_df, "Baseline (Full Model)")
    results["Baseline (Full Model)"] = baseline_rmse

    # Experiment 2: Ablation of Hyperparameter Tuning
    # Use default LGBMRegressor parameters instead of the manually tuned ones.
    default_lgbm_params = {'random_state': SEED}
    ablation_rmse_hparams = run_experiment(train_df, val_df, "Ablation: Default LGBM Hyperparameters", model_params=default_lgbm_params)
    results["Ablation: Default LGBM Hyperparameters"] = ablation_rmse_hparams

    # Experiment 3: Ablation of fillna(0) Strategy
    # Use the global mean to fill missing values from merges instead of zero.
    ablation_rmse_fill = run_experiment(train_df, val_df, "Ablation: Global Mean FillNA Strategy", fill_na_strategy='global_mean')
    results["Ablation: Global Mean FillNA Strategy"] = ablation_rmse_fill

    # Experiment 4: Ablation of Boroname One-Hot Encoding
    # Remove the 'boroname' column from the one-hot encoding step.
    ablation_rmse_boro_ohe = run_experiment(train_df, val_df, "Ablation: No 'boroname' One-Hot Encoding", use_boroname_ohe=False)
    results["Ablation: No 'boroname' One-Hot Encoding"] = ablation_rmse_boro_ohe

    # --- Analysis and Conclusion ---
    print("\n\n--- Ablation Study Summary ---")
    print(f"{'Experiment':<45} | {'Validation RMSE':<20} | {'Change from Baseline':<20}")
    print("-" * 90)

    performance_changes = {}
    for name, rmse in results.items():
        change = rmse - baseline_rmse
        change_str = f"{change:+.4f}" if name != "Baseline (Full Model)" else "-"
        print(f"{name:<45} | {rmse:<20.4f} | {change_str:<20}")
        if name != "Baseline (Full Model)":
            performance_changes[name] = change

    # Determine which ablation had the most negative impact (largest increase in RMSE)
    if not performance_changes:
        print("\nCould not run ablation studies to determine impact.")
        return
        
    most_impactful_component = max(performance_changes, key=performance_changes.get)
    # Rephrase the component name for the conclusion
    most_impactful_component_clean = most_impactful_component.replace("Ablation: ", "")

    print("\n--- Conclusion ---")
    print(f"The ablation study shows that '{most_impactful_component_clean}' contributes the most to the model's performance.")
    print("Its removal or alteration resulted in the largest increase in RMSE, indicating its critical role in the model's accuracy.")


if __name__ == '__main__':
    main()
