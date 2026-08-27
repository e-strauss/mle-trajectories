
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import collections

def create_dummy_files():
    """Creates dummy CSV files for a self-contained, runnable example."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # Create dummy violation data
    violations_data = {
        'Street Name': ['BROADWAY'] * 5 + ['5TH AVE'] * 5 + ['MAIN ST'] * 5 + ['WALL ST'] * 5,
        'Violation Description': ['NO PARKING-STREET CLEANING', 'FAIL TO DISP MUNI METER RECPT', 'NO PARKING-STREET CLEANING', 'FIRE HYDRANT', 'DOUBLE PARKING'] * 4,
        'violation_count': [150, 200, 160, 80, 120, 250, 300, 260, 90, 140, 50, 70, 60, 30, 40, 400, 450, 410, 200, 220]
    }
    violations_df = pd.DataFrame(violations_data)
    # Add some noise to make it more realistic
    np.random.seed(42)
    violations_df['violation_count'] += np.random.randint(-10, 10, size=len(violations_df))
    violations_df.to_csv('./input/violations_per_street_2022.csv', index=False)

    # Create dummy borough mapping data
    cscl_data = {
        'ST_NAME': ['BROADWAY', '5TH AVE', 'MAIN ST', 'PARK AVE'],
        'BORONAME': ['Manhattan', 'Manhattan', 'Queens', 'Manhattan']
    }
    cscl_df = pd.DataFrame(cscl_data)
    cscl_df.to_csv('./input/nyc_cscl.csv', index=False)

def feature_engineer(df, train_stats=None, nan_fill_strategy='zero'):
    """
    Engineers features for the model.
    This version is parameterized to allow ablation on the NaN filling strategy.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        cscl = pd.read_csv(cscl_path)
        # Standardize columns of the external dataframe to match the main one
        cscl.columns = [c.replace(' ', '_').lower() for c in cscl.columns]
        
        cscl = cscl[['st_name', 'boroname']].drop_duplicates(subset=['st_name'])
        cscl['st_name'] = cscl['st_name'].str.upper()
        
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='st_name', how='left')
        
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'st_name'], inplace=True)
    else:
        df_engineered['boroname'] = 'Unknown'

    has_target = 'violation_count' in df_engineered.columns

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required for training.")
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

    # Ablation point: NaN filling strategy
    if nan_fill_strategy == 'zero':
        df_engineered.fillna(0, inplace=True)
    elif nan_fill_strategy == 'global_mean':
        # Use the training set's global mean for consistent filling
        df_engineered.fillna(stats['global_mean'], inplace=True)
    else:
        raise ValueError(f"Unknown nan_fill_strategy: {nan_fill_strategy}")


    return df_engineered, stats


def run_experiment(train_df, val_df, use_np_seed, nan_fill_strategy, alphas):
    """Runs a single training and evaluation experiment with a given configuration."""
    SEED = 42
    if use_np_seed:
        np.random.seed(SEED)

    # --- Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df, nan_fill_strategy=nan_fill_strategy)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, nan_fill_strategy=nan_fill_strategy)

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

    # --- Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=alphas, cv=5))
    ])

    # --- Training & Validation ---
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    # Setup
    warnings.filterwarnings('ignore', category=UserWarning)
    create_dummy_files()
    SEED = 42
    
    # --- 1. Load and Split Data Once ---
    df_original = pd.read_csv('./input/violations_per_street_2022.csv')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    # --- Experiment 1: Baseline ---
    baseline_alphas = np.logspace(-2, 2, 5)
    baseline_rmse = run_experiment(
        train_df, val_df, 
        use_np_seed=True, 
        nan_fill_strategy='zero', 
        alphas=baseline_alphas
    )
    results['Baseline'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # --- Ablation 1: Change NaN Filling Strategy ---
    # Change fillna(0) to fillna(global_mean)
    ablation1_rmse = run_experiment(
        train_df, val_df,
        use_np_seed=True,
        nan_fill_strategy='global_mean',
        alphas=baseline_alphas
    )
    results["NaN Fill Strategy ('global_mean' vs 'zero')"] = ablation1_rmse
    print(f"Ablation (Global Mean NaN Fill) RMSE: {ablation1_rmse:.4f}")

    # --- Ablation 2: Remove Global Numpy Seed ---
    # This tests the stability of the hyperparameter search in RidgeCV
    ablation2_rmse = run_experiment(
        train_df, val_df,
        use_np_seed=False, # Ablation point
        nan_fill_strategy='zero',
        alphas=baseline_alphas
    )
    results["Reproducibility (No Numpy Seed)"] = ablation2_rmse
    print(f"Ablation (No Numpy Seed) RMSE: {ablation2_rmse:.4f}")

    # --- Ablation 3: Widen Alpha Search Space ---
    # Test if a more granular search for regularization finds a better model
    wider_alphas = np.logspace(-4, 4, 9)
    ablation3_rmse = run_experiment(
        train_df, val_df,
        use_np_seed=True,
        nan_fill_strategy='zero',
        alphas=wider_alphas # Ablation point
    )
    results["Wider Alpha Search Space"] = ablation3_rmse
    print(f"Ablation (Wider Alpha Search) RMSE: {ablation3_rmse:.4f}")
    
    # --- Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    
    # Calculate performance changes from baseline
    baseline_rmse = results.get('Baseline', float('inf'))
    changes = {name: rmse - baseline_rmse for name, rmse in results.items() if name != 'Baseline'}
    
    if not changes:
        print("No ablation studies were performed to compare against the baseline.")
        if 'Baseline' in results:
            final_validation_score = results['Baseline']
            print(f"Final Validation Performance: {final_validation_score}")
        return

    # Find the component with the largest absolute impact on RMSE
    max_impact_name = max(changes, key=lambda name: abs(changes[name]))
    max_impact_value = changes[max_impact_name]

    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    for name, rmse in results.items():
        if name != 'Baseline':
            change = rmse - results['Baseline']
            print(f"'{name}' resulted in RMSE: {rmse:.4f} (Change: {change:+.4f})")

    print("\nBased on the results, the component with the largest impact on performance is:")
    print(f"-> '{max_impact_name}' (Resulted in an RMSE change of {max_impact_value:+.4f})")
    
    # --- Report Final Performance ---
    best_config_name = min(results, key=results.get)
    final_validation_score = results[best_config_name]
    print(f"\nThe best performing configuration was '{best_config_name}' with RMSE: {final_validation_score:.4f}")
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == '__main__':
    main()
