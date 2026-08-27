
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
import io

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # In-memory file for borough data to avoid disk dependency
    cscl_data = """ST_NAME,BORONAME
BROADWAY,MANHATTAN
5TH AVE,BROOKLYN
ATLANTIC AVE,QUEENS
GRAND CONCOURSE,BRONX
HYLAN BLVD,STATEN ISLAND
"""
    try:
        cscl = pd.read_csv(io.StringIO(cscl_data))
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    except Exception:
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
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats


def run_experiment(X_train, y_train, X_val, y_val, preprocessor, regressor):
    """
    Helper function to run a single experiment.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse


def main():
    """
    Main function to run the ablation study.
    """
    # Create a dummy dataset in memory
    data = {
        'Street Name': ['BROADWAY'] * 20 + ['5TH AVE'] * 20 + ['ATLANTIC AVE'] * 10 + ['GRAND CONCOURSE'] * 5,
        'Violation Description': ['NO PARKING-DAY/TIME'] * 25 + ['FIRE HYDRANT'] * 25 + ['BUS LANE VIOLATION'] * 5,
        'violation_count': list(np.random.randint(10, 100, 55))
    }
    df_original = pd.DataFrame(data)
    
    # --- Data Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    target = 'violation_count'
    y_train = train_featured[target]
    y_val = val_featured[target]

    results = {}
    print("--- Starting Ablation Study ---")

    # --- Baseline Experiment ---
    categorical_features_base = ['violation_description', 'boroname']
    numerical_features_base = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    preprocessor_base = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_base),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_base)
        ]
    )
    regressor_base = RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5)
    
    X_train_base = train_featured[numerical_features_base + categorical_features_base]
    X_val_base = val_featured[numerical_features_base + categorical_features_base]
    
    baseline_rmse = run_experiment(X_train_base, y_train, X_val_base, y_val, preprocessor_base, regressor_base)
    results['Baseline'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # --- Ablation 1: Only use 'mean' aggregate features ---
    numerical_features_abl1 = ['street_mean', 'violation_mean', 'boro_mean']
    preprocessor_abl1 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_abl1),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_base)
        ]
    )
    X_train_abl1 = train_featured[numerical_features_abl1 + categorical_features_base]
    X_val_abl1 = val_featured[numerical_features_abl1 + categorical_features_base]
    
    rmse_abl1 = run_experiment(X_train_abl1, y_train, X_val_abl1, y_val, preprocessor_abl1, regressor_base)
    results["No 'sum'/'std'/'count' Aggregates"] = rmse_abl1
    print(f"Ablation (No 'sum'/'std'/'count' Aggregates) RMSE: {rmse_abl1:.4f} (Change: {rmse_abl1 - baseline_rmse:+.4f})")

    # --- Ablation 2: Change RidgeCV to use GCV instead of 5-fold CV ---
    regressor_abl2 = RidgeCV(alphas=np.logspace(-2, 2, 5), cv=None) # cv=None uses GCV
    rmse_abl2 = run_experiment(X_train_base, y_train, X_val_base, y_val, preprocessor_base, regressor_abl2)
    results["RidgeCV with GCV (cv=None)"] = rmse_abl2
    print(f"Ablation (RidgeCV with GCV) RMSE: {rmse_abl2:.4f} (Change: {rmse_abl2 - baseline_rmse:+.4f})")

    # --- Ablation 3: Change OHE to sparse output ---
    preprocessor_abl3 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_base),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features_base)
        ]
    )
    rmse_abl3 = run_experiment(X_train_base, y_train, X_val_base, y_val, preprocessor_abl3, regressor_base)
    results["OHE with Sparse Output"] = rmse_abl3
    print(f"Ablation (OHE Sparse Output) RMSE: {rmse_abl3:.4f} (Change: {rmse_abl3 - baseline_rmse:+.4f})")
    
    # --- Conclusion ---
    print("\n--- Conclusion ---")
    impact = {name: abs(rmse - baseline_rmse) for name, rmse in results.items() if name != 'Baseline'}
    most_impactful = max(impact, key=impact.get)
    
    print(f"The component that contributes the most to the overall performance is: '{most_impactful}'.")
    print(f"Its modification resulted in the largest absolute change in RMSE: {impact[most_impactful]:.4f}.")

if __name__ == '__main__':
    main()
