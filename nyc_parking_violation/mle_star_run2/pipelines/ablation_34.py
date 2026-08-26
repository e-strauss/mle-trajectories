
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV, PoissonRegressor
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

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns if isinstance(c, str)]

    # --- Augment with Borough Data ---
    # In a real scenario, this would read from './input/nyc_cscl.csv'
    # For this self-contained script, we simulate its presence or absence.
    # We will assume the file does not exist to make the script runnable anywhere.
    df_engineered['boroname'] = 'Unknown'

    # The target column might not be present in a keys-only test file
    has_target = 'violation_count' in df_engineered.columns

    # --- Create Aggregate Features ---
    if train_stats is None:
        # Training mode: Calculate stats from the dataframe itself
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")

        # Aggregate by street name
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']

        # Aggregate by violation description
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']

        # Aggregate by borough
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']

        # Store calculated stats for later use
        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        # Inference mode: Apply pre-calculated stats
        stats = train_stats

    # Merge aggregate features onto the dataframe
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Fill NaNs created by left merges.
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(train_df, val_df, model_pipeline):
    """
    Runs a single experiment with a given pipeline.
    """
    # Feature Engineering
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

    # Training
    model_pipeline.fit(X_train, y_train)

    # Validation
    val_predictions = model_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    # Create a dummy dataset for the ablation study
    data = """Street Name,Violation Description,violation_count
Mott Street,FAILURE TO STOP AT RED LIGHT,150
Canal Street,FAILURE TO STOP AT RED LIGHT,250
Mott Street,PARKING AT A BUS STOP,50
Canal Street,PARKING AT A BUS STOP,80
Baxter Street,FAILURE TO STOP AT RED LIGHT,120
Baxter Street,NO STANDING-DAY/TIME LIMITS,70
Mott Street,FAILURE TO STOP AT RED LIGHT,160
Canal Street,FAILURE TO STOP AT RED LIGHT,260
Mott Street,PARKING AT A BUS STOP,55
Canal Street,PARKING AT A BUS STOP,85
Baxter Street,FAILURE TO STOP AT RED LIGHT,125
Baxter Street,NO STANDING-DAY/TIME LIMITS,75
Bayard Street,FAILURE TO STOP AT RED LIGHT,90
Bayard Street,NO STANDING-DAY/TIME LIMITS,40
"""
    df_original = pd.read_csv(io.StringIO(data))

    # --- Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # Define preprocessor
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    categorical_features = ['violation_description', 'boroname']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Ablation Study ---
    results = {}

    # 1. Baseline Experiment (Poisson Regressor with alpha=1.0)
    baseline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', PoissonRegressor(alpha=1.0, max_iter=500))
    ])
    baseline_rmse = run_experiment(train_df.copy(), val_df.copy(), baseline_model)
    results['Baseline (Poisson Regressor)'] = baseline_rmse

    # 2. Ablation: Change Model Type to RidgeCV
    ridge_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5)))
    ])
    results['Ablation: Model Type (RidgeCV)'] = run_experiment(train_df.copy(), val_df.copy(), ridge_model)

    # 3. Ablation: Remove Regularization from Poisson Regressor
    no_reg_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', PoissonRegressor(alpha=0, max_iter=500)) # alpha=0 means no regularization
    ])
    results['Ablation: No Regularization (alpha=0)'] = run_experiment(train_df.copy(), val_df.copy(), no_reg_model)
    
    # 4. Ablation: Reduce Solver Iterations
    fewer_iter_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', PoissonRegressor(alpha=1.0, max_iter=50)) # Reduced max_iter
    ])
    results['Ablation: Fewer Solver Iterations (max_iter=50)'] = run_experiment(train_df.copy(), val_df.copy(), fewer_iter_model)


    # --- Print Results ---
    print("\nAblation Study Results:")
    print("-" * 50)
    print(f"Baseline (Poisson Regressor) RMSE: {results['Baseline (Poisson Regressor)']:.4f}")
    print("-" * 50)

    impacts = {}
    for name, rmse in results.items():
        if name != 'Baseline (Poisson Regressor)':
            change = rmse - baseline_rmse
            print(f"{name}:")
            print(f"  RMSE: {rmse:.4f} (Change from Baseline: {change:+.4f})")
            impacts[name] = abs(change)

    # --- Conclusion ---
    most_impactful = max(impacts, key=impacts.get)
    print("-" * 50)
    print(f"The component that contributes the most to the overall performance is: '{most_impactful.split(': ')[1]}'")


if __name__ == '__main__':
    main()
