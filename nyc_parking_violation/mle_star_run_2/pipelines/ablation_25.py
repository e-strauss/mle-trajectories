
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV, Ridge
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

def feature_engineer(df, train_stats=None, use_delta_features=True, use_street_count=True):
    """
    Engineers features for the model, with ablations.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data (simplified for script execution) ---
    df_engineered['boroname'] = 'Unknown' # Assume no file to avoid dependency

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

    # --- Ablation: Control inclusion of street_key_count ---
    if not use_street_count and 'street_key_count' in stats['street_agg'].columns:
        # Drop from stats so it's not merged
        stats['street_agg'] = stats['street_agg'].drop(columns=['street_key_count'])

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # --- Ablation: Control creation of hierarchical delta features ---
    if use_delta_features:
        # This captures how much a specific street's stats differ from the borough average.
        for stat in ['mean', 'sum', 'std']:
            street_col = f'street_{stat}'
            boro_col = f'boro_{stat}'
            delta_col = f'street_{stat}_boro_delta'
            if street_col in df_engineered.columns and boro_col in df_engineered.columns:
                df_engineered[delta_col] = df_engineered[street_col] - df_engineered[boro_col]

    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(train_df, val_df, use_delta_features, use_street_count, use_ridge_cv):
    """
    Runs a single training and evaluation experiment with a given configuration.
    """
    # --- Feature Engineering ---
    train_featured, train_stats = feature_engineer(
        train_df,
        use_delta_features=use_delta_features,
        use_street_count=use_street_count
    )
    val_featured, _ = feature_engineer(
        val_df,
        train_stats=train_stats,
        use_delta_features=use_delta_features,
        use_street_count=use_street_count
    )

    # Define features based on the experiment configuration
    numerical_features = [
        'street_mean', 'street_sum', 'street_std',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    if use_street_count:
        numerical_features.append('street_key_count')
    if use_delta_features:
        numerical_features.extend(['street_mean_boro_delta', 'street_sum_boro_delta', 'street_std_boro_delta'])

    categorical_features = ['violation_description', 'boroname']
    
    all_features = numerical_features + categorical_features
    
    # Filter to columns that actually exist after engineering
    train_cols = train_featured.columns
    numerical_features = [f for f in numerical_features if f in train_cols]
    all_features = [f for f in all_features if f in train_cols]

    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # --- Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # --- Ablation: Control model type (CV vs. fixed alpha) ---
    if use_ridge_cv:
        model = RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5)
    else:
        model = Ridge(alpha=1.0) # Use default Ridge with fixed alpha

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    # --- Training & Evaluation ---
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    return rmse

def main():
    # Create a dummy dataset in-memory
    data = {
        'Street Name': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'F', 'G', 'H'] * 10,
        'Violation Description': ['V1', 'V2', 'V1', 'V3', 'V2', 'V3', 'V1', 'V4', 'V1', 'V2', 'V5', 'V5'] * 10,
        'violation_count': np.random.randint(1, 100, 120) + np.repeat([10, 20, 10, 50, 20, 50, 10, 80, 10, 20, 90, 90], 10)
    }
    df_original = pd.DataFrame(data)
    
    # Validation Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Run Ablation Study ---
    results = {}
    
    # Baseline
    baseline_rmse = run_experiment(train_df, val_df, 
                                   use_delta_features=True, 
                                   use_street_count=True, 
                                   use_ridge_cv=True)
    results['Baseline (All Features & RidgeCV)'] = baseline_rmse
    
    # Ablation 1: No Delta Features
    rmse_no_delta = run_experiment(train_df, val_df, 
                                   use_delta_features=False, 
                                   use_street_count=True, 
                                   use_ridge_cv=True)
    results['Ablation: No Hierarchical Delta Features'] = rmse_no_delta
    
    # Ablation 2: No Street Count Feature
    rmse_no_count = run_experiment(train_df, val_df, 
                                   use_delta_features=True, 
                                   use_street_count=False, 
                                   use_ridge_cv=True)
    results['Ablation: No Street Count Feature'] = rmse_no_count

    # Ablation 3: No Hyperparameter Tuning (use fixed Ridge)
    rmse_no_cv = run_experiment(train_df, val_df, 
                                use_delta_features=True, 
                                use_street_count=True, 
                                use_ridge_cv=False)
    results['Ablation: No Hyperparameter Tuning (Ridge)'] = rmse_no_cv

    # --- Print and Analyze Results ---
    print("--- Ablation Study Results ---")
    
    max_impact = -1
    most_impactful_component = "None"
    
    for name, rmse in results.items():
        if name.startswith('Ablation'):
            change = rmse - baseline_rmse
            print(f"{name}: RMSE = {rmse:.4f} (Change from Baseline: {change:+.4f})")
            if abs(change) > max_impact:
                max_impact = abs(change)
                most_impactful_component = name.replace("Ablation: ", "")
        else:
            print(f"{name}: RMSE = {rmse:.4f}")

    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")

if __name__ == '__main__':
    # Set seed for consistent dummy data generation
    np.random.seed(SEED)
    main()

