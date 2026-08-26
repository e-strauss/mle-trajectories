
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

def feature_engineer(df, train_stats=None, smoothing_factor=100, nan_fill_strategy='smart'):
    """
    Engineers features for the model, with ablation options.

    Args:
        df (pd.DataFrame): The dataframe to engineer features for.
        train_stats (dict, optional): Pre-calculated stats for inference mode.
        smoothing_factor (int): Regularization for target encoding. 0 means no smoothing.
        nan_fill_strategy (str): 'smart' (global mean for means, 0 otherwise) or 'simple' (0 for all).

    Returns:
        pd.DataFrame: The dataframe with new features.
        dict: A dictionary of the calculated stats.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # Use a dummy placeholder for borough as it's not the focus of this study
    df_engineered['boroname'] = 'Unknown'

    has_target = 'violation_count' in df_engineered.columns

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        global_mean = df_engineered['violation_count'].mean()

        def calculate_smoothed_mean(group_df, global_mean, smoothing):
            if smoothing == 0:
                return group_df['mean']
            group_mean = group_df['mean']
            group_count = group_df['count']
            return (group_mean * group_count + global_mean * smoothing) / (group_count + smoothing)

        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg['mean'] = calculate_smoothed_mean(street_agg, global_mean, smoothing_factor)
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        street_agg['street_std'].fillna(0, inplace=True)

        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        violation_agg['mean'] = calculate_smoothed_mean(violation_agg, global_mean, smoothing_factor)
        violation_agg = violation_agg.drop(columns=['count'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        violation_agg['violation_std'].fillna(0, inplace=True)

        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'global_mean': global_mean
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')

    if nan_fill_strategy == 'smart':
        mean_cols = ['street_mean', 'violation_mean']
        for col in mean_cols:
            if col in df_engineered.columns:
                df_engineered[col].fillna(stats['global_mean'], inplace=True)
        df_engineered.fillna(0, inplace=True)
    elif nan_fill_strategy == 'simple':
        df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(train_df_original, val_df_original, ablation_name, smoothing_factor, nan_fill_strategy, use_street_key_count):
    """
    Runs a single training and validation experiment with specific settings.
    """
    print(f"--- Running Experiment: {ablation_name} ---")
    
    # Feature Engineering
    train_df = train_df_original.copy()
    val_df = val_df_original.copy()
    train_featured, train_stats = feature_engineer(train_df, smoothing_factor=smoothing_factor, nan_fill_strategy=nan_fill_strategy)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, nan_fill_strategy=nan_fill_strategy)

    # Define features based on ablation setting
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std',
        'violation_mean', 'violation_sum', 'violation_std',
    ]
    if use_street_key_count:
        numerical_features.append('street_key_count')
    
    all_features = numerical_features + categorical_features
    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # Model Pipeline
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

    # Training and Validation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Validation RMSE: {rmse:.4f}\n")
    return rmse, ablation_name

def main():
    # Create a dummy dataset in memory
    data = """Street Name,Violation Description,violation_count
ST JOHNS PL,FAILURE TO STOP AT RED LIGHT,250
ST JOHNS PL,FAILURE TO DISPLAY MUNI METER RECEIPT,120
ST JOHNS PL,NO STANDING-DAY/TIME LIMITS,80
42 ST,FAILURE TO STOP AT RED LIGHT,300
42 ST,FAILURE TO DISPLAY MUNI METER RECEIPT,150
MADISON AVE,NO PARKING-STREET CLEANING,200
MADISON AVE,FAILURE TO STOP AT RED LIGHT,180
PARK AVE,NO STANDING-DAY/TIME LIMITS,90
5 AVE,FAILURE TO STOP AT RED LIGHT,220
5 AVE,NO PARKING-STREET CLEANING,110
BROADWAY,FAILURE TO STOP AT RED LIGHT,400
BROADWAY,FAILURE TO DISPLAY MUNI METER RECEIPT,180
BROADWAY,NO STANDING-DAY/TIME LIMITS,130
BROADWAY,INSP. STICKER-EXPIRED/MISSING,50
UNKNOWN ST,FAILURE TO STOP AT RED LIGHT,10
"""
    df_original = pd.read_csv(io.StringIO(data))

    # Validation Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Ablation Study ---
    experiments = []

    # 1. Baseline
    baseline_rmse, _ = run_experiment(
        train_df, val_df, 
        ablation_name="Baseline (Smoothed Mean, Smart NaN Fill, Street Count)",
        smoothing_factor=100,
        nan_fill_strategy='smart',
        use_street_key_count=True
    )
    experiments.append({'name': 'Baseline', 'rmse': baseline_rmse})
    
    # 2. Ablation: No Smoothing
    rmse, name = run_experiment(
        train_df, val_df, 
        ablation_name="No Smoothed Target Encoding",
        smoothing_factor=0, # Set smoothing to 0
        nan_fill_strategy='smart',
        use_street_key_count=True
    )
    experiments.append({'name': 'No Smoothed Target Encoding', 'rmse': rmse})
    
    # 3. Ablation: Simple NaN Fill
    rmse, name = run_experiment(
        train_df, val_df, 
        ablation_name="Simple NaN Fill (fillna with 0)",
        smoothing_factor=100,
        nan_fill_strategy='simple', # Change NaN fill strategy
        use_street_key_count=True
    )
    experiments.append({'name': 'Simple NaN Fill (fillna with 0)', 'rmse': rmse})

    # 4. Ablation: No Street Key Count Feature
    rmse, name = run_experiment(
        train_df, val_df, 
        ablation_name="No 'street_key_count' Feature",
        smoothing_factor=100,
        nan_fill_strategy='smart',
        use_street_key_count=False # Exclude the feature
    )
    experiments.append({'name': "No 'street_key_count' Feature", 'rmse': rmse})

    # --- Conclusion ---
    results = {}
    for exp in experiments:
        # We don't calculate diff for baseline itself
        if exp['name'] == 'Baseline':
            continue
        diff = exp['rmse'] - baseline_rmse
        results[exp['name']] = diff

    print("--- Ablation Study Summary ---")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    for name, diff in results.items():
        print(f"Impact of '{name}': {diff:+.4f} RMSE")

    if not results:
        print("\nNo ablations were run to compare against the baseline.")
        return

    # Find the component with the largest absolute impact
    most_impactful_component = max(results, key=lambda k: abs(results[k]))
    print(f"\nConclusion: '{most_impactful_component}' contributes the most to the overall performance, as its modification caused the largest change in RMSE.")


if __name__ == '__main__':
    main()
