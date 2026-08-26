
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

def feature_engineer(df, train_stats=None, use_median=True, smoothing_factor=20):
    """
    Engineers features for the model. This version is adapted for ablation.

    Args:
        df (pd.DataFrame): The dataframe to engineer features for.
        train_stats (dict, optional): Pre-calculated stats for inference.
        use_median (bool): If True, calculates median as an aggregate feature.
        smoothing_factor (int): The strength of smoothing for target encoding.
                                A value of 0 means no smoothing.
    Returns:
        pd.DataFrame: The dataframe with new features.
        dict: A dictionary of the calculated stats.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data (Simplified for script) ---
    df_engineered['boroname'] = 'Unknown' # Placeholder

    has_target = 'violation_count' in df_engineered.columns
    target_series = None
    if has_target:
        target_series = df_engineered['violation_count']
        df_engineered = df_engineered.drop(columns=['violation_count'])

    # --- Create Aggregate Features ---
    if train_stats is None:
        if not has_target:
            raise ValueError("`violation_count` column is required for training.")

        global_mean = target_series.mean()

        def get_smoothed_aggregates(group_by_col, prefix):
            temp_df = df_engineered.copy()
            temp_df['violation_count'] = target_series
            
            agg_list = ['mean', 'sum', 'std', 'count']
            if use_median:
                agg_list.append('median')

            agg = temp_df.groupby(group_by_col)['violation_count'].agg(agg_list)
            
            # Apply smoothing to the mean
            if smoothing_factor > 0:
                agg['mean'] = (agg['count'] * agg['mean'] + smoothing_factor * global_mean) / (agg['count'] + smoothing_factor)
            
            agg.columns = [f'{prefix}_{stat}' for stat in agg.columns]
            return agg

        street_agg = get_smoothed_aggregates('street_name', 'street')
        violation_agg = get_smoothed_aggregates('violation_description', 'violation')
        boro_agg = get_smoothed_aggregates('boroname', 'boro')
        
        # Rename count columns to avoid name collisions
        street_agg.rename(columns={'street_count': 'street_key_count'}, inplace=True)
        violation_agg.rename(columns={'violation_count': 'violation_type_count'}, inplace=True)
        boro_agg.rename(columns={'boro_count': 'boro_key_count'}, inplace=True)

        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            'global_mean': global_mean,
            'use_median': use_median
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    if target_series is not None:
        df_engineered['violation_count'] = target_series

    # Smart NaN filling
    all_agg_cols = list(stats['street_agg'].columns) + \
                   list(stats['violation_agg'].columns) + \
                   list(stats['boro_agg'].columns)
    
    fill_values = {}
    for col in all_agg_cols:
        if col in df_engineered.columns:
            if col.endswith('_mean'):
                fill_values[col] = stats['global_mean']
            else:
                fill_values[col] = 0

    df_engineered.fillna(value=fill_values, inplace=True)

    return df_engineered, stats

def run_experiment(exp_name, train_df, val_df, use_median, smoothing_factor, use_violation_ohe):
    """
    Runs a single experiment with a specific configuration.
    """
    # --- Feature Engineering ---
    train_featured, train_stats = feature_engineer(
        train_df, use_median=use_median, smoothing_factor=smoothing_factor
    )
    val_featured, _ = feature_engineer(
        val_df, train_stats=train_stats, use_median=use_median, smoothing_factor=smoothing_factor
    )

    # --- Define Features ---
    categorical_features = ['boroname']
    if use_violation_ohe:
        categorical_features.append('violation_description')

    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std', 'violation_type_count',
        'boro_mean', 'boro_sum', 'boro_std', 'boro_key_count'
    ]
    if use_median:
        numerical_features.extend(['street_median', 'violation_median', 'boro_median'])

    # Filter for columns that actually exist
    numerical_features = [f for f in numerical_features if f in train_featured.columns]
    all_features = numerical_features + categorical_features
    
    # --- Model Pipeline ---
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])
    
    # --- Training & Evaluation ---
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"{exp_name}: Validation RMSE = {rmse:.4f}")
    return rmse


def main():
    """
    Main function to run the ablation study.
    """
    # Create a dummy dataset in memory
    data = """Street Name,Violation Description,violation_count
STREET A,DOUBLE PARKING,150
STREET A,NO STANDING,50
STREET B,DOUBLE PARKING,200
STREET B,NO PARKING,10
STREET C,FAILURE TO DISPLAY METER RECEIPT,300
STREET D,DOUBLE PARKING,5
STREET D,NO STANDING,25
STREET E,DOUBLE PARKING,10
STREET F,NO STANDING,80
STREET F,FAILURE TO STOP AT RED LIGHT,120
STREET G,PHTO SCHOOL ZN SPEED VIOLATION,500
STREET H,FAILURE TO DISPLAY METER RECEIPT,20
STREET I,PHTO SCHOOL ZN SPEED VIOLATION,450
STREET J,DOUBLE PARKING,15
STREET K,NO STANDING,90
STREET L,DOUBLE PARKING,12
STREET M,NO PARKING,30
STREET N,FAILURE TO DISPLAY METER RECEIPT,350
STREET O,DOUBLE PARKING,8
STREET O,NO STANDING,22
STREET P,DOUBLE PARKING,18
STREET Q,NO STANDING,75
STREET R,FAILURE TO STOP AT RED LIGHT,110
STREET S,PHTO SCHOOL ZN SPEED VIOLATION,550
STREET T,FAILURE TO DISPLAY METER RECEIPT,18
"""
    df_original = pd.read_csv(io.StringIO(data))

    # --- Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Run Ablation Experiments ---
    results = {}

    # Baseline (Full Model)
    results['Baseline'] = run_experiment(
        "Baseline (Full Model)", train_df, val_df,
        use_median=True, smoothing_factor=20, use_violation_ohe=True
    )

    # Ablation 1: Remove Median Aggregate
    results['No Median Aggregate'] = run_experiment(
        "Ablation: No 'Median' Aggregate", train_df, val_df,
        use_median=False, smoothing_factor=20, use_violation_ohe=True
    )

    # Ablation 2: Remove Smoothing on Target Encoding
    results['No Smoothing'] = run_experiment(
        "Ablation: No Smoothing", train_df, val_df,
        use_median=True, smoothing_factor=0, use_violation_ohe=True
    )

    # Ablation 3: Remove One-Hot Encoding for Violation Description
    results['No Violation OHE'] = run_experiment(
        "Ablation: No 'violation_description' OHE", train_df, val_df,
        use_median=True, smoothing_factor=20, use_violation_ohe=False
    )
    
    # --- Analyze Results ---
    baseline_rmse = results['Baseline']
    performance_impact = {}
    for name, rmse in results.items():
        if name != 'Baseline':
            # We care about the magnitude of change, so we use abs()
            performance_impact[name] = abs(rmse - baseline_rmse)

    if not performance_impact:
        print("\nCould not determine the most impactful component.")
        return

    most_impactful_component = max(performance_impact, key=performance_impact.get)
    
    print("\n---")
    print(f"Conclusion: '{most_impactful_component}' contributes the most to the model's performance, as its removal caused the largest change in RMSE.")

if __name__ == '__main__':
    main()
