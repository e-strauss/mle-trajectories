
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

def feature_engineer(df, train_stats=None, use_ratio_features=True, use_std_features=True, use_count_feature=True):
    """
    Engineers features for the model, with ablations.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data (placeholder if file not found) ---
    df_engineered['boroname'] = 'Unknown' # Keep it simple for this study

    has_target = 'violation_count' in df_engineered.columns

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        # --- Base Aggregates ---
        agg_funcs = ['mean', 'sum']
        if use_std_features:
            agg_funcs.append('std')
        if use_count_feature:
            agg_funcs.append('count')

        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(agg_funcs)
        street_agg.columns = [f'street_{func}' for func in agg_funcs]

        # Reset funcs for other groups that don't need 'count'
        agg_funcs = ['mean', 'sum']
        if use_std_features:
            agg_funcs.append('std')
        
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(agg_funcs)
        violation_agg.columns = [f'violation_{func}' for func in agg_funcs]
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(agg_funcs)
        boro_agg.columns = [f'boro_{func}' for func in agg_funcs]

        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }

        # --- Ratio Features (Ablation Target) ---
        if use_ratio_features:
            stats['violation_agg']['violation_global_ratio'] = stats['violation_agg']['violation_mean'] / stats['global_mean']
            
            street_to_boro = df_engineered.groupby('street_name')['boroname'].first()
            stats['street_agg'] = stats['street_agg'].join(street_to_boro)
            stats['street_agg'] = stats['street_agg'].merge(stats['boro_agg'][['boro_mean']], left_on='boroname', right_index=True, how='left')
            stats['street_agg']['street_boro_ratio'] = stats['street_agg']['street_mean'] / stats['street_agg']['boro_mean']
            stats['street_agg'] = stats['street_agg'].drop(columns=['boroname', 'boro_mean'], errors='ignore')

    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(train_df, val_df, use_ratio_features=True, use_std_features=True, use_count_feature=True):
    """
    Runs a single training and evaluation experiment with specified components.
    """
    # Feature Engineering
    train_featured, train_stats = feature_engineer(
        train_df, use_ratio_features=use_ratio_features, use_std_features=use_std_features, use_count_feature=use_count_feature
    )
    val_featured, _ = feature_engineer(
        val_df, train_stats=train_stats, use_ratio_features=use_ratio_features, use_std_features=use_std_features, use_count_feature=use_count_feature
    )

    # Define features based on toggles
    numerical_features = [
        'street_mean', 'street_sum',
        'violation_mean', 'violation_sum',
        'boro_mean', 'boro_sum'
    ]
    if use_std_features:
        numerical_features.extend(['street_std', 'violation_std', 'boro_std'])
    if use_count_feature:
        numerical_features.append('street_key_count')
    if use_ratio_features:
        numerical_features.extend(['violation_global_ratio', 'street_boro_ratio'])

    categorical_features = ['violation_description', 'boroname']
    
    # Filter for existing columns only, as some might not be created
    numerical_features = [col for col in numerical_features if col in train_featured.columns]

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
    return rmse

def main():
    # --- 1. Load and Prepare Data (using in-memory data for simplicity) ---
    csv_data = """Street Name,Violation Description,violation_count
WALL STREET,FAILURE TO STOP AT RED LIGHT,150
WALL STREET,NO PARKING-STREET CLEANING,300
BROADWAY,FAILURE TO STOP AT RED LIGHT,120
BROADWAY,NO PARKING-STREET CLEANING,250
LEXINGTON AVENUE,FAILURE TO STOP AT RED LIGHT,80
LEXINGTON AVENUE,DOUBLE PARKING,180
MADISON AVENUE,NO STANDING-DAY/TIME LIMITS,220
MADISON AVENUE,DOUBLE PARKING,190
5TH AVENUE,NO STANDING-DAY/TIME LIMITS,240
5TH AVENUE,FAILURE TO STOP AT RED LIGHT,95
PARK AVENUE,NO PARKING-STREET CLEANING,210
PARK AVENUE,BUS LANE VIOLATION,350
34TH STREET,BUS LANE VIOLATION,400
34TH STREET,NO STANDING-DAY/TIME LIMITS,280
42ND STREET,DOUBLE PARKING,310
42ND STREET,BUS LANE VIOLATION,450
"""
    df_original = pd.read_csv(io.StringIO(csv_data))

    # Create a larger, more varied dataset for a meaningful study
    np.random.seed(SEED)
    num_samples = 500
    streets = ['WALL STREET', 'BROADWAY', 'LEXINGTON AVENUE', 'MADISON AVENUE', '5TH AVENUE', 'PARK AVENUE', '34TH STREET', '42ND STREET', '1ST AVENUE', '2ND AVENUE']
    violations = ['FAILURE TO STOP AT RED LIGHT', 'NO PARKING-STREET CLEANING', 'DOUBLE PARKING', 'BUS LANE VIOLATION', 'NO STANDING-DAY/TIME LIMITS']
    
    data = {
        'Street Name': np.random.choice(streets, num_samples),
        'Violation Description': np.random.choice(violations, num_samples)
    }
    df_large = pd.DataFrame(data)
    df_large['violation_count'] = np.random.randint(10, 500, size=num_samples) + np.random.randn(num_samples) * 20
    df_large['violation_count'] = df_large['violation_count'].astype(int).clip(0)
    df_original = df_large
    
    # --- 2. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- 3. Ablation Study ---
    results = {}
    print("Running ablation study...")

    # Baseline: All features enabled
    baseline_rmse = run_experiment(train_df, val_df, use_ratio_features=True, use_std_features=True, use_count_feature=True)
    results['Baseline (All Features)'] = baseline_rmse
    print(f"  - Baseline RMSE: {baseline_rmse:.4f}")

    # Ablation 1: No Ratio Features
    rmse_no_ratio = run_experiment(train_df, val_df, use_ratio_features=False, use_std_features=True, use_count_feature=True)
    results['No Ratio Features'] = rmse_no_ratio
    print(f"  - Ablation (No Ratio Features) RMSE: {rmse_no_ratio:.4f}")

    # Ablation 2: No 'std' aggregates
    rmse_no_std = run_experiment(train_df, val_df, use_ratio_features=True, use_std_features=False, use_count_feature=True)
    results['No STD Features'] = rmse_no_std
    print(f"  - Ablation (No 'std' Features) RMSE: {rmse_no_std:.4f}")
    
    # Ablation 3: No 'street_key_count' feature
    rmse_no_count = run_experiment(train_df, val_df, use_ratio_features=True, use_std_features=True, use_count_feature=False)
    results['No Street Count Feature'] = rmse_no_count
    print(f"  - Ablation (No 'street_key_count' Feature) RMSE: {rmse_no_count:.4f}")

    print("\n--- Ablation Study Conclusion ---")
    
    # --- 4. Analyze Results ---
    impacts = {
        'Ratio Features': abs(results['No Ratio Features'] - baseline_rmse),
        "'std' Features": abs(results['No STD Features'] - baseline_rmse),
        "'street_key_count' Feature": abs(results['No Street Count Feature'] - baseline_rmse)
    }

    most_impactful_component = max(impacts, key=impacts.get)
    
    print(f"Baseline Performance: {results['Baseline (All Features)']:.4f}\n")
    print(f"Impact of removing Ratio Features: {results['No Ratio Features'] - baseline_rmse:+.4f}")
    print(f"Impact of removing 'std' Features: {results['No STD Features'] - baseline_rmse:+.4f}")
    print(f"Impact of removing 'street_key_count' Feature: {results['No Street Count Feature'] - baseline_rmse:+.4f}\n")

    print(f"The component that contributes the most to the overall performance is: {most_impactful_component}")

if __name__ == '__main__':
    main()
