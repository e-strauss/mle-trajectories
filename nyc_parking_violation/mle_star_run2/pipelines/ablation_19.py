
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

def feature_engineer(df, train_stats=None, use_boro_augmentation=True):
    """
    Engineers features for the model.
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    if use_boro_augmentation:
        # Dummy augmentation for this script
        # In a real scenario, this would load and merge a file.
        borough_mapping = {
            'JAMAICA AVE': 'QUEENS',
            'FLATBUSH AVE': 'BROOKLYN',
            'BROADWAY': 'MANHATTAN',
            'GRAND CONCOURSE': 'BRONX'
        }
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered['boroname'] = df_engineered['street_name_upper'].map(borough_mapping).fillna('Unknown')
        df_engineered.drop(columns=['street_name_upper'], inplace=True)
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

def run_pipeline(X_train, y_train, X_val, y_val, numerical_features, categorical_features, clip_negatives=True):
    """
    Defines, trains, and evaluates a pipeline.
    """
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

    if clip_negatives:
        val_predictions[val_predictions < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse


# --- Main Ablation Study ---
if __name__ == '__main__':

    # Create a dummy CSV in memory to make the script self-contained
    csv_data = """Street Name,Violation Description,violation_count
BROADWAY,PHTO SCHOOL ZN SPEED VIOLATION,350
BROADWAY,FAILURE TO STOP AT RED LIGHT,120
FLATBUSH AVE,PHTO SCHOOL ZN SPEED VIOLATION,450
FLATBUSH AVE,FAILURE TO STOP AT RED LIGHT,150
JAMAICA AVE,PHTO SCHOOL ZN SPEED VIOLATION,280
JAMAICA AVE,BUS LANE VIOLATION,90
GRAND CONCOURSE,PHTO SCHOOL ZN SPEED VIOLATION,320
GRAND CONCOURSE,BUS LANE VIOLATION,110
5TH AVE,FAILURE TO STOP AT RED LIGHT,80
5TH AVE,PHTO SCHOOL ZN SPEED VIOLATION,200
ATLANTIC AVE,BUS LANE VIOLATION,75
ATLANTIC AVE,FAILURE TO STOP AT RED LIGHT,95
MAIN ST,PHTO SCHOOL ZN SPEED VIOLATION,180
MAIN ST,DOUBLE PARKING,50
QUEENS BLVD,PHTO SCHOOL ZN SPEED VIOLATION,400
QUEENS BLVD,FAILURE TO STOP AT RED LIGHT,130
"""
    df_original = pd.read_csv(io.StringIO(csv_data))

    # --- 1. Data Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)
    
    results = {}

    # --- 2. Baseline Experiment ---
    train_featured_base, train_stats_base = feature_engineer(train_df)
    val_featured_base, _ = feature_engineer(val_df, train_stats=train_stats_base)

    baseline_numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    baseline_categorical_features = ['violation_description', 'boroname']
    
    X_train_base = train_featured_base[baseline_numerical_features + baseline_categorical_features]
    y_train_base = train_featured_base['violation_count']
    X_val_base = val_featured_base[baseline_numerical_features + baseline_categorical_features]
    y_val_base = val_featured_base['violation_count']

    baseline_rmse = run_pipeline(
        X_train_base, y_train_base, X_val_base, y_val_base,
        baseline_numerical_features, baseline_categorical_features,
        clip_negatives=True
    )
    results['Baseline (Full Model)'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Final Validation Performance: {baseline_rmse}")

    # --- 3. Ablation Experiments ---

    # Ablation 1: No Borough Augmentation during Feature Engineering
    train_featured_ab1, train_stats_ab1 = feature_engineer(train_df, use_boro_augmentation=False)
    val_featured_ab1, _ = feature_engineer(val_df, train_stats=train_stats_ab1, use_boro_augmentation=False)
    X_train_ab1 = train_featured_ab1[baseline_numerical_features + baseline_categorical_features]
    X_val_ab1 = val_featured_ab1[baseline_numerical_features + baseline_categorical_features]
    
    rmse_ab1 = run_pipeline(
        X_train_ab1, y_train_base, X_val_ab1, y_val_base,
        baseline_numerical_features, baseline_categorical_features,
        clip_negatives=True
    )
    results['Ablation: No Borough Augmentation'] = rmse_ab1
    print(f"RMSE (No Borough Augmentation): {rmse_ab1:.4f}")

    # Ablation 2: Remove 'violation_description' from features
    # The original error occurred because 'violation_description' was not in the categorical list,
    # but was still in the dataframe, causing it to be passed through by `remainder='passthrough'`.
    # The fix is to create a dataframe for this ablation that does NOT include the 'violation_description' column.
    ab2_categorical_features = ['boroname']
    ab2_features = baseline_numerical_features + ab2_categorical_features
    X_train_ab2 = train_featured_base[ab2_features]
    X_val_ab2 = val_featured_base[ab2_features]

    rmse_ab2 = run_pipeline(
        X_train_ab2, y_train_base, X_val_ab2, y_val_base,
        baseline_numerical_features, ab2_categorical_features,
        clip_negatives=True
    )
    results["Ablation: No OHE for 'violation_description'"] = rmse_ab2
    print(f"RMSE (No OHE for 'violation_description'): {rmse_ab2:.4f}")
    
    # Ablation 3: No Negative Clipping
    rmse_ab3 = run_pipeline(
        X_train_base, y_train_base, X_val_base, y_val_base,
        baseline_numerical_features, baseline_categorical_features,
        clip_negatives=False
    )
    results["Ablation: No Negative Clipping"] = rmse_ab3
    print(f"RMSE (No Negative Clipping): {rmse_ab3:.4f}")

    # --- 4. Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    
    impact = {}
    for name, score in results.items():
        if name != 'Baseline (Full Model)':
            change = score - baseline_rmse
            impact[name] = abs(change)
            print(f"Impact of '{name}': {change:+.4f} RMSE")
    
    if not impact:
        most_impactful = "No ablations performed"
    else:
        most_impactful = max(impact, key=impact.get)

    print(f"\nComponent with the most impact on performance: {most_impactful.replace('Ablation: ', '')}")
