
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
    Engineers features for the model, including standard aggregates and relative "hotspot" features.
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # To ensure the script is self-contained and runnable without external files,
    # we create a dummy 'boroname' feature. This allows the ratio features to be calculated.
    if 'street_name' in df_engineered.columns:
        df_engineered['boroname'] = df_engineered['street_name'].apply(lambda x: f"Boro_{ord(x[0]) % 4}")
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

    # --- Create Relative "Hotspot" Features ---
    epsilon = 1e-6
    df_engineered['street_boro_mean_ratio'] = df_engineered['street_mean'] / (df_engineered['boro_mean'] + epsilon)
    df_engineered['street_boro_sum_ratio'] = df_engineered['street_sum'] / (df_engineered['boro_sum'] + epsilon)

    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(train_df, val_df, use_ratio_features=True, clip_negatives=True, use_street_count=True):
    """
    Runs a single training and validation experiment with a specific configuration.
    """
    # 1. Feature Engineering
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    # 2. Define Features based on configuration
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    if use_ratio_features:
        numerical_features.extend(['street_boro_mean_ratio', 'street_boro_sum_ratio'])
    if use_street_count:
        numerical_features.append('street_key_count')

    all_features = numerical_features + categorical_features

    # 3. Prepare Data for Model
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # 4. Model Pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    # 5. Training and Validation
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)

    # 6. Post-processing based on configuration
    if clip_negatives:
        val_predictions[val_predictions < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def perform_ablation_study():
    """
    Orchestrates the ablation study by loading data and running experiments
    with different configurations.
    """
    # Create a dummy dataset in memory to ensure the script is runnable.
    # This data is augmented to provide enough samples for stable group-based splitting and aggregation.
    data = """Street Name,Violation Description,violation_count
BERGEN STREET,FAILURE TO STOP AT RED LIGHT,150
BERGEN STREET,PARKING AT AN EXPIRED METER,200
FLATBUSH AVENUE,NO PARKING-STREET CLEANING,350
FLATBUSH AVENUE,FAILURE TO STOP AT RED LIGHT,120
ATLANTIC AVENUE,PARKING AT AN EXPIRED METER,250
ATLANTIC AVENUE,NO PARKING-STREET CLEANING,400
5TH AVENUE,FAILURE TO STOP AT RED LIGHT,90
5TH AVENUE,PARKING AT AN EXPIRED METER,180
DEAN STREET,NO PARKING-STREET CLEANING,300
DEAN STREET,FAILURE TO STOP AT RED LIGHT,110
UNION STREET,PARKING AT AN EXPIRED METER,220
UNION STREET,NO PARKING-STREET CLEANING,380
"""
    df_original = pd.read_csv(io.StringIO(data))

    # Augment data to create a larger, more varied dataset for the study
    df_augmented_list = []
    for i in range(30):
        temp_df = df_original.copy()
        # Introduce noise and variations
        temp_df['violation_count'] = temp_df['violation_count'] * (1 + (i % 5 - 2) * 0.1) + np.random.randint(-20, 20, size=len(temp_df))
        temp_df['Street Name'] = temp_df['Street Name'].apply(lambda x: x if i % 2 == 0 else f"{x}_{i}")
        df_augmented_list.append(temp_df)
    df_original = pd.concat(df_augmented_list, ignore_index=True)
    df_original['violation_count'] = df_original['violation_count'].astype(int).clip(lower=0)


    # Validation Split using GroupShuffleSplit to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    # --- Run Experiments ---

    # Baseline: Full model with all features and steps
    results['Baseline (Full Model)'] = run_experiment(
        train_df, val_df,
        use_ratio_features=True,
        clip_negatives=True,
        use_street_count=True
    )
    print(f"Baseline (Full Model) RMSE: {results['Baseline (Full Model)']:.4f}")

    # Ablation 1: Remove the "Hotspot" Ratio Features
    results['Ablation: No Ratio Features'] = run_experiment(
        train_df, val_df,
        use_ratio_features=False,
        clip_negatives=True,
        use_street_count=True
    )
    print(f"Ablation (No Ratio Features) RMSE: {results['Ablation: No Ratio Features']:.4f}")

    # Ablation 2: Remove the post-processing step of clipping negative predictions
    results['Ablation: No Negative Clipping'] = run_experiment(
        train_df, val_df,
        use_ratio_features=True,
        clip_negatives=False,
        use_street_count=True
    )
    print(f"Ablation (No Negative Clipping) RMSE: {results['Ablation: No Negative Clipping']:.4f}")

    # Ablation 3: Remove the 'street_key_count' feature
    results['Ablation: No street_key_count'] = run_experiment(
        train_df, val_df,
        use_ratio_features=True,
        clip_negatives=True,
        use_street_count=False
    )
    print(f"Ablation (No street_key_count) RMSE: {results['Ablation: No street_key_count']:.4f}")

    # --- Conclusion Logic ---
    baseline_rmse = results['Baseline (Full Model)']
    impact = {}

    # A positive change means performance got worse when the component was removed,
    # indicating the component was helpful.
    impact['Ratio Features'] = results['Ablation: No Ratio Features'] - baseline_rmse
    impact['Negative Clipping'] = results['Ablation: No Negative Clipping'] - baseline_rmse
    impact['street_key_count Feature'] = results['Ablation: No street_key_count'] - baseline_rmse

    # Find the component whose removal caused the largest increase in RMSE.
    # This is the most beneficial component.
    if not any(v > 0 for v in impact.values()):
        most_impactful = "None, as no single component's removal worsened performance"
    else:
        most_impactful = max(impact, key=impact.get)

    print(f"\nConclusion: The '{most_impactful}' contributes the most to the model's performance.")


if __name__ == '__main__':
    perform_ablation_study()
