
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

def feature_engineer(df, train_stats=None, create_missing_indicators=True, imputation_strategy='median'):
    """
    Engineers features for the model, with configurable NaN handling.

    Args:
        df (pd.DataFrame): The dataframe to engineer features for.
        train_stats (dict, optional): Stats from the training set.
        create_missing_indicators (bool): If True, create binary features indicating missingness.
        imputation_strategy (str): 'median', 'mean', or 'zero'.

    Returns:
        pd.DataFrame: The dataframe with new features.
        dict: A dictionary of the calculated stats.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data (placeholder if file not found) ---
    df_engineered['boroname'] = 'Unknown' # Simplified for this study

    has_target = 'violation_count' in df_engineered.columns

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column required for training.")
        
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
            'global_mean': df_engineered['violation_count'].mean(),
            'global_median': df_engineered['violation_count'].median()
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    agg_cols = list(stats['street_agg'].columns) + list(stats['violation_agg'].columns) + list(stats['boro_agg'].columns)

    if imputation_strategy == 'zero':
        df_engineered.fillna(0, inplace=True)
    else: # 'median' or 'mean'
        impute_value = stats['global_median'] if imputation_strategy == 'median' else stats['global_mean']
        for col in agg_cols:
            if create_missing_indicators:
                df_engineered[f'{col}_is_missing'] = df_engineered[col].isnull().astype(int)
            df_engineered[col].fillna(impute_value, inplace=True)
        # Final safeguard
        df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(df_original, fe_config):
    """Encapsulates a single training and validation run."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    train_featured, train_stats = feature_engineer(train_df, **fe_config)
    
    # Get feature names after engineering on the training set
    base_numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    
    missing_indicator_features = [f'{col}_is_missing' for col in base_numerical_features if f'{col}_is_missing' in train_featured.columns]
    numerical_features = base_numerical_features + missing_indicator_features
    categorical_features = ['violation_description', 'boroname']
    all_features = numerical_features + categorical_features
    
    # Apply same FE to validation set
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, **fe_config)

    # Align columns for safety, filling missing feature columns with 0
    for col in all_features:
        if col not in val_featured.columns:
            val_featured[col] = 0
    
    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

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
    val_predictions[val_predictions < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def main():
    # Simulate loading data for the study
    data = """"Street Name","Violation Description","violation_count"
"5TH AVE","FAILURE TO STOP AT RED LIGHT",150
"BROADWAY","FAILURE TO STOP AT RED LIGHT",200
"LEXINGTON AVE","FAILURE TO STOP AT RED LIGHT",120
"MADISON AVE","FAILURE TO STOP AT RED LIGHT",110
"PARK AVE","FAILURE TO STOP AT RED LIGHT",100
"5TH AVE","NO PARKING-STREET CLEANING",300
"BROADWAY","NO PARKING-STREET CLEANING",400
"LEXINGTON AVE","NO PARKING-STREET CLEANING",250
"WALL ST","NO PARKING-STREET CLEANING",50
"WALL ST","DOUBLE PARKING",30
"CHAMBERS ST", "FAILURE TO DISPLAY MUNI METER", 20
"FDR DRIVE", "SPEEDING", 500
"WEST SIDE HWY", "SPEEDING", 450
"FDR DRIVE", "IMPROPER LANE CHANGE", 100
"5TH AVE", "DOUBLE PARKING", 80
"BROADWAY", "BLOCKING THE BOX", 150
"SPRING ST", "NO STANDING-DAYTIME", 90
"HOUSTON ST", "NO STANDING-DAYTIME", 85
"14TH ST", "BUS LANE VIOLATION", 180
"34TH ST", "BUS LANE VIOLATION", 220
"42ND ST", "BUS LANE VIOLATION", 300
"""
    df_original = pd.read_csv(io.StringIO(data))
    
    # Expand the dummy data to have a more realistic validation set
    np.random.seed(SEED)
    new_streets = [f"NEW_ST_{i}" for i in range(5)]
    new_violations = list(df_original["Violation Description"].unique())
    new_data = []
    for street in new_streets:
        for violation in np.random.choice(new_violations, 2, replace=False):
            new_data.append({"Street Name": street, "Violation Description": violation, "violation_count": np.random.randint(10, 50)})
    df_original = pd.concat([df_original, pd.DataFrame(new_data)], ignore_index=True)


    experiments = {
        "Baseline (Median Impute + Indicators)": {"create_missing_indicators": True, "imputation_strategy": 'median'},
        "Ablation: No Missingness Indicators": {"create_missing_indicators": False, "imputation_strategy": 'median'},
        "Ablation: Impute with Global Mean": {"create_missing_indicators": True, "imputation_strategy": 'mean'},
        "Ablation: Impute with Zero": {"create_missing_indicators": False, "imputation_strategy": 'zero'}
    }

    results = {}
    print("--- Running Ablation Study on NaN Handling Strategy ---")
    for name, config in experiments.items():
        rmse = run_experiment(df_original, config)
        results[name] = rmse
        print(f"'{name}': Validation RMSE = {rmse:.4f}")

    baseline_rmse = results["Baseline (Median Impute + Indicators)"]
    performance_changes = {}

    for name, rmse in results.items():
        if name != "Baseline (Median Impute + Indicators)":
            change = rmse - baseline_rmse
            performance_changes[name] = change

    # Find the component with the largest absolute impact
    if not performance_changes:
        most_impactful_component = "N/A"
    else:
        most_impactful_component = max(performance_changes, key=lambda k: abs(performance_changes[k]))
        # Clean up the name for the conclusion
        most_impactful_component = most_impactful_component.replace("Ablation: ", "")

    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is '{most_impactful_component}'.")
    for name, change in performance_changes.items():
        clean_name = name.replace("Ablation: ", "")
        print(f"  - Modifying '{clean_name}' changed the RMSE by {change:+.4f}.")


if __name__ == '__main__':
    main()

