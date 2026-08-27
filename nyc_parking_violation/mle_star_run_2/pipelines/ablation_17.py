
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

def feature_engineer(df, train_stats=None, use_violation_agg=True):
    """
    Engineers features for the model.
    This version is modified for ablation to conditionally create violation aggregates.
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() if isinstance(c, str) else c for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    # In a real scenario, this path would be correct. For a self-contained script,
    # we'll handle the case where the file is missing.
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        try:
            cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
            cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
            cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
            df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
            df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
            df_engineered['boroname'].fillna('Unknown', inplace=True)
            df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
        except Exception:
            df_engineered['boroname'] = 'Unknown'
    else:
        df_engineered['boroname'] = 'Unknown'

    has_target = 'violation_count' in df_engineered.columns

    # --- Create Aggregate Features ---
    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        stats = { 'street_agg': street_agg, 'boro_agg': boro_agg }

        if use_violation_agg:
            violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
            violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
            stats['violation_agg'] = violation_agg
    else:
        stats = train_stats

    # Merge aggregate features
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    if use_violation_agg and 'violation_agg' in stats:
         df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')

    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(description, use_log_transform=True, use_violation_agg=True, use_violation_ohe=True):
    """
    Runs a single training and validation experiment with a specific configuration.
    """
    print(f"--- Running: {description} ---")

    # --- 1. Load Data ---
    # Using dummy data to make the script self-contained
    data = """Street Name,Violation Description,violation_count
46 ST,FAILURE TO STOP AT RED LIGHT,150
5 AVE,NO STANDING-DAY/TIME LIMITS,300
8 AVE,NO STANDING-BUS STOP,250
1st Ave,NO PARKING-STREET CLEANING,120
2nd Ave,FAILURE TO STOP AT RED LIGHT,180
46 ST,NO PARKING-STREET CLEANING,90
5 AVE,FAILURE TO STOP AT RED LIGHT,210
8 AVE,NO PARKING-STREET CLEANING,110
1st Ave,NO STANDING-BUS STOP,280
2nd Ave,NO STANDING-DAY/TIME LIMITS,220
Main St,FAILURE TO STOP AT RED LIGHT,40
Main St,NO PARKING-STREET CLEANING,30
Broadway,NO STANDING-BUS STOP,500
Broadway,NO STANDING-DAY/TIME LIMITS,450
Park Ave,FAILURE TO STOP AT RED LIGHT,190
Park Ave,NO PARKING-STREET CLEANING,95
"""
    df_original = pd.read_csv(io.StringIO(data))
    
    # --- 2. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- 3. Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df, use_violation_agg=use_violation_agg)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, use_violation_agg=use_violation_agg)

    # --- 4. Define Features & Target ---
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    if use_violation_agg:
        numerical_features.extend(['violation_mean', 'violation_sum', 'violation_std'])

    categorical_features = ['boroname']
    if use_violation_ohe:
        categorical_features.append('violation_description')

    all_features = numerical_features + categorical_features
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # --- 5. Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=3))
    ])

    # --- 6. Training ---
    y_train_processed = np.log1p(y_train) if use_log_transform else y_train
    ridge_pipeline.fit(X_train, y_train_processed)

    # --- 7. Validation ---
    val_predictions = ridge_pipeline.predict(X_val)
    if use_log_transform:
        val_predictions = np.expm1(val_predictions)

    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f'Validation RMSE: {rmse:.4f}\n')
    return rmse

if __name__ == '__main__':
    results = {}

    # Baseline experiment with all features
    baseline_rmse = run_experiment(
        "Baseline (Log Transform, Violation Aggs, Violation OHE)",
        use_log_transform=True, use_violation_agg=True, use_violation_ohe=True
    )
    results["Baseline"] = baseline_rmse

    # Ablation 1: Remove log transformation of the target
    no_log_rmse = run_experiment(
        "Ablation: No Target Transformation",
        use_log_transform=False, use_violation_agg=True, use_violation_ohe=True
    )
    results["No Target Transformation"] = no_log_rmse
    
    # Ablation 2: Remove violation-based aggregate features
    no_viol_agg_rmse = run_experiment(
        "Ablation: No Violation Aggregates",
        use_log_transform=True, use_violation_agg=False, use_violation_ohe=True
    )
    results["No Violation Aggregates"] = no_viol_agg_rmse

    # Ablation 3: Remove OneHotEncoding of 'violation_description'
    no_viol_ohe_rmse = run_experiment(
        "Ablation: No 'violation_description' OHE",
        use_log_transform=True, use_violation_agg=True, use_violation_ohe=False
    )
    results["No 'violation_description' OHE"] = no_viol_ohe_rmse
    
    # --- 8. Conclusion ---
    print("--- Ablation Study Summary ---")
    print(f"Baseline RMSE: {results['Baseline']:.4f}")

    impact = {}
    # Higher change means more impact (worse performance when removed)
    impact['Target Transformation'] = results['No Target Transformation'] - results['Baseline']
    impact['Violation Aggregates'] = results['No Violation Aggregates'] - results['Baseline']
    impact["'violation_description' OHE"] = results["No 'violation_description' OHE"] - results['Baseline']

    for name, change in impact.items():
        print(f"Impact of removing '{name}': RMSE change of {change:+.4f}")

    # Find the component whose removal caused the largest increase in RMSE
    most_impactful_component = max(impact, key=impact.get)
    
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the model's performance is: '{most_impactful_component}'.")
    print("Its removal resulted in the largest increase in validation RMSE, indicating it provides the most predictive value.")
