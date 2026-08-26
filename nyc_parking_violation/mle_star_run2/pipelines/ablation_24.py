
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

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

# --- Create dummy data for self-contained execution ---
if not os.path.exists('./input'):
    os.makedirs('./input')

# Dummy violations data
violations_data = """Street Name,Violation Description,violation_count
5TH AVE,FAILURE TO STOP AT RED LIGHT,150
5TH AVE,DOUBLE PARKING,80
PARK AVE,NO PARKING-STREET CLEANING,200
PARK AVE,FAILURE TO STOP AT RED LIGHT,70
BROADWAY,DOUBLE PARKING,120
BROADWAY,NO STANDING-DAY/TIME LIMITS,90
MADISON AVE,FAILURE TO STOP AT RED LIGHT,60
MADISON AVE,DOUBLE PARKING,40
LEXINGTON AVE,NO PARKING-STREET CLEANING,180
LEXINGTON AVE,NO STANDING-DAY/TIME LIMITS,50
34TH ST,BUS LANE VIOLATION,300
34TH ST,DOUBLE PARKING,110
42ND ST,BUS LANE VIOLATION,400
42ND ST,FAILURE TO STOP AT RED LIGHT,95
TIMES SQ,PHTO SCHOOL ZN SPEED VIOLATION,250
"""
with open('./input/violations_per_street_2022.csv', 'w') as f:
    f.write(violations_data)

# Dummy borough data
cscl_data = """ST_NAME,BORONAME
5TH AVE,Manhattan
PARK AVE,Manhattan
BROADWAY,Manhattan
MADISON AVE,Manhattan
LEXINGTON AVE,Manhattan
34TH ST,Manhattan
42ND ST,Manhattan
TIMES SQ,Manhattan
FLATBUSH AVE,Brooklyn
ATLANTIC AVE,Brooklyn
"""
with open('./input/nyc_cscl.csv', 'w') as f:
    f.write(cscl_data)

# --- Feature Engineering Functions for Ablation ---

def feature_engineer_full(df, train_stats=None, use_ratio_features=True, use_hierarchical_imputation=True):
    """ The full feature engineering pipeline with flags for ablation. """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

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

    global_mean = stats.get('global_mean', 0)

    if use_hierarchical_imputation:
        df_engineered['boro_mean'].fillna(global_mean, inplace=True)
        df_engineered['street_mean'].fillna(df_engineered['boro_mean'], inplace=True)
        df_engineered['violation_mean'].fillna(global_mean, inplace=True)

    if use_ratio_features:
        epsilon = 1e-6
        df_engineered['street_to_boro_mean_ratio'] = df_engineered['street_mean'] / (df_engineered['boro_mean'] + epsilon)
        df_engineered['street_to_global_mean_ratio'] = df_engineered['street_mean'] / (global_mean + epsilon)
        df_engineered['boro_to_global_mean_ratio'] = df_engineered['boro_mean'] / (global_mean + epsilon)

    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(name, train_df, val_df, use_ratio_features=True, use_hierarchical_imputation=True, use_scaler=True):
    """
    Runs a single experiment with specified configurations.
    """
    print(f"--- Running: {name} ---")

    # 1. Feature Engineering
    train_featured, train_stats = feature_engineer_full(train_df, use_ratio_features=use_ratio_features, use_hierarchical_imputation=use_hierarchical_imputation)
    val_featured, _ = feature_engineer_full(val_df, train_stats=train_stats, use_ratio_features=use_ratio_features, use_hierarchical_imputation=use_hierarchical_imputation)

    # 2. Define Features
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    if use_ratio_features:
        numerical_features.extend(['street_to_boro_mean_ratio', 'street_to_global_mean_ratio', 'boro_to_global_mean_ratio'])
    
    all_features = numerical_features + categorical_features
    
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # 3. Model Pipeline
    num_transformer = StandardScaler() if use_scaler else 'passthrough'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=3))
    ])

    # 4. Training and Evaluation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"  Validation RMSE: {rmse:.4f}\n")
    return rmse


def main():
    # Load and Split Data (Done once)
    train_path = './input/violations_per_street_2022.csv'
    df_original = pd.read_csv(train_path)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    # --- Run Experiments ---
    results['Baseline'] = run_experiment("Baseline (Full Model)", train_df, val_df)
    results['No Ratio Features'] = run_experiment("Ablation: No Ratio Features", train_df, val_df, use_ratio_features=False)
    results['No Hierarchical Imputation'] = run_experiment("Ablation: No Hierarchical Imputation", train_df, val_df, use_hierarchical_imputation=False)
    results['No StandardScaler'] = run_experiment("Ablation: No StandardScaler", train_df, val_df, use_scaler=False)

    # --- Print Conclusion ---
    print("--- Ablation Study Summary ---")
    baseline_rmse = results['Baseline']
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    impacts = {}
    for key, rmse in results.items():
        if key != 'Baseline':
            change = rmse - baseline_rmse
            print(f"Ablation '{key}': RMSE={rmse:.4f}, Change from Baseline: {change:+.4f}")
            impacts[key] = abs(change)
    
    if impacts:
        most_impactful_component = max(impacts, key=impacts.get)
        print(f"\nConclusion: The component that contributes the most to the overall performance is '{most_impactful_component}'.")
    else:
        print("\nConclusion: No measurable impact from the tested ablations.")

if __name__ == '__main__':
    main()
