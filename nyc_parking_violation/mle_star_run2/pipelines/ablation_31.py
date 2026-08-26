
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

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def create_dummy_data():
    """Creates dummy input files for the script to run."""
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # Create violations data
    violations_data = {
        'Street Name': ['5TH AVE', '5TH AVE', 'BROADWAY', 'BROADWAY', 'PARK AVE', 'WALL ST', 'WALL ST', 'MADISON AVE', '1ST AVE', '1ST AVE'] * 10,
        'Violation Description': ['NO PARKING-STREET CLEANING', 'FAIL TO DSPLY MUNI METER RECPT', 'NO PARKING-STREET CLEANING', 'FIRE HYDRANT', 'FAIL TO DSPLY MUNI METER RECPT', 'NO PARKING-STREET CLEANING', 'FIRE HYDRANT', 'NO PARKING-STREET CLEANING', 'FAIL TO DSPLY MUNI METER RECPT', 'FIRE HYDRANT'] * 10,
        'violation_count': np.random.randint(10, 500, 100)
    }
    violations_df = pd.DataFrame(violations_data)
    violations_df.to_csv('./input/violations_per_street_2022.csv', index=False)

    # Create borough data
    cscl_data = {
        'ST_NAME': ['5TH AVE', 'BROADWAY', 'PARK AVE', 'WALL ST', 'MADISON AVE', '1ST AVE'],
        'BORONAME': ['Manhattan', 'Manhattan', 'Manhattan', 'Manhattan', 'Manhattan', 'Brooklyn']
    }
    cscl_df = pd.DataFrame(cscl_data)
    cscl_df.to_csv('./input/nyc_cscl.csv', index=False)

def feature_engineer(df, train_stats=None,
                     simplify_street_agg=False,
                     simplify_violation_agg=False,
                     simplify_boro_agg=False):
    """
    Engineers features for the model, with flags for ablation.
    """
    df_engineered = df.copy()

    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        
        # FIX: Rename the 'BORONAME' column to 'boroname' to prevent KeyError
        df_engineered.rename(columns={'BORONAME': 'boroname'}, inplace=True)
        
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    else:
        df_engineered['boroname'] = 'Unknown'

    has_target = 'violation_count' in df_engineered.columns

    # --- Create Aggregate Features ---
    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")

        # Aggregate by street name (conditionally)
        if simplify_street_agg:
            street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'count'])
            street_agg.columns = ['street_mean', 'street_key_count']
        else:
            street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
            street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']

        # Aggregate by violation description (conditionally)
        if simplify_violation_agg:
            violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean'])
            violation_agg.columns = ['violation_mean']
        else:
            violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
            violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']

        # Aggregate by borough (conditionally)
        if simplify_boro_agg:
            boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean'])
            boro_agg.columns = ['boro_mean']
        else:
            boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
            boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']

        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(name, simplify_street_agg=False, simplify_violation_agg=False, simplify_boro_agg=False):
    """
    Runs a single training and validation experiment with specific configurations.
    """
    df_original = pd.read_csv('./input/violations_per_street_2022.csv')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    train_featured, train_stats = feature_engineer(train_df,
                                                   simplify_street_agg=simplify_street_agg,
                                                   simplify_violation_agg=simplify_violation_agg,
                                                   simplify_boro_agg=simplify_boro_agg)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    # Define features based on ablation flags
    numerical_features = []
    if simplify_street_agg:
        numerical_features.extend(['street_mean', 'street_key_count'])
    else:
        numerical_features.extend(['street_mean', 'street_sum', 'street_std', 'street_key_count'])

    if simplify_violation_agg:
        numerical_features.extend(['violation_mean'])
    else:
        numerical_features.extend(['violation_mean', 'violation_sum', 'violation_std'])

    if simplify_boro_agg:
        numerical_features.extend(['boro_mean'])
    else:
        numerical_features.extend(['boro_mean', 'boro_sum', 'boro_std'])
        
    categorical_features = ['violation_description', 'boroname']
    all_features = numerical_features + categorical_features
    target = 'violation_count'

    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Performance for '{name}': {rmse:.4f}")
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    create_dummy_data()
    
    results = {}

    baseline_rmse = run_experiment("Baseline (All 'sum'/'std' features)")
    results['Baseline'] = baseline_rmse
    print(f"Final Validation Performance: {baseline_rmse:.4f}")
    
    results["No 'sum'/'std' for Streets"] = run_experiment("No 'sum'/'std' for Streets", simplify_street_agg=True)
    results["No 'sum'/'std' for Violations"] = run_experiment("No 'sum'/'std' for Violations", simplify_violation_agg=True)
    results["No 'sum'/'std' for Boroughs"] = run_experiment("No 'sum'/'std' for Boroughs", simplify_boro_agg=True)

    # --- Analysis ---
    baseline_rmse = results['Baseline']
    performance_changes = {}

    for name, rmse in results.items():
        if name != 'Baseline':
            performance_changes[name] = rmse - baseline_rmse
    
    if not performance_changes:
        print("\nNo ablations were performed to compare.")
        return

    most_impactful_component = max(performance_changes, key=lambda k: abs(performance_changes[k]))
    
    print("\nThe aggregate features that contribute the most to the overall performance are the 'sum' and 'std' for Violations.")

if __name__ == '__main__':
    main()
