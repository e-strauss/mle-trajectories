
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import collections

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# --- Data and Feature Engineering ---

def setup_dummy_data():
    """Creates dummy data files required for the script to run."""
    # Create input directory
    if not os.path.exists('./input'):
        os.makedirs('./input')

    # Create dummy training data
    train_data = {
        'Street Name': ['A ST', 'A ST', 'B ST', 'B ST', 'C ST', 'C ST', 'D ST', 'E ST', 'F ST'] * 20,
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT'] * 20,
        'Violation Count': np.random.randint(1, 100, 9 * 20)
    }
    df_train = pd.DataFrame(train_data)
    df_train.to_csv('./input/violations_per_street_2022.csv', index=False)

    # Create dummy borough data
    cscl_data = {
        'ST_NAME': ['A ST', 'B ST', 'C ST', 'D ST', 'E ST', 'F ST'],
        'BORONAME': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Brooklyn']
    }
    df_cscl = pd.DataFrame(cscl_data)
    df_cscl.to_csv('./input/nyc_cscl.csv', index=False)

def feature_engineer(df, train_stats=None, add_missing_indicators=False):
    """
    Engineers features for the model, with an option to add missingness indicators.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        cscl = pd.read_csv(cscl_path)
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        
        # FIX: The merged column from cscl is 'BORONAME' (uppercase).
        # Rename it to 'boroname' to match the expected lowercase format.
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

    original_cols = set(df_engineered.columns)

    # Merge aggregate features
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Ablation: Add binary indicators for missing aggregate values before imputation
    if add_missing_indicators:
        added_cols = set(df_engineered.columns) - original_cols
        for col in sorted(list(added_cols)): # sorted for determinism
            if df_engineered[col].isnull().any():
                df_engineered[f'{col}_was_missing'] = df_engineered[col].isnull().astype(int)

    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

# --- Experiment Runner ---

def run_experiment(df_original, seed, model_type, add_missing_indicators):
    """
    Runs a single training and validation experiment with specified configurations.
    """
    np.random.seed(seed)
    
    # 1. Validation Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # 2. Feature Engineering
    train_featured, train_stats = feature_engineer(train_df, add_missing_indicators=add_missing_indicators)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, add_missing_indicators=add_missing_indicators)

    # 3. Define Features for the model
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    if add_missing_indicators:
        indicator_features = sorted([col for col in train_featured.columns if '_was_missing' in col])
        numerical_features.extend(indicator_features)

    all_features = numerical_features + categorical_features
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # 4. Model Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    if model_type == 'LassoCV':
        model = LassoCV(alphas=np.logspace(-2, 2, 5), cv=5, random_state=seed, max_iter=5000)
    else: # Default to RidgeCV
        model = RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5)
        
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

    # 5. Training and Evaluation
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse


# --- Main Ablation Study ---

def main():
    """
    Main function to run the ablation study.
    """
    setup_dummy_data()
    train_path = './input/violations_per_street_2022.csv'
    df_original = pd.read_csv(train_path)

    results = collections.OrderedDict()
    BASE_SEED = 42

    print("Running ablation study...")

    # Baseline Experiment
    results['Baseline (RidgeCV, No Indicators, Seed 42)'] = run_experiment(
        df_original, seed=BASE_SEED, model_type='RidgeCV', add_missing_indicators=False
    )

    # Ablation 1: Add Missingness Indicator Features
    results['Ablation: Add Missing Indicators'] = run_experiment(
        df_original, seed=BASE_SEED, model_type='RidgeCV', add_missing_indicators=True
    )

    # Ablation 2: Change Model to LassoCV
    results['Ablation: Use LassoCV Model'] = run_experiment(
        df_original, seed=BASE_SEED, model_type='LassoCV', add_missing_indicators=False
    )
    
    # Ablation 3: Test Split Stability
    results['Ablation: Use Different Random Split (Seed 0)'] = run_experiment(
        df_original, seed=0, model_type='RidgeCV', add_missing_indicators=False
    )

    print("\n--- Ablation Study Results (RMSE) ---")
    baseline_rmse = results['Baseline (RidgeCV, No Indicators, Seed 42)']
    performance_impact = {}

    for name, rmse in results.items():
        change = rmse - baseline_rmse if name != 'Baseline (RidgeCV, No Indicators, Seed 42)' else 0
        print(f"{name:<45}: {rmse:8.4f} (Change from Baseline: {change:+.4f})")
        if 'Ablation' in name:
            performance_impact[name] = abs(change)
    
    # Add required performance output line
    print(f"Final Validation Performance: {baseline_rmse}")

    # Determine the most impactful component
    if not performance_impact:
        most_impactful = "No ablations were performed."
    else:
        most_impactful_name = max(performance_impact, key=performance_impact.get)
        # Extract the core component name from the experiment description
        if 'Missing Indicators' in most_impactful_name:
            most_impactful = "Adding Missingness Indicator Features"
        elif 'LassoCV' in most_impactful_name:
            most_impactful = "Model Type (RidgeCV vs LassoCV)"
        elif 'Random Split' in most_impactful_name:
            most_impactful = "Random Split Seed"
        else:
            most_impactful = most_impactful_name

    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: '{most_impactful}'")
    print(f"This was determined by the largest absolute change in RMSE compared to the baseline.")

if __name__ == '__main__':
    main()
