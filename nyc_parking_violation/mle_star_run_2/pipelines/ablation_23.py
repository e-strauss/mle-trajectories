
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


def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
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
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats


def run_experiment(numerical_features, categorical_features, experiment_name):
    """
    Runs a single training and validation experiment with a given feature set.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")
    
    # --- 1. Load Data ---
    try:
        df_original = pd.read_csv('./input/violations_per_street_2022.csv')
    except FileNotFoundError:
        print("Error: Training file './input/violations_per_street_2022.csv' not found. Aborting.")
        return np.inf

    # --- 2. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- 3. Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    all_features = numerical_features + categorical_features
    for col in all_features:
        if col not in train_featured.columns:
            raise ValueError(f"Feature column '{col}' not found after feature engineering.")

    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- 4. Model Pipeline ---
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

    # --- 5. Training & 6. Validation ---
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

def perform_ablation_study():
    """
    Conducts an ablation study on different feature sets.
    """
    # --- Baseline Experiment ---
    baseline_categorical_features = ['violation_description', 'boroname']
    baseline_numerical_features = [
        'street_mean', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std'
    ]
    baseline_rmse = run_experiment(
        baseline_numerical_features, 
        baseline_categorical_features, 
        "Baseline (Curated Features)"
    )

    results = {}

    # --- Ablation 1: No 'street_key_count' feature ---
    # This tests the contribution of the street frequency feature.
    abl1_numerical_features = [f for f in baseline_numerical_features if f != 'street_key_count']
    abl1_rmse = run_experiment(
        abl1_numerical_features,
        baseline_categorical_features,
        "Ablation: No 'street_key_count' Feature"
    )
    change1 = abl1_rmse - baseline_rmse
    results["'street_key_count' Feature"] = change1
    print(f"Impact of removing 'street_key_count': RMSE changed by {change1:+.4f}")


    # --- Ablation 2: Simplified Violation Features (mean only) ---
    # This tests if 'violation_sum' and 'violation_std' add value beyond 'violation_mean'.
    abl2_numerical_features = [f for f in baseline_numerical_features if f not in ['violation_sum', 'violation_std']]
    abl2_rmse = run_experiment(
        abl2_numerical_features,
        baseline_categorical_features,
        "Ablation: Simplified Violation Features"
    )
    change2 = abl2_rmse - baseline_rmse
    results["'violation_sum' and 'violation_std' Features"] = change2
    print(f"Impact of simplifying violation features: RMSE changed by {change2:+.4f}")


    # --- Ablation 3: No 'boroname' feature ---
    # This tests the contribution of the borough as a one-hot encoded feature.
    abl3_categorical_features = [f for f in baseline_categorical_features if f != 'boroname']
    abl3_rmse = run_experiment(
        baseline_numerical_features,
        abl3_categorical_features,
        "Ablation: No 'boroname' OHE Feature"
    )
    change3 = abl3_rmse - baseline_rmse
    results["'boroname' OHE Feature"] = change3
    print(f"Impact of removing 'boroname' OHE: RMSE changed by {change3:+.4f}")

    # --- Conclusion ---
    if not results:
        print("\nCould not run ablation studies.")
        return

    most_impactful_component = max(results, key=lambda k: abs(results[k]))
    print(f"\nBased on the results, the component that contributes the most to the overall performance is: '{most_impactful_component}'.")


if __name__ == '__main__':
    perform_ablation_study()
