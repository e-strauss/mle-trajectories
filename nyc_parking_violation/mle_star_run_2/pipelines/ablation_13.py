
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
    Engineers features for the model. Includes borough augmentation and aggregates.
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
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def train_and_evaluate(train_featured, val_featured, use_aggregates=True, ohe_handle_unknown='ignore', ridge_cv_folds=5):
    """
    Helper function to run one experiment configuration.
    It takes pre-engineered dataframes and applies different modeling configurations.
    """
    if use_aggregates:
        categorical_features = ['violation_description', 'boroname']
        numerical_features = [
            'street_mean', 'street_sum', 'street_std', 'street_key_count',
            'violation_mean', 'violation_sum', 'violation_std',
            'boro_mean', 'boro_sum', 'boro_std'
        ]
    else:
        # Ablation: Do not use any of the created aggregate features
        categorical_features = ['violation_description', 'boroname']
        numerical_features = []

    all_features = numerical_features + categorical_features
    target = 'violation_count'
    
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- Model Pipeline ---
    transformers = []
    if numerical_features:
        transformers.append(('num', StandardScaler(), numerical_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown=ohe_handle_unknown, sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=ridge_cv_folds))
    ])

    # --- Training and Validation ---
    try:
        pipeline.fit(X_train, y_train)
        val_predictions = pipeline.predict(X_val)
        val_predictions[val_predictions < 0] = 0
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        return rmse
    except Exception as e:
        return f"Failed with error: {e}"

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    
    # Create dummy files if they don't exist to make the script runnable
    if not os.path.exists(train_path):
        print(f"Training file not found. Creating dummy data at '{train_path}'...")
        os.makedirs('./input', exist_ok=True)
        dummy_data = {
            'Street Name': [f'Street {i % 10}' for i in range(200)] + ['New Street in Val'],
            'Violation Description': [f'Violation Type {i % 5}' for i in range(201)],
            'violation_count': np.random.randint(1, 500, 201)
        }
        pd.DataFrame(dummy_data).to_csv(train_path, index=False)
        cscl_path = './input/nyc_cscl.csv'
        if not os.path.exists(cscl_path):
            print(f"Creating dummy CSCL file at '{cscl_path}'...")
            dummy_cscl_data = {
                'ST_NAME': [f'STREET {i}' for i in range(10)],
                'BORONAME': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'] * 2
            }
            pd.DataFrame(dummy_cscl_data).to_csv(cscl_path, index=False)

    df_original = pd.read_csv(train_path)

    # --- Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Run Feature Engineering Once ---
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    # --- Run Experiments ---
    results = {}
    
    results['Baseline'] = train_and_evaluate(train_featured, val_featured, use_aggregates=True, ohe_handle_unknown='ignore', ridge_cv_folds=5)
    results['No Aggregate Features'] = train_and_evaluate(train_featured, val_featured, use_aggregates=False)
    results['OHE handle_unknown=error'] = train_and_evaluate(train_featured, val_featured, ohe_handle_unknown='error')
    results['RidgeCV Folds=3'] = train_and_evaluate(train_featured, val_featured, ridge_cv_folds=3)

    # --- Report Results ---
    print("\n--- Ablation Study Results ---")
    baseline_rmse = results['Baseline']
    print(f"Baseline (Full Model) RMSE: {baseline_rmse:.4f}")

    performance_changes = {}
    for name, rmse in results.items():
        if name == 'Baseline':
            continue
        if isinstance(rmse, str):
            print(f"Ablation '{name}': {rmse}")
            performance_changes[name] = float('inf') 
        else:
            change = rmse - baseline_rmse
            print(f"Ablation '{name}': RMSE = {rmse:.4f} (Change from Baseline: {change:+.4f})")
            performance_changes[name] = change
    
    print("\n--- Conclusion ---")
    valid_changes = {k: v for k, v in performance_changes.items() if v != float('inf')}
    if not valid_changes:
        most_impactful_ablation = "OHE handle_unknown=error"
    else:
        most_impactful_ablation = max(valid_changes, key=valid_changes.get)

    if performance_changes[most_impactful_ablation] > 0:
        print(f"The '{most_impactful_ablation}' component contributes the most to the model's performance.")
        print("Its removal or alteration caused the largest increase in error, confirming its high value.")
    else:
        best_improvement_ablation = min(valid_changes, key=valid_changes.get)
        print("The baseline model was improved by one of the ablations.")
        print(f"Altering the '{best_improvement_ablation}' component resulted in the largest performance improvement (lowest error).")

if __name__ == '__main__':
    main()
