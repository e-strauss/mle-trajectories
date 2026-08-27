
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

# --- Create dummy data for self-contained script ---
def create_dummy_data():
    os.makedirs('./input', exist_ok=True)
    # violations data
    data = {
        'Street Name': ['MAIN ST'] * 10 + ['OAK AVE'] * 10 + ['PINE LN'] * 10 + ['ELM ST'] * 10,
        'Violation Description': ['NO PARKING-STREET CLEANING'] * 20 + ['FIRE HYDRANT'] * 20,
        'violation_count': list(np.random.randint(5, 50, 20)) + list(np.random.randint(50, 200, 20))
    }
    df_violations = pd.DataFrame(data)
    df_violations.to_csv('./input/violations_per_street_2022.csv', index=False)
    # borough data
    cscl_data = {
        'ST_NAME': ['MAIN ST', 'OAK AVE', 'PINE LN', 'MAPLE DR'],
        'BORONAME': ['Manhattan', 'Brooklyn', 'Brooklyn', 'Queens']
    }
    df_cscl = pd.DataFrame(cscl_data)
    df_cscl.to_csv('./input/nyc_cscl.csv', index=False)

def feature_engineer(df, train_stats=None, fill_na_strategy='zero', global_mean_val=None):
    """
    Engineers features, allowing for ablation on NaN filling strategy.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    cscl = pd.read_csv(cscl_path)
    # FIX: Standardize column names to lowercase to avoid KeyError
    cscl.columns = [col.lower() for col in cscl.columns]
    
    cscl['st_name'] = cscl['st_name'].str.upper()
    df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
    
    # FIX: Use lowercase column names ('st_name') for merge and subsequent operations
    df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='st_name', how='left')
    
    # This line will now work as 'boroname' column exists
    df_engineered['boroname'].fillna('Unknown', inplace=True)
    
    # FIX: Use correct column names to drop
    df_engineered.drop(columns=['street_name_upper', 'st_name'], inplace=True)
    
    has_target = 'violation_count' in df_engineered.columns

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required for training stats.")
        
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        # Interaction features
        boro_violation_agg = df_engineered.groupby(['boroname', 'violation_description'])['violation_count'].agg(['mean', 'sum', 'std'])
        boro_violation_agg.columns = ['boro_violation_mean', 'boro_violation_sum', 'boro_violation_std']
        
        stats = {
            'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg,
            'boro_violation_agg': boro_violation_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        stats = train_stats

    # Merge features
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_violation_agg'], on=['boroname', 'violation_description'], how='left')

    # --- NaN Filling Strategy ---
    # `std` for single-member groups is NaN. This should always be 0.
    std_cols = [c for c in df_engineered.columns if c.endswith('_std')]
    df_engineered[std_cols] = df_engineered[std_cols].fillna(0)
    
    fill_value = 0
    if fill_na_strategy == 'global_mean':
        # For train set, use its own mean. For val/test, use the provided one from training.
        fill_value = stats['global_mean'] if global_mean_val is None else global_mean_val
    
    # Apply filling to all remaining NaNs (from unseen keys in val/test)
    df_engineered.fillna(fill_value, inplace=True)
    
    return df_engineered, stats

def run_experiment(ablation_name, use_interaction_features=True, nan_fill_strategy='zero', ridge_cv_folds=5):
    """
    Runs a single training and validation experiment with specified configurations.
    """
    # Load data
    df_original = pd.read_csv('./input/violations_per_street_2022.csv')

    # Validation Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # Feature Engineering
    train_featured, train_stats = feature_engineer(train_df, fill_na_strategy=nan_fill_strategy)
    global_mean_for_val = train_stats['global_mean']
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, fill_na_strategy=nan_fill_strategy, global_mean_val=global_mean_for_val)

    # Define features based on ablation flags
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    if use_interaction_features:
        numerical_features.extend(['boro_violation_mean', 'boro_violation_sum', 'boro_violation_std'])

    all_features = numerical_features + categorical_features
    X_train = train_featured[all_features]
    y_train = train_featured['violation_count']
    X_val = val_featured[all_features]
    y_val = val_featured['violation_count']

    # Model Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=ridge_cv_folds))
    ])

    # Training and Validation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"{ablation_name} RMSE: {rmse:.4f}")
    return rmse

if __name__ == '__main__':
    create_dummy_data()
    
    results = {}

    # --- Run Experiments ---
    # Baseline: Full model with all features and default settings
    baseline_rmse = run_experiment("Baseline")
    results['Baseline'] = baseline_rmse
    print(f"Final Validation Performance: {baseline_rmse:.4f}")

    # Ablation 1: Remove the new interaction features
    results['No Interaction Features'] = run_experiment("Ablation (No Interaction Features)", use_interaction_features=False)

    # Ablation 2: Change NaN filling strategy from 'zero' to 'global_mean'
    results['Global Mean NaN Fill'] = run_experiment("Ablation (Global Mean NaN Fill)", nan_fill_strategy='global_mean')
    
    # Ablation 3: Reduce the number of cross-validation folds in RidgeCV
    results['Fewer CV Folds'] = run_experiment("Ablation (Fewer CV Folds)", ridge_cv_folds=3)

    # --- Analyze Results ---
    # The baseline_rmse is already captured from the first run.
    impacts = {}
    for name, rmse in results.items():
        if name != 'Baseline':
            # Positive impact means RMSE increased, so the component was helpful
            impacts[name] = rmse - baseline_rmse

    # Find the component with the largest absolute change in RMSE
    if impacts:
        most_impactful_component = max(impacts, key=lambda k: abs(impacts[k]))
        impact_value = impacts[most_impactful_component]

        print("\n--- Conclusion ---")
        if impact_value > 0:
            conclusion = (f"The ablation study shows that '{most_impactful_component}' contributes the most positively to performance. "
                          f"Removing or altering it degraded the model, increasing RMSE by {impact_value:.4f}.")
        else:
            conclusion = (f"The ablation study shows that modifying '{most_impactful_component}' had the most impact. "
                          f"This change improved performance, decreasing RMSE by {abs(impact_value):.4f}, suggesting the baseline was suboptimal.")
        print(conclusion)
    else:
        print("\n--- Conclusion ---\nNo ablations were run.")
