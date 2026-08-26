
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import copy

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
    # In a real scenario, the path would be './input/nyc_cscl.csv'
    # For this self-contained script, we'll handle the case where it doesn't exist.
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

def run_ablation_study():
    """
    Runs an ablation study on the LGBM training pipeline.
    """
    # Create a dummy dataset since we can't access the original files
    print("Creating a dummy dataset for the ablation study...")
    num_samples = 1000
    streets = [f'Street_{i}' for i in range(50)]
    violations = [f'Violation_{i}' for i in range(10)]
    data = {
        'Street Name': np.random.choice(streets, num_samples),
        'Violation Description': np.random.choice(violations, num_samples),
        'violation_count': np.random.poisson(20, num_samples) + np.random.randint(0, 50, num_samples)
    }
    df_original = pd.DataFrame(data)
    # Add some correlation for features to have an effect
    street_map = {street: i for i, street in enumerate(streets)}
    violation_map = {vio: i for i, vio in enumerate(violations)}
    df_original['violation_count'] += df_original['Street Name'].map(street_map) * 2
    df_original['violation_count'] += df_original['Violation Description'].map(violation_map) * 5


    # --- 1. Common Setup: Data Splitting and Feature Engineering ---
    print("Splitting data and engineering features...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    all_features = numerical_features + categorical_features
    target = 'violation_count'

    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    results = {}

    # --- 2. Baseline Experiment ---
    print("\n--- Running Baseline Experiment ---")
    preprocessor_base = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough')

    pipeline_base = Pipeline(steps=[
        ('preprocessor', preprocessor_base),
        ('regressor', LGBMRegressor(objective='poisson', random_state=SEED))
    ])
    pipeline_base.fit(X_train, y_train)
    val_preds = pipeline_base.predict(X_val)
    val_preds[val_preds < 0] = 0
    baseline_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    results['Baseline'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # --- 3. Ablation 1: Change LGBM Objective ---
    print("\n--- Ablation 1: Change LGBM Objective (from 'poisson' to 'regression_l2') ---")
    pipeline_ab1 = copy.deepcopy(pipeline_base)
    pipeline_ab1.named_steps['regressor'].set_params(objective='regression_l2')
    pipeline_ab1.fit(X_train, y_train)
    val_preds_ab1 = pipeline_ab1.predict(X_val)
    val_preds_ab1[val_preds_ab1 < 0] = 0
    ab1_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ab1))
    results["LGBM Objective ('regression_l2')"] = ab1_rmse
    print(f"Ablation RMSE: {ab1_rmse:.4f} (Change from Baseline: {ab1_rmse - baseline_rmse:+.4f})")

    # --- 4. Ablation 2: No Feature Scaling ---
    print("\n--- Ablation 2: No Feature Scaling ---")
    preprocessor_ab2 = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough')
    # Reorder columns to match remainder
    X_train_ab2 = X_train[categorical_features + numerical_features]
    X_val_ab2 = X_val[categorical_features + numerical_features]
    pipeline_ab2 = Pipeline(steps=[
        ('preprocessor', preprocessor_ab2),
        ('regressor', LGBMRegressor(objective='poisson', random_state=SEED))
    ])
    pipeline_ab2.fit(X_train_ab2, y_train)
    val_preds_ab2 = pipeline_ab2.predict(X_val_ab2)
    val_preds_ab2[val_preds_ab2 < 0] = 0
    ab2_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ab2))
    results['No Feature Scaling'] = ab2_rmse
    print(f"Ablation RMSE: {ab2_rmse:.4f} (Change from Baseline: {ab2_rmse - baseline_rmse:+.4f})")
    
    # --- 5. Ablation 3: Native Categorical Handling ---
    print("\n--- Ablation 3: Native LGBM Categorical Handling ---")
    X_train_ab3 = X_train.copy()
    X_val_ab3 = X_val.copy()
    # Convert categorical columns to pandas 'category' dtype
    for col in categorical_features:
        X_train_ab3[col] = X_train_ab3[col].astype('category')
        X_val_ab3[col] = X_val_ab3[col].astype('category')
    
    # Manually scale numerical features
    scaler = StandardScaler()
    X_train_ab3[numerical_features] = scaler.fit_transform(X_train_ab3[numerical_features])
    X_val_ab3[numerical_features] = scaler.transform(X_val_ab3[numerical_features])

    model_ab3 = LGBMRegressor(objective='poisson', random_state=SEED)
    # Fit model directly, letting LGBM know which features are categorical
    model_ab3.fit(X_train_ab3, y_train, categorical_feature=categorical_features)
    val_preds_ab3 = model_ab3.predict(X_val_ab3)
    val_preds_ab3[val_preds_ab3 < 0] = 0
    ab3_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ab3))
    results['Native Categorical Handling'] = ab3_rmse
    print(f"Ablation RMSE: {ab3_rmse:.4f} (Change from Baseline: {ab3_rmse - baseline_rmse:+.4f})")

    # --- 6. Conclusion ---
    print("\n" + "="*50)
    print("Ablation Study Summary:")
    print(f"{'Experiment':<35} | {'RMSE':<10} | {'Change from Baseline':<20}")
    print("-"*70)
    print(f"{'Baseline':<35} | {results['Baseline']:<10.4f} | {'-':<20}")
    
    impacts = {}
    for name, score in results.items():
        if name != 'Baseline':
            change = score - baseline_rmse
            impacts[name] = abs(change)
            print(f"{name:<35} | {score:<10.4f} | {change:<+20.4f}")

    most_impactful = max(impacts, key=impacts.get)
    print("="*50)
    print(f"\nConclusion: The component that contributes the most to the overall performance is '{most_impactful}'.")

if __name__ == '__main__':
    run_ablation_study()
