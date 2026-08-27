
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
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
    This function is copied from the original solution for consistency.
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

if __name__ == '__main__':
    train_path = './input/violations_per_street_2022.csv'
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Warning: Training file not found at {train_path}. Creating dummy data for demonstration.")
        os.makedirs('./input', exist_ok=True)
        dummy_data = {
            'Street Name': [f'STREET_{i%20}' for i in range(200)],
            'Violation Description': [f'VIOLATION_{i%5}' for i in range(200)],
            'violation_count': np.random.randint(1, 500, 200) + np.repeat(np.arange(20) * 20, 10)
        }
        df_original = pd.DataFrame(dummy_data)
        
        dummy_cscl_data = {
            'ST_NAME': [f'STREET_{i}' for i in range(20)],
            'BORONAME': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'] * 4
        }
        pd.DataFrame(dummy_cscl_data).to_csv('./input/nyc_cscl.csv', index=False)

    results = {}
    
    # Define experiment configurations
    ablation_configs = {
        "Baseline (Stratified Split, All Features, 5 CV Folds)": {
            "validation_strategy": "stratified",
            "use_std_features": True,
            "ridge_cv_folds": 5
        },
        "Ablation: Use GroupShuffleSplit": {
            "validation_strategy": "group_shuffle",
            "use_std_features": True,
            "ridge_cv_folds": 5
        },
        "Ablation: No STD Aggregate Features": {
            "validation_strategy": "stratified",
            "use_std_features": False,
            "ridge_cv_folds": 5
        },
        "Ablation: RidgeCV with 3 Folds": {
            "validation_strategy": "stratified",
            "use_std_features": True,
            "ridge_cv_folds": 3
        }
    }

    # Run experiments
    for name, config in ablation_configs.items():
        np.random.seed(SEED)  # Reset seed for each run for fair comparison

        # 1. Validation Split
        if config["validation_strategy"] == "stratified":
            street_median_target = df_original.groupby('Street Name')['violation_count'].transform('median')
            num_bins = min(10, len(df_original['Street Name'].unique()) // 2)
            if num_bins < 2: num_bins = 2
            target_bins = pd.qcut(street_median_target, q=num_bins, labels=False, duplicates='drop')
            
            n_splits = 5
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            try:
                train_idx, val_idx = next(sgkf.split(X=df_original, y=target_bins, groups=df_original['Street Name']))
            except ValueError:
                gss = GroupShuffleSplit(n_splits=1, test_size=1/n_splits, random_state=SEED)
                train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))

        else:  # group_shuffle
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))

        train_df = df_original.iloc[train_idx].reset_index(drop=True)
        val_df = df_original.iloc[val_idx].reset_index(drop=True)

        # 2. Feature Engineering
        train_featured, train_stats = feature_engineer(train_df)
        val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

        # 3. Define Features
        categorical_features = ['violation_description', 'boroname']
        numerical_features = [
            'street_mean', 'street_sum', 'street_key_count',
            'violation_mean', 'violation_sum',
            'boro_mean', 'boro_sum'
        ]
        if config["use_std_features"]:
            numerical_features.extend(['street_std', 'violation_std', 'boro_std'])

        all_features = numerical_features + categorical_features
        X_train, y_train = train_featured[all_features], train_featured['violation_count']
        X_val, y_val = val_featured[all_features], val_featured['violation_count']
        
        # 4. Model Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])

        ridge_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=config["ridge_cv_folds"]))
        ])
        
        # 5. Training & Validation
        ridge_pipeline.fit(X_train, y_train)
        val_predictions = ridge_pipeline.predict(X_val)
        val_predictions[val_predictions < 0] = 0
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        results[name] = rmse

    # 6. Conclusion
    baseline_name = "Baseline (Stratified Split, All Features, 5 CV Folds)"
    baseline_rmse = results.get(baseline_name)
    
    print("\n--- Ablation Study Summary ---")
    print(f"Validation RMSE for '{baseline_name}': {baseline_rmse:.4f}")

    performance_change = {}
    for name, rmse in results.items():
        if name != baseline_name:
            change = rmse - baseline_rmse
            performance_change[name] = change
            print(f"Validation RMSE for '{name}': {rmse:.4f} (Change from Baseline: {change:+.4f})")
    
    # Find the most impactful component based on the absolute change in RMSE
    most_impactful_component = max(performance_change, key=lambda k: abs(performance_change[k]))
    
    print(f"\nConclusion: The component '{most_impactful_component}' contributes the most to the overall performance.")
