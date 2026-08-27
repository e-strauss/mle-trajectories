
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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

def feature_engineer(df, train_stats=None, use_borough_data=True):
    """
    Engineers features for the model, with an option to ablate borough data.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns if c in ['Street Name', 'Violation Description', 'violation_count']]

    if use_borough_data:
        cscl_path = './input/nyc_cscl.csv'
        if os.path.exists(cscl_path):
            try:
                cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
                cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
                cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
                df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
                df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
                df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
                
                # FIX: Rename the merged 'BORONAME' column to lowercase 'boroname'
                # to be consistent with other code paths and expectations.
                df_engineered.rename(columns={'BORONAME': 'boroname'}, inplace=True)

            except Exception:
                df_engineered['boroname'] = 'Unknown'
        else:
            df_engineered['boroname'] = 'Unknown'
    else:
        df_engineered['boroname'] = 'Unknown' # Ablation case

    # This check is now safe because the column is consistently named 'boroname'.
    if 'boroname' not in df_engineered.columns:
        df_engineered['boroname'] = 'Unknown' # Defensive: ensure column exists
    df_engineered['boroname'].fillna('Unknown', inplace=True)
    
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

def run_experiment(description, X_train, y_train, X_val, y_val, pipeline):
    """Fits a pipeline and evaluates it, printing the result."""
    print(f"--- Running: {description} ---")
    
    # Deepcopy to avoid fitting on an already fitted model in subsequent runs
    model_pipeline = copy.deepcopy(pipeline)
    
    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate on the validation set
    val_predictions = model_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Validation RMSE for '{description}': {rmse:.4f}\n")
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    print(f"Loading training data from {train_path}...")
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Exiting.")
        return

    print("Splitting data into train and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Feature Engineering (for most experiments) ---
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

    X_train_base = train_featured[all_features].copy()
    y_train = train_featured[target]
    X_val_base = val_featured[all_features].copy()
    y_val = val_featured[target]
    
    for col in categorical_features:
        X_train_base[col] = X_train_base[col].astype('category')
        X_val_base[col] = X_val_base[col].astype('category')
    
    ablation_results = {}

    # --- BASELINE PIPELINE ---
    preprocessor_base = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features)], remainder='passthrough')
    hgb_regressor = HistGradientBoostingRegressor(
        categorical_features=[i for i, col in enumerate(X_train_base.columns) if col in categorical_features],
        random_state=SEED)
    transformed_hgb_regressor = TransformedTargetRegressor(regressor=hgb_regressor, func=np.log1p, inverse_func=np.expm1)
    baseline_pipeline = Pipeline(steps=[('preprocessor', preprocessor_base), ('regressor', transformed_hgb_regressor)])
    
    baseline_rmse = run_experiment("Baseline (Full Model)", X_train_base, y_train, X_val_base, y_val, baseline_pipeline)
    ablation_results['Baseline'] = baseline_rmse

    # --- ABLATION 1: NO TARGET TRANSFORMATION ---
    pipeline_no_transform = Pipeline(steps=[('preprocessor', preprocessor_base), ('regressor', hgb_regressor)])
    rmse_no_transform = run_experiment("No Target Transformation", X_train_base, y_train, X_val_base, y_val, pipeline_no_transform)
    ablation_results['No Target Transformation'] = rmse_no_transform

    # --- ABLATION 2: NO FEATURE SCALING ---
    preprocessor_no_scale = ColumnTransformer(transformers=[], remainder='passthrough') # Pass all features untouched
    pipeline_no_scale = Pipeline(steps=[('preprocessor', preprocessor_no_scale), ('regressor', transformed_hgb_regressor)])
    rmse_no_scale = run_experiment("No Feature Scaling", X_train_base, y_train, X_val_base, y_val, pipeline_no_scale)
    ablation_results['No Feature Scaling'] = rmse_no_scale

    # --- ABLATION 3: NO BOROUGH DATA ---
    train_no_boro, stats_no_boro = feature_engineer(train_df, use_borough_data=False)
    val_no_boro, _ = feature_engineer(val_df, train_stats=stats_no_boro, use_borough_data=False)
    X_train_no_boro = train_no_boro[all_features].copy()
    X_val_no_boro = val_no_boro[all_features].copy()
    for col in categorical_features:
        X_train_no_boro[col] = X_train_no_boro[col].astype('category')
        X_val_no_boro[col] = X_val_no_boro[col].astype('category')
        
    rmse_no_boro = run_experiment("No Borough Data", X_train_no_boro, y_train, X_val_no_boro, y_val, baseline_pipeline)
    ablation_results['No Borough Data'] = rmse_no_boro

    # --- Conclusion ---
    print("\n--- Ablation Study Summary ---")
    baseline_score = ablation_results['Baseline']
    performance_changes = {}
    for name, score in ablation_results.items():
        if name != 'Baseline':
            change = score - baseline_score
            performance_changes[name] = change
            print(f"'{name}' resulted in an RMSE change of: {change:+.4f}")
    
    if not performance_changes:
        print("No ablations were run to compare against the baseline.")
    else:
        most_impactful_component = max(performance_changes, key=lambda k: abs(performance_changes[k]))
        print(f"\nConclusion: '{most_impactful_component}' contributes the most to the model's performance.")

    final_validation_score = baseline_rmse
    print(f'Final Validation Performance: {final_validation_score}')


if __name__ == '__main__':
    main()
