
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
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model. This function is kept as in the original script.
    """
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

def run_experiment(name, model_pipeline, X_train, y_train, X_val, y_val):
    """Trains and evaluates a given model pipeline, printing and returning the RMSE."""
    print(f"--- Running Experiment: {name} ---")
    model_pipeline.fit(X_train, y_train)
    val_predictions = model_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE for '{name}': {rmse:.4f}\n")
    return rmse

def perform_ablation_study():
    """
    Performs an ablation study on the model choice and target transformation.
    """
    print("Starting ablation study...")
    print("Focusing on: Model Choice (LGBM vs Ridge) and Target Transformation.\n")
    
    # --- 1. Load and Prepare Data (Common for all experiments) ---
    train_path = './input/violations_per_street_2022.csv'
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Please check the path.")
        return

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # --- 2. Define and Run Experiments ---
    results = {}

    # Experiment 1: Baseline (LGBM with Target Transformation)
    lgbm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(random_state=SEED))
    ])
    baseline_model = TransformedTargetRegressor(
        regressor=lgbm_pipeline, func=np.log1p, inverse_func=np.expm1
    )
    baseline_rmse = run_experiment("Baseline (LGBM + Target Transform)", baseline_model, X_train, y_train, X_val, y_val)
    results["Baseline"] = baseline_rmse

    # Experiment 2: Ablation - No Target Transformation
    no_transform_model = lgbm_pipeline
    ablation1_rmse = run_experiment("Ablation (LGBM without Target Transform)", no_transform_model, X_train, y_train, X_val, y_val)
    results["No Target Transform"] = ablation1_rmse
    
    # Experiment 3: Ablation - Use Ridge Model instead of LGBM (with target transform)
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])
    ridge_model_transformed = TransformedTargetRegressor(
        regressor=ridge_pipeline, func=np.log1p, inverse_func=np.expm1
    )
    ablation2_rmse = run_experiment("Ablation (RidgeCV + Target Transform)", ridge_model_transformed, X_train, y_train, X_val, y_val)
    results["RidgeCV Model"] = ablation2_rmse
    
    # --- 3. Conclusion ---
    print("--- Ablation Study Summary ---")
    print(f"Baseline (LGBM + Target Transform) RMSE: {results['Baseline']:.4f}")
    print(f"Ablation (LGBM without Target Transform) RMSE: {results['No Target Transform']:.4f}")
    print(f"Ablation (RidgeCV + Target Transform) RMSE: {results['RidgeCV Model']:.4f}")
    
    # Calculate performance drop (higher drop means more important component)
    impact_of_target_transform = results["No Target Transform"] - results["Baseline"]
    impact_of_model_choice = results["RidgeCV Model"] - results["Baseline"]
    
    impacts = {
        "Target Transformation (np.log1p)": impact_of_target_transform,
        "The choice of model (LGBMRegressor over RidgeCV)": impact_of_model_choice
    }
    
    # Find the component with the largest positive impact (most detrimental to remove/replace)
    if max(impacts.values()) <= 0:
        conclusion = "Neither of the tested components showed a clear positive contribution to performance."
    else:
        most_impactful_component = max(impacts, key=impacts.get)
        conclusion = f"{most_impactful_component} contributes the most to the overall performance."

    print(f"\n{conclusion}")

if __name__ == '__main__':
    perform_ablation_study()
