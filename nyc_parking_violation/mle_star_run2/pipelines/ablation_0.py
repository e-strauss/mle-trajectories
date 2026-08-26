
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
import shutil

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def setup_dummy_data():
    """Creates dummy data files for a self-contained run."""
    print("Setting up dummy data files...")
    if os.path.exists('./input'):
        shutil.rmtree('./input')
    os.makedirs('./input', exist_ok=True)

    # Create dummy main violations data
    data = {
        'Street Name': ['MAIN ST', 'MAIN ST', 'FIRST AVE', 'FIRST AVE', 'OAK ST', 'MAPLE AVE'] * 20,
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT'] * 20,
        'violation_count': [150, 25, 120, 50, 80, 30, 160, 22, 115, 55, 85, 28] * 10
    }
    df_violations = pd.DataFrame(data)
    df_violations['violation_count'] = df_violations['violation_count'] + np.random.randint(-10, 10, size=len(df_violations))
    df_violations.to_csv('./input/violations_per_street_2022.csv', index=False)

    # Create dummy borough data
    borough_data = {
        'ST_NAME': ['MAIN ST', 'FIRST AVE', 'OAK ST', 'ELM ST'],
        'BORONAME': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens']
    }
    df_cscl = pd.DataFrame(borough_data)
    df_cscl.to_csv('./input/nyc_cscl.csv', index=False)
    print("Dummy data created.")

def feature_engineer(df, use_borough_data=True, train_stats=None):
    """
    Engineers features for the model, with an option to disable borough augmentation.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data (Ablation Point) ---
    if use_borough_data:
        cscl_path = './input/nyc_cscl.csv'
        if os.path.exists(cscl_path):
            cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
            cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
            cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
            df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
            df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
            df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    
    # Always have the column, fill with 'Unknown' if data wasn't used or merge failed
    if 'boroname' not in df_engineered.columns:
        df_engineered['boroname'] = 'Unknown'
    df_engineered['boroname'].fillna('Unknown', inplace=True)

    # --- Create Aggregate Features ---
    if train_stats is None:
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        stats = {'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg}
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(train_df, val_df, use_borough_data, use_street_aggregates):
    """Runs a single experiment with a specific configuration."""
    
    # --- Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df, use_borough_data=use_borough_data)
    val_featured, _ = feature_engineer(val_df, use_borough_data=use_borough_data, train_stats=train_stats)
    
    # --- Define Features based on configuration (Ablation Points) ---
    numerical_features = [
        'violation_mean', 'violation_sum', 'violation_std',
    ]
    categorical_features = ['violation_description']

    if use_street_aggregates:
        numerical_features.extend(['street_mean', 'street_sum', 'street_std', 'street_key_count'])

    if use_borough_data:
        numerical_features.extend(['boro_mean', 'boro_sum', 'boro_std'])
        categorical_features.append('boroname')

    target = 'violation_count'
    X_train = train_featured[numerical_features + categorical_features]
    y_train = train_featured[target]
    X_val = val_featured[numerical_features + categorical_features]
    y_val = val_featured[target]
    
    # --- Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=3)) # Reduced cv for speed
    ])
    
    # --- Training & Evaluation ---
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def main():
    """Main function to run the ablation study."""
    setup_dummy_data()
    
    # --- 1. Load and Split Data ---
    df_original = pd.read_csv('./input/violations_per_street_2022.csv')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    # --- Experiment 1: Baseline (Full Model) ---
    print("\nRunning baseline experiment (all features)...")
    results['Baseline'] = run_experiment(
        train_df, val_df, 
        use_borough_data=True, 
        use_street_aggregates=True
    )
    print(f"  -> Validation RMSE: {results['Baseline']:.4f}")

    # --- Experiment 2: Ablation of Borough Augmentation ---
    print("\nRunning ablation: No Borough Data Augmentation...")
    results['No Borough Data'] = run_experiment(
        train_df, val_df, 
        use_borough_data=False, 
        use_street_aggregates=True
    )
    print(f"  -> Validation RMSE: {results['No Borough Data']:.4f}")

    # --- Experiment 3: Ablation of Street-level Aggregates ---
    print("\nRunning ablation: No Street-level Aggregate Features...")
    results['No Street Aggregates'] = run_experiment(
        train_df, val_df, 
        use_borough_data=True, 
        use_street_aggregates=False
    )
    print(f"  -> Validation RMSE: {results['No Street Aggregates']:.4f}")

    # --- 4. Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    baseline_rmse = results['Baseline']
    perf_drop_borough = results['No Borough Data'] - baseline_rmse
    perf_drop_street = results['No Street Aggregates'] - baseline_rmse

    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Performance drop without Borough Data: {perf_drop_borough:+.4f} (New RMSE: {results['No Borough Data']:.4f})")
    print(f"Performance drop without Street Aggregates: {perf_drop_street:+.4f} (New RMSE: {results['No Street Aggregates']:.4f})")

    if perf_drop_street > perf_drop_borough:
        print("\nConclusion: Street-level aggregate features contribute the most to the model's performance.")
    elif perf_drop_borough > perf_drop_street:
        print("\nConclusion: Borough data augmentation contributes the most to the model's performance.")
    else:
        print("\nConclusion: Both features seem to have a similar impact on performance.")
        
    # --- Cleanup ---
    if os.path.exists('./input'):
        shutil.rmtree('./input')
    print("\nCleaned up dummy data files.")

if __name__ == '__main__':
    main()
