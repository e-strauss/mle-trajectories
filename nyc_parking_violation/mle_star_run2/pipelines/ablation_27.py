
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
import io

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

# --- Data Setup ---
# Create dummy data files in memory to make the script self-contained
# This simulates the files being present on disk.

# Dummy violations data
violations_csv_data = """Street Name,Violation Description,violation_count
STREET A,DOUBLE PARKING,150
STREET A,FIRE HYDRANT,50
STREET B,NO STANDING,200
STREET C,DOUBLE PARKING,30
STREET D,FIRE HYDRANT,25
STREET E,NO STANDING,180
STREET A,NO STANDING,10
STREET B,FIRE HYDRANT,60
STREET C,NO STANDING,45
STREET F,DOUBLE PARKING,90
STREET G,FIRE HYDRANT,120
STREET H,NO STANDING,300
STREET A,DOUBLE PARKING,140
STREET B,NO STANDING,210
STREET D,DOUBLE PARKING,35
STREET E,FIRE HYDRANT,20
STREET F,NO STANDING,95
STREET G,DOUBLE PARKING,110
STREET H,FIRE HYDRANT,290
STREET I,NO STANDING,55
STREET J,DOUBLE PARKING,75
STREET K,FIRE HYDRANT,85
STREET L,NO STANDING,125
"""

# Dummy borough data
cscl_csv_data = """ST_NAME,BORONAME
STREET A,Manhattan
STREET B,Brooklyn
STREET C,Manhattan
STREET D,Brooklyn
STREET E,Queens
STREET F,Queens
STREET G,Bronx
STREET H,Staten Island
STREET I,Manhattan
STREET J,Brooklyn
STREET K,Queens
STREET L,Bronx
"""

# Create in-memory directories and files
if not os.path.exists('./input'):
    os.makedirs('./input')

with open('./input/violations_per_street_2022.csv', 'w') as f:
    f.write(violations_csv_data)

with open('./input/nyc_cscl.csv', 'w') as f:
    f.write(cscl_csv_data)


# --- Original Functions (from the problem description) ---

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.
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

def run_experiment(train_df, val_df, scaler_with_std=True, ohe_drop_policy=None, ridge_alphas=np.logspace(-2, 2, 5)):
    """
    Runs a single training and validation experiment with configurable components.
    """
    # --- 3. Feature Engineering ---
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

    # --- 4. Model Pipeline (with configurable components) ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_std=scaler_with_std), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=ohe_drop_policy), categorical_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=ridge_alphas, cv=5))
    ])

    # --- 5. Training & 6. Validation ---
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    # --- 1. Load Data ---
    train_path = './input/violations_per_street_2022.csv'
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Exiting.")
        return

    # --- 2. Validation Split (do this once for all experiments) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- Run Ablation Experiments ---
    results = {}
    print("--- Running Ablation Study ---")

    # Experiment 1: Baseline
    baseline_rmse = run_experiment(train_df.copy(), val_df.copy())
    results['Baseline'] = baseline_rmse

    # Experiment 2: Ablation of StandardScaler 'with_std'
    ablation_rmse_scaler = run_experiment(train_df.copy(), val_df.copy(), scaler_with_std=False)
    results['No Scaling (with_std=False)'] = ablation_rmse_scaler

    # Experiment 3: Ablation of OneHotEncoder 'drop'
    ablation_rmse_ohe = run_experiment(train_df.copy(), val_df.copy(), ohe_drop_policy='first')
    results['OHE drop=\'first\''] = ablation_rmse_ohe

    # Experiment 4: Ablation of RidgeCV alpha grid
    finer_alphas = np.logspace(-2, 2, 50)
    ablation_rmse_ridge = run_experiment(train_df.copy(), val_df.copy(), ridge_alphas=finer_alphas)
    results['Finer Alpha Grid'] = ablation_rmse_ridge

    # --- Summarize and Conclude ---
    print("\n--- Ablation Study Summary ---")
    baseline_score = results['Baseline']
    performance_changes = {}

    for name, score in results.items():
        change = score - baseline_score
        print(f"Experiment: {name:<28} | RMSE: {score:.4f} | Change from Baseline: {change:+.4f}")
        if name != 'Baseline':
            performance_changes[name] = abs(change)

    # Find the component with the largest impact
    if not performance_changes:
        most_impactful = "No ablations were run."
    else:
        most_impactful = max(performance_changes, key=performance_changes.get)

    print(f"\nThe component that contributes the most to the overall performance is: '{most_impactful}'")


if __name__ == '__main__':
    main()
