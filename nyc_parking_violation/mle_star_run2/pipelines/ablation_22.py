
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
from io import StringIO

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

# --- Mock Data Setup ---
# Create a mock environment to ensure the script is self-contained and runnable.
if not os.path.exists('./input'):
    os.makedirs('./input')

# Mock violations data
violations_data = """
Street Name,Violation Description,violation_count
5TH AVE,NO PARKING-STREET CLEANING,150
5TH AVE,FAIL TO DSPLY MUNI METER RECPT,200
BROADWAY,NO PARKING-STREET CLEANING,300
BROADWAY,PHTO SCHOOL ZN SPEED VIOLATION,450
MADISON AVE,NO PARKING-STREET CLEANING,120
MADISON AVE,FAIL TO DSPLY MUNI METER RECPT,180
LEXINGTON AVE,BUS LANE VIOLATION,500
LEXINGTON AVE,FAIL TO DSPLY MUNI METER RECPT,175
PARK AVE,PHTO SCHOOL ZN SPEED VIOLATION,400
PARK AVE,NO PARKING-STREET CLEANING,110
WALL ST,NO PARKING-EXC AS AUTH,250
WALL ST,NO STANDING-EXC AUTH VEHICLE,350
"""
with open('./input/violations_per_street_2022.csv', 'w') as f:
    f.write(violations_data)

# Mock borough data
cscl_data = """
ST_NAME,BORONAME
5TH AVE,MANHATTAN
BROADWAY,MANHATTAN
MADISON AVE,MANHATTAN
LEXINGTON AVE,MANHATTAN
PARK AVE,MANHATTAN
WALL ST,MANHATTAN
FLATBUSH AVE,BROOKLYN
"""
with open('./input/nyc_cscl.csv', 'w') as f:
    f.write(cscl_data)
# --- End Mock Data Setup ---


def feature_engineer(df, train_stats=None, use_street_aggs=True, use_violation_aggs=True, use_boro_aggs=True):
    """
    Engineers features for the model, with flags to control ablations.
    """
    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    cscl = pd.read_csv(cscl_path)
    cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
    cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()

    df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
    df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
    
    # FIX: The merge adds 'BORONAME' (uppercase) but downstream code expects 'boroname' (lowercase).
    # This is because the initial columns of df_engineered were lowercased *before* the merge.
    if 'BORONAME' in df_engineered.columns:
        df_engineered.rename(columns={'BORONAME': 'boroname'}, inplace=True)
    
    # Ensure 'boroname' column exists before filling NaNs, making the pipeline robust.
    if 'boroname' not in df_engineered.columns:
        df_engineered['boroname'] = 'Unknown'
    
    df_engineered['boroname'].fillna('Unknown', inplace=True)
    
    # Drop temporary and redundant columns, ignoring errors if they don't exist.
    df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True, errors='ignore')

    has_target = 'violation_count' in df_engineered.columns

    # --- Create Aggregate Features ---
    if train_stats is None:
        if not has_target:
            raise ValueError("`violation_count` column is required to build training statistics.")
        
        stats = {}
        aggregations = ['mean', 'sum', 'std', 'count']

        if use_street_aggs:
            street_agg = df_engineered.groupby('street_name')['violation_count'].agg(aggregations)
            street_agg.columns = [f'street_{agg}' for agg in aggregations]
            stats['street_agg'] = street_agg

        if use_violation_aggs:
            violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(aggregations)
            violation_agg.columns = [f'violation_{agg}' for agg in aggregations]
            violation_agg.rename(columns={'violation_count': 'violation_group_count'}, inplace=True)
            stats['violation_agg'] = violation_agg

        if use_boro_aggs:
            boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(aggregations)
            boro_agg.columns = [f'boro_{agg}' for agg in aggregations]
            stats['boro_agg'] = boro_agg
    else:
        stats = train_stats

    # Merge aggregate features
    if use_street_aggs and 'street_agg' in stats:
        df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    if use_violation_aggs and 'violation_agg' in stats:
        df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    if use_boro_aggs and 'boro_agg' in stats:
        df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    df_engineered.fillna(0, inplace=True)
    return df_engineered, stats

def run_experiment(label, df_original, use_street_aggs=True, use_violation_aggs=True, use_boro_aggs=True):
    """
    Runs a single training and validation experiment with specific settings.
    """
    # 1. Validation Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    # Handle cases where there are not enough groups for a split
    if df_original['Street Name'].nunique() < 2:
         print(f"{label}: Skipped - Not enough unique groups to split.")
         return np.nan

    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # 2. Feature Engineering
    train_featured, train_stats = feature_engineer(
        train_df, use_street_aggs=use_street_aggs, use_violation_aggs=use_violation_aggs, use_boro_aggs=use_boro_aggs
    )
    val_featured, _ = feature_engineer(
        val_df, train_stats=train_stats, use_street_aggs=use_street_aggs, use_violation_aggs=use_violation_aggs, use_boro_aggs=use_boro_aggs
    )

    # 3. Define Features for the model
    categorical_features = ['violation_description', 'boroname']
    numerical_features = []
    if use_street_aggs:
        numerical_features.extend(['street_mean', 'street_sum', 'street_std', 'street_count'])
    if use_violation_aggs:
        numerical_features.extend(['violation_mean', 'violation_sum', 'violation_std', 'violation_group_count'])
    if use_boro_aggs:
        numerical_features.extend(['boro_mean', 'boro_sum', 'boro_std', 'boro_count'])
        
    all_features = numerical_features + categorical_features
    
    # Ensure all defined features exist in the dataframe before proceeding
    for col in all_features:
        if col not in train_featured.columns:
            # Create dummy columns if an entire group was disabled
            train_featured[col] = 0
            val_featured[col] = 0

    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # 4. Model Pipeline
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

    # 5. Training and Validation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    print(f"{label}: Validation RMSE = {rmse:.4f}")
    return rmse

# --- Ablation Study ---
df_main = pd.read_csv('./input/violations_per_street_2022.csv')

results = {}

# Baseline
results['Baseline (All Aggregates)'] = run_experiment(
    'Baseline (All Aggregates)', df_main, 
    use_street_aggs=True, use_violation_aggs=True, use_boro_aggs=True
)

# Ablation 1: No Street-level aggregates
results['Ablation: No Street Aggregates'] = run_experiment(
    'Ablation: No Street Aggregates', df_main, 
    use_street_aggs=False, use_violation_aggs=True, use_boro_aggs=True
)

# Ablation 2: No Violation-level aggregates
results['Ablation: No Violation Aggregates'] = run_experiment(
    'Ablation: No Violation Aggregates', df_main, 
    use_street_aggs=True, use_violation_aggs=False, use_boro_aggs=True
)

# Ablation 3: No Borough-level aggregates
results['Ablation: No Borough Aggregates'] = run_experiment(
    'Ablation: No Borough Aggregates', df_main, 
    use_street_aggs=True, use_violation_aggs=True, use_boro_aggs=False
)

# --- Conclusion ---
print("\n--- Ablation Study Conclusion ---")
baseline_rmse = results['Baseline (All Aggregates)']
if not np.isnan(baseline_rmse):
    print(f"Final Validation Performance: {baseline_rmse}")
    impacts = {}
    for name, rmse in results.items():
        if name != 'Baseline (All Aggregates)' and not np.isnan(rmse):
            # Calculate the absolute change in RMSE from the baseline
            impact = abs(rmse - baseline_rmse)
            impacts[name] = impact

    # Find the component with the highest impact
    if impacts:
        most_impactful_component = max(impacts, key=impacts.get)
        # Clean up the name for the final print statement
        component_name = most_impactful_component.replace('Ablation: No ', '')
        print(f"The component that contributes the most to the overall performance is: '{component_name}'")
        print(f"Its removal caused an RMSE change of {impacts[most_impactful_component]:.4f}.")
    else:
        print("Could not determine the most impactful component from the ablation study.")
else:
    print("Baseline run failed, could not determine performance.")

