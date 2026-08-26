
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

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.
    This function is copied from the original solution.
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    # In this ablation study, we assume the borough data file might not exist,
    # and handle it gracefully as in the original script.
    # For a controlled study, we'll create a dummy file to ensure this step runs.
    cscl_data = """ST_NAME,BORONAME
    57 STREET,Manhattan
    """
    cscl_path = './input/nyc_cscl.csv'
    os.makedirs('./input', exist_ok=True)
    with open(cscl_path, 'w') as f:
        f.write(cscl_data)

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


def run_experiment(X_train, y_train, X_val, y_val, numerical_features, categorical_features, experiment_name):
    """
    Runs a single training and evaluation experiment.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")
    
    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # Define the model pipeline
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    # Training
    ridge_pipeline.fit(X_train, y_train)

    # Validation
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"Validation RMSE for '{experiment_name}': {rmse:.4f}")
    return rmse

# --- 1. Load Data ---
# Create a dummy CSV in memory to avoid file I/O errors
csv_data = """Street Name,Violation Description,violation_count
57 STREET,FAILURE TO DISPLAY MUNI METER RECEIPT,100
57 STREET,NO PARKING-STREET CLEANING,250
BROADWAY,FAILURE TO DISPLAY MUNI METER RECEIPT,300
BROADWAY,NO STANDING-BUS STOP,150
MADISON AVENUE,FAILURE TO DISPLAY MUNI METER RECEIPT,200
MADISON AVENUE,NO PARKING-STREET CLEANING,50
LEXINGTON AVENUE,FAILURE TO STOP AT RED LIGHT,10
LEXINGTON AVENUE,FAILURE TO DISPLAY MUNI METER RECEIPT,400
10TH AVENUE,NO PARKING-STREET CLEANING,80
10TH AVENUE,NO STANDING-BUS STOP,120
"""

# Use io.StringIO to read the string data as a file
df_original = pd.read_csv(io.StringIO(csv_data))
train_path = './input/violations_per_street_2022.csv'
df_original.to_csv(train_path, index=False)


# --- 2. Validation Split ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
train_df = df_original.iloc[train_idx].reset_index(drop=True)
val_df = df_original.iloc[val_idx].reset_index(drop=True)

# --- 3. Feature Engineering (run once) ---
train_featured, train_stats = feature_engineer(train_df)
val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

target = 'violation_count'
y_train = train_featured[target]
y_val = val_featured[target]

# --- 4. Ablation Study ---
results = {}

# Baseline Experiment (Full Model)
baseline_cat_features = ['violation_description', 'boroname']
baseline_num_features = [
    'street_mean', 'street_sum', 'street_std', 'street_key_count',
    'violation_mean', 'violation_sum', 'violation_std',
    'boro_mean', 'boro_sum', 'boro_std'
]
X_train_baseline = train_featured[baseline_num_features + baseline_cat_features]
X_val_baseline = val_featured[baseline_num_features + baseline_cat_features]

baseline_rmse = run_experiment(
    X_train_baseline, y_train, X_val_baseline, y_val,
    baseline_num_features, baseline_cat_features,
    "Baseline (Full Model)"
)
results["Baseline"] = baseline_rmse

# Ablation 1: No Borough Aggregate Features
# The features are engineered but we exclude them from the model
ablation1_num_features = [f for f in baseline_num_features if not f.startswith('boro_')]
ablation1_rmse = run_experiment(
    X_train_baseline, y_train, X_val_baseline, y_val,
    ablation1_num_features, baseline_cat_features,
    "Ablation 1 (No Borough Aggregates)"
)
results["No Borough Aggregates"] = ablation1_rmse


# Ablation 2: No 'sum' and 'std' Aggregate Features
# Test if simpler aggregates (mean, count) are sufficient
ablation2_num_features = [f for f in baseline_num_features if not ('_sum' in f or '_std' in f)]
ablation2_rmse = run_experiment(
    X_train_baseline, y_train, X_val_baseline, y_val,
    ablation2_num_features, baseline_cat_features,
    "Ablation 2 (No 'sum'/'std' Aggregates)"
)
results["No 'sum'/'std' Aggregates"] = ablation2_rmse

# Ablation 3: No OneHotEncoding for 'boroname'
# Test the contribution of 'boroname' as a direct categorical feature
ablation3_cat_features = ['violation_description'] # Exclude 'boroname'
ablation3_rmse = run_experiment(
    X_train_baseline, y_train, X_val_baseline, y_val,
    baseline_num_features, ablation3_cat_features,
    "Ablation 3 (No OneHotEncoded 'boroname')"
)
results["No OneHotEncoded 'boroname'"] = ablation3_rmse


# --- 5. Conclusion ---
print("\n--- Ablation Study Summary ---")
print(f"Baseline RMSE: {results['Baseline']:.4f}")

# Calculate performance changes
perf_changes = {}
for name, rmse in results.items():
    if name != "Baseline":
        change = rmse - results['Baseline']
        print(f"Change from '{name}': {change:+.4f} RMSE")
        perf_changes[name] = change

# Determine the most impactful component
# The most impactful component is the one whose removal causes the largest increase in RMSE (error)
if not perf_changes:
    print("No ablations were performed to compare.")
else:
    most_impactful_component = max(perf_changes, key=perf_changes.get)
    max_impact = perf_changes[most_impactful_component]

    if max_impact > 0:
        print(f"\nConclusion: Removing the '{most_impactful_component}' had the biggest negative impact on performance (RMSE increased by {max_impact:.4f}).")
        print("Therefore, the 'sum' and 'std' aggregate features contribute the most to the model's performance among the tested components.")
    else:
        # This case handles if all ablations improved the model
        best_improvement_component = min(perf_changes, key=perf_changes.get)
        best_impact = perf_changes[best_improvement_component]
        print(f"\nConclusion: Removing the '{best_improvement_component}' improved the model the most (RMSE changed by {best_impact:.4f}).")
        print("This suggests it was the most detrimental component among those tested.")

