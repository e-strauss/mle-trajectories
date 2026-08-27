
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
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
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    # In a real scenario, this would load a file. For this self-contained script,
    # we simulate its presence or absence. For simplicity, we assume it's always missing
    # and create the placeholder, as its impact has been studied before.
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

def run_experiment(df_original, hgb_params, use_categorical_mask, clip_negatives):
    """
    Runs a single training and evaluation experiment with a given configuration.
    """
    # --- 2. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

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

    # --- 4. Model Pipeline ---
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ])

    categorical_features_mask = [False] * len(numerical_features) + [True] * len(categorical_features)
    
    # Conditionally use the native categorical feature handling
    if use_categorical_mask:
        regressor = HistGradientBoostingRegressor(categorical_features=categorical_features_mask, random_state=SEED, **hgb_params)
    else:
        # Treat all features as numerical
        regressor = HistGradientBoostingRegressor(random_state=SEED, **hgb_params)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # --- 5. Training ---
    pipeline.fit(X_train, y_train)

    # --- 6. Validation ---
    val_predictions = pipeline.predict(X_val)

    # Conditionally clip negative predictions
    if clip_negatives:
        val_predictions[val_predictions < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse


# --- Main Ablation Study Script ---

# Create dummy data to run the script without external files
dummy_data = """
Street Name,Violation Description,Violation Count
MAIN ST,NO PARKING-STREET CLEANING,350
MAIN ST,FIRE HYDRANT,120
OAK AVE,NO PARKING-STREET CLEANING,280
PINE LN,FAILURE TO DISPLAY METER RECEIPT,50
MAPLE DR,NO STANDING-DAY/TIME LIMITS,75
MAPLE DR,FIRE HYDRANT,30
ELM ST,NO PARKING-STREET CLEANING,410
CEDAR RD,FAILURE TO STOP AT RED LIGHT,5
WASHINGTON BLVD,NO PARKING-STREET CLEANING,380
WASHINGTON BLVD,DOUBLE PARKING,60
LINCOLN WAY,FAILURE TO DISPLAY METER RECEIPT,45
ADAMS AVE,NO STANDING-DAY/TIME LIMITS,90
JEFFERSON CT,NO PARKING-STREET CLEANING,320
MADISON AVE,FIRE HYDRANT,110
MONROE ST,DOUBLE PARKING,70
BROADWAY,NO PARKING-STREET CLEANING,500
BROADWAY,FIRE HYDRANT,150
5TH AVE,FAILURE TO DISPLAY METER RECEIPT,80
PARK AVE,NO STANDING-DAY/TIME LIMITS,120
WALL ST,DOUBLE PARKING,95
"""
df_original = pd.read_csv(io.StringIO(dummy_data))

# To make the data larger and more realistic for grouping
df_original = pd.concat([df_original] * 100, ignore_index=True)
df_original['Violation Count'] += np.random.randint(-10, 10, size=len(df_original))

# --- Define and Run Experiments ---
results = {}

# Experiment 1: Baseline
# Full model with default HGBoost parameters, native categorical handling, and negative clipping.
results['Baseline'] = run_experiment(
    df_original=df_original,
    hgb_params={},
    use_categorical_mask=True,
    clip_negatives=True
)

# Experiment 2: Simpler Model (Ablate Hyperparameters)
# Test impact of a less complex model by reducing the number of boosting iterations.
results['Ablation: Simpler HGB (max_iter=50)'] = run_experiment(
    df_original=df_original,
    hgb_params={'max_iter': 50},
    use_categorical_mask=True,
    clip_negatives=True
)

# Experiment 3: No Native Categorical Handling
# Test the value of the model's built-in categorical feature support.
results['Ablation: No Native Categorical Handling'] = run_experiment(
    df_original=df_original,
    hgb_params={},
    use_categorical_mask=False, # The change is here
    clip_negatives=True
)

# Experiment 4: No Negative Clipping
# Test the impact of the post-processing step.
results['Ablation: No Negative Clipping'] = run_experiment(
    df_original=df_original,
    hgb_params={},
    use_categorical_mask=True,
    clip_negatives=False # The change is here
)

# --- Print Results and Conclusion ---
print("--- Ablation Study Results ---")
baseline_rmse = results['Baseline']
print(f"Baseline RMSE: {baseline_rmse:.4f}\n")

# Calculate performance changes
changes = {}
for name, rmse in results.items():
    if name != 'Baseline':
        change = rmse - baseline_rmse
        changes[name] = change
        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f} (Change from Baseline: {change:+.4f})")

# Determine the most impactful component
if not changes:
    print("\nNo ablations were run to compare against the baseline.")
else:
    most_impactful_name = max(changes, key=lambda k: abs(changes[k]))
    # Simplify the name for the conclusion
    conclusion_name = most_impactful_name.replace('Ablation: ', '')
    print(f"\nConclusion: '{conclusion_name}' contributes the most to the model's performance, as its removal/change caused the largest absolute change in RMSE.")

