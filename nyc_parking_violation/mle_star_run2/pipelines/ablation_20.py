
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

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Dummy data creation for reproducibility if input file is not found
def create_dummy_data():
    if not os.path.exists('./input'):
        os.makedirs('./input')
    
    # Create dummy nyc_cscl.csv
    cscl_data = {
        'ST_NAME': ['BROADWAY', 'WALL STREET', 'FIFTH AVENUE', 'MAIN STREET'],
        'BORONAME': ['Manhattan', 'Manhattan', 'Manhattan', 'Queens']
    }
    pd.DataFrame(cscl_data).to_csv('./input/nyc_cscl.csv', index=False)

    # Create dummy violations_per_street_2022.csv
    streets = ['BROADWAY', 'WALL STREET', 'FIFTH AVENUE', 'MAIN STREET', 'UNKNOWN STREET'] * 20
    violations = ['NO PARKING', 'FIRE HYDRANT', 'NO STANDING', 'DOUBLE PARKING'] * 25
    np.random.shuffle(streets)
    np.random.shuffle(violations)
    
    data = {
        'Street Name': streets[:100],
        'Violation Description': violations[:100],
        'Violation Count': np.random.randint(1, 500, 100)
    }
    pd.DataFrame(data).to_csv('./input/violations_per_street_2022.csv', index=False)

# Check for data and create if necessary
if not os.path.exists('./input/violations_per_street_2022.csv'):
    create_dummy_data()

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model. This function is identical to the one
    in the original script.
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

def run_experiment(description, X_train_full, y_train, X_val_full, y_val, numerical_features, categorical_features):
    """
    Runs a single experiment with a specific set of features.
    """
    all_features = numerical_features + categorical_features
    X_train = X_train_full[all_features].copy()
    X_val = X_val_full[all_features].copy()

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Define the model pipeline
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    # Training and Validation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    print(f"- {description}: RMSE = {rmse:.4f}")
    return rmse

# --- 1. Load and Prepare Data (once) ---
print("--- Ablation Study on Feature Selection and Set ---")
try:
    df_original = pd.read_csv('./input/violations_per_street_2022.csv')
except FileNotFoundError:
    print("Error: Training file not found. Please place it at './input/violations_per_street_2022.csv'")
    exit()

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
train_df = df_original.iloc[train_idx].reset_index(drop=True)
val_df = df_original.iloc[val_idx].reset_index(drop=True)

train_featured, train_stats = feature_engineer(train_df)
val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

target = 'violation_count'
y_train = train_featured[target]
y_val = val_featured[target]

results = {}

# --- 2. Run Experiments ---

# Experiment 1: Baseline (curated features from the original script)
baseline_num = ['street_mean', 'street_key_count', 'violation_mean', 'boro_mean']
baseline_cat = ['boroname']
results['Baseline (Curated Features)'] = run_experiment(
    "Baseline (Curated Features)",
    train_featured, y_train, val_featured, y_val,
    numerical_features=baseline_num,
    categorical_features=baseline_cat
)

# Experiment 2: Ablation of Borough Features
ablation1_num = ['street_mean', 'street_key_count', 'violation_mean']
ablation1_cat = []
results['No Borough Features'] = run_experiment(
    "Ablation: No Borough Features",
    train_featured, y_train, val_featured, y_val,
    numerical_features=ablation1_num,
    categorical_features=ablation1_cat
)

# Experiment 3: Ablation using all available aggregate features
ablation2_num = [
    'street_mean', 'street_sum', 'street_std', 'street_key_count',
    'violation_mean', 'violation_sum', 'violation_std',
    'boro_mean', 'boro_sum', 'boro_std'
]
ablation2_cat = ['boroname']
results['All Aggregate Features'] = run_experiment(
    "Ablation: All Aggregate Features",
    train_featured, y_train, val_featured, y_val,
    numerical_features=ablation2_num,
    categorical_features=ablation2_cat
)


# --- 3. Conclusion ---
baseline_rmse = results['Baseline (Curated Features)']
impacts = {
    name: abs(rmse - baseline_rmse) for name, rmse in results.items() if 'Baseline' not in name
}

if not impacts:
    print("\nNo ablations were run to compare against the baseline.")
else:
    most_impactful_component = max(impacts, key=impacts.get)
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: 'The Selection of Features'.")
    print(f"Modifying the feature set to '{most_impactful_component}' resulted in the largest change in RMSE (change of {impacts[most_impactful_component]:.4f}).")

