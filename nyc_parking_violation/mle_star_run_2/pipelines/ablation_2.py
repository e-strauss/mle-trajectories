
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.linear_model import Ridge, RidgeCV
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
    (Function is copied from the original solution for consistency)
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    # In-memory CSV to simulate file presence without needing the actual file
    cscl_data = """physicalid,ST_NAME,BORONAME
1,WALL STREET,MANHATTAN
2,BROADWAY,MANHATTAN
3,FLATBUSH AVENUE,BROOKLYN
"""
    try:
        cscl = pd.read_csv(io.StringIO(cscl_data))
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    except Exception:
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

def run_ablation_study():
    """
    Performs an ablation study on the modeling pipeline.
    """
    # Use in-memory data to simulate the input file
    train_data = """Street Name,Violation Description,violation_count
WALL STREET,FAILURE TO STOP AT RED LIGHT,150
WALL STREET,NO PARKING-STREET CLEANING,200
BROADWAY,FAILURE TO STOP AT RED LIGHT,300
BROADWAY,NO PARKING-STREET CLEANING,450
FLATBUSH AVENUE,FAILURE TO STOP AT RED LIGHT,50
FLATBUSH AVENUE,NO PARKING-STREET CLEANING,80
FLATBUSH AVENUE,PARKING AT A BUS STOP,25
MADISON AVENUE,FAILURE TO STOP AT RED LIGHT,120
MADISON AVENUE,NO PARKING-STREET CLEANING,180
FIFTH AVENUE,FAILURE TO STOP AT RED LIGHT,250
FIFTH AVENUE,NO PARKING-STREET CLEANING,350
"""
    df_original = pd.read_csv(io.StringIO(train_data))

    results = {}

    # --- Experiment Configurations ---
    ablation_configs = {
        "Baseline (Full Model)": {
            "use_group_split": True,
            "use_ridge_cv": True,
            "use_categorical_features": True
        },
        "Ablation: No GroupShuffleSplit": {
            "use_group_split": False,
            "use_ridge_cv": True,
            "use_categorical_features": True
        },
        "Ablation: No RidgeCV Tuning": {
            "use_group_split": True,
            "use_ridge_cv": False,
            "use_categorical_features": True
        },
        "Ablation: No Categorical Features": {
            "use_group_split": True,
            "use_ridge_cv": True,
            "use_categorical_features": False
        }
    }

    for name, config in ablation_configs.items():
        print(f"--- Running: {name} ---")

        # --- 1. Validation Split ---
        if config["use_group_split"]:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
        else:
            # Use a standard random split if not using GroupShuffleSplit
            ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
            train_idx, val_idx = next(ss.split(df_original))

        train_df = df_original.iloc[train_idx].reset_index(drop=True)
        val_df = df_original.iloc[val_idx].reset_index(drop=True)

        # --- 2. Feature Engineering ---
        train_featured, train_stats = feature_engineer(train_df)
        val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

        # --- 3. Define Features ---
        numerical_features = [
            'street_mean', 'street_sum', 'street_std', 'street_key_count',
            'violation_mean', 'violation_sum', 'violation_std',
            'boro_mean', 'boro_sum', 'boro_std'
        ]
        
        if config["use_categorical_features"]:
            categorical_features = ['violation_description', 'boroname']
        else:
            categorical_features = []
        
        all_features = numerical_features + categorical_features
        target = 'violation_count'
        
        X_train = train_featured[all_features]
        y_train = train_featured[target]
        X_val = val_featured[all_features]
        y_val = val_featured[target]

        # --- 4. Model Pipeline ---
        transformers = [('num', StandardScaler(), numerical_features)]
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
        
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')

        if config["use_ridge_cv"]:
            model = RidgeCV(alphas=np.logspace(-2, 2, 5), cv=3) # cv=3 due to small sample size
        else:
            model = Ridge(alpha=1.0) # Fixed alpha, no CV

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

        # --- 5. Training and Evaluation ---
        pipeline.fit(X_train, y_train)
        val_predictions = pipeline.predict(X_val)
        val_predictions[val_predictions < 0] = 0
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        print(f'Validation RMSE: {rmse:.4f}\n')
        results[name] = rmse

    # --- 6. Conclusion ---
    baseline_rmse = results["Baseline (Full Model)"]
    performance_impact = {}
    for name, rmse in results.items():
        if name != "Baseline (Full Model)":
            # A higher RMSE means the ablated component was important
            performance_impact[name] = rmse - baseline_rmse

    if not performance_impact:
        print("Could not run ablations to determine impact.")
        return
        
    most_impactful_component = max(performance_impact, key=performance_impact.get)
    
    print("--- Ablation Study Conclusion ---")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    for name, impact in performance_impact.items():
        print(f"Impact of removing '{name.replace('Ablation: No ', '')}': {impact:+.4f} change in RMSE")

    print(f"\nConclusion: Based on the largest increase in error when removed, "
          f"'{most_impactful_component.replace('Ablation: No ', '')}' contributes the most to the model's performance.")


if __name__ == '__main__':
    run_ablation_study()
