
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV, Ridge
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

def get_data():
    """Creates a dummy dataset in memory."""
    csv_data = """
"Street Name","Violation Description","violation_count"
"5th Ave","NO PARKING-DAY/TIME LIMITS",150
"5th Ave","PHTO SCHOOL ZN SPEED VIOLATION",250
"Broadway","NO PARKING-DAY/TIME LIMITS",200
"Broadway","FAILURE TO DISPLAY MUNI METER RECPT",300
"Lexington Ave","NO PARKING-DAY/TIME LIMITS",120
"Lexington Ave","FAILURE TO STOP AT RED LIGHT",80
"Madison Ave","NO PARKING-DAY/TIME LIMITS",180
"Madison Ave","PHTO SCHOOL ZN SPEED VIOLATION",220
"Park Ave","FAILURE TO DISPLAY MUNI METER RECPT",250
"Park Ave","NO STANDING-DAY/TIME LIMITS",90
"7th Ave","NO PARKING-DAY/TIME LIMITS",130
"7th Ave","FAILURE TO STOP AT RED LIGHT",70
"8th Ave","PHTO SCHOOL ZN SPEED VIOLATION",280
"8th Ave","NO STANDING-DAY/TIME LIMITS",110
"9th Ave","FAILURE TO DISPLAY MUNI METER RECPT",270
"9th Ave","NO PARKING-DAY/TIME LIMITS",160
"10th Ave","PHTO SCHOOL ZN SPEED VIOLATION",290
"10th Ave","FAILURE TO STOP AT RED LIGHT",95
"1st Ave","NO STANDING-DAY/TIME LIMITS",105
"1st Ave","FAILURE TO DISPLAY MUNI METER RECPT",265
    """
    df = pd.read_csv(io.StringIO(csv_data))
    # Create more data for robust splitting
    df_extended = pd.concat([df] * 10, ignore_index=True)
    df_extended['violation_count'] += np.random.randint(-20, 20, size=len(df_extended))
    return df_extended

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model, including borough augmentation.
    """
    df_engineered = df.copy()

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Dummy Borough Augmentation ---
    # In a real scenario, this would use an external file.
    # For this self-contained script, we'll create a dummy mapping.
    dummy_boro_map = {
        '5TH AVE': 'MANHATTAN', 'BROADWAY': 'MANHATTAN', 'LEXINGTON AVE': 'MANHATTAN',
        'MADISON AVE': 'MANHATTAN', 'PARK AVE': 'MANHATTAN', '7TH AVE': 'BROOKLYN',
        '8TH AVE': 'BROOKLYN', '9TH AVE': 'BROOKLYN', '10TH AVE': 'QUEENS', '1ST AVE': 'QUEENS'
    }
    df_engineered['boroname'] = df_engineered['street_name'].str.upper().map(dummy_boro_map).fillna('Unknown')
    
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
            'street_agg': street_agg, 'violation_agg': violation_agg, 'boro_agg': boro_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Smart NaN filling from the plan_implement_agent_1
    mean_cols = ['street_mean', 'violation_mean', 'boro_mean']
    global_mean_val = stats.get('global_mean', 0)
    for col in mean_cols:
        if col in df_engineered.columns:
            df_engineered[col].fillna(global_mean_val, inplace=True)
    
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats


def run_experiment(train_df, val_df, with_mean_centering=True, use_borough_features=True, ridge_cv_method=5):
    """
    Runs a single training and validation experiment with a specific configuration.
    """
    # Feature Engineering
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    # Define feature sets based on ablation
    categorical_features = ['violation_description']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std'
    ]
    if use_borough_features:
        categorical_features.append('boroname')
        numerical_features.extend(['boro_mean', 'boro_sum', 'boro_std'])

    all_features = numerical_features + categorical_features
    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # Model Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=with_mean_centering), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=ridge_cv_method))
    ])

    # Training and Validation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    return rmse


def main():
    """
    Main function to run the ablation study.
    """
    df_original = get_data()

    # Validation Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    # --- 1. Baseline Experiment ---
    baseline_rmse = run_experiment(
        train_df.copy(), val_df.copy(),
        with_mean_centering=True,
        use_borough_features=True,
        ridge_cv_method=5
    )
    results['Baseline (Full Model)'] = baseline_rmse
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    # --- 2. Ablation: No Mean Centering ---
    # Test the impact of centering the data in StandardScaler
    no_mean_centering_rmse = run_experiment(
        train_df.copy(), val_df.copy(),
        with_mean_centering=False,
        use_borough_features=True,
        ridge_cv_method=5
    )
    results['Ablation: No Mean Centering'] = no_mean_centering_rmse
    print(f"Ablation (No Mean Centering) RMSE: {no_mean_centering_rmse:.4f}")

    # --- 3. Ablation: No Borough-related Features ---
    # Test the impact of all features derived from the borough augmentation
    no_boro_features_rmse = run_experiment(
        train_df.copy(), val_df.copy(),
        with_mean_centering=True,
        use_borough_features=False,
        ridge_cv_method=5
    )
    results['Ablation: No Borough Features'] = no_boro_features_rmse
    print(f"Ablation (No Borough Features) RMSE: {no_boro_features_rmse:.4f}")

    # --- 4. Ablation: Use Leave-One-Out CV in RidgeCV ---
    # Test the impact of the hyperparameter tuning strategy within RidgeCV
    loo_cv_rmse = run_experiment(
        train_df.copy(), val_df.copy(),
        with_mean_centering=True,
        use_borough_features=True,
        ridge_cv_method=None # Use efficient Leave-One-Out CV
    )
    results['Ablation: RidgeCV with LOOCV'] = loo_cv_rmse
    print(f"Ablation (RidgeCV with LOOCV) RMSE: {loo_cv_rmse:.4f}")

    # --- Conclusion ---
    print("\n--- Ablation Study Conclusion ---")
    impacts = {
        'Mean Centering': abs(results['Ablation: No Mean Centering'] - baseline_rmse),
        'Borough Features': abs(results['Ablation: No Borough Features'] - baseline_rmse),
        'RidgeCV Internal Method (5-fold vs LOOCV)': abs(results['Ablation: RidgeCV with LOOCV'] - baseline_rmse)
    }

    if not impacts:
        most_impactful = "No component had a measurable impact."
    else:
        most_impactful = max(impacts, key=impacts.get)

    print(f"Component with the most impact on performance: {most_impactful}")

if __name__ == '__main__':
    main()
