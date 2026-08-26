
import argparse
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


def run_experiment(df_original, use_violation_aggs=True, use_scaler=True, clip_negatives=True):
    """
    Runs a single experiment with a specific configuration.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'boro_mean', 'boro_sum', 'boro_std'
    ]

    # Ablation: Conditionally include violation aggregates
    if use_violation_aggs:
        numerical_features.extend(['violation_mean', 'violation_sum', 'violation_std'])

    all_features = numerical_features + categorical_features
    target = 'violation_count'

    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # Ablation: Conditionally use StandardScaler
    num_transformer = StandardScaler() if use_scaler else 'passthrough'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)

    # Ablation: Conditionally clip negative predictions
    if clip_negatives:
        val_predictions[val_predictions < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    
    try:
        df = pd.read_csv(train_path)
    except FileNotFoundError:
        # Create a dummy dataframe if the file is not found
        print(f"Warning: Training file not found at {train_path}. Using a dummy dataframe for demonstration.")
        dummy_data = """Street Name,Violation Description,violation_count
45 ST,FAILURE TO STOP AT RED LIGHT,150
5 AVE,FAILURE TO STOP AT RED LIGHT,200
5 AVE,NO STANDING-DAY/TIME LIMITS,300
86 ST,FAILURE TO STOP AT RED LIGHT,80
86 ST,NO STANDING-DAY/TIME LIMITS,120
MAIN ST,BUS LANE VIOLATION,500
MAIN ST,FAILURE TO STOP AT RED LIGHT,250
BROADWAY,BUS LANE VIOLATION,450
BROADWAY,NO STANDING-DAY/TIME LIMITS,350
"""
        df = pd.read_csv(io.StringIO(dummy_data))
        # Create dummy borough file
        if not os.path.exists('./input'):
            os.makedirs('./input')
        dummy_cscl_data = """ST_NAME,BORONAME
45 ST,Brooklyn
5 AVE,Manhattan
86 ST,Brooklyn
MAIN ST,Queens
BROADWAY,Manhattan
"""
        with open('./input/nyc_cscl.csv', 'w') as f:
            f.write(dummy_cscl_data)


    results = {}

    # --- Baseline Experiment ---
    results['Baseline (Full Model)'] = run_experiment(df, use_violation_aggs=True, use_scaler=True, clip_negatives=True)

    # --- Ablation 1: No Violation Description Aggregates ---
    results['Ablation: No Violation Aggregates'] = run_experiment(df, use_violation_aggs=False, use_scaler=True, clip_negatives=True)

    # --- Ablation 2: No StandardScaler ---
    results['Ablation: No Feature Scaling'] = run_experiment(df, use_violation_aggs=True, use_scaler=False, clip_negatives=True)

    # --- Ablation 3: No Clipping of Negative Predictions ---
    results['Ablation: No Negative Clipping'] = run_experiment(df, use_violation_aggs=True, use_scaler=True, clip_negatives=False)

    print("--- Ablation Study Results (Validation RMSE) ---")
    for name, rmse in results.items():
        print(f'{name}: {rmse:.4f}')

    # --- Conclusion ---
    baseline_rmse = results['Baseline (Full Model)']
    impacts = {
        'Violation Description Aggregates': results['Ablation: No Violation Aggregates'] - baseline_rmse,
        'Feature Scaling (StandardScaler)': results['Ablation: No Feature Scaling'] - baseline_rmse,
        'Clipping Negative Predictions': results['Ablation: No Negative Clipping'] - baseline_rmse,
    }

    # The most impactful feature is the one whose removal causes the largest change in error (largest absolute value)
    most_impactful_feature = max(impacts, key=lambda k: abs(impacts[k]))
    impact_value = impacts[most_impactful_feature]
    
    print("\n--- Conclusion ---")
    if impact_value > 0:
        print(f"The removed component '{most_impactful_feature}' was beneficial. Its removal increased the RMSE by {impact_value:.4f}.")
        print(f"Therefore, '{most_impactful_feature}' contributes the most to the model's performance.")
    else:
        print(f"The removed component '{most_impactful_feature}' was detrimental. Its removal decreased the RMSE by {-impact_value:.4f}.")
        print(f"Therefore, removing '{most_impactful_feature}' improves the model's performance the most.")


if __name__ == '__main__':
    main()
