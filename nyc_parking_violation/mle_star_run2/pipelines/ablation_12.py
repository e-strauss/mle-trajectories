
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

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model, including aggregates and interaction terms.
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
    df_engineered['boro_violation_interaction'] = df_engineered['boroname'].astype(str) + '_' + df_engineered['violation_description'].astype(str)

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std', 'violation_key_count']
        
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std', 'boro_key_count']

        interaction_agg = df_engineered.groupby('boro_violation_interaction')['violation_count'].agg(['mean', 'sum', 'std'])
        interaction_agg.columns = ['interaction_mean', 'interaction_sum', 'interaction_std']
        
        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            'interaction_agg': interaction_agg,
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')
    df_engineered = pd.merge(df_engineered, stats['interaction_agg'], on='boro_violation_interaction', how='left')

    df_engineered.drop(columns=['boro_violation_interaction'], inplace=True)
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats

def run_experiment(
    train_df,
    val_df,
    alphas=np.logspace(-2, 2, 5),
    use_count_features=True,
    remainder_strategy='passthrough'
):
    """
    Runs a single experiment with a specific configuration.
    """
    # Feature Engineering
    train_featured, train_stats = feature_engineer(train_df)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    # Feature Definition
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std',
        'interaction_mean', 'interaction_sum', 'interaction_std'
    ]
    
    if use_count_features:
        count_features = ['street_key_count', 'violation_key_count', 'boro_key_count']
        numerical_features.extend(count_features)
    
    all_features = numerical_features + categorical_features
    target = 'violation_count'
    
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # Model Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder=remainder_strategy
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=alphas, cv=5))
    ])

    # Training & Evaluation
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    return rmse

def main():
    """
    Main function to run the ablation study.
    """
    train_path = './input/violations_per_street_2022.csv'
    try:
        df_original = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Creating dummy data to proceed.")
        data = {
            'Street Name': [f'Street {i}' for i in range(100) for _ in range(5)],
            'Violation Description': [f'Violation {j}' for _ in range(100) for j in range(5)],
            'violation_count': np.random.randint(1, 500, 500)
        }
        df_original = pd.DataFrame(data)
        if not os.path.exists('./input'):
            os.makedirs('./input')
        df_original.to_csv(train_path, index=False)
        cscl_data = {
            'ST_NAME': [f'STREET {i}' for i in range(100)],
            'BORONAME': [f'Boro {i % 5}' for i in range(100)]
        }
        pd.DataFrame(cscl_data).to_csv('./input/nyc_cscl.csv', index=False)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    results = {}

    print("Running: Baseline Model")
    baseline_rmse = run_experiment(train_df, val_df)
    results['Baseline'] = baseline_rmse
    print(f"  - Validation RMSE: {baseline_rmse:.4f}\n")

    print("Running: Ablation with narrower Alpha range for RidgeCV")
    narrower_alphas = np.logspace(-1, 1, 10)
    ablation_alpha_rmse = run_experiment(train_df, val_df, alphas=narrower_alphas)
    results['Narrow Alpha Range'] = ablation_alpha_rmse
    print(f"  - Validation RMSE: {ablation_alpha_rmse:.4f}\n")

    print("Running: Ablation without count features")
    ablation_counts_rmse = run_experiment(train_df, val_df, use_count_features=False)
    results['No Count Features'] = ablation_counts_rmse
    print(f"  - Validation RMSE: {ablation_counts_rmse:.4f}\n")
    
    print("Running: Ablation with ColumnTransformer remainder='drop'")
    ablation_remainder_rmse = run_experiment(train_df, val_df, remainder_strategy='drop')
    results['No Remainder Passthrough'] = ablation_remainder_rmse
    print(f"  - Validation RMSE: {ablation_remainder_rmse:.4f}\n")

    print("--- Ablation Study Results ---")
    baseline_score = results['Baseline']
    performance_impact = {}

    for name, score in results.items():
        if name != 'Baseline':
            impact = score - baseline_score
            performance_impact[name] = impact
            print(f"Impact of '{name}': RMSE changed by {impact:+.4f}")

    if not performance_impact:
        print("\nNo ablations were run to compare against the baseline.")
        return

    most_impactful_component = max(performance_impact, key=lambda k: abs(performance_impact[k]))
    impact_value = performance_impact[most_impactful_component]

    print("\n--- Conclusion ---")
    if impact_value > 0.0001:
        print(f"The component that contributes most to the model's performance is '{most_impactful_component}'.")
        print(f"Removing or changing it increased the error (worsened performance) by {impact_value:.4f}.")
    elif impact_value < -0.0001:
        print(f"The component that is most detrimental to the model's performance is '{most_impactful_component}'.")
        print(f"Removing or changing it decreased the error (improved performance) by {abs(impact_value):.4f}.")
    else:
        print(f"The component with the largest (but negligible) impact was '{most_impactful_component}'.")

if __name__ == '__main__':
    main()
