
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import os

# Suppress LightGBM warnings which can be verbose
warnings.filterwarnings("ignore", category=UserWarning)

def load_data(path='./input/violations_per_street_2022.csv'):
    """Loads data or creates a dummy dataframe if the file is not found."""
    if not os.path.exists(path):
        print(f"Warning: The file {path} was not found. Using a small dummy dataset for demonstration.")
        data = {
            'street_name': [f'STREET_{i%10}' for i in range(100)],
            'violation_description': [f'DESC_{i%5}' for i in range(100)],
            'violation_count': np.random.randint(1, 150, 100) + np.repeat([10, 20, 5, 80, 40], 20)
        }
        df = pd.DataFrame(data)
        return df

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_ablation_study():
    """Performs an ablation study on target encoding strategies."""
    df_full = load_data()
    if df_full is None:
        return

    ablations = {}

    # --- Experiment 1: Baseline (Leaky Target Encoding) ---
    # This reflects the original script where encoding is done before splitting, causing data leakage.
    df = df_full.copy()
    desc_agg = df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    df = pd.merge(df, desc_agg, on='violation_description', how='left')
    df = pd.merge(df, street_agg, on='street_name', how='left')
    df['street_name_encoded'] = LabelEncoder().fit_transform(df['street_name'])
    df['violation_description_encoded'] = LabelEncoder().fit_transform(df['violation_description'])
    
    features = ['street_name_encoded', 'violation_description_encoded', 'description_mean_count', 'street_mean_count']
    X = df[features]
    y = np.log1p(df['violation_count'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    val_preds = np.expm1(lgbm.predict(X_val))
    val_preds[val_preds < 0] = 0
    baseline_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_preds))
    ablations['Baseline (Leaky Target Encoding)'] = baseline_rmse

    # --- Setup for Proper (Non-Leaky) Experiments ---
    df = df_full.copy()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    y_val_original = val_df['violation_count'].copy()

    # --- Experiment 2: Ablation - Fixing Data Leakage in Target Encoding ---
    # Aggregates are created ONLY from the training set.
    desc_agg = train_df.groupby('violation_description')['violation_count'].mean().to_frame('description_mean_count')
    street_agg = train_df.groupby('street_name')['violation_count'].mean().to_frame('street_mean_count')
    global_train_mean = train_df['violation_count'].mean()

    # Fit encoders ONCE on training data, including a token for unknown values. [1]
    le_street = LabelEncoder().fit(np.append(train_df['street_name'].unique(), '<unknown>'))
    le_desc = LabelEncoder().fit(np.append(train_df['violation_description'].unique(), '<unknown>'))

    # Create temporary list to hold processed dataframes.
    processed_dfs = []
    for d in [train_df, val_df]:
        d_copy = d.copy()
        
        # Handle unseen values by mapping them to '<unknown>' token before transforming.
        d_copy['street_name_encoded'] = le_street.transform(
            d_copy['street_name'].map(lambda s: s if s in le_street.classes_ else '<unknown>')
        )
        d_copy['violation_description_encoded'] = le_desc.transform(
            d_copy['violation_description'].map(lambda s: s if s in le_desc.classes_ else '<unknown>')
        )

        # Merge target encoded features. This operation returns a new DataFrame. [4]
        d_copy = pd.merge(d_copy, desc_agg, on='violation_description', how='left')
        d_copy = pd.merge(d_copy, street_agg, on='street_name', how='left')
        
        # Fill NaNs that result from merging (e.g., unseen categories in val set).
        d_copy.fillna({'description_mean_count': global_train_mean, 'street_mean_count': global_train_mean}, inplace=True)
        
        processed_dfs.append(d_copy)
    
    # Reassign the dataframes to the processed versions to make changes persistent. [6, 12]
    train_df, val_df = processed_dfs[0], processed_dfs[1]

    X_train, X_val = train_df[features], val_df[features]
    y_train = np.log1p(train_df['violation_count'])
    
    lgbm.fit(X_train, y_train)
    val_preds = np.expm1(lgbm.predict(X_val))
    val_preds[val_preds < 0] = 0
    no_leak_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    ablations['Proper (Non-Leaky) Target Encoding'] = no_leak_rmse

    # --- Experiment 3: Ablation - Adding Smoothing to Target Encoding ---
    # Builds on the non-leaky approach by adding smoothing.
    smoothing_factor = 20
    # Aggregates are based on the original train_df's violation_count, which is correct.
    desc_agg_smooth = train_df.groupby('violation_description')['violation_count'].agg(['count', 'mean'])
    street_agg_smooth = train_df.groupby('street_name')['violation_count'].agg(['count', 'mean'])
    
    smoothed_desc = (desc_agg_smooth['count'] * desc_agg_smooth['mean'] + smoothing_factor * global_train_mean) / (desc_agg_smooth['count'] + smoothing_factor)
    smoothed_street = (street_agg_smooth['count'] * street_agg_smooth['mean'] + smoothing_factor * global_train_mean) / (street_agg_smooth['count'] + smoothing_factor)

    # This loop modifies the dataframes in-place by overwriting the mean_count columns.
    for d in [train_df, val_df]:
        d['description_mean_count'] = d['violation_description'].map(smoothed_desc)
        d['street_mean_count'] = d['street_name'].map(smoothed_street)
        d.fillna({'description_mean_count': global_train_mean, 'street_mean_count': global_train_mean}, inplace=True)

    X_train, X_val = train_df[features], val_df[features]
    lgbm.fit(X_train, y_train)
    val_preds = np.expm1(lgbm.predict(X_val))
    val_preds[val_preds < 0] = 0
    smoothed_rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    ablations['Smoothed Target Encoding'] = smoothed_rmse

    # --- Results and Conclusion ---
    print("--- Ablation Study on Target Encoding ---")
    for name, score in ablations.items():
        print(f"RMSE for {name}: {score:.4f}")

    # Calculate impact (degradation). Note: Fixing a leak often increases RMSE, which is a good thing (more realistic score).
    impacts = {
        "Fixing Data Leakage": ablations['Proper (Non-Leaky) Target Encoding'] - ablations['Baseline (Leaky Target Encoding)'],
        "Adding Smoothing": ablations['Smoothed Target Encoding'] - ablations['Proper (Non-Leaky) Target Encoding']
    }

    print("\n--- Performance Impact Analysis ---")
    print(f"Impact of Fixing Leakage (vs. Baseline): {impacts['Fixing Data Leakage']:.4f} RMSE")
    print(f"Impact of Adding Smoothing (vs. Proper Encoding): {impacts['Adding Smoothing']:.4f} RMSE")
    
    # Determine most impactful component by magnitude of change.
    most_impactful_component = max(impacts, key=lambda k: abs(impacts[k]))
    
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance measurement is: **{most_impactful_component}**.")
    print("A large positive impact from 'Fixing Data Leakage' indicates the baseline score was artificially inflated due to data leakage.")
    print("A negative impact (improvement) from 'Adding Smoothing' suggests it helps the model generalize better.")
    
    final_validation_score = smoothed_rmse
    print(f'Final Validation Performance: {final_validation_score}')


if __name__ == '__main__':
    run_ablation_study()
