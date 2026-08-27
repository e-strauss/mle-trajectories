
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore')

# --- Utility Functions from the original script ---

def create_smoothed_target_encoding(train_series, target_series, test_series, smoothing=10):
    """Creates smoothed target encoding for a categorical feature."""
    global_mean = np.mean(target_series)
    stats = target_series.groupby(train_series).agg(['mean', 'count'])
    smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    
    encoded_train = train_series.map(smoothed_mean)
    encoded_test = test_series.map(smoothed_mean)
    encoded_test.fillna(global_mean, inplace=True)
    
    return encoded_train, encoded_test

# --- Experiment Definitions ---

def run_baseline(df):
    """
    Runs the baseline experiment using 5-Fold Cross-Validation and smoothed target encoding.
    """
    print("Running: Baseline (K-Fold with Smoothing)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    features = ['street_name_encoded', 'violation_description_encoded']

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_fold, val_fold = df.iloc[train_idx], df.iloc[val_idx]
        y_train_fold, y_val_fold = train_fold['log_violation_count'], val_fold['log_violation_count']

        X_train_fold = pd.DataFrame(index=train_fold.index)
        X_val_fold = pd.DataFrame(index=val_fold.index)

        # Smoothed Target Encoding for 'violation_description'
        X_train_fold['violation_description_encoded'], X_val_fold['violation_description_encoded'] = \
            create_smoothed_target_encoding(train_fold['violation_description'], y_train_fold, val_fold['violation_description'], smoothing=10)

        # Smoothed Target Encoding for 'street_name'
        X_train_fold['street_name_encoded'], X_val_fold['street_name_encoded'] = \
            create_smoothed_target_encoding(train_fold['street_name'], y_train_fold, val_fold['street_name'], smoothing=10)
        
        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train_fold[features], y_train_fold)
        val_preds_log = lgbm.predict(X_val_fold[features])
        oof_preds[val_idx] = val_preds_log

    oof_preds_unlogged = np.expm1(oof_preds)
    oof_preds_unlogged[oof_preds_unlogged < 0] = 0
    return np.sqrt(mean_squared_error(df['violation_count'], oof_preds_unlogged))

def run_ablation_no_kfold(df):
    """
    Ablation 1: Removes K-Fold and uses a simple train/test split.
    This introduces data leakage in the target encoding.
    """
    print("Running: Ablation 1 (No K-Fold, simple split)")
    features = ['street_name_encoded', 'violation_description_encoded']

    X_train, X_val, y_train, y_val = train_test_split(df, df['log_violation_count'], test_size=0.2, random_state=42)
    
    # Create leaky target encoding on the training set
    desc_map = X_train.groupby('violation_description')['log_violation_count'].mean()
    street_map = X_train.groupby('street_name')['log_violation_count'].mean()

    # Apply to both train and validation sets
    X_train_encoded = pd.DataFrame(index=X_train.index)
    X_val_encoded = pd.DataFrame(index=X_val.index)
    
    X_train_encoded['violation_description_encoded'] = X_train['violation_description'].map(desc_map)
    X_train_encoded['street_name_encoded'] = X_train['street_name'].map(street_map)
    
    global_mean_desc = y_train.mean()
    global_mean_street = y_train.mean()
    X_val_encoded['violation_description_encoded'] = X_val['violation_description'].map(desc_map).fillna(global_mean_desc)
    X_val_encoded['street_name_encoded'] = X_val['street_name'].map(street_map).fillna(global_mean_street)

    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train_encoded[features], y_train)

    val_preds_log = lgbm.predict(X_val_encoded[features])
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0

    return np.sqrt(mean_squared_error(np.expm1(y_val), val_preds))


def run_ablation_no_smoothing(df):
    """
    Ablation 2: Keeps K-Fold but removes smoothing from target encoding.
    """
    print("Running: Ablation 2 (K-Fold with No Smoothing)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    features = ['street_name_encoded', 'violation_description_encoded']

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_fold, val_fold = df.iloc[train_idx], df.iloc[val_idx]
        y_train_fold, y_val_fold = train_fold['log_violation_count'], val_fold['log_violation_count']

        X_train_fold = pd.DataFrame(index=train_fold.index)
        X_val_fold = pd.DataFrame(index=val_fold.index)

        # Target Encoding with smoothing=0
        X_train_fold['violation_description_encoded'], X_val_fold['violation_description_encoded'] = \
            create_smoothed_target_encoding(train_fold['violation_description'], y_train_fold, val_fold['violation_description'], smoothing=0)
        X_train_fold['street_name_encoded'], X_val_fold['street_name_encoded'] = \
            create_smoothed_target_encoding(train_fold['street_name'], y_train_fold, val_fold['street_name'], smoothing=0)
        
        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train_fold[features], y_train_fold)
        val_preds_log = lgbm.predict(X_val_fold[features])
        oof_preds[val_idx] = val_preds_log

    oof_preds_unlogged = np.expm1(oof_preds)
    oof_preds_unlogged[oof_preds_unlogged < 0] = 0
    return np.sqrt(mean_squared_error(df['violation_count'], oof_preds_unlogged))


def main():
    """Main function to run the ablation study."""
    parser = argparse.ArgumentParser(description="Ablation study for NYC parking violations model.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data file.')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}")
        # Create a small dummy dataframe for demonstration if file not found
        print("Using a small dummy dataset for demonstration.")
        data = {'street_name': [f'Street {i%2}' for i in range(100)],
                'violation_description': [f'Desc {i%5}' for i in range(100)],
                'violation_count': np.random.randint(1, 100, 100)}
        df = pd.DataFrame(data)

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['log_violation_count'] = np.log1p(df['violation_count'])

    # --- Run Experiments ---
    results = {}
    results['Baseline'] = run_baseline(df.copy())
    results['No K-Fold'] = run_ablation_no_kfold(df.copy())
    results['No Smoothing'] = run_ablation_no_smoothing(df.copy())

    # --- Print and Analyze Results ---
    print("\n--- Ablation Study Results ---")
    baseline_score = results['Baseline']
    print(f"Baseline RMSE: {baseline_score:.4f}")

    degradations = {}
    for name, score in results.items():
        if name != 'Baseline':
            degradation = score - baseline_score
            degradations[name] = degradation
            print(f"Ablation '{name}' RMSE: {score:.4f} (Degradation: {degradation:+.4f})")

    if not degradations:
        print("\nNo ablations were run to compare.")
        return
        
    most_impactful = max(degradations, key=degradations.get)
    print(f"\nConclusion: The component that contributes most to performance is '{most_impactful}'.")
    print("Removing it caused the largest increase in RMSE, indicating its high importance for model accuracy and generalization.")


if __name__ == '__main__':
    # Create dummy data if input folder doesn't exist, for portability
    if not os.path.exists('./input'):
        os.makedirs('./input')
        print("Creating dummy input file: ./input/violations_per_street_2022.csv")
        num_rows = 5000
        streets = [f'STREET_{i}' for i in range(100)]
        descriptions = [f'VIOLATION_DESC_{i}' for i in range(50)]
        dummy_data = {
            'Street Name': np.random.choice(streets, num_rows),
            'Violation Description': np.random.choice(descriptions, num_rows),
            'violation_count': np.random.lognormal(3, 1, num_rows).astype(int) + 1
        }
        pd.DataFrame(dummy_data).to_csv('./input/violations_per_street_2022.csv', index=False)

    main()
