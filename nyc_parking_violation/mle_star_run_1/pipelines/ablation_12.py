
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_dummy_data_if_not_exists(path='./input'):
    """Creates a dummy CSV file if it doesn't exist to ensure the script is runnable."""
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, 'violations_per_street_2022.csv')
    if not os.path.exists(file_path):
        print("Dummy data not found, creating 'violations_per_street_2022.csv'...")
        data = {
            'street_name': ['MAIN ST'] * 20 + ['ELM ST'] * 20 + ['OAK AVE'] * 20 + ['PINE LN'] * 20 + ['MAPLE DR'] * 20,
            'violation_description': (['NO PARKING', 'FIRE HYDRANT'] * 10) + 
                                     (['NO STANDING', 'DOUBLE PARKING'] * 10) +
                                     (['BIKE LANE', 'NO PARKING'] * 10) +
                                     (['FIRE HYDRANT', 'NO STANDING'] * 10) +
                                     (['DOUBLE PARKING', 'BIKE LANE'] * 10),
            'violation_count': np.abs(np.random.randn(100) * 50 + np.repeat([10, 50, 20, 80, 40], 20)).astype(int) + 1
        }
        df = pd.DataFrame(data)
        # Introduce order to make shuffling matter
        df = df.sort_values(by=['violation_count']).reset_index(drop=True)
        df.to_csv(file_path, index=False)


def run_ablation_experiment(name, shuffle_kfold, nan_fill_strategy, train_path='./input/violations_per_street_2022.csv'):
    """
    Runs a single training and validation experiment with specified ablation configurations.
    
    Args:
        name (str): The name of the experiment for logging.
        shuffle_kfold (bool): Controls whether KFold shuffles data before splitting.
        nan_fill_strategy (str): 'mean' to fill with the fold's training mean, or 'zero' to fill with 0.

    Returns:
        float: The mean validation RMSE across all folds.
    """
    try:
        df_train = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_path}. Aborting experiment.")
        return float('inf')

    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')

    # Basic Feature Engineering (Label Encoding)
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])
    
    df_train['log_target'] = np.log1p(df_train['violation_count'])

    base_features = ['street_name_encoded', 'violation_description_encoded']
    target = 'log_target'

    # --- K-Fold Cross-Validation with Ablation Points ---
    n_splits = 5
    # Ablation Point 1: KFold shuffling behavior
    kf = KFold(n_splits=n_splits, shuffle=shuffle_kfold, random_state=42 if shuffle_kfold else None)

    val_scores = []
    
    for train_idx, val_idx in kf.split(df_train):
        train_fold = df_train.iloc[train_idx].copy()
        val_fold = df_train.iloc[val_idx].copy()

        # Target Encoding performed within each fold to prevent data leakage
        desc_map = train_fold.groupby('violation_description_encoded')[target].mean()
        street_map = train_fold.groupby('street_name_encoded')[target].mean()

        train_fold['description_mean_target'] = train_fold['violation_description_encoded'].map(desc_map)
        val_fold['description_mean_target'] = val_fold['violation_description_encoded'].map(desc_map)
        
        train_fold['street_mean_target'] = train_fold['street_name_encoded'].map(street_map)
        val_fold['street_mean_target'] = val_fold['street_name_encoded'].map(street_map)

        # Ablation Point 2: Strategy for filling NaNs from target encoding
        if nan_fill_strategy == 'mean':
            global_mean_target = train_fold[target].mean()
            val_fold.fillna({'description_mean_target': global_mean_target, 'street_mean_target': global_mean_target}, inplace=True)
        elif nan_fill_strategy == 'zero':
            val_fold.fillna(0, inplace=True)

        features = base_features + ['description_mean_target', 'street_mean_target']
        X_train, y_train = train_fold[features], train_fold[target]
        X_val, y_val = val_fold[features], val_fold[target]
        y_val_original = np.expm1(y_val)

        # Model Training
        lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=100)
        lgbm.fit(X_train, y_train)

        # Validation
        val_preds_log = lgbm.predict(X_val)
        val_preds = np.expm1(val_preds_log)
        val_preds[val_preds < 0] = 0

        fold_score = np.sqrt(mean_squared_error(y_val_original, val_preds))
        val_scores.append(fold_score)

    final_score = np.mean(val_scores)
    print(f"Experiment '{name}': Validation RMSE = {final_score:.4f}")
    return final_score

if __name__ == '__main__':
    create_dummy_data_if_not_exists()
    
    results = {}

    # --- 1. Baseline Experiment ---
    # The standard configuration with robust practices.
    results['Baseline'] = run_ablation_experiment(
        name='Baseline (Shuffled KFold, Mean NaN Fill)',
        shuffle_kfold=True,
        nan_fill_strategy='mean'
    )

    # --- 2. Ablation 1: No Shuffling in KFold ---
    # Tests the impact of data order by disabling shuffling in cross-validation.
    results['Ablation_No_Shuffle'] = run_ablation_experiment(
        name='Ablation 1 (No KFold Shuffle)',
        shuffle_kfold=False,
        nan_fill_strategy='mean'
    )

    # --- 3. Ablation 2: Naive NaN Filling in Target Encoding ---
    # Replaces the robust mean-filling strategy with a simple zero-fill.
    results['Ablation_Zero_Fill'] = run_ablation_experiment(
        name='Ablation 2 (Zero NaN Fill)',
        shuffle_kfold=True,
        nan_fill_strategy='zero'
    )
    
    print("\n--- Ablation Study Summary ---")

    baseline_score = results.get('Baseline', float('inf'))
    if baseline_score == float('inf'):
        print("Baseline experiment failed. Cannot determine impact.")
    else:
        degradations = {
            # Higher degradation value means worse performance (higher RMSE)
            'KFold Shuffling': results['Ablation_No_Shuffle'] - baseline_score,
            'Target Encoding NaN Fill Strategy': results['Ablation_Zero_Fill'] - baseline_score,
        }

        print(f"Baseline RMSE: {baseline_score:.4f}")
        for component, degradation in degradations.items():
            print(f"Degradation from removing '{component}': {degradation:+.4f} RMSE")
        
        # Determine the most impactful component based on the largest performance drop
        if not degradations or max(degradations.values()) <= 0:
            most_impactful_component = "None of the tested components worsened performance"
        else:
            most_impactful_component = max(degradations, key=degradations.get)

        print(f"\nMost impactful component: The '{most_impactful_component}' contributes the most to performance.")

