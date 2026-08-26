
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import io
import warnings

# Suppress LightGBM warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_evaluate(df_train, n_splits, shuffle_kfold):
    """
    Trains and evaluates the model using K-Fold cross-validation with specified parameters.
    
    Args:
        df_train (pd.DataFrame): The training dataframe.
        n_splits (int): The number of folds for KFold.
        shuffle_kfold (bool): Whether to shuffle the data before splitting.

    Returns:
        float: The out-of-fold Root Mean Squared Error.
    """
    
    # --- Feature Engineering ---
    le_street = LabelEncoder()
    le_desc = LabelEncoder()
    
    df_train['street_name_encoded'] = le_street.fit_transform(df_train['street_name'])
    df_train['violation_description_encoded'] = le_desc.fit_transform(df_train['violation_description'])
    
    # --- Model Training with K-Fold Cross-Validation ---
    features_to_use = [
        'street_name_encoded', 
        'violation_description_encoded',
        'description_mean_count',
        'street_mean_count'
    ]
    target = 'violation_count'
    
    # FIX: Use random_state only when shuffle is True to avoid ValueError
    kf = KFold(n_splits=n_splits, shuffle=shuffle_kfold, random_state=42 if shuffle_kfold else None)
    
    oof_preds = np.zeros(len(df_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        train_fold = df_train.iloc[train_idx].copy()
        val_fold = df_train.iloc[val_idx].copy()

        # --- Target Encoding within the training fold to prevent leakage ---
        street_mean_map = train_fold.groupby('street_name')[target].mean()
        desc_mean_map = train_fold.groupby('violation_description')[target].mean()
        
        train_fold['street_mean_count'] = train_fold['street_name'].map(street_mean_map)
        train_fold['description_mean_count'] = train_fold['violation_description'].map(desc_mean_map)
        
        val_fold['street_mean_count'] = val_fold['street_name'].map(street_mean_map)
        val_fold['description_mean_count'] = val_fold['violation_description'].map(desc_mean_map)
        
        # Fill NaNs for categories in validation but not in training
        global_target_mean = train_fold[target].mean()
        val_fold['street_mean_count'].fillna(global_target_mean, inplace=True)
        val_fold['description_mean_count'].fillna(global_target_mean, inplace=True)

        # Log-transform the target variable
        train_fold['log_target'] = np.log1p(train_fold[target])
        
        X_train = train_fold[features_to_use]
        y_train = train_fold['log_target']
        X_val = val_fold[features_to_use]

        lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
        lgbm.fit(X_train, y_train)

        val_preds_log = lgbm.predict(X_val)
        oof_preds[val_idx] = val_preds_log

    # --- Validation Performance on Out-of-Fold Predictions ---
    oof_preds_orig_scale = np.expm1(oof_preds)
    oof_preds_orig_scale[oof_preds_orig_scale < 0] = 0

    final_validation_score = np.sqrt(mean_squared_error(df_train[target], oof_preds_orig_scale))
    print(f'Final Validation Performance: {final_validation_score}')
    return final_validation_score

def run_ablation_study():
    """
    Runs the ablation study by executing different configurations of the training pipeline.
    """
    # Create a reproducible, ordered dummy dataset to test the effect of shuffling
    np.random.seed(42)
    data = {
        'street_name': [
            'A ST', 'B ST', 'A ST', 'C ST', 'B ST', 'D ST', 'A ST', 'C ST', 'D ST', 'E ST',
            'F ST', 'G ST', 'F ST', 'H ST', 'G ST', 'I ST', 'F ST', 'H ST', 'I ST', 'J ST'
        ] * 10,
        'violation_description': [
            'NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'NO STANDING', 'FIRE HYDRANT', 
            'NO PARKING', 'NO PARKING', 'NO STANDING', 'FIRE HYDRANT', 'BUS LANE',
            'CROSSWALK', 'SIDEWALK', 'CROSSWALK', 'DOUBLE PARK', 'SIDEWALK', 'NO PARKING',
            'CROSSWALK', 'DOUBLE PARK', 'SIDEWALK', 'BUS LANE'
        ] * 10,
        'violation_count': np.random.randint(1, 200, 200)
    }
    df = pd.DataFrame(data).sort_values('violation_count').reset_index(drop=True)

    experiments = {}

    # --- Baseline Experiment ---
    # Using a robust 5-fold shuffled CV
    print("Running Baseline (5-Fold, Shuffled)...")
    baseline_score = train_and_evaluate(df.copy(), n_splits=5, shuffle_kfold=True)
    experiments['Baseline (5-Fold, Shuffled)'] = baseline_score
    print(f"Baseline (5-Fold, Shuffled) RMSE: {baseline_score:.4f}")

    # --- Ablation 1: Reduce Number of Folds ---
    # This tests the stability of the CV score. Fewer folds mean larger validation sets but fewer training iterations.
    print("\nRunning Ablation (2-Fold, Shuffled)...")
    ablation_1_score = train_and_evaluate(df.copy(), n_splits=2, shuffle_kfold=True)
    experiments['Ablation (2-Fold, Shuffled)'] = ablation_1_score
    print(f"Ablation (2-Fold, Shuffled) RMSE: {ablation_1_score:.4f}")

    # --- Ablation 2: Disable Shuffling ---
    # This tests sensitivity to data order. If the data has an inherent order (like our sorted dummy data),
    # performance can change dramatically.
    print("\nRunning Ablation (5-Fold, Not Shuffled)...")
    ablation_2_score = train_and_evaluate(df.copy(), n_splits=5, shuffle_kfold=False)
    experiments['Ablation (5-Fold, Not Shuffled)'] = ablation_2_score
    print(f"Ablation (5-Fold, Not Shuffled) RMSE: {ablation_2_score:.4f}")

    print("\n--- Ablation Analysis ---")
    
    impact = {}
    impact['Number of Folds'] = ablation_1_score - baseline_score
    impact['KFold Shuffling'] = ablation_2_score - baseline_score

    print(f"Impact of changing Number of Folds (5 -> 2): {impact['Number of Folds']:.4f}")
    print(f"Impact of disabling KFold Shuffling: {impact['KFold Shuffling']:.4f}")

    if not impact:
        print("\nNo ablations were performed.")
        return

    # Determine the component with the largest absolute impact on performance
    most_impactful_component = max(impact, key=lambda k: abs(impact[k]))
    
    print("\n--- Conclusion ---")
    print(f"The component that contributes the most to the overall performance is: {most_impactful_component}")

if __name__ == '__main__':
    run_ablation_study()
