
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
import lightgbm as lgb

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)

def feature_engineer(df, train_stats=None):
    """
    Engineers features for the model.

    This function performs two main tasks:
    1. Augments the data with borough information by joining with an external file.
    2. Creates aggregate features (mean, sum, std, count) based on street,
       violation type, and borough.

    To prevent data leakage, it can operate in two modes:
    - Training mode (train_stats is None): Calculates and returns new statistics.
    - Inference mode (train_stats is provided): Applies pre-calculated statistics
      to a new dataset (validation or test).

    Args:
        df (pd.DataFrame): The dataframe to engineer features for.
        train_stats (dict, optional): A dictionary containing statistics (aggregates)
                                      from the training set. If None, stats are
                                      calculated from df itself.

    Returns:
        pd.DataFrame: The dataframe with new features.
        dict: A dictionary of the calculated stats (if train_stats was None).
    """
    df_engineered = df.copy()

    # Standardize column names for easier access - already done in main, but good practice
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
        cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
        cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()

        df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
        df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
        df_engineered['boroname'].fillna('Unknown', inplace=True)
        df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
    else:
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


def main():
    """
    Main function to run the training, validation, and optional prediction pipeline.
    """
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction")
    parser.add_argument('--train-path', default='/Users/USER/Documents/UNI/SS26/nyc_open_data/train/all_data/violations_per_street_2022.csv', help='Path to the training data CSV.')
    parser.add_argument('--test-path', default="/Users/USER/Documents/UNI/SS26/nyc_open_data/violations_per_street_2023.csv", help='(Optional) Path to the test data CSV for prediction.')
    args = parser.parse_args()

    # --- 1. Load and Prepare Training Data ---
    print(f"Loading training data from {args.train_path}...")
    try:
        df_original = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}")
        return

    # Standardize column names at the beginning
    df_original.columns = [c.replace(' ', '_').lower() for c in df_original.columns]

    # --- 2. Development Validation (Holdout Set) ---
    print("Creating a validation set from the training data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['street_name']))
    
    df_train = df_original.iloc[train_idx]
    df_val = df_original.iloc[val_idx]

    print("Engineering features for training and validation sets...")
    train_featured, train_stats = feature_engineer(df_train)
    val_featured, _ = feature_engineer(df_val, train_stats=train_stats)

    # --- 3. Define Features and Target for Validation ---
    categorical_features = ['violation_description', 'boroname']
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    all_features = numerical_features + categorical_features
    target = 'violation_count'

    X_train_val_split = train_featured[all_features]
    y_train_val_split = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- 4. Model Pipeline Definition ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    lgbm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=SEED))
    ])

    # --- 5. Train and Evaluate on Validation Set ---
    print("Training models for validation...")
    ridge_pipeline.fit(X_train_val_split, y_train_val_split)
    lgbm_pipeline.fit(X_train_val_split, y_train_val_split)

    print("Generating predictions for the validation set...")
    val_pred_ridge = ridge_pipeline.predict(X_val)
    val_pred_lgbm = lgbm_pipeline.predict(X_val)

    val_pred_ridge[val_pred_ridge < 0] = 0
    val_pred_lgbm[val_pred_lgbm < 0] = 0
    
    # Apply dynamic ensemble blending to validation predictions
    val_confidence_proxy = val_featured['street_key_count'].values
    min_weight, max_weight, transition_count = 0.25, 0.75, 40.0
    val_scaled_count = np.minimum(val_confidence_proxy / transition_count, 1.0)
    val_lgbm_weight = min_weight + (max_weight - min_weight) * val_scaled_count
    val_ridge_weight = 1.0 - val_lgbm_weight
    val_predictions_ensembled = (val_lgbm_weight * val_pred_lgbm) + (val_ridge_weight * val_pred_ridge)

    final_validation_score = np.sqrt(mean_squared_error(y_val, val_predictions_ensembled))
    print(f"Final Validation Performance: {final_validation_score}")

    # --- 6. Optional: Generate Test Predictions if test_path is provided ---
    if args.test_path:
        print(f"\n--- Test path provided: {args.test_path} ---")
        print("Retraining models on the full training dataset...")
        
        # a. Re-engineer features on full training data to create final stats
        train_full_featured, train_full_stats = feature_engineer(df_original)
        X_train_full = train_full_featured[all_features]
        y_train_full = train_full_featured[target]
        
        # b. Retrain pipelines on full data
        ridge_pipeline.fit(X_train_full, y_train_full)
        print(f"Ridge training complete. Best alpha found: {ridge_pipeline.named_steps['regressor'].alpha_}")
        lgbm_pipeline.fit(X_train_full, y_train_full)
        print("LightGBM training complete.")

        # c. Load and process test data
        print(f"Loading test data from {args.test_path}...")
        try:
            test_df_original = pd.read_csv(args.test_path)
            test_df_original.columns = [c.replace(' ', '_').lower() for c in test_df_original.columns]
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}")
            return
            
        print("Engineering features for the test set...")
        test_featured, _ = feature_engineer(test_df_original, train_stats=train_full_stats)
        X_test = test_featured[all_features]

        # d. Report on unseen keys
        train_keys = set(zip(df_original['street_name'], df_original['violation_description']))
        test_keys = set(zip(test_df_original['street_name'], test_df_original['violation_description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} key pairs in test set not present in training data.")

        # e. Predict on test data
        print("Generating predictions on the test set...")
        test_pred_ridge = ridge_pipeline.predict(X_test)
        test_pred_lgbm = lgbm_pipeline.predict(X_test)

        test_pred_ridge[test_pred_ridge < 0] = 0
        test_pred_lgbm[test_pred_lgbm < 0] = 0
        
        # f. Dynamic ensemble blending for test set
        test_confidence_proxy = test_featured['street_key_count'].values
        test_scaled_count = np.minimum(test_confidence_proxy / transition_count, 1.0)
        test_lgbm_weight = min_weight + (max_weight - min_weight) * test_scaled_count
        test_ridge_weight = 1.0 - test_lgbm_weight
        test_predictions_ensembled = (test_lgbm_weight * test_pred_lgbm) + (test_ridge_weight * test_pred_ridge)
        
        # g. Create submission file
        print("Creating submission file...")
        submission_df = test_df_original[['street_name', 'violation_description']].copy()
        submission_df.rename(columns={'violation_description': 'violation_type'}, inplace=True)
        submission_df['predicted_count'] = np.round(test_predictions_ensembled).astype(int)
        
        os.makedirs('./final', exist_ok=True)
        submission_path = './final/submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Successfully created {submission_path}")

        # h. Score test set if ground truth is available
        if 'violation_count' in test_df_original.columns:
            test_ground_truth = test_df_original['violation_count']
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_predictions_ensembled))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
