
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
    2. Creates regularized aggregate features (smoothed mean, sum, std, median, count)
       based on street, violation type, and borough.

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

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
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

    # The target column might not be present in a keys-only test file
    has_target = 'violation_count' in df_engineered.columns

    # To prevent issues with feature name collisions, we temporarily remove the target
    # column and add it back later.
    target_series = None
    if has_target:
        target_series = df_engineered['violation_count']
        df_engineered = df_engineered.drop(columns=['violation_count'])


    # --- Create Aggregate Features ---
    if train_stats is None:
        # Training mode: Calculate stats from the dataframe itself
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")

        global_mean = target_series.mean()
        smoothing_factor = 20

        def get_smoothed_aggregates(group_by_col, prefix):
            temp_df = df_engineered.copy()
            temp_df['violation_count'] = target_series

            agg = temp_df.groupby(group_by_col)['violation_count'].agg(['mean', 'sum', 'std', 'count', 'median'])
            agg['mean'] = (agg['count'] * agg['mean'] + smoothing_factor * global_mean) / (agg['count'] + smoothing_factor)
            agg.columns = [f'{prefix}_{stat}' for stat in ['mean', 'sum', 'std', 'count', 'median']]
            return agg

        street_agg = get_smoothed_aggregates('street_name', 'street')
        violation_agg = get_smoothed_aggregates('violation_description', 'violation')
        
        # FIX: Rename the aggregate count for 'violation' group to avoid collision with the target column name.
        violation_agg.rename(columns={'violation_count': 'violation_type_count'}, inplace=True)
        
        boro_agg = get_smoothed_aggregates('boroname', 'boro')

        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            'global_mean': global_mean
        }
    else:
        # Inference mode: Apply pre-calculated stats
        stats = train_stats

    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    if target_series is not None:
        df_engineered['violation_count'] = target_series

    all_agg_cols = list(stats['street_agg'].columns) + \
                   list(stats['violation_agg'].columns) + \
                   list(stats['boro_agg'].columns)

    fill_values = {}
    global_mean_val = stats['global_mean']
    for col in all_agg_cols:
        if col in df_engineered.columns:
            if col.endswith('_mean'):
                fill_values[col] = global_mean_val
            else:
                fill_values[col] = 0

    df_engineered.fillna(value=fill_values, inplace=True)

    return df_engineered, stats


def main():
    """
    Main function to run the training and prediction pipeline.
    """
    parser = argparse.ArgumentParser(description="Predict NYC parking violations using Ridge Regression.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='(Optional) Path to the test/evaluation data CSV file.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"Loading training data from {args.train_path}...")
    try:
        df_original = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}. Exiting.")
        return

    # --- 2. Validation Split ---
    print("Splitting data into train and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

    # --- 3. Feature Engineering ---
    print("Engineering features for training set...")
    train_featured, train_stats = feature_engineer(train_df)

    print("Engineering features for validation set...")
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    # Define categorical and numerical features for the model
    categorical_features = ['violation_description', 'boroname']

    # The aggregate feature for violation count has been renamed to 'violation_type_count'
    # to avoid collision with the target column 'violation_count'.
    numerical_features = [
        'street_mean', 'street_sum', 'street_std', 'street_count',
        'violation_mean', 'violation_sum', 'violation_std', 'violation_type_count',
        'boro_mean', 'boro_sum', 'boro_std', 'boro_count'
    ]
    numerical_features = [f for f in numerical_features if f in train_featured.columns]

    all_features = numerical_features + categorical_features
    for col in all_features:
        if col not in train_featured.columns:
            raise ValueError(f"Feature column '{col}' not found after feature engineering.")

    # Define the target variable and create feature/target sets
    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- 4. Model Pipeline ---
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

    # --- 5. Training ---
    print("Training the Ridge Regression model...")
    # X_train now contains only feature columns, so we can fit directly.
    ridge_pipeline.fit(X_train, y_train)
    print(f"Training complete. Best alpha found: {ridge_pipeline.named_steps['regressor'].alpha_}")

    # --- 6. Validation ---
    print("Evaluating model on the validation set...")
    val_predictions = ridge_pipeline.predict(X_val)

    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Final Validation Performance: {rmse:.4f}")

    # --- 7. Test Prediction (if applicable) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_original = pd.read_csv(args.test_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}.")
            return

        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()

        has_ground_truth = 'violation_count' in test_df_original.columns
        test_ground_truth = test_df_original['violation_count'] if has_ground_truth else None

        train_keys = set(zip(df_original['Street Name'], df_original['Violation Description']))
        test_keys = set(zip(test_df_original['Street Name'], test_df_original['Violation Description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} key pairs (street_name, violation_type) in test set not present in training data.")
        print("These are handled by using broader group statistics (e.g., violation type-level) and OHE's `handle_unknown` mechanism.")

        print("Engineering features for the test set...")
        test_featured, _ = feature_engineer(test_df_original, train_stats=train_stats)
        X_test = test_featured[all_features]

        print("Generating predictions on the test set...")
        test_predictions = ridge_pipeline.predict(X_test)
        test_predictions[test_predictions < 0] = 0

        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = np.round(test_predictions).astype(int)

        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Successfully created {submission_path}")

        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_predictions))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
