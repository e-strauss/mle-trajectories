
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

    # Standardize column names for easier access
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- Augment with Borough Data ---
    cscl_path = './input/nyc_cscl.csv'
    if os.path.exists(cscl_path):
        try:
            cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
            # Select relevant columns and create a unique mapping from street to borough
            cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
            cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()

            # Merge borough information
            df_engineered['street_name_upper'] = df_engineered['street_name'].str.upper()
            df_engineered = pd.merge(df_engineered, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
            df_engineered['boroname'].fillna('Unknown', inplace=True)
            df_engineered.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
        except Exception:
            # If augmentation fails for any reason, create a placeholder column
            df_engineered['boroname'] = 'Unknown'
    else:
        # If the augmentation file is not present, create a placeholder column
        df_engineered['boroname'] = 'Unknown'

    # The target column might not be present in a keys-only test file
    has_target = 'violation_count' in df_engineered.columns

    # --- Create Aggregate Features ---
    if train_stats is None:
        # Training mode: Calculate stats from the dataframe itself
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")
        
        # Aggregate by street name
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
        street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']
        
        # Aggregate by violation description
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
        violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']
        
        # Aggregate by borough
        boro_agg = df_engineered.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
        boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']
        
        # Store calculated stats for later use
        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
            'boro_agg': boro_agg,
            # Global mean to fill in for completely new entities
            'global_mean': df_engineered['violation_count'].mean()
        }
    else:
        # Inference mode: Apply pre-calculated stats
        stats = train_stats

    # Merge aggregate features onto the dataframe
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, stats['boro_agg'], on='boroname', how='left')

    # Fill NaNs created by left merges.
    # NaNs can occur for unseen keys (e.g., a new street in the test set)
    # or for std deviation where a group has only one member.
    # We fill with 0, assuming no prior information implies a zero effect.
    df_engineered.fillna(0, inplace=True)

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
    # Use GroupShuffleSplit to ensure all records for a given street are either
    # in the training or validation set, preventing data leakage.
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
    # Based on an ablation study, street and borough-related features were removed
    # as they were found to be noisy or uninformative.
    categorical_features = ['violation_description']
    numerical_features = [
        'violation_mean', 'violation_sum', 'violation_std'
    ]
    
    # Ensure all defined features exist in the dataframe
    all_features = numerical_features + categorical_features
    for col in all_features:
        if col not in train_featured.columns:
            raise ValueError(f"Feature column '{col}' not found after feature engineering.")

    # Define the target variable
    target = 'violation_count'
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- 4. Model Pipeline ---
    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though none are expected
    )

    # Define the model pipeline with preprocessing and the RidgeCV regressor.
    # RidgeCV automatically finds the best alpha from the provided list using cross-validation.
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    # --- 5. Training ---
    print("Training the Ridge Regression model...")
    ridge_pipeline.fit(X_train, y_train)
    print(f"Training complete. Best alpha found: {ridge_pipeline.named_steps['regressor'].alpha_}")

    # --- 6. Validation ---
    print("Evaluating model on the validation set...")
    val_predictions = ridge_pipeline.predict(X_val)

    # Post-processing: Ensure predictions are non-negative
    val_predictions[val_predictions < 0] = 0

    # Calculate and report RMSE
    # The 'squared' argument is not available in older versions of scikit-learn.
    # Calculate MSE and take the square root for compatibility.
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f'Final Validation Performance: {rmse:.4f}')

    # --- 7. Test Prediction (if applicable) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_original = pd.read_csv(args.test_path)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}.")
            return
        
        # Keep original keys for the submission file
        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()
        
        # Check for ground truth for scoring
        test_ground_truth = test_df_original.get('violation_count')

        # Report on unseen keys
        train_keys = set(zip(df_original['Street Name'], df_original['Violation Description']))
        test_keys = set(zip(test_df_original['Street Name'], test_df_original['Violation Description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} key pairs (street_name, violation_type) in test set not present in training data.")
        print("These are handled by using broader group statistics (e.g., violation type-level) and OHE's `handle_unknown` mechanism.")
        
        # Apply the same feature engineering and prediction steps
        print("Engineering features for the test set...")
        test_featured, _ = feature_engineer(test_df_original, train_stats=train_stats)
        X_test = test_featured[all_features]

        print("Generating predictions on the test set...")
        test_predictions = ridge_pipeline.predict(X_test)
        test_predictions[test_predictions < 0] = 0
        
        # Create submission file with integer predictions
        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = np.round(test_predictions).astype(int)
        
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Successfully created {submission_path}")

        # If ground truth was provided, score the predictions
        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_predictions))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
