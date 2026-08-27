
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

# Suppress warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set a seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def feature_engineer(df, train_stats_df=None):
    """
    Engineers features for the model.

    Args:
        df (pd.DataFrame): The dataframe to engineer features for.
        train_stats_df (pd.DataFrame, optional): The dataframe to calculate statistics from.
                                                  If None, stats are calculated from df itself.
                                                  This is used to apply training set stats to the test set.

    Returns:
        pd.DataFrame: The dataframe with new features.
    """
    df_engineered = df.copy()

    # --- 1. Standardize Column Names ---
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    # --- 2. Augment with Borough Data ---
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
        print("Borough augmentation file not found. Skipping.")
        df_engineered['boroname'] = 'Unknown'

    # --- 3. Create Aggregate Features ---
    # Use the provided train_stats_df for stats if available (for test set), otherwise use df itself (for train set)
    source_df = train_stats_df if train_stats_df is not None else df_engineered

    # Ensure source_df has standardized columns if it's the raw train_df
    if train_stats_df is not None:
        source_df.columns = [c.replace(' ', '_').lower() for c in source_df.columns]

    # If source_df needs borough info (i.e., it's the training set itself being processed for stats)
    if 'boroname' not in source_df.columns:
        source_df_copy = source_df.copy()
        if os.path.exists(cscl_path):
            # This logic is duplicated for when source_df is the training set itself
            cscl = pd.read_csv(cscl_path, on_bad_lines='skip', low_memory=False)
            cscl = cscl[['ST_NAME', 'BORONAME']].drop_duplicates(subset=['ST_NAME'])
            cscl['ST_NAME'] = cscl['ST_NAME'].str.upper()
            source_df_copy['street_name_upper'] = source_df_copy['street_name'].str.upper()
            source_df_copy = pd.merge(source_df_copy, cscl, left_on='street_name_upper', right_on='ST_NAME', how='left')
            source_df_copy['boroname'].fillna('Unknown', inplace=True)
            source_df_copy.drop(columns=['street_name_upper', 'ST_NAME'], inplace=True)
        else:
            source_df_copy['boroname'] = 'Unknown'
        source_df = source_df_copy

    # Aggregates by street_name
    street_agg = source_df.groupby('street_name')['violation_count'].agg(['mean', 'sum', 'std', 'count'])
    street_agg.columns = ['street_mean', 'street_sum', 'street_std', 'street_key_count']

    # Aggregates by violation_description
    violation_agg = source_df.groupby('violation_description')['violation_count'].agg(['mean', 'sum', 'std'])
    violation_agg.columns = ['violation_mean', 'violation_sum', 'violation_std']

    # Aggregates by boroname
    boro_agg = source_df.groupby('boroname')['violation_count'].agg(['mean', 'sum', 'std'])
    boro_agg.columns = ['boro_mean', 'boro_sum', 'boro_std']

    # Merge aggregates
    df_engineered = pd.merge(df_engineered, street_agg, on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, violation_agg, on='violation_description', how='left')
    df_engineered = pd.merge(df_engineered, boro_agg, on='boroname', how='left')

    # Fill NaNs created by left merges (for unseen keys in test data) and from std calc (where count=1)
    df_engineered.fillna(0, inplace=True)

    return df_engineered


def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violations using TabNet.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='(Optional) Path to the test data CSV file for prediction.')
    args, _ = parser.parse_known_args()

    # --- 1. Load and Prepare Training Data ---
    print(f"Loading training data from {args.train_path}...")
    try:
        train_df_original = pd.read_csv(args.train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}. Exiting.")
        return

    print("Engineering features for the training data...")
    train_df_featured = feature_engineer(train_df_original)

    # --- 2. Label Encode Categorical Features ---
    categorical_features = ['street_name', 'violation_description', 'boroname']
    encoders = {}
    for col in categorical_features:
        train_df_featured[col] = train_df_featured[col].astype(str)
        le = LabelEncoder()
        train_df_featured[col] = le.fit_transform(train_df_featured[col])
        encoders[col] = le

    # --- 3. Define Features and Target ---
    TARGET = 'violation_count'
    features = [col for col in train_df_featured.columns if col != TARGET and 'violation_count' not in col]
    
    if TARGET in features:
        features.remove(TARGET)
    
    X = train_df_featured[features]
    y = train_df_featured[[TARGET]]

    # --- 4. Validation Split (Grouped by Street Name) ---
    print("Splitting data for validation...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    # The 'groups' parameter requires the original, un-encoded street names for grouping
    train_idx, val_idx = next(gss.split(X, y, groups=train_df_original['Street Name']))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    # Convert to numpy arrays for TabNet
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_val_np = X_val.values
    y_val_np = y_val.values

    # --- 5. Model Training ---
    print("Training TabNet model...")
    # Prepare TabNet parameters
    cat_idxs = [features.index(col) for col in categorical_features]
    # Add +1 to dims to account for potential unseen categories in the test set
    cat_dims = [len(encoders[col].classes_) + 1 for col in categorical_features]

    tabnet = TabNetRegressor(
        cat_dims=cat_dims,
        cat_idxs=cat_idxs,
        cat_emb_dim=4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax',
        seed=SEED,
        verbose=10
    )

    tabnet.fit(
        X_train=X_train_np, y_train=y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        drop_last=False
    )

    # --- 6. Validation Performance ---
    print("Evaluating model on the hold-out validation set...")
    val_preds = tabnet.predict(X_val_np)
    val_preds[val_preds < 0] = 0  # Ensure predictions are non-negative
    final_validation_score = np.sqrt(mean_squared_error(y_val_np, val_preds))

    print(f'Final Validation Performance: {final_validation_score}')

    # --- 7. Test Prediction (if requested) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_original = pd.read_csv(args.test_path, low_memory=False)
        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}. Skipping prediction.")
            return

        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()
        has_target = 'violation_count' in [c.lower().replace(' ', '_') for c in test_df_original.columns]
        if has_target:
            test_ground_truth = test_df_original['violation_count']

        print("Engineering features for the test data...")
        test_df_featured = feature_engineer(test_df_original, train_stats_df=train_df_original)

        print("Applying learned encodings to test data...")
        for col in categorical_features:
            test_df_featured[col] = test_df_featured[col].astype(str)
            le = encoders[col]
            class_to_int = {c: i for i, c in enumerate(le.classes_)}
            unknown_val_index = len(le.classes_)  # This will be the new index for unseen values
            test_df_featured[col] = [class_to_int.get(s, unknown_val_index) for s in test_df_featured[col]]
        
        X_test = test_df_featured[features]
        X_test_np = X_test.values

        # Report on unseen keys
        train_keys = set(zip(train_df_original['Street Name'], train_df_original['Violation Description']))
        test_keys = set(zip(test_df_original['Street Name'], test_df_original['Violation Description']))
        unseen_keys_count = len(test_keys - train_keys)
        print(f"Found {unseen_keys_count} (street_name, violation_type) pairs in test set not present in training data.")
        print("These pairs were handled by encoding them as a special 'unseen' category and using aggregate features from broader groups.")

        print("Generating predictions on the test set...")
        test_preds = tabnet.predict(X_test_np)
        test_preds[test_preds < 0] = 0
        test_preds = np.round(test_preds).astype(int)

        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = test_preds
        submission_df.to_csv('submission.csv', index=False)
        print("Successfully created submission.csv")

        if has_target:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_preds))
            print(f"RMSE on provided test set: {test_rmse:.4f}")


if __name__ == '__main__':
    main()
