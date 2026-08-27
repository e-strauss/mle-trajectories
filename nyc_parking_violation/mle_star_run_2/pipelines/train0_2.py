
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# Set a seed for reproducibility across all libraries
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats


def main():
    """
    Main function to run the training and prediction pipeline for an ensemble of Ridge and TabNet.
    """
    parser = argparse.ArgumentParser(description="Predict NYC parking violations using an ensemble of Ridge and TabNet.")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data CSV file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='(Optional) Path to the test/evaluation data CSV file.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"Loading training data from {args.train_path}...")
    df_original = pd.read_csv(args.train_path)

    # --- 2. Validation Split ---
    print("Splitting data into train and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

    # --- 3. Feature Engineering (common for both models) ---
    print("Engineering features for training set...")
    train_featured, train_stats = feature_engineer(train_df)
    
    print("Engineering features for validation set...")
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats)

    target = 'violation_count'
    y_train = train_featured[target]
    y_val = val_featured[target]

    # --- 4. RIDGE MODEL ---
    print("\n--- Training Ridge Model ---")
    
    ridge_cat_features = ['violation_description', 'boroname']
    ridge_num_features = [
        'street_mean', 'street_sum', 'street_std', 'street_key_count',
        'violation_mean', 'violation_sum', 'violation_std',
        'boro_mean', 'boro_sum', 'boro_std'
    ]
    ridge_all_features = ridge_num_features + ridge_cat_features
    
    X_train_ridge = train_featured[ridge_all_features]
    X_val_ridge = val_featured[ridge_all_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ridge_num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ridge_cat_features)
        ],
        remainder='passthrough'
    )

    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    ridge_pipeline.fit(X_train_ridge, y_train)
    print(f"Ridge training complete. Best alpha found: {ridge_pipeline.named_steps['regressor'].alpha_}")
    
    val_predictions_ridge = ridge_pipeline.predict(X_val_ridge)
    val_predictions_ridge[val_predictions_ridge < 0] = 0

    # --- 5. TABNET MODEL ---
    print("\n--- Training TabNet Model ---")
    
    tabnet_cat_features = ['street_name', 'violation_description', 'boroname']
    
    # Label Encode Categorical Features for TabNet
    encoders = {}
    for col in tabnet_cat_features:
        train_featured[col] = train_featured[col].astype(str)
        val_featured[col] = val_featured[col].astype(str)
        le = LabelEncoder()
        train_featured[col] = le.fit_transform(train_featured[col])
        # Transform validation set, handling unseen values later if needed
        val_featured[col] = val_featured[col].map(lambda s: s if s in le.classes_ else '<unknown>')
        le_classes = le.classes_.tolist()
        if '<unknown>' not in le_classes:
            le_classes.append('<unknown>')
        le.classes_ = np.array(le_classes)
        val_featured[col] = le.transform(val_featured[col])
        encoders[col] = le
        
    tabnet_features = ridge_num_features + tabnet_cat_features
    X_train_tabnet = train_featured[tabnet_features]
    y_train_tabnet = train_featured[[target]]
    X_val_tabnet = val_featured[tabnet_features]
    y_val_tabnet = val_featured[[target]]

    # Convert to numpy arrays for TabNet
    X_train_np = X_train_tabnet.values
    y_train_np = y_train_tabnet.values
    X_val_np = X_val_tabnet.values
    y_val_np = y_val_tabnet.values

    # Prepare TabNet parameters
    cat_idxs = [tabnet_features.index(col) for col in tabnet_cat_features]
    cat_dims = [len(encoders[col].classes_) for col in tabnet_cat_features]

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

    val_predictions_tabnet = tabnet.predict(X_val_np)
    val_predictions_tabnet[val_predictions_tabnet < 0] = 0

    # --- 6. ENSEMBLE & FINAL VALIDATION ---
    print("\n--- Ensembling Models and Final Evaluation ---")
    
    # Simple averaging ensemble
    ensemble_val_predictions = (val_predictions_ridge + val_predictions_tabnet.flatten()) / 2.0
    
    rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_predictions))
    print(f'Final Validation Performance: {rmse:.4f}')

    # --- 7. TEST PREDICTION (if applicable) ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        test_df_original = pd.read_csv(args.test_path)
        
        submission_keys = test_df_original[['Street Name', 'Violation Description']].copy()
        test_ground_truth = test_df_original.get('violation_count')

        print("Engineering features for the test set...")
        test_featured, _ = feature_engineer(test_df_original, train_stats=train_stats)
        
        # Ridge predictions
        X_test_ridge = test_featured[ridge_all_features]
        test_predictions_ridge = ridge_pipeline.predict(X_test_ridge)
        test_predictions_ridge[test_predictions_ridge < 0] = 0

        # TabNet predictions
        print("Applying learned encodings to test data for TabNet...")
        for col in tabnet_cat_features:
            test_featured[col] = test_featured[col].astype(str)
            le = encoders[col]
            class_to_int = {c: i for i, c in enumerate(le.classes_)}
            # Use the index for '<unknown>' for any unseen categories
            unknown_val_index = class_to_int.get('<unknown>', len(le.classes_)-1) 
            test_featured[col] = [class_to_int.get(s, unknown_val_index) for s in test_featured[col]]
        
        X_test_tabnet = test_featured[tabnet_features]
        X_test_np = X_test_tabnet.values
        
        test_predictions_tabnet = tabnet.predict(X_test_np)
        test_predictions_tabnet[test_predictions_tabnet < 0] = 0

        # Ensemble test predictions
        ensemble_test_predictions = (test_predictions_ridge + test_predictions_tabnet.flatten()) / 2.0
        
        # Create submission file
        submission_df = submission_keys
        submission_df.columns = ['street_name', 'violation_type']
        submission_df['predicted_count'] = np.round(ensemble_test_predictions).astype(int)
        
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Successfully created {submission_path}")

        if test_ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(test_ground_truth, ensemble_test_predictions))
            print(f"RMSE on provided test set: {test_rmse:.4f}")

if __name__ == '__main__':
    main()
