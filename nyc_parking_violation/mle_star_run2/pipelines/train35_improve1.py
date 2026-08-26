
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import argparse
import os
import gc

# Import the LGBMRegressor class to fix the NameError
from lightgbm import LGBMRegressor

def feature_engineering(df, is_train=True):
    """
    Creates features for the model.
    Adds simple features based on string lengths and aggregates.
    """
    print("Performing feature engineering...")
    
    df['street_name_len'] = df['Street Name'].astype(str).apply(len)
    df['violation_desc_len'] = df['Violation Description'].astype(str).apply(len)
    
    # Interaction feature
    df['street_violation_interaction'] = df['Street Name'].astype(str) + "_" + df['Violation Description'].astype(str)

    # On training data, create aggregation features to be applied to both train and test
    if is_train and 'violation_count' in df.columns:
        street_stats = df.groupby('Street Name')['violation_count'].agg(['mean', 'sum', 'std']).reset_index()
        street_stats.columns = ['Street Name', 'street_mean_violations', 'street_sum_violations', 'street_std_violations']
        df = pd.merge(df, street_stats, on='Street Name', how='left')
        
        violation_stats = df.groupby('Violation Description')['violation_count'].agg(['mean', 'sum', 'std']).reset_index()
        violation_stats.columns = ['Violation Description', 'violation_mean_violations', 'violation_sum_violations', 'violation_std_violations']
        df = pd.merge(df, violation_stats, on='Violation Description', how='left')
        
    df.fillna(0, inplace=True)
    
    print("Feature engineering complete.")
    return df

def train_and_predict(train_path, test_path=None):
    """
    Main function to load data, train the model, and generate predictions.
    """
    
    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path)

    # Subsample for memory efficiency if the dataset is large, as per requirements
    if len(df_train) > 500000:
        print(f"Subsampling data from {len(df_train)} to 500000 rows to manage memory.")
        df_train = df_train.sample(n=500000, random_state=42)
        gc.collect()

    # --- Feature Engineering ---
    # Create aggregation features from the full training data before splitting
    street_aggregates = df_train.groupby('Street Name')['violation_count'].agg(['mean', 'sum', 'std']).reset_index()
    street_aggregates.columns = ['Street Name', 'street_mean_violations', 'street_sum_violations', 'street_std_violations']
    
    violation_aggregates = df_train.groupby('Violation Description')['violation_count'].agg(['mean', 'sum', 'std']).reset_index()
    violation_aggregates.columns = ['Violation Description', 'violation_mean_violations', 'violation_sum_violations', 'violation_std_violations']

    # Apply feature engineering to the training set
    df_train['street_name_len'] = df_train['Street Name'].astype(str).apply(len)
    df_train['violation_desc_len'] = df_train['Violation Description'].astype(str).apply(len)
    df_train = pd.merge(df_train, street_aggregates, on='Street Name', how='left')
    df_train = pd.merge(df_train, violation_aggregates, on='Violation Description', how='left')
    df_train.fillna(0, inplace=True)

    # --- Label Encoding ---
    categorical_cols = ['Street Name', 'Violation Description']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        encoders[col] = le

    # --- Model Training ---
    features = [col for col in df_train.columns if col not in ['violation_count']]
    target = 'violation_count'
    
    X = df_train[features]
    y = np.log1p(df_train[target])  # Log transform target for better performance with RMSE

    # Use GroupKFold to prevent data leakage from the same street being in train and validation sets
    kf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df_train))
    models = []

    print("\nStarting model training with 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups=df_train['Street Name'])):
        print(f"--- Fold {fold+1} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # The fix is applied here by using the imported LGBMRegressor
        model = LGBMRegressor(
            random_state=42,
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = np.expm1(val_preds)  # Inverse transform predictions
        models.append(model)
        gc.collect()

    # Clip negative predictions that might result from floating point inaccuracies
    oof_preds[oof_preds < 0] = 0
    final_validation_score = np.sqrt(mean_squared_error(df_train[target], oof_preds))
    print(f"\nFinal Validation Performance: {final_validation_score}")

    # --- Prediction on Test Set ---
    if test_path:
        print(f"\n--- Generating predictions for test file: {test_path} ---")
        df_test = pd.read_csv(test_path)
        
        ground_truth = None
        if 'violation_count' in df_test.columns:
            ground_truth = df_test['violation_count'].copy()
            df_test = df_test.drop(columns=['violation_count'])

        submission_keys = df_test[['Street Name', 'Violation Description']].copy()
        
        # Apply feature engineering to the test set
        df_test['street_name_len'] = df_test['Street Name'].astype(str).apply(len)
        df_test['violation_desc_len'] = df_test['Violation Description'].astype(str).apply(len)
        df_test = pd.merge(df_test, street_aggregates, on='Street Name', how='left')
        df_test = pd.merge(df_test, violation_aggregates, on='Violation Description', how='left')
        df_test.fillna(0, inplace=True)
        
        # Handle unseen categorical values
        unseen_count = 0
        for col in categorical_cols:
            seen_labels = set(encoders[col].classes_)
            unseen_labels = set(df_test[col].astype(str)) - seen_labels
            if unseen_labels:
                unseen_count += len(unseen_labels)
                # Add a label for "unseen"
                encoders[col].classes_ = np.append(encoders[col].classes_, 'unseen')
            
            # Apply transformation, replacing unseen with the new 'unseen' label
            df_test[col] = df_test[col].astype(str).map(lambda s: s if s in seen_labels else 'unseen')
            df_test[col] = encoders[col].transform(df_test[col])

        print(f"Handled {unseen_count} total unseen keys in the test set.")
        
        X_test = df_test[features]

        # Average predictions from all fold models
        test_predictions = np.zeros(len(X_test))
        for model in models:
            preds = model.predict(X_test)
            test_predictions += np.expm1(preds) / len(models)
        
        test_predictions[test_predictions < 0] = 0
        
        # Create submission file
        submission_df = submission_keys
        submission_df['predicted_count'] = test_predictions
        submission_df.to_csv('submission.csv', index=False)
        print("submission.csv created successfully.")

        # Score test set if ground truth is available
        if ground_truth is not None:
            test_rmse = np.sqrt(mean_squared_error(ground_truth, test_predictions))
            print(f"Test RMSE (on {test_path}): {test_rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Parking Violation Prediction Pipeline")
    parser.add_argument('--train-path', type=str, default='./input/violations_per_street_2022.csv',
                        help='Path to the training data file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data file for prediction.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_path):
        print(f"Error: Training file not found at {args.train_path}")
    elif args.test_path and not os.path.exists(args.test_path):
        print(f"Error: Test file not found at {args.test_path}")
    else:
        train_and_predict(args.train_path, args.test_path)
