
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import argparse
import os
import gc
import sys

def print_log(message):
    """Prints a message to stdout with a timestamp for tracing."""
    print(f"[INFO] {message}", flush=True)

def handle_unseen_labels(series, encoder):
    """
    Transforms a series using a fitted LabelEncoder, handling unseen labels by mapping them to the 'unseen' category.
    """
    known_labels = set(encoder.classes_)
    # Map values not present in the encoder's known classes to the string 'unseen'
    series_with_unseen_category = series.map(lambda s: s if s in known_labels else 'unseen')
    # Now, transform the entire prepared series
    return encoder.transform(series_with_unseen_category)

def main():
    """
    Main function to run the training and prediction pipeline.
    """
    try:
        print_log("Script starting.")

        # --- 1. Argument Parsing ---
        parser = argparse.ArgumentParser(description="Predict NYC Parking Violations.")
        parser.add_argument("--train-path", type=str, default="./input/violations_per_street_2022.csv", help="Path to the training data CSV.")
        parser.add_argument("--test-path", type=str, default=None, help="Optional path to the test data CSV.")
        parser.add_argument("--subsample", type=float, default=None, help="Fraction of data to use for quick runs (e.g., 0.1).")
        args = parser.parse_args()
        print_log(f"Arguments: {args}")

        # --- 2. Data Loading ---
        print_log(f"Loading training data from: {args.train_path}")
        if not os.path.exists(args.train_path):
            print(f"[ERROR] Training file not found at {args.train_path}", file=sys.stderr)
            return

        train_df = pd.read_csv(args.train_path)
        print_log(f"Training data loaded. Initial shape: {train_df.shape}")

        if args.subsample:
            print_log(f"Subsampling data to {args.subsample} fraction.")
            train_df = train_df.sample(frac=args.subsample, random_state=42).reset_index(drop=True)
            print_log(f"New training data shape: {train_df.shape}")

        if 'violation_count' not in train_df.columns:
            print("[ERROR] 'violation_count' column not found in training data.", file=sys.stderr)
            return

        # --- 3. Feature Engineering ---
        print_log("Starting feature engineering.")
        
        categorical_cols = ['Street Name', 'Violation Description']
        encoders = {}

        for col in categorical_cols:
            print_log(f"Encoding column: {col}")
            # Use pd.concat as pd.Series.append is deprecated and removed in pandas >= 2.0
            unique_train_vals = pd.Series(train_df[col].astype(str).unique())
            unseen_val = pd.Series(['unseen'])
            unique_vals = pd.concat([unique_train_vals, unseen_val], ignore_index=True)
            
            le = LabelEncoder().fit(unique_vals)
            encoders[col] = le
            train_df[col] = le.transform(train_df[col].astype(str))

        print_log("Feature engineering complete.")

        # --- 4. Model Training (Cross-Validation) ---
        TARGET = 'violation_count'
        FEATURES = categorical_cols.copy()
        
        print_log(f"Using features: {FEATURES}")
        X = train_df[FEATURES]
        y = np.log1p(train_df[TARGET])  # Log-transform target for better performance with count data

        NFOLDS = 5
        kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(X.shape[0])
        models = []

        print_log(f"Starting training with {NFOLDS}-fold cross-validation.")
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print_log(f"--- Fold {fold+1}/{NFOLDS} ---")
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_val, y_val = X.iloc[val_index], y.iloc[val_index]

            lgb_params = {
                'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
                'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
                'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42 + fold,
                'boosting_type': 'gbdt',
            }

            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(100, verbose=False)])

            val_preds = model.predict(X_val)
            oof_preds[val_index] = val_preds
            models.append(model)
            
            del X_train, y_train, X_val, y_val
            gc.collect()

        print_log("Cross-validation finished.")

        # --- 5. Validation Score ---
        oof_preds_final = np.expm1(oof_preds)
        oof_preds_final[oof_preds_final < 0] = 0

        final_validation_score = np.sqrt(mean_squared_error(train_df[TARGET], oof_preds_final))
        print(f"Final Validation Performance: {final_validation_score}")

        # --- 6. Test Set Prediction ---
        if args.test_path:
            print_log(f"Processing test file: {args.test_path}")
            if not os.path.exists(args.test_path):
                 print(f"[ERROR] Test file not found at {args.test_path}", file=sys.stderr)
            else:
                test_df = pd.read_csv(args.test_path)
                print_log(f"Test data loaded. Shape: {test_df.shape}")
                
                original_test_keys = test_df[['Street Name', 'Violation Description']].copy()
                
                X_test = pd.DataFrame()
                total_unseen_keys = 0
                for col in categorical_cols:
                    # Check for unseen values before transformation
                    unseen_mask = ~test_df[col].astype(str).isin(encoders[col].classes_)
                    total_unseen_keys += unseen_mask.sum()
                    
                    # Apply the corrected handler for unseen labels
                    X_test[col] = handle_unseen_labels(test_df[col].astype(str), encoders[col])
                
                print_log(f"Handled {total_unseen_keys} total unseen key instances in the test set by mapping them to the 'unseen' category.")

                X_test = X_test[FEATURES]
                
                test_preds_agg = np.mean([model.predict(X_test) for model in models], axis=0)

                final_test_preds = np.expm1(test_preds_agg)
                final_test_preds[final_test_preds < 0] = 0

                submission_df = original_test_keys.copy()
                submission_df['predicted_count'] = final_test_preds
                submission_df.to_csv("submission.csv", index=False)
                print_log("submission.csv created.")

                if 'violation_count' in test_df.columns:
                    test_rmse = np.sqrt(mean_squared_error(test_df['violation_count'], final_test_preds))
                    print_log(f"Test RMSE: {test_rmse}")

        print_log("Script finished successfully.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
