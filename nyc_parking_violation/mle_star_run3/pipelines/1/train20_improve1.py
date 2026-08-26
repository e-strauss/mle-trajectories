
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
import os
import sys

def print_memory_usage(df, stage=""):
    """Prints memory usage of a dataframe."""
    mem = df.memory_usage(index=True).sum()
    print(f"Memory usage at stage '{stage}': {mem / 1024**2:.2f} MB")

def preprocess_data(df, fine_data, is_train=False, categorizers=None):
    """
    Preprocesses the data: feature engineering, categorical encoding.
    """
    # Standardize column names
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    
    ground_truth = None
    if not is_train and 'violation_count' in df.columns:
        ground_truth = df['violation_count']

    # --- Feature Augmentation: Merge with fine data ---
    if fine_data is not None:
        df['violation_description'] = df['violation_description'].str.strip()
        df = pd.merge(df, fine_data, left_on='violation_description', right_on='definition', how='left')
        
        fine_cols = ['manhattan_96th_st._&_below', 'all_other_areas']
        for col in fine_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        if 'definition' in df.columns:
            df = df.drop(columns=['definition'])
        if 'code' in df.columns:
            df = df.drop(columns=['code'])

    # --- Categorical Feature Encoding ---
    cat_cols = ['street_name', 'violation_description']
    unseen_counts = {}

    if is_train:
        categorizers = {}
        for col in cat_cols:
            df[col] = df[col].astype('category')
            categorizers[col] = df[col].dtype
        
        for col in cat_cols:
            df[f'{col}_code'] = df[col].cat.codes
    else: 
        if categorizers is None:
            raise ValueError("Categorizers must be provided for test data processing.")
            
        for col in cat_cols:
            df[col] = df[col].astype(categorizers[col])
            unseen_mask = df[col].isna()
            unseen_counts[col] = unseen_mask.sum()
            df[f'{col}_code'] = df[col].cat.codes

    return df, ground_truth, categorizers, unseen_counts

def main():
    parser = argparse.ArgumentParser(description="Predict NYC parking violation counts.")
    parser.add_argument("--train-path", type=str, default="./input/violations_per_street_2022.csv",
                        help="Path to the training data file.")
    parser.add_argument("--test-path", type=str,
                        help="Path to the test data file (optional).")
    args = parser.parse_args()

    # --- Load Auxiliary Data ---
    print("Loading auxiliary fine data...")
    fine_data_path = './input/DOF_Parking_Violation_Codes.csv'
    fine_data = None
    try:
        fine_data = pd.read_csv(fine_data_path)
        fine_data.columns = fine_data.columns.str.replace(' ', '_').str.lower()
        
        for col in ['manhattan_96th_st._&_below', 'all_other_areas']:
             if col in fine_data.columns:
                fine_data[col] = pd.to_numeric(fine_data[col].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
        
        if 'definition' in fine_data.columns:
            fine_data['definition'] = fine_data['definition'].str.strip()
        
    except FileNotFoundError:
        print(f"Warning: Fine data file not found at {fine_data_path}. Skipping feature augmentation.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error processing fine data file: {e}. Skipping feature augmentation.", file=sys.stderr)

    # --- Load and Process Training Data ---
    print(f"Loading training data from {args.train_path}...")
    try:
        train_df_full = pd.read_csv(args.train_path,
                                    dtype={'Street Name': 'category', 'Violation Description': 'category'})
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error loading training data: {e}", file=sys.stderr)
        return

    print_memory_usage(train_df_full, "initial train load")

    # --- Subsampling ---
    sample_size = 1_500_000 
    if len(train_df_full) > sample_size:
        print(f"Subsampling training data from {len(train_df_full)} to {sample_size} rows.")
        train_df = train_df_full.sample(n=sample_size, random_state=42)
        print_memory_usage(train_df, "after subsampling")
    else:
        train_df = train_df

    print("Preprocessing training data...")
    train_df, _, categorizers, _ = preprocess_data(
        train_df, 
        fine_data=fine_data,
        is_train=True
    )
    print_memory_usage(train_df, "after preprocessing train")

    # --- Model Training ---
    feature_cols = [col for col in train_df.columns if '_code' in col or '_below' in col or 'all_other_areas' in col]
    target_col = 'violation_count'
    
    feature_cols = [f for f in feature_cols if f in train_df.columns]
    
    print(f"Using features: {feature_cols}")

    X = train_df[feature_cols]
    y = np.log1p(train_df[target_col])

    # --- Validation Split ---
    print("Splitting data for validation...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=train_df['street_name_code']))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    y_val_original = train_df.iloc[val_idx][target_col]

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

    model = HistGradientBoostingRegressor(random_state=42, l2_regularization=0.1, max_iter=200)
    
    print("Training model...")
    model.fit(X_train, y_train)

    # --- Validation ---
    print("Validating model...")
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds = np.maximum(0, val_preds)

    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    print(f"Final Validation Performance: {rmse}")

    # --- Prediction on Test Set ---
    if args.test_path:
        print(f"\nProcessing test file: {args.test_path}")
        try:
            test_df_orig = pd.read_csv(args.test_path)
            
            test_df, test_ground_truth, _, unseen_counts = preprocess_data(
                test_df_orig.copy(),
                fine_data=fine_data,
                is_train=False,
                categorizers=categorizers
            )

            total_unseen = sum(unseen_counts.values())
            print(f"Found {total_unseen} total key pairs in test set not present in training data.")
            if total_unseen > 0:
                for col, count in unseen_counts.items():
                    if count > 0:
                        print(f"  - Handled {count} unseen values in '{col}' by assigning code -1.")
            
            X_test = test_df[feature_cols]

            print("Generating predictions on test data...")
            test_preds_log = model.predict(X_test)
            test_preds = np.expm1(test_preds_log)
            test_preds = np.maximum(0, test_preds)

            submission_df = pd.DataFrame({
                'street_name': test_df_orig['Street Name'],
                'violation_type': test_df_orig['Violation Description'],
                'predicted_count': test_preds.round()
            })

            submission_df.to_csv("submission.csv", index=False)
            print("Submission file 'submission.csv' created successfully.")

            if test_ground_truth is not None:
                test_rmse = np.sqrt(mean_squared_error(test_ground_truth, test_preds))
                print(f"Test RMSE (on file {args.test_path}): {test_rmse}")

        except FileNotFoundError:
            print(f"Error: Test file not found at {args.test_path}", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred during test set processing: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
