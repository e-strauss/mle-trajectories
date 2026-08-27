
import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

def load_data(path, required_cols=None):
    """
    Safely loads a CSV file from the './input' directory, checking for existence and required columns.
    """
    full_path = os.path.join("./input", path)
    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}", file=sys.stderr)
        return None
    
    try:
        df = pd.read_csv(full_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV file {full_path}: {e}", file=sys.stderr)
        return None

    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in {path}: {missing_cols}", file=sys.stderr)
            return None
            
    return df

def feature_engineer(df, cameras_df, centerlines_df):
    """
    Engineers features for the model by augmenting the main dataframe with external data.
    """
    
    # --- Street Name Normalization ---
    def normalize_street_name(name):
        if not isinstance(name, str):
            return ""
        return name.upper().strip()

    df['street_name_norm'] = df['Street Name'].apply(normalize_street_name)
    
    # FIX: The original code produced a KeyError because 'st_label' does not exist.
    # The correct column is likely 'full_stree'. We check for its existence to make the pipeline robust.
    use_centerline_features = False
    if centerlines_df is not None and not centerlines_df.empty:
        centerline_street_col = 'full_stree'
        if centerline_street_col in centerlines_df.columns:
            centerlines_df['street_name_norm'] = centerlines_df[centerline_street_col].apply(normalize_street_name)
            use_centerline_features = True
        else:
            print(f"Warning: Expected column '{centerline_street_col}' not found in centerlines data. Skipping these features.", file=sys.stderr)

    if cameras_df is not None and not cameras_df.empty:
        cameras_df['street_name_norm'] = cameras_df['street'].apply(normalize_street_name)

    # --- Augmentation 1: Camera Counts ---
    if cameras_df is not None and not cameras_df.empty:
        camera_counts = cameras_df.groupby('street_name_norm').size().reset_index(name='camera_count')
        df = pd.merge(df, camera_counts, on='street_name_norm', how='left')
        df['camera_count'].fillna(0, inplace=True)
    else:
        df['camera_count'] = 0

    # --- Augmentation 2: Street Centerline Features ---
    if use_centerline_features:
        centerline_features = centerlines_df[[
            'street_name_norm', 'borocode', 'st_width', 'bike_lane', 
            'trafdir', 'rw_type'
        ]].copy()
        
        centerline_features['borocode'] = centerline_features['borocode'].astype('category')
        
        def get_mode(x):
            modes = x.mode()
            return modes[0] if not modes.empty else np.nan

        agg_funcs = {
            'st_width': 'mean',
            'borocode': get_mode, 
            'bike_lane': get_mode,
            'trafdir': get_mode,
            'rw_type': get_mode
        }
        
        centerline_agg = centerline_features.groupby('street_name_norm').agg(agg_funcs).reset_index()

        df = pd.merge(df, centerline_agg, on='street_name_norm', how='left')

        # Fill NaNs created by the merge
        for col in ['borocode', 'bike_lane', 'trafdir', 'rw_type']:
            if col in df.columns and (pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object'):
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        if 'st_width' in df.columns:
            if not df['st_width'].isnull().all():
                df['st_width'].fillna(df['st_width'].median(), inplace=True)
            else:
                df['st_width'].fillna(0, inplace=True)
    else:
        # Create placeholder columns if centerline data is missing or unusable
        for col in ['borocode', 'bike_lane', 'trafdir', 'rw_type']:
            df[col] = 'Unknown'
        df['st_width'] = 0
    
    # Final check for any remaining NaNs in feature columns
    for col in ['st_width', 'camera_count', 'borocode', 'bike_lane', 'trafdir', 'rw_type']:
        if col not in df.columns:
            df[col] = 0 if col in ['st_width', 'camera_count'] else 'Unknown'

    df.drop(columns=['street_name_norm'], inplace=True, errors='ignore')
    return df

def main():
    """Main function to run the training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict NYC parking violations.")
    parser.add_argument('--train-path', type=str, default='violations_per_street_2022.csv',
                        help='Path to the training data file.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Optional path to the test data file.')
    args = parser.parse_args()

    # --- 1. Load Data ---
    print("Loading data...")
    train_df = load_data(args.train_path, required_cols=['Street Name', 'Violation Description', 'violation_count'])
    if train_df is None:
        return 1

    red_light_cameras = load_data('red_light_camera_locations.csv')
    speed_cameras = load_data('speed_camera_locations.csv')
    centerlines = load_data('nyc_cscl.csv')
    
    cameras_df = None
    all_camera_streets = []
    if red_light_cameras is not None and 'STREET' in red_light_cameras.columns:
        all_camera_streets.append(red_light_cameras[['STREET']].rename(columns={'STREET': 'street'}))
    if speed_cameras is not None and 'street' in speed_cameras.columns:
        all_camera_streets.append(speed_cameras[['street']])
    
    if all_camera_streets:
        cameras_df = pd.concat(all_camera_streets, ignore_index=True)

    # --- 2. Feature Engineering ---
    print("Engineering features...")
    if len(train_df) > 500000:
        print(f"Subsampling training data from {len(train_df)} to 500000 rows.")
        train_df = train_df.sample(n=500000, random_state=42)
        
    train_featured = feature_engineer(train_df.copy(), cameras_df, centerlines)

    # --- 3. Prepare for Modeling ---
    print("Preparing for model training...")
    
    categorical_features = ['Violation Description', 'borocode', 'bike_lane', 'trafdir', 'rw_type']
    numerical_features = ['camera_count', 'st_width']
    
    for col in categorical_features + numerical_features:
        if col not in train_featured.columns:
            print(f"FATAL: Engineered dataframe is missing column '{col}'. Aborting.", file=sys.stderr)
            return 1
            
    train_featured['log_violation_count'] = np.log1p(train_featured['violation_count'])
    
    X = train_featured[categorical_features + numerical_features]
    y = train_featured['log_violation_count']
    groups = train_featured['Street Name']

    # --- 4. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # --- 5. Create Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough',
        n_jobs=1
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 4, 10)))
    ])

    # --- 6. Train the Model ---
    print("Training the model...")
    model.fit(X_train, y_train)

    # --- 7. Validate the Model ---
    print("Validating the model...")
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds[val_preds < 0] = 0
    
    y_val_orig = np.expm1(y_val)
    
    validation_rmse = np.sqrt(mean_squared_error(y_val_orig, val_preds))
    print(f"Final Validation Performance: {validation_rmse}")

    # --- 8. Full Retraining ---
    print("Retraining model on all available 2022 data...")
    model.fit(X, y)

    # --- 9. Handle Test Data ---
    if args.test_path:
        print(f"Processing test file: {args.test_path}")
        test_df = load_data(args.test_path)
        if test_df is None:
            return 1
            
        has_ground_truth = 'violation_count' in test_df.columns

        test_featured = feature_engineer(test_df.copy(), cameras_df, centerlines)

        train_keys = set(train_df['Street Name'].astype(str) + ' | ' + train_df['Violation Description'].astype(str))
        test_featured['composite_key'] = test_featured['Street Name'].astype(str) + ' | ' + test_featured['Violation Description'].astype(str)
        
        is_seen = test_featured['composite_key'].isin(train_keys)
        seen_df = test_featured[is_seen]
        unseen_df = test_featured[~is_seen]

        print(f"Test set contains {len(seen_df)} seen key pairs and {len(unseen_df)} unseen key pairs.")

        predictions_list = []
        default_pred = np.expm1(y.mean()) # Global default prediction on original scale
        
        if not seen_df.empty:
            seen_preds_log = model.predict(seen_df[categorical_features + numerical_features])
            seen_preds = np.expm1(seen_preds_log)
            seen_preds[seen_preds < 0] = 0
            
            seen_results = seen_df[['Street Name', 'Violation Description']].copy()
            seen_results['predicted_count'] = seen_preds
            predictions_list.append(seen_results)

        if not unseen_df.empty:
            unseen_results = unseen_df[['Street Name', 'Violation Description']].copy()
            unseen_results['predicted_count'] = default_pred
            predictions_list.append(unseen_results)
            
        if not predictions_list:
            print("Warning: No predictions were generated.", file=sys.stderr)
            final_predictions = pd.DataFrame(columns=['street_name', 'violation_type', 'predicted_count'])
        else:
            predictions = pd.concat(predictions_list, ignore_index=True)
            # Ensure final predictions are merged back in the original test file order
            final_predictions = pd.merge(test_df[['Street Name', 'Violation Description']], predictions,
                                         on=['Street Name', 'Violation Description'], how='left')
            # Fill any potential NaNs from merge with the default value
            final_predictions['predicted_count'].fillna(default_pred, inplace=True)
            final_predictions['predicted_count'] = final_predictions['predicted_count'].round().astype(int)
            final_predictions.rename(columns={'Street Name': 'street_name', 'Violation Description': 'violation_type'}, inplace=True)

        submission_path = 'submission.csv'
        final_predictions.to_csv(submission_path, index=False)
        print(f"Submission file created at {submission_path}")

        if has_ground_truth:
            # Ensure alignment before scoring
            merged_for_scoring = pd.merge(test_df, final_predictions, left_on=['Street Name', 'Violation Description'], right_on=['street_name', 'violation_type'])
            test_rmse = np.sqrt(mean_squared_error(merged_for_scoring['violation_count'], merged_for_scoring['predicted_count']))
            print(f"Test RMSE (on {args.test_path}): {test_rmse}")

    return 0

if __name__ == "__main__":
    main()
