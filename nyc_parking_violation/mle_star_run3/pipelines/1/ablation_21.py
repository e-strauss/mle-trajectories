
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def setup_dummy_data():
    """Creates dummy CSV files for the script to run."""
    os.makedirs("./input", exist_ok=True)
    
    train_data = {
        'Street Name': ['5TH AVE', '5TH AVE', 'BROADWAY', 'BROADWAY', '1ST AVE', 'LEXINGTON AVE', 'LEXINGTON AVE', 'PARK AVE', 'MADISON AVE'],
        'Violation Description': ['NO PARKING-STREET CLEANING', 'FAIL TO DISP. MUNI METER RECPT', 'NO PARKING-STREET CLEANING', 'NO STANDING-DAY/TIME LIMITS', 'DOUBLE PARKING', 'NO PARKING-STREET CLEANING', 'FAIL TO DISP. MUNI METER RECPT', 'NO PARKING-STREET CLEANING', 'BUS LANE VIOLATION'],
        'Violation Count': [150, 200, 120, 180, 90, 110, 130, 250, 80]
    }
    pd.DataFrame(train_data).to_csv("./input/violations_per_street_2022.csv", index=False)

    fine_data = {
        'CODE': [21, 14, 46, 38, 99],
        'DEFINITION': ['NO PARKING-STREET CLEANING', 'NO STANDING-DAY/TIME LIMITS', 'DOUBLE PARKING', 'FAIL TO DISP. MUNI METER RECPT', 'BUS LANE VIOLATION'],
        'Manhattan 96th St. & below': ['$65', '$115', '$115', '$65', '$115'],
        'All Other Areas': ['$45', '$115', '$115', '$35', '$115']
    }
    pd.DataFrame(fine_data).to_csv("./input/DOF_Parking_Violation_Codes.csv", index=False)

def preprocess_data(df, fine_data, is_train=False, categorizers=None):
    """
    Preprocesses the data: feature engineering, categorical encoding.
    """
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    
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
            df[col] = pd.Categorical(df[col], categories=categorizers[col].categories, ordered=False)
            df[f'{col}_code'] = df[col].cat.codes

    return df, categorizers

def train_and_evaluate(use_fine_data, use_subsampling, l2_reg_value):
    """
    A single training and evaluation pipeline run.
    """
    # --- Load Auxiliary Data ---
    fine_data_path = './input/DOF_Parking_Violation_Codes.csv'
    fine_data_df = None
    if use_fine_data:
        try:
            fine_data_df = pd.read_csv(fine_data_path)
            fine_data_df.columns = fine_data_df.columns.str.replace(' ', '_').str.lower()
            
            for col in ['manhattan_96th_st._&_below', 'all_other_areas']:
                 if col in fine_data_df.columns:
                    fine_data_df[col] = pd.to_numeric(fine_data_df[col].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0)
            
            if 'definition' in fine_data_df.columns:
                fine_data_df['definition'] = fine_data_df['definition'].str.strip()
            
        except FileNotFoundError:
            print(f"Warning: Fine data file not found.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error processing fine data file: {e}.", file=sys.stderr)

    # --- Load and Process Training Data ---
    train_df_full = pd.read_csv("./input/violations_per_street_2022.csv")

    # --- Subsampling ---
    if use_subsampling:
        sample_size = 7
        if len(train_df_full) > sample_size:
            train_df = train_df_full.sample(n=sample_size, random_state=42)
        else:
            train_df = train_df_full
    else:
        train_df = train_df_full

    # --- Preprocessing ---
    train_df, categorizers = preprocess_data(
        train_df, 
        fine_data=fine_data_df,
        is_train=True
    )

    # --- Model Training ---
    feature_cols = [col for col in train_df.columns if '_code' in col or '_below' in col or 'all_other_areas' in col]
    target_col = 'violation_count'
    feature_cols = [f for f in feature_cols if f in train_df.columns]
    
    X = train_df[feature_cols]
    y = np.log1p(train_df[target_col])

    # --- Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=train_df['street_name_code']))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    y_val_original = train_df.iloc[val_idx][target_col]

    # --- Model Fitting and Validation ---
    model = HistGradientBoostingRegressor(random_state=42, l2_regularization=l2_reg_value, max_iter=200)
    model.fit(X_train, y_train)

    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    val_preds = np.maximum(0, val_preds)

    rmse = np.sqrt(mean_squared_error(y_val_original, val_preds))
    return rmse

def main():
    """Main function to run the ablation study."""
    setup_dummy_data()

    results = {}

    # --- 1. Baseline Experiment ---
    print("Running: Baseline (All features enabled)")
    baseline_rmse = train_and_evaluate(
        use_fine_data=True,
        use_subsampling=True,
        l2_reg_value=0.1
    )
    results["Baseline"] = baseline_rmse

    # --- 2. Ablation: No Fine Data Augmentation ---
    print("Running: Ablation (No Fine Data Augmentation)")
    no_fine_data_rmse = train_and_evaluate(
        use_fine_data=False,
        use_subsampling=True,
        l2_reg_value=0.1
    )
    results["No Fine Data Augmentation"] = no_fine_data_rmse
    
    # --- 3. Ablation: No Subsampling ---
    print("Running: Ablation (No Subsampling)")
    no_subsampling_rmse = train_and_evaluate(
        use_fine_data=True,
        use_subsampling=False,
        l2_reg_value=0.1
    )
    results["No Subsampling"] = no_subsampling_rmse

    # --- 4. Ablation: No L2 Regularization ---
    print("Running: Ablation (No L2 Regularization)")
    no_l2_reg_rmse = train_and_evaluate(
        use_fine_data=True,
        use_subsampling=True,
        l2_reg_value=0.0
    )
    results["No L2 Regularization"] = no_l2_reg_rmse

    # --- Summary ---
    print("\n--- Ablation Study Results ---")
    impacts = {
        "Fine Data Augmentation": abs(results["No Fine Data Augmentation"] - baseline_rmse),
        "Subsampling": abs(results["No Subsampling"] - baseline_rmse),
        "L2 Regularization": abs(results["No L2 Regularization"] - baseline_rmse)
    }

    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    print(f"Ablation 'No Fine Data Augmentation' RMSE: {results['No Fine Data Augmentation']:.4f} (Impact of removing: {impacts['Fine Data Augmentation']:.4f})")
    print(f"Ablation 'No Subsampling' RMSE: {results['No Subsampling']:.4f} (Impact of removing: {impacts['Subsampling']:.4f})")
    print(f"Ablation 'No L2 Regularization' RMSE: {results['No L2 Regularization']:.4f} (Impact of removing: {impacts['L2 Regularization']:.4f})")
    
    most_impactful_component = max(impacts, key=impacts.get)
    print(f"\nThe most impactful component is '{most_impactful_component}'.")

if __name__ == '__main__':
    main()
