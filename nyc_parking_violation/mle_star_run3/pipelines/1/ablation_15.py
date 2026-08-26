
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Data Generation for Standalone Script ---
def create_dummy_data():
    """Creates dummy data files for the ablation study to run."""
    if not os.path.exists('input'):
        os.makedirs('input')

    # Main training data
    train_data = {
        'Street Name': ['MAIN ST', 'OAK AVE', 'MAIN ST', 'PINE ST', 'OAK AVE', 'ELM ST', 'MAPLE RD'] * 10,
        'Violation Description': ['NO PARKING', 'FIRE HYDRANT', 'NO PARKING', 'DOUBLE PARKING', 'NO PARKING', 'FIRE HYDRANT', 'NO STANDING'] * 10,
        'violation_count': np.random.randint(10, 200, 70)
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv('input/violations_per_street_2022.csv', index=False)

    # Camera location data for feature augmentation
    camera_data = {
        'Street Name': ['MAIN ST', 'ELM ST', 'BROADWAY'],
        'Location': ['Intersection A', 'Intersection B', 'Intersection C']
    }
    camera_df = pd.DataFrame(camera_data)
    camera_df.to_csv('input/dot_camera_locations.csv', index=False)
    
# --- Core Components from Original Script ---

TARGET_COL = 'violation_count'
STREET_NAME_COL = 'Street Name'
VIOLATION_DESC_COL = 'Violation Description'
CATEGORICAL_COLS = [STREET_NAME_COL, VIOLATION_DESC_COL]

class TargetEncoder:
    """Target encoder using K-fold cross-validation."""
    def __init__(self, cols_to_encode, n_splits=5, random_state=42):
        self.cols_to_encode = cols_to_encode
        self.n_splits = n_splits
        self.random_state = random_state
        self.encoders = {}
        self.global_means = {}

    def fit_transform(self, df):
        encoded_df = df.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for col in self.cols_to_encode:
            self.global_means[col] = df[TARGET_COL].mean()
            encoded_col_name = f'{col}_encoded'
            encoded_df[encoded_col_name] = np.nan
            for train_idx, val_idx in kf.split(df):
                train_fold = df.iloc[train_idx]
                encoder = train_fold.groupby(col)[TARGET_COL].mean()
                encoded_df.iloc[val_idx, encoded_df.columns.get_loc(encoded_col_name)] = df.iloc[val_idx][col].map(encoder)
            encoded_df[encoded_col_name].fillna(self.global_means[col], inplace=True)
            self.encoders[col] = df.groupby(col)[TARGET_COL].mean()
        return encoded_df

    def transform(self, df):
        transformed_df = df.copy()
        for col in self.cols_to_encode:
            encoded_col_name = f'{col}_encoded'
            transformed_df[encoded_col_name] = transformed_df[col].map(self.encoders.get(col, pd.Series()))
            transformed_df[encoded_col_name].fillna(self.global_means.get(col, 0), inplace=True)
        return transformed_df

def run_experiment(use_camera_feature=True, use_kfold_encoder=True, model_max_iter=500):
    """
    Runs a single training and validation experiment with specified configurations.
    """
    # 1. Load Data
    train_df = pd.read_csv('input/violations_per_street_2022.csv')

    # 2. Feature Engineering
    if use_camera_feature:
        camera_df = pd.read_csv('input/dot_camera_locations.csv')
        camera_streets = set(camera_df[STREET_NAME_COL].str.upper().str.strip())
        train_df['has_camera'] = train_df[STREET_NAME_COL].str.upper().str.strip().isin(camera_streets).astype(int)
    else:
        train_df['has_camera'] = 0

    # 3. Validation Split
    dev_train_df, dev_val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # 4. Target Encoding
    if use_kfold_encoder:
        encoder = TargetEncoder(cols_to_encode=CATEGORICAL_COLS, n_splits=5)
        dev_train_encoded = encoder.fit_transform(dev_train_df)
        dev_val_encoded = encoder.transform(dev_val_df)
    else: # Simplified (leaky) target encoding
        dev_train_encoded = dev_train_df.copy()
        dev_val_encoded = dev_val_df.copy()
        for col in CATEGORICAL_COLS:
            encoding_map = dev_train_encoded.groupby(col)[TARGET_COL].mean()
            dev_train_encoded[f'{col}_encoded'] = dev_train_encoded[col].map(encoding_map)
            dev_val_encoded[f'{col}_encoded'] = dev_val_encoded[col].map(encoding_map)
            # Fill unseen categories in validation with global mean
            global_mean = dev_train_encoded[TARGET_COL].mean()
            dev_val_encoded[f'{col}_encoded'].fillna(global_mean, inplace=True)
            
    # 5. Prepare data for model
    feature_cols = [col for col in dev_train_encoded.columns if col.endswith('_encoded') or col == 'has_camera']
    X_train = dev_train_encoded[feature_cols]
    y_train = dev_train_encoded[TARGET_COL]
    X_val = dev_val_encoded[feature_cols]
    y_val = dev_val_encoded[TARGET_COL]

    # 6. Model Training and Evaluation
    model = HistGradientBoostingRegressor(random_state=42, max_iter=model_max_iter, learning_rate=0.05)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_preds = np.maximum(0, val_preds) # Clip at 0
    
    return np.sqrt(mean_squared_error(y_val, val_preds))

def main():
    """Main function to run the ablation study."""
    create_dummy_data()
    
    results = {}

    # Experiment 1: Baseline
    print("Running Baseline experiment...")
    baseline_rmse = run_experiment(use_camera_feature=True, use_kfold_encoder=True, model_max_iter=500)
    results['Baseline'] = baseline_rmse
    print(f"  - Baseline RMSE: {baseline_rmse:.4f}\n")

    # Ablation 1: Remove the 'has_camera' feature
    print("Running Ablation: No 'has_camera' Feature...")
    no_camera_rmse = run_experiment(use_camera_feature=False, use_kfold_encoder=True, model_max_iter=500)
    results["No 'has_camera' Feature"] = no_camera_rmse
    print(f"  - RMSE without camera feature: {no_camera_rmse:.4f}\n")

    # Ablation 2: Use a simplified, leaky target encoder instead of K-Fold
    print("Running Ablation: Simplified (Leaky) Target Encoder...")
    simple_encoder_rmse = run_experiment(use_camera_feature=True, use_kfold_encoder=False, model_max_iter=500)
    results['Simplified Target Encoder'] = simple_encoder_rmse
    print(f"  - RMSE with simplified encoder: {simple_encoder_rmse:.4f}\n")

    # Ablation 3: Reduce model complexity
    print("Running Ablation: Reduced Model Complexity...")
    simple_model_rmse = run_experiment(use_camera_feature=True, use_kfold_encoder=True, model_max_iter=100)
    results['Reduced Model Complexity'] = simple_model_rmse
    print(f"  - RMSE with reduced model complexity (max_iter=100): {simple_model_rmse:.4f}\n")

    # --- Analysis ---
    impacts = {
        "'has_camera' Feature": results["No 'has_camera' Feature"] - baseline_rmse,
        "K-Fold Target Encoder": results['Simplified Target Encoder'] - baseline_rmse,
        "Model Complexity (max_iter=500)": results['Reduced Model Complexity'] - baseline_rmse
    }

    # Determine the most impactful component
    most_impactful_component = max(impacts, key=lambda k: abs(impacts[k]))
    impact_value = impacts[most_impactful_component]

    print("-" * 50)
    print("Ablation Study Summary:")
    print(f"Baseline RMSE: {results['Baseline']:.4f}")
    for name, impact in impacts.items():
        print(f"  - Impact of removing/changing {name}: {impact:+.4f} RMSE")
    print("-" * 50)

    print(f"The most impactful component is the '{most_impactful_component}'. Changing it alters the RMSE by {impact_value:.4f}.")
    
if __name__ == '__main__':
    main()
