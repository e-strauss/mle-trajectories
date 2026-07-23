

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Configuration for reproducibility and CV ---
N_SPLITS = 3  # As per task description for 3-Fold CV
RANDOM_STATE = 42

# --- Load data ---
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

# --- Feature Engineering (common for all experiments) ---
# Create a composite 'Hillshade' feature by averaging the three existing 'Hillshade' columns
train_df['Hillshade_composite'] = (train_df['Hillshade_9am'] + train_df['Hillshade_Noon'] + train_df['Hillshade_3pm']) / 3

# Separate features and target, dropping 'Id', 'Cover_Type', and original 'Hillshade' columns
X_base = train_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)
y_original = train_df['Cover_Type']

# Map target to 0-6 for sklearn compatibility (Original: 1-7, Transformed: 0-6)
y_transformed = y_original - 1

# --- Function to run cross-validation for a given feature set and model configuration ---
def run_cv_experiment(X_features, y_target, model_config):
    """
    Runs Stratified K-Fold Cross-Validation for a given feature set and model configuration.
    Handles problematic classes for StratifiedKFold.
    """
    # --- Handle classes with fewer samples than N_SPLITS for StratifiedKFold ---
    # StratifiedKFold requires each class to have at least N_SPLITS samples.
    class_counts = y_target.value_counts()
    problematic_classes = class_counts[class_counts < N_SPLITS].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_target[y_target.isin(problematic_classes)].index.tolist()
        X_cv = X_features.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_target.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        # Note: In a full ablation study, this warning might be printed once for the baseline.
        # For concise output of ablations, we suppress repeated warnings here.
    else:
        X_cv = X_features
        y_cv = y_target

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Model initialization
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **model_config)

    # Store fold performances
    fold_accuracies = []

    # Perform Cross-Validation
    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the validation set
        y_pred_val = model.predict(X_val)

        # Calculate accuracy for the current fold
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    # Return average performance across all folds
    return np.mean(fold_accuracies)

# --- Baseline Experiment (using original solution's features and model params) ---
print("--- Running Baseline Experiment ---")
baseline_model_params = {'n_estimators': 100}  # Default model parameters from the original solution
baseline_accuracy = run_cv_experiment(X_base, y_transformed, baseline_model_params)
print(f'Baseline Performance: {baseline_accuracy:.4f}')
print("-" * 40)

# --- Ablation 1: Remove 'Horizontal_Distance_To_Fire_Points' feature ---
print("\n--- Running Ablation 1: Removed 'Horizontal_Distance_To_Fire_Points' ---")
X_ablation1 = X_base.drop(['Horizontal_Distance_To_Fire_Points'], axis=1)
ablation1_accuracy = run_cv_experiment(X_ablation1, y_transformed, baseline_model_params)
print(f"Ablation 1 Performance: {ablation1_accuracy:.4f} (Change: {ablation1_accuracy - baseline_accuracy:.4f})")
print("-" * 40)

# --- Ablation 2: Remove 'Slope' feature ---
print("\n--- Running Ablation 2: Removed 'Slope' ---")
X_ablation2 = X_base.drop(['Slope'], axis=1)
ablation2_accuracy = run_cv_experiment(X_ablation2, y_transformed, baseline_model_params)
print(f"Ablation 2 Performance: {ablation2_accuracy:.4f} (Change: {ablation2_accuracy - baseline_accuracy:.4f})")
print("-" * 40)

# --- Ablation 3: Reduce RandomForestClassifier's 'max_depth' to 10 ---
print("\n--- Running Ablation 3: RandomForestClassifier with max_depth=10 ---")
# The default max_depth is None, allowing trees to grow fully.
# Limiting it reduces model complexity.
ablation3_model_params = {'n_estimators': 100, 'max_depth': 10}
ablation3_accuracy = run_cv_experiment(X_base, y_transformed, ablation3_model_params)
print(f"Ablation 3 Performance: {ablation3_accuracy:.4f} (Change: {ablation3_accuracy - baseline_accuracy:.4f})")
print("-" * 40)

# --- Ablation Study Conclusion ---
print("\n--- Ablation Study Conclusion ---")

# Collect performance changes relative to baseline for analysis
changes_from_baseline = {
    "Horizontal_Distance_To_Fire_Points_removal_impact": baseline_accuracy - ablation1_accuracy,
    "Slope_removal_impact": baseline_accuracy - ablation2_accuracy,
    "Original_RF_max_depth_impact": baseline_accuracy - ablation3_accuracy # if positive, original unlimited depth was better
}

# Find the modification that resulted in the largest performance drop (most positive 'change' value)
largest_drop_component_key = None
max_drop_value = -float('inf')

for component_key, drop_value in changes_from_baseline.items():
    if drop_value > max_drop_value:
        max_drop_value = drop_value
        largest_drop_component_key = component_key

# Map the internal component keys to user-friendly descriptions for the conclusion
component_description_map = {
    "Horizontal_Distance_To_Fire_Points_removal_impact": "The 'Horizontal_Distance_To_Fire_Points' feature",
    "Slope_removal_impact": "The 'Slope' feature",
    "Original_RF_max_depth_impact": "The default (unlimited) 'max_depth' parameter of RandomForestClassifier"
}

if max_drop_value > 0.0001:  # Consider a meaningful drop (e.g., greater than 0.0001)
    contributing_part_desc = component_description_map.get(largest_drop_component_key, largest_drop_component_key)
    print(f"The part that contributes the most to the overall performance (i.e., its removal or modification caused the largest performance drop) is: {contributing_part_desc}.")
    print(f"This resulted in a performance drop of {max_drop_value:.4f}.")
elif max_drop_value < -0.0001:  # A negative drop means an improvement
    # Find the ablation that resulted in the highest accuracy among all runs (baseline included)
    all_accuracies = {
        "Baseline": baseline_accuracy,
        "Ablation 1 (Removed Horizontal_Distance_To_Fire_Points)": ablation1_accuracy,
        "Ablation 2 (Removed Slope)": ablation2_accuracy,
        "Ablation 3 (Reduced RF max_depth to 10)": ablation3_accuracy
    }
    best_config_name = max(all_accuracies, key=all_accuracies.get)
    best_accuracy_value = all_accuracies[best_config_name]

    print(f"Interestingly, none of the specific ablations caused a significant drop. The best performance was achieved by: '{best_config_name}' with an accuracy of {best_accuracy_value:.4f}.")
    print("This suggests that the original component or parameter setting related to this ablation might be suboptimal, as removing/modifying it led to higher accuracy.")
else:
    print("All ablations had a negligible impact on performance.")

