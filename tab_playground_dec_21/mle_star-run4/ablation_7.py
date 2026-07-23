
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Configuration for reproducibility ---
N_SPLITS = 3
RANDOM_STATE = 42
N_ESTIMATORS = 100

def calculate_accuracy(X_param, y_param, n_splits=N_SPLITS, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS):
    """
    Performs StratifiedKFold Cross-Validation and returns the average accuracy.
    Handles problematic classes for StratifiedKFold.
    """
    # Map target to 0-6 for sklearn compatibility if not already done
    if y_param.min() == 1:
        y_param = y_param - 1

    # --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
    class_counts = y_param.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_param[y_param.isin(problematic_classes)].index.tolist()
        X_cv = X_param.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_param.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    else:
        X_cv = X_param
        y_cv = y_param

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# Load data (ensure this is done only once)
try:
    original_train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

print("Starting ablation study...")

# --- Baseline Performance ---
print("\n--- Baseline Model ---")
train_df_baseline = original_train_df.copy()

# Feature Engineering for Baseline
train_df_baseline['Hillshade_composite'] = (train_df_baseline['Hillshade_9am'] + train_df_baseline['Hillshade_Noon'] + train_df_baseline['Hillshade_3pm']) / 3
train_df_baseline['Elevation_at_Hydrology'] = train_df_baseline['Elevation'] - train_df_baseline['Vertical_Distance_To_Hydrology']
train_df_baseline['Aspect_sin'] = np.sin(np.deg2rad(train_df_baseline['Aspect']))
train_df_baseline['Aspect_cos'] = np.cos(np.deg2rad(train_df_baseline['Aspect']))
train_df_baseline['Euclidean_Distance_To_Hydrology'] = np.sqrt(
    train_df_baseline['Horizontal_Distance_To_Hydrology']**2 + train_df_baseline['Vertical_Distance_To_Hydrology']**2
)
train_df_baseline['Slope_x_Euclidean_Distance_To_Hydrology'] = train_df_baseline['Slope'] * train_df_baseline['Euclidean_Distance_To_Hydrology']

# Define features to drop for X (Original Aspect is retained in X in this solution)
features_to_drop_for_X_baseline = ['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
X_baseline = train_df_baseline.drop(features_to_drop_for_X_baseline, axis=1)
y_baseline = train_df_baseline['Cover_Type']

baseline_accuracy = calculate_accuracy(X_baseline, y_baseline)
print(f'Baseline Validation Performance: {baseline_accuracy:.4f}')

# Store results
results = {
    'Baseline': baseline_accuracy
}

# --- Ablation 1: Remove 'Slope_x_Euclidean_Distance_To_Hydrology' feature ---
print("\n--- Ablation 1: Remove 'Slope_x_Euclidean_Distance_To_Hydrology' ---")
train_df_ablation1 = original_train_df.copy()

train_df_ablation1['Hillshade_composite'] = (train_df_ablation1['Hillshade_9am'] + train_df_ablation1['Hillshade_Noon'] + train_df_ablation1['Hillshade_3pm']) / 3
train_df_ablation1['Elevation_at_Hydrology'] = train_df_ablation1['Elevation'] - train_df_ablation1['Vertical_Distance_To_Hydrology']
train_df_ablation1['Aspect_sin'] = np.sin(np.deg2rad(train_df_ablation1['Aspect']))
train_df_ablation1['Aspect_cos'] = np.cos(np.deg2rad(train_df_ablation1['Aspect']))
train_df_ablation1['Euclidean_Distance_To_Hydrology'] = np.sqrt(
    train_df_ablation1['Horizontal_Distance_To_Hydrology']**2 + train_df_ablation1['Vertical_Distance_To_Hydrology']**2
)
# 'Slope_x_Euclidean_Distance_To_Hydrology' is NOT created here

features_to_drop_for_X_ablation1 = ['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
X_ablation1 = train_df_ablation1.drop(features_to_drop_for_X_ablation1, axis=1)
y_ablation1 = train_df_ablation1['Cover_Type']

ablation1_accuracy = calculate_accuracy(X_ablation1, y_ablation1)
print(f"Ablation 1 (No 'Slope_x_Euclidean_Distance_To_Hydrology') Performance: {ablation1_accuracy:.4f}")
print(f"Change from Baseline: {ablation1_accuracy - baseline_accuracy:.4f}")
results['No Slope_x_Euclidean_Distance_To_Hydrology'] = ablation1_accuracy


# --- Ablation 2: Remove 'Aspect', 'Aspect_sin', and 'Aspect_cos' ---
print("\n--- Ablation 2: Remove 'Aspect', 'Aspect_sin', and 'Aspect_cos' ---")
train_df_ablation2 = original_train_df.copy()

train_df_ablation2['Hillshade_composite'] = (train_df_ablation2['Hillshade_9am'] + train_df_ablation2['Hillshade_Noon'] + train_df_ablation2['Hillshade_3pm']) / 3
train_df_ablation2['Elevation_at_Hydrology'] = train_df_ablation2['Elevation'] - train_df_ablation2['Vertical_Distance_To_Hydrology']
# 'Aspect_sin' and 'Aspect_cos' are NOT created here
train_df_ablation2['Euclidean_Distance_To_Hydrology'] = np.sqrt(
    train_df_ablation2['Horizontal_Distance_To_Hydrology']**2 + train_df_ablation2['Vertical_Distance_To_Hydrology']**2
)
train_df_ablation2['Slope_x_Euclidean_Distance_To_Hydrology'] = train_df_ablation2['Slope'] * train_df_ablation2['Euclidean_Distance_To_Hydrology']

# Drop original 'Aspect' along with other standard drops
features_to_drop_for_X_ablation2 = ['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect']
X_ablation2 = train_df_ablation2.drop(features_to_drop_for_X_ablation2, axis=1)
y_ablation2 = train_df_ablation2['Cover_Type']

ablation2_accuracy = calculate_accuracy(X_ablation2, y_ablation2)
print(f"Ablation 2 (No Aspect or Aspect_sin/cos) Performance: {ablation2_accuracy:.4f}")
print(f"Change from Baseline: {ablation2_accuracy - baseline_accuracy:.4f}")
results['No Aspect or Aspect_sin/cos'] = ablation2_accuracy


# --- Ablation 3: Remove 'Euclidean_Distance_To_Hydrology' feature (and its dependent feature) ---
print("\n--- Ablation 3: Remove 'Euclidean_Distance_To_Hydrology' (and derived 'Slope_x_Euclidean_Distance_To_Hydrology') ---")
train_df_ablation3 = original_train_df.copy()

train_df_ablation3['Hillshade_composite'] = (train_df_ablation3['Hillshade_9am'] + train_df_ablation3['Hillshade_Noon'] + train_df_ablation3['Hillshade_3pm']) / 3
train_df_ablation3['Elevation_at_Hydrology'] = train_df_ablation3['Elevation'] - train_df_ablation3['Vertical_Distance_To_Hydrology']
train_df_ablation3['Aspect_sin'] = np.sin(np.deg2rad(train_df_ablation3['Aspect']))
train_df_ablation3['Aspect_cos'] = np.cos(np.deg2rad(train_df_ablation3['Aspect']))
# 'Euclidean_Distance_To_Hydrology' is NOT created here
# 'Slope_x_Euclidean_Distance_To_Hydrology' is NOT created here as it depends on 'Euclidean_Distance_To_Hydrology'

features_to_drop_for_X_ablation3 = ['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
X_ablation3 = train_df_ablation3.drop(features_to_drop_for_X_ablation3, axis=1)
y_ablation3 = train_df_ablation3['Cover_Type']

ablation3_accuracy = calculate_accuracy(X_ablation3, y_ablation3)
print(f"Ablation 3 (No 'Euclidean_Distance_To_Hydrology' and derived 'Slope_x_Euclidean_Distance_To_Hydrology') Performance: {ablation3_accuracy:.4f}")
print(f"Change from Baseline: {ablation3_accuracy - baseline_accuracy:.4f}")
results['No Euclidean_Distance_To_Hydrology (and derived Slope_x_Euclidean_Distance_To_Hydrology)'] = ablation3_accuracy


# --- Conclusion ---
print("\n--- Ablation Study Conclusion ---")
performance_changes = {}
for name, accuracy in results.items():
    if name != 'Baseline':
        change = accuracy - baseline_accuracy
        performance_changes[name] = change

most_contributing = None
largest_drop = 0
for name, change in performance_changes.items():
    if change < largest_drop: # Looking for the largest negative change (largest drop in accuracy)
        largest_drop = change
        most_contributing = name

if most_contributing and largest_drop < 0:
    print(f"\nThe part of the code that contributes the most to the overall performance (among those ablated) is: '{most_contributing}'. Its removal resulted in a performance drop of {-largest_drop:.4f}.")
elif all(change >= 0 for change in performance_changes.values()):
    print("\nAmong the ablated parts, none showed a significant positive contribution (or their removal improved performance). The baseline is strong, or removed features were detrimental.")
else:
    print("\nCould not determine the most contributing part from the observed changes.")

