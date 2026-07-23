
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# All the provided input data is stored in "./input" directory.
# Load data
try:
    original_train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

n_splits = 3 # As per task description for 3-Fold CV

def evaluate_model_performance(current_train_df, features_to_drop_list):
    """
    Performs 3-Fold Stratified Cross-Validation on the given dataset with a RandomForestClassifier.
    Assumes `current_train_df` already contains all necessary features for the current run.
    """
    X = current_train_df.drop(features_to_drop_list, axis=1)
    y_original = current_train_df['Cover_Type']
    y_transformed = y_original - 1

    # --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
    class_counts = y_transformed.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_transformed[y_transformed.isin(problematic_classes)].index.tolist()
        X_cv = X.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_transformed.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    else:
        X_cv = X
        y_cv = y_transformed

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# Define the base columns to drop in all scenarios, consistent with the original solution
base_drop_cols = ['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect']

# --- Baseline Run ---
print("--- Running Baseline ---")
train_df_baseline = original_train_df.copy()

# Apply all feature engineering steps for the baseline
train_df_baseline['Hillshade_composite'] = (train_df_baseline['Hillshade_9am'] + train_df_baseline['Hillshade_Noon'] + train_df_baseline['Hillshade_3pm']) / 3
train_df_baseline['Elevation_at_Hydrology'] = train_df_baseline['Elevation'] - train_df_baseline['Vertical_Distance_To_Hydrology']
train_df_baseline['Hillshade_Noon_to_3pm_Diff'] = train_df_baseline['Hillshade_Noon'] - train_df_baseline['Hillshade_3pm']
train_df_baseline['Total_Horizontal_Distance'] = np.log1p(np.maximum(0, train_df_baseline['Horizontal_Distance_To_Hydrology'])) + \
                                               np.log1p(np.maximum(0, train_df_baseline['Horizontal_Distance_To_Roadways'])) + \
                                               np.log1p(np.maximum(0, train_df_baseline['Horizontal_Distance_To_Fire_Points']))
train_df_baseline['Slope_x_Vertical_Distance_To_Hydrology'] = train_df_baseline['Slope'] * train_df_baseline['Vertical_Distance_To_Hydrology']

baseline_score = evaluate_model_performance(train_df_baseline, base_drop_cols)
print(f'Baseline Performance: {baseline_score:.4f}')

ablation_results = {}
ablation_results['Baseline'] = baseline_score

# --- Ablation 1: Remove 'Hillshade_Noon_to_3pm_Diff' feature ---
print("\n--- Running Ablation: Remove 'Hillshade_Noon_to_3pm_Diff' ---")
train_df_ablation1 = original_train_df.copy()
train_df_ablation1['Hillshade_composite'] = (train_df_ablation1['Hillshade_9am'] + train_df_ablation1['Hillshade_Noon'] + train_df_ablation1['Hillshade_3pm']) / 3
train_df_ablation1['Elevation_at_Hydrology'] = train_df_ablation1['Elevation'] - train_df_ablation1['Vertical_Distance_To_Hydrology']
# 'Hillshade_Noon_to_3pm_Diff' is not created in this ablation
train_df_ablation1['Total_Horizontal_Distance'] = np.log1p(np.maximum(0, train_df_ablation1['Horizontal_Distance_To_Hydrology'])) + \
                                               np.log1p(np.maximum(0, train_df_ablation1['Horizontal_Distance_To_Roadways'])) + \
                                               np.log1p(np.maximum(0, train_df_ablation1['Horizontal_Distance_To_Fire_Points']))
train_df_ablation1['Slope_x_Vertical_Distance_To_Hydrology'] = train_df_ablation1['Slope'] * train_df_ablation1['Vertical_Distance_To_Hydrology']

score_ablation1 = evaluate_model_performance(train_df_ablation1, base_drop_cols)
print(f"Performance without 'Hillshade_Noon_to_3pm_Diff': {score_ablation1:.4f}")
print(f"Change from Baseline: {score_ablation1 - baseline_score:.4f}")
ablation_results['Hillshade_Noon_to_3pm_Diff'] = score_ablation1

# --- Ablation 2: Remove 'Total_Horizontal_Distance' feature ---
print("\n--- Running Ablation: Remove 'Total_Horizontal_Distance' ---")
train_df_ablation2 = original_train_df.copy()
train_df_ablation2['Hillshade_composite'] = (train_df_ablation2['Hillshade_9am'] + train_df_ablation2['Hillshade_Noon'] + train_df_ablation2['Hillshade_3pm']) / 3
train_df_ablation2['Elevation_at_Hydrology'] = train_df_ablation2['Elevation'] - train_df_ablation2['Vertical_Distance_To_Hydrology']
train_df_ablation2['Hillshade_Noon_to_3pm_Diff'] = train_df_ablation2['Hillshade_Noon'] - train_df_ablation2['Hillshade_3pm']
# 'Total_Horizontal_Distance' is not created in this ablation
train_df_ablation2['Slope_x_Vertical_Distance_To_Hydrology'] = train_df_ablation2['Slope'] * train_df_ablation2['Vertical_Distance_To_Hydrology']

score_ablation2 = evaluate_model_performance(train_df_ablation2, base_drop_cols)
print(f"Performance without 'Total_Horizontal_Distance': {score_ablation2:.4f}")
print(f"Change from Baseline: {score_ablation2 - baseline_score:.4f}")
ablation_results['Total_Horizontal_Distance'] = score_ablation2

# --- Ablation 3: Remove 'Slope_x_Vertical_Distance_To_Hydrology' feature ---
print("\n--- Running Ablation: Remove 'Slope_x_Vertical_Distance_To_Hydrology' ---")
train_df_ablation3 = original_train_df.copy()
train_df_ablation3['Hillshade_composite'] = (train_df_ablation3['Hillshade_9am'] + train_df_ablation3['Hillshade_Noon'] + train_df_ablation3['Hillshade_3pm']) / 3
train_df_ablation3['Elevation_at_Hydrology'] = train_df_ablation3['Elevation'] - train_df_ablation3['Vertical_Distance_To_Hydrology']
train_df_ablation3['Hillshade_Noon_to_3pm_Diff'] = train_df_ablation3['Hillshade_Noon'] - train_df_ablation3['Hillshade_3pm']
train_df_ablation3['Total_Horizontal_Distance'] = np.log1p(np.maximum(0, train_df_ablation3['Horizontal_Distance_To_Hydrology'])) + \
                                               np.log1p(np.maximum(0, train_df_ablation3['Horizontal_Distance_To_Roadways'])) + \
                                               np.log1p(np.maximum(0, train_df_ablation3['Horizontal_Distance_To_Fire_Points']))
# 'Slope_x_Vertical_Distance_To_Hydrology' is not created in this ablation

score_ablation3 = evaluate_model_performance(train_df_ablation3, base_drop_cols)
print(f"Performance without 'Slope_x_Vertical_Distance_To_Hydrology': {score_ablation3:.4f}")
print(f"Change from Baseline: {score_ablation3 - baseline_score:.4f}")
ablation_results['Slope_x_Vertical_Distance_To_Hydrology'] = score_ablation3

# --- Ablation Study Conclusion ---
print("\n--- Ablation Study Conclusion ---")
most_contributing_part = None
max_drop = 0.0

for part_name, score in ablation_results.items():
    if part_name == 'Baseline':
        continue
    # Calculate the drop in performance relative to the baseline
    drop = baseline_score - score
    print(f"Ablation '{part_name}': Baseline - Ablation = {baseline_score:.4f} - {score:.4f} = {drop:.4f}")
    if drop > max_drop:
        max_drop = drop
        most_contributing_part = part_name

if most_contributing_part and max_drop > 0:
    print(f"\nThe part contributing the most to overall performance is: '{most_contributing_part}' with a performance drop of {max_drop:.4f} when removed.")
elif not most_contributing_part:
    print("\nNo single part showed a positive contribution when removed, or all removals led to improvements/no change.")
