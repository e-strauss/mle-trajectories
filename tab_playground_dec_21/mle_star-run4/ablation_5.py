
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Function to run the model with a given set of features
def run_ablation(X_data, y_data, n_splits=3, model_params=None):
    if model_params is None:
        model_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}

    # Handle classes with fewer samples than n_splits for StratifiedKFold
    class_counts = y_data.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_data[y_data.isin(problematic_classes)].index.tolist()
        X_cv = X_data.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_data.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    else:
        X_cv = X_data
        y_cv = y_data

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = RandomForestClassifier(**model_params)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# Load data
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

y_original = train_df['Cover_Type']
y_transformed = y_original - 1 # Map target to 0-6

# --- Baseline Feature Engineering ---
# This mirrors the provided solution exactly for the baseline
baseline_df = train_df.copy()
baseline_df['Hillshade_composite'] = (baseline_df['Hillshade_9am'] + baseline_df['Hillshade_Noon'] + baseline_df['Hillshade_3pm']) / 3
baseline_df['Euclidean_Distance_To_Hydrology'] = np.sqrt(baseline_df['Horizontal_Distance_To_Hydrology']**2 + baseline_df['Vertical_Distance_To_Hydrology']**2)
baseline_df['Elevation_at_Hydrology'] = baseline_df['Elevation'] - baseline_df['Vertical_Distance_To_Hydrology']

# Drop columns for baseline
X_baseline = baseline_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)

# Run Baseline
print("--- Running Baseline Model ---")
baseline_score = run_ablation(X_baseline, y_transformed)
print(f'Baseline Performance: {baseline_score:.4f}\n')

# --- Ablation 1: Remove 'Euclidean_Distance_To_Hydrology' feature ---
# This feature is present in the baseline. We remove it to test its contribution.
ablation1_df = train_df.copy()
ablation1_df['Hillshade_composite'] = (ablation1_df['Hillshade_9am'] + ablation1_df['Hillshade_Noon'] + ablation1_df['Hillshade_3pm']) / 3
# 'Euclidean_Distance_To_Hydrology' is intentionally NOT created in this ablation
ablation1_df['Elevation_at_Hydrology'] = ablation1_df['Elevation'] - ablation1_df['Vertical_Distance_To_Hydrology']

X_ablation1 = ablation1_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)

print("--- Running Ablation 1: Removing 'Euclidean_Distance_To_Hydrology' feature ---")
ablation1_score = run_ablation(X_ablation1, y_transformed)
print(f"Ablation 1 Performance: {ablation1_score:.4f} (Change vs Baseline: {ablation1_score - baseline_score:.4f})\n")

# --- Ablation 2: Remove 'Elevation_at_Hydrology' feature ---
# This feature is present in the baseline. We remove it to test its contribution.
ablation2_df = train_df.copy()
ablation2_df['Hillshade_composite'] = (ablation2_df['Hillshade_9am'] + ablation2_df['Hillshade_Noon'] + ablation2_df['Hillshade_3pm']) / 3
ablation2_df['Euclidean_Distance_To_Hydrology'] = np.sqrt(ablation2_df['Horizontal_Distance_To_Hydrology']**2 + ablation2_df['Vertical_Distance_To_Hydrology']**2)
# 'Elevation_at_Hydrology' is intentionally NOT created in this ablation

X_ablation2 = ablation2_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)

print("--- Running Ablation 2: Removing 'Elevation_at_Hydrology' feature ---")
ablation2_score = run_ablation(X_ablation2, y_transformed)
print(f"Ablation 2 Performance: {ablation2_score:.4f} (Change vs Baseline: {ablation2_score - baseline_score:.4f})\n")

# --- Ablation 3: Retain original 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm' features (along with composite) ---
# In the baseline, these three original Hillshade features are dropped after 'Hillshade_composite' is created.
# In this ablation, they are *retained* in the feature set 'X' to see if they add value or cause redundancy/noise.
ablation3_df = train_df.copy()
ablation3_df['Hillshade_composite'] = (ablation3_df['Hillshade_9am'] + ablation3_df['Hillshade_Noon'] + ablation3_df['Hillshade_3pm']) / 3
ablation3_df['Euclidean_Distance_To_Hydrology'] = np.sqrt(ablation3_df['Horizontal_Distance_To_Hydrology']**2 + ablation3_df['Vertical_Distance_To_Hydrology']**2)
ablation3_df['Elevation_at_Hydrology'] = ablation3_df['Elevation'] - ablation3_df['Vertical_Distance_To_Hydrology']

# Drop columns for ablation 3 - only 'Id' and 'Cover_Type'.
# Original Hillshade columns ('Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm') are RETAINED.
X_ablation3 = ablation3_df.drop(['Id', 'Cover_Type'], axis=1)

print("--- Running Ablation 3: Retaining original 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm' features (along with composite) ---")
ablation3_score = run_ablation(X_ablation3, y_transformed)
print(f"Ablation 3 Performance: {ablation3_score:.4f} (Change vs Baseline: {ablation3_score - baseline_score:.4f})\n")

# --- Determine the most contributing part ---
# A positive 'contribution' is measured by how much the performance drops if a feature/modification is absent or changed (relative to baseline).
# So, for Ablation 1 and 2, a positive (baseline_score - ablation_score) means the feature's presence contributes positively.
# For Ablation 3, the baseline *drops* the original Hillshade features. Ablation 3 *keeps* them.
# If (baseline_score - ablation3_score) is POSITIVE, it means the *decision to drop* them (as in baseline) was beneficial.
# If (baseline_score - ablation3_score) is NEGATIVE, it means the *decision to keep* them would have been beneficial.

contributions = {
    'Presence of "Euclidean_Distance_To_Hydrology" feature': baseline_score - ablation1_score,
    'Presence of "Elevation_at_Hydrology" feature': baseline_score - ablation2_score,
    'Decision to drop original "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm" features': baseline_score - ablation3_score
}

max_contribution_val = 0
most_contributing_part = "None of the ablated components showed a positive contribution to performance."

for part, value in contributions.items():
    if value > max_contribution_val:
        max_contribution_val = value
        most_contributing_part = part

if max_contribution_val > 0:
    print(f"Based on this ablation study, the part of the code that contributes the most to the overall performance is: {most_contributing_part}, with a positive impact of {max_contribution_val:.4f}.")
else:
    # If no component showed a positive contribution (i.e., its removal or change caused no drop or an increase),
    # we identify the component that caused the largest magnitude change, and interpret its effect.
    absolute_changes = {
        'Euclidean_Distance_To_Hydrology': abs(baseline_score - ablation1_score),
        'Elevation_at_Hydrology': abs(baseline_score - ablation2_score),
        'Hillshade feature handling (dropping vs. retaining originals)': abs(baseline_score - ablation3_score)
    }
    
    most_impactful_component_name = max(absolute_changes, key=absolute_changes.get)
    
    if most_impactful_component_name == 'Euclidean_Distance_To_Hydrology':
        change = baseline_score - ablation1_score
        if change > 0:
            print(f"Based on this ablation study, the '{most_impactful_component_name}' feature contributes positively, as its removal caused a drop of {change:.4f}.")
        else:
            print(f"Based on this ablation study, the '{most_impactful_component_name}' feature has a neutral or negative contribution, as its removal caused an increase of {abs(change):.4f}.")
    elif most_impactful_component_name == 'Elevation_at_Hydrology':
        change = baseline_score - ablation2_score
        if change > 0:
            print(f"Based on this ablation study, the '{most_impactful_component_name}' feature contributes positively, as its removal caused a drop of {change:.4f}.")
        else:
            print(f"Based on this ablation study, the '{most_impactful_component_name}' feature has a neutral or negative contribution, as its removal caused an increase of {abs(change):.4f}.")
    elif most_impactful_component_name == 'Hillshade feature handling (dropping vs. retaining originals)':
        change = baseline_score - ablation3_score
        if change > 0:
            print(f"Based on this ablation study, the *decision to drop* the original Hillshade features (9am, Noon, 3pm) in the baseline contributes positively, as retaining them caused a drop of {change:.4f}.")
        else:
            print(f"Based on this ablation study, the *decision to retain* the original Hillshade features (9am, Noon, 3pm) would have contributed positively or neutrally, as the baseline's choice to drop them resulted in a change of {change:.4f}.")

