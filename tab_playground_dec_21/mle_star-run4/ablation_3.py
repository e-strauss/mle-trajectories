
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# All the provided input data is stored in "./input" directory.
# Load data
try:
    train_df_original = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

# Prepare the target variable (always the same)
y_transformed = train_df_original['Cover_Type'] - 1

# Store baseline performance and ablation results
baseline_performance = 0
ablation_results = {}

# --- Helper function to run a single experiment ---
def run_experiment(X_features, y_target, experiment_name):
    """
    Runs a 3-Fold Stratified Cross-Validation for a given set of features and target.
    """
    print(f"\n--- Running Experiment: {experiment_name} ---")

    n_splits = 3

    # --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
    class_counts = y_target.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_target[y_target.isin(problematic_classes)].index.tolist()
        X_cv = X_features.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_target.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        # print(f"Warning: {len(indices_to_exclude_from_cv)} samples excluded from CV due to problematic classes for '{experiment_name}'.")
    else:
        X_cv = X_features
        y_cv = y_target
        # print(f"All classes have at least {n_splits} samples. CV will be performed on all {len(X_cv)} samples for '{experiment_name}'.")

    # Initialize StratifiedKFold for reproducibility
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Model initialization (using RandomForestClassifier as in the original solution)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    fold_accuracies = []

    # Perform 3-Fold Cross-Validation
    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)
        # print(f"  Fold {fold+1} Accuracy: {accuracy:.4f}")

    final_validation_score = np.mean(fold_accuracies)
    print(f'  Final Validation Performance: {final_validation_score:.4f}')
    return final_validation_score

# --- Baseline Experiment: With all features from the current `plan_implement_agent_1` ---
# This includes Hillshade composite, Aspect sine/cosine, and Slope-Aspect interactions.
current_train_df_baseline = train_df_original.copy()

current_train_df_baseline['Hillshade_composite'] = (current_train_df_baseline['Hillshade_9am'] + current_train_df_baseline['Hillshade_Noon'] + current_train_df_baseline['Hillshade_3pm']) / 3
current_train_df_baseline['Aspect_rad'] = np.deg2rad(current_train_df_baseline['Aspect'])
current_train_df_baseline['Aspect_sin'] = np.sin(current_train_df_baseline['Aspect_rad'])
current_train_df_baseline['Aspect_cos'] = np.cos(current_train_df_baseline['Aspect_rad'])
current_train_df_baseline['Slope_Aspect_sin'] = current_train_df_baseline['Slope'] * current_train_df_baseline['Aspect_sin']
current_train_df_baseline['Slope_Aspect_cos'] = current_train_df_baseline['Slope'] * current_train_df_baseline['Aspect_cos']

X_baseline = current_train_df_baseline.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect', 'Aspect_rad'], axis=1)

baseline_performance = run_experiment(X_baseline, y_transformed, "Baseline (All current features)")
ablation_results["Baseline"] = baseline_performance

# --- Ablation 1: Remove Slope-Aspect Interaction Features (Slope_Aspect_sin, Slope_Aspect_cos) ---
# Keep Aspect sine/cosine, but remove the interaction features derived from them.
current_train_df_ablation1 = train_df_original.copy()
current_train_df_ablation1['Hillshade_composite'] = (current_train_df_ablation1['Hillshade_9am'] + current_train_df_ablation1['Hillshade_Noon'] + current_train_df_ablation1['Hillshade_3pm']) / 3
current_train_df_ablation1['Aspect_rad'] = np.deg2rad(current_train_df_ablation1['Aspect'])
current_train_df_ablation1['Aspect_sin'] = np.sin(current_train_df_ablation1['Aspect_rad'])
current_train_df_ablation1['Aspect_cos'] = np.cos(current_train_df_ablation1['Aspect_rad'])
# Slope_Aspect_sin and Slope_Aspect_cos are NOT created here.

X_ablation1 = current_train_df_ablation1.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect', 'Aspect_rad'], axis=1)
ablation_results["No Slope-Aspect Interaction Features"] = run_experiment(X_ablation1, y_transformed, "Ablation 1: No Slope-Aspect Interaction Features")

# --- Ablation 2: Remove all Aspect-related features (original Aspect and engineered Aspect_sin, Aspect_cos, Slope_Aspect_sin, Slope_Aspect_cos) ---
# This means neither the original 'Aspect' nor any features derived from it are included.
current_train_df_ablation2 = train_df_original.copy()
current_train_df_ablation2['Hillshade_composite'] = (current_train_df_ablation2['Hillshade_9am'] + current_train_df_ablation2['Hillshade_Noon'] + current_train_df_ablation2['Hillshade_3pm']) / 3
# No Aspect_rad, Aspect_sin, Aspect_cos, Slope_Aspect_sin, Slope_Aspect_cos are created.

X_ablation2 = current_train_df_ablation2.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1) # Drop original Aspect
ablation_results["No Aspect-related Features (original + engineered)"] = run_experiment(X_ablation2, y_transformed, "Ablation 2: No Aspect-related Features")

# --- Ablation 3: Remove Elevation Feature ---
# This means removing 'Elevation' from the fully engineered feature set (X_baseline).
X_ablation3 = X_baseline.copy().drop('Elevation', axis=1)
ablation_results["No Elevation Feature"] = run_experiment(X_ablation3, y_transformed, "Ablation 3: No Elevation Feature")


# --- Summarize Results ---
print("\n--- Ablation Study Summary ---")
for name, score in ablation_results.items():
    if name == "Baseline":
        print(f"Baseline Performance (All current features): {score:.4f}")
    else:
        change = score - baseline_performance
        print(f"{name}: {score:.4f} (Change from Baseline: {change:+.4f})")

# Determine the most impactful feature/component by finding the largest drop in performance
most_impactful_change = 0
most_impactful_component = ""

# Exclude Baseline from finding the most impactful change
impact_scores = {k: v for k, v in ablation_results.items() if k != "Baseline"}

if impact_scores:
    # Find the ablation that resulted in the largest drop in performance (most negative change)
    sorted_impact = sorted(impact_scores.items(), key=lambda item: item[1] - baseline_performance)

    most_impactful_component = sorted_impact[0][0]
    most_impactful_change = sorted_impact[0][1] - baseline_performance

    print(f"\nConclusion: The part of the code that contributes the most to the overall performance is removing '{most_impactful_component}', which resulted in the largest performance drop of {abs(most_impactful_change):.4f}.")
else:
    print("\nNo ablation results to compare.")

