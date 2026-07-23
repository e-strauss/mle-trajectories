
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Configuration ---
N_SPLITS = 3
RANDOM_STATE = 42
N_ESTIMATORS = 100

def run_experiment(X_experiment, y_experiment, description="Baseline"):
    """
    Runs the RandomForestClassifier with StratifiedKFold cross-validation
    and returns the average accuracy.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    fold_accuracies = []

    # Handle classes with fewer samples than n_splits for StratifiedKFold
    class_counts = y_experiment.value_counts()
    problematic_classes = class_counts[class_counts < N_SPLITS].index.tolist()

    if problematic_classes:
        # Separate problematic classes to avoid StratifiedKFold error
        indices_to_exclude_from_cv = y_experiment[y_experiment.isin(problematic_classes)].index.tolist()
        X_cv = X_experiment.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_experiment.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        
        # If there are problematic classes, we cannot perform CV on them.
        # For simplicity in this ablation study, we exclude them from CV and training.
        # In a real scenario, one might consider alternative strategies like
        # oversampling, undersampling, or custom validation splits.
        if not X_cv.empty:
            print(f"Warning: Excluding classes {problematic_classes} from StratifiedKFold due to too few samples ({N_SPLITS} folds).")
        else:
            print(f"Warning: All samples belong to problematic classes {problematic_classes}. Cannot perform StratifiedKFold.")
            # If all samples are problematic, we can't do CV. Return 0 accuracy.
            return 0.0 
    else:
        X_cv = X_experiment
        y_cv = y_experiment

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    avg_accuracy = np.mean(fold_accuracies)
    print(f"{description} Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy

# --- Load data (shared across all experiments) ---
try:
    train_df_original = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

y_original = train_df_original['Cover_Type']
y_transformed = y_original - 1 # Map target to 0-6 (original 1-7)

print("Starting ablation study...")

# --- BASELINE ---
# Perform the exact feature engineering as provided in the current solution
train_df_baseline = train_df_original.copy()

train_df_baseline['Hillshade_composite'] = (train_df_baseline['Hillshade_9am'] + train_df_baseline['Hillshade_Noon'] + train_df_baseline['Hillshade_3pm']) / 3
train_df_baseline['Euclidean_Distance_To_Hydrology'] = np.sqrt(train_df_baseline['Horizontal_Distance_To_Hydrology']**2 + train_df_baseline['Vertical_Distance_To_Hydrology']**2)
train_df_baseline['Elevation_at_Hydrology'] = train_df_baseline['Elevation'] - train_df_baseline['Vertical_Distance_To_Hydrology']
train_df_baseline['Aspect_sin'] = np.sin(np.deg2rad(train_df_baseline['Aspect']))
train_df_baseline['Aspect_cos'] = np.cos(np.deg2rad(train_df_baseline['Aspect']))
train_df_baseline['Proximity_To_Human_Features'] = train_df_baseline['Horizontal_Distance_To_Roadways'] + train_df_baseline['Horizontal_Distance_To_Fire_Points']

X_baseline = train_df_baseline.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)

baseline_accuracy = run_experiment(X_baseline, y_transformed, "Baseline")

# --- ABLATION 1: Remove 'Proximity_To_Human_Features' ---
train_df_ablation1 = train_df_original.copy()

train_df_ablation1['Hillshade_composite'] = (train_df_ablation1['Hillshade_9am'] + train_df_ablation1['Hillshade_Noon'] + train_df_ablation1['Hillshade_3pm']) / 3
train_df_ablation1['Euclidean_Distance_To_Hydrology'] = np.sqrt(train_df_ablation1['Horizontal_Distance_To_Hydrology']**2 + train_df_ablation1['Vertical_Distance_To_Hydrology']**2)
train_df_ablation1['Elevation_at_Hydrology'] = train_df_ablation1['Elevation'] - train_df_ablation1['Vertical_Distance_To_Hydrology']
train_df_ablation1['Aspect_sin'] = np.sin(np.deg2rad(train_df_ablation1['Aspect']))
train_df_ablation1['Aspect_cos'] = np.cos(np.deg2rad(train_df_ablation1['Aspect']))
# 'Proximity_To_Human_Features' is NOT created here

X_ablation1 = train_df_ablation1.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)
ablation1_accuracy = run_experiment(X_ablation1, y_transformed, "Ablation 1: Removed 'Proximity_To_Human_Features'")


# --- ABLATION 2: Remove 'Euclidean_Distance_To_Hydrology' ---
train_df_ablation2 = train_df_original.copy()

train_df_ablation2['Hillshade_composite'] = (train_df_ablation2['Hillshade_9am'] + train_df_ablation2['Hillshade_Noon'] + train_df_ablation2['Hillshade_3pm']) / 3
# 'Euclidean_Distance_To_Hydrology' is NOT created here
train_df_ablation2['Elevation_at_Hydrology'] = train_df_ablation2['Elevation'] - train_df_ablation2['Vertical_Distance_To_Hydrology']
train_df_ablation2['Aspect_sin'] = np.sin(np.deg2rad(train_df_ablation2['Aspect']))
train_df_ablation2['Aspect_cos'] = np.cos(np.deg2rad(train_df_ablation2['Aspect']))
train_df_ablation2['Proximity_To_Human_Features'] = train_df_ablation2['Horizontal_Distance_To_Roadways'] + train_df_ablation2['Horizontal_Distance_To_Fire_Points']

X_ablation2 = train_df_ablation2.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)
ablation2_accuracy = run_experiment(X_ablation2, y_transformed, "Ablation 2: Removed 'Euclidean_Distance_To_Hydrology'")


# --- ABLATION 3: Revert Aspect feature engineering: keep original 'Aspect' instead of sin/cos transformation ---
train_df_ablation3 = train_df_original.copy()

train_df_ablation3['Hillshade_composite'] = (train_df_ablation3['Hillshade_9am'] + train_df_ablation3['Hillshade_Noon'] + train_df_ablation3['Hillshade_3pm']) / 3
train_df_ablation3['Euclidean_Distance_To_Hydrology'] = np.sqrt(train_df_ablation3['Horizontal_Distance_To_Hydrology']**2 + train_df_ablation3['Vertical_Distance_To_Hydrology']**2)
train_df_ablation3['Elevation_at_Hydrology'] = train_df_ablation3['Elevation'] - train_df_ablation3['Vertical_Distance_To_Hydrology']
# 'Aspect_sin' and 'Aspect_cos' are NOT created here
train_df_ablation3['Proximity_To_Human_Features'] = train_df_ablation3['Horizontal_Distance_To_Roadways'] + train_df_ablation3['Horizontal_Distance_To_Fire_Points']

# Crucially, do NOT drop original 'Aspect' here, and do NOT add 'Aspect_sin'/'Aspect_cos'
# The original 'Aspect' column is kept in X_ablation3
X_ablation3 = train_df_ablation3.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)

ablation3_accuracy = run_experiment(X_ablation3, y_transformed, "Ablation 3: Kept original 'Aspect', removed sin/cos transformation")


# --- Final Conclusion ---
performance_changes = {
    "Removed 'Proximity_To_Human_Features'": baseline_accuracy - ablation1_accuracy,
    "Removed 'Euclidean_Distance_To_Hydrology'": baseline_accuracy - ablation2_accuracy,
    "Kept original 'Aspect', removed sin/cos transformation": baseline_accuracy - ablation3_accuracy
}

print("\n--- Ablation Study Summary ---")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
# Fixed SyntaxError: Use correct string literal for dictionary keys
print(f"Ablation 1 (Removed 'Proximity_To_Human_Features') Accuracy: {ablation1_accuracy:.4f} (Change from Baseline: {performance_changes['Removed \'Proximity_To_Human_Features\'']:.4f})")
print(f"Ablation 2 (Removed 'Euclidean_Distance_To_Hydrology') Accuracy: {ablation2_accuracy:.4f} (Change from Baseline: {performance_changes['Removed \'Euclidean_Distance_To_Hydrology\'']:.4f})")
print(f"Ablation 3 (Kept original 'Aspect', removed sin/cos transformation) Accuracy: {ablation3_accuracy:.4f} (Change from Baseline: {performance_changes['Kept original \'Aspect\', removed sin/cos transformation']:.4f})")

# Determine the most contributing part.
# The 'most contributing' part (positively) is the one whose removal (or modification) leads to the largest drop in performance.
max_drop = 0
most_contributing_part = "None of the ablated parts showed a significant positive contribution."

positive_contributions = {k: v for k, v in performance_changes.items() if v > 0} # Filter for actual drops in performance

if positive_contributions:
    most_contributing_part_name = max(positive_contributions, key=positive_contributions.get)
    max_drop = positive_contributions[most_contributing_part_name]
    most_contributing_part = f"The part whose removal caused the largest drop in performance is '{most_contributing_part_name}'. Its presence in the baseline contributed positively by {max_drop:.4f} to the overall performance."
else:
    # If all ablations either improve or have negligible negative change
    min_change_label = min(performance_changes, key=performance_changes.get)
    min_change_value = performance_changes[min_change_label]
    if min_change_value < 0:
        most_contributing_part = f"All ablated parts either had negligible or negative contributions. The largest improvement ({abs(min_change_value):.4f}) was observed when '{min_change_label}' was performed, suggesting that the baseline inclusion of this part was detrimental."
    else:
        most_contributing_part = "None of the ablated parts showed a significant positive or negative contribution; all changes were negligible."

print(f"\nConclusion: {most_contributing_part}")
print(f"Final Validation Performance: {baseline_accuracy}")
