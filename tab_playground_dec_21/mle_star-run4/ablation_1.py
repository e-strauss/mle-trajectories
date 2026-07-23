

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Base path for data
DATA_PATH = './input/train.csv'

# Load data - Encapsulated to be called multiple times for different ablations
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: train.csv not found at {DATA_PATH}. Please ensure the file is present.")
        raise FileNotFoundError(f"train.csv not found at {DATA_PATH}.")

# Function to run the model with specified features
def run_model_ablation(processed_X, y_transformed, ablation_name=""):
    print(f"\n--- Running Ablation: {ablation_name} ---")

    n_splits = 3 # As per task description for 3-Fold CV

    # Handle classes with fewer samples than n_splits for StratifiedKFold
    class_counts = y_transformed.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_transformed[y_transformed.isin(problematic_classes)].index.tolist()
        X_cv = processed_X.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_transformed.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        # print(f"Warning: The following classes have fewer than {n_splits} samples and will be excluded from cross-validation for '{ablation_name}':")
        # for cls, count in y_transformed[y_transformed.isin(problematic_classes)].value_counts().items():
        #     print(f"  - Original Cover_Type: {cls + 1} (transformed: {cls}) has {count} sample(s).")
        # print(f"Total {len(indices_to_exclude_from_cv)} samples excluded from CV. CV will be performed on remaining {len(X_cv)} samples.")
    else:
        X_cv = processed_X
        y_cv = y_transformed
        # print(f"All classes have at least {n_splits} samples for '{ablation_name}'. Cross-validation will be performed on all {len(X_cv)} samples.")

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
        # print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

    final_validation_score = np.mean(fold_accuracies)
    print(f"Final Validation Performance for '{ablation_name}': {final_validation_score:.4f}")
    return final_validation_score


# Dictionary to store results
ablation_results = {}

# --- 1. Baseline Run (Original Solution) ---
print("--- Starting Baseline Run (Original Solution) ---")
base_train_df = load_data()
base_df = base_train_df.copy()

# Transform 'Aspect' into sine and cosine components
base_df['Aspect_sin'] = np.sin(np.radians(base_df['Aspect']))
base_df['Aspect_cos'] = np.cos(np.radians(base_df['Aspect']))

# Create a composite 'Hillshade' feature
base_df['Hillshade_composite'] = (base_df['Hillshade_9am'] + base_df['Hillshade_Noon'] + base_df['Hillshade_3pm']) / 3

# Separate features and target for baseline
X_baseline = base_df.drop(['Id', 'Cover_Type', 'Aspect', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)
y_baseline = base_df['Cover_Type'] - 1 # Map target to 0-6

baseline_score = run_model_ablation(X_baseline, y_baseline, "Baseline (Original Solution)")
ablation_results["Baseline"] = baseline_score


# --- 2. Ablation: Remove Aspect Sine/Cosine Transformation ---
print("\n--- Starting Ablation: Remove Aspect Sine/Cosine Transformation ---")
abl_aspect_df = load_data()

# DO NOT create 'Aspect_sin' and 'Aspect_cos'
# Keep original 'Aspect' column

# Create a composite 'Hillshade' feature (as in baseline)
abl_aspect_df['Hillshade_composite'] = (abl_aspect_df['Hillshade_9am'] + abl_aspect_df['Hillshade_Noon'] + abl_aspect_df['Hillshade_3pm']) / 3

# Separate features and target
# Note: 'Aspect' is now kept instead of its sin/cos components, and the original Hillshade columns are dropped.
X_abl_aspect = abl_aspect_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis=1)
y_abl_aspect = abl_aspect_df['Cover_Type'] - 1

abl_aspect_score = run_model_ablation(X_abl_aspect, y_abl_aspect, "No Aspect Sine/Cosine")
ablation_results["No Aspect Sine/Cosine"] = abl_aspect_score


# --- 3. Ablation: Remove Hillshade Composite Feature ---
print("\n--- Starting Ablation: Remove Hillshade Composite Feature ---")
abl_hillshade_df = load_data()

# Transform 'Aspect' into sine and cosine components (as in baseline)
abl_hillshade_df['Aspect_sin'] = np.sin(np.radians(abl_hillshade_df['Aspect']))
abl_hillshade_df['Aspect_cos'] = np.cos(np.radians(abl_hillshade_df['Aspect']))

# DO NOT create 'Hillshade_composite'
# Keep original 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm' columns

# Separate features and target
# Note: Original 'Hillshade' columns are kept, and 'Aspect' is dropped in favor of its sin/cos components.
X_abl_hillshade = abl_hillshade_df.drop(['Id', 'Cover_Type', 'Aspect'], axis=1)
y_abl_hillshade = abl_hillshade_df['Cover_Type'] - 1

abl_hillshade_score = run_model_ablation(X_abl_hillshade, y_abl_hillshade, "No Hillshade Composite")
ablation_results["No Hillshade Composite"] = abl_hillshade_score


# --- 4. Ablation: Remove Wilderness_Area Features ---
print("\n--- Starting Ablation: Remove Wilderness_Area Features ---")
abl_wilderness_df = load_data()

# Transform 'Aspect' into sine and cosine components (as in baseline)
abl_wilderness_df['Aspect_sin'] = np.sin(np.radians(abl_wilderness_df['Aspect']))
abl_wilderness_df['Aspect_cos'] = np.cos(np.radians(abl_wilderness_df['Aspect']))

# Create a composite 'Hillshade' feature (as in baseline)
abl_wilderness_df['Hillshade_composite'] = (abl_wilderness_df['Hillshade_9am'] + abl_wilderness_df['Hillshade_Noon'] + abl_wilderness_df['Hillshade_3pm']) / 3

# Identify Wilderness_Area columns to drop
wilderness_cols_to_drop = [col for col in abl_wilderness_df.columns if 'Wilderness_Area' in col]

# Separate features and target
# Note: Original 'Aspect' and 'Hillshade' related columns are dropped as per baseline, plus Wilderness_Area columns.
X_abl_wilderness = abl_wilderness_df.drop(
    ['Id', 'Cover_Type', 'Aspect', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'] + wilderness_cols_to_drop,
    axis=1
)
y_abl_wilderness = abl_wilderness_df['Cover_Type'] - 1

abl_wilderness_score = run_model_ablation(X_abl_wilderness, y_abl_wilderness, "No Wilderness_Area Features")
ablation_results["No Wilderness_Area Features"] = abl_wilderness_score


# --- Summarize Ablation Results ---
print("\n--- Ablation Study Summary ---")
for ablation, score in ablation_results.items():
    print(f"{ablation}: {score:.4f}")

# Determine the most contributing part based on performance drop
baseline = ablation_results["Baseline"]
performance_drops = {}
for ablation, score in ablation_results.items():
    if ablation != "Baseline":
        drop = baseline - score
        performance_drops[ablation] = drop
        print(f"Performance drop for '{ablation}': {drop:.4f}")

if performance_drops:
    most_contributing_part = max(performance_drops, key=performance_drops.get)
    max_drop = performance_drops[most_contributing_part]
    print(f"\nBased on this ablation study, the part of the code that contributes the most to the overall performance is: '{most_contributing_part}', with a performance drop of {max_drop:.4f} when removed.")
else:
    print("\nNo ablations performed to determine the most contributing part.")

