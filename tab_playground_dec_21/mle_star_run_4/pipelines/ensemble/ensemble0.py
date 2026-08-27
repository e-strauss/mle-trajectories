
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# All the provided input data is stored in "./input" directory.
# Load data
try:
    train_df = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")


# Separate features and target


# Create a composite 'Hillshade' feature by averaging the three existing 'Hillshade' columns
train_df['Hillshade_composite'] = (train_df['Hillshade_9am'] + train_df['Hillshade_Noon'] + train_df['Hillshade_3pm']) / 3

# Create 'Elevation_at_Hydrology'
train_df['Elevation_at_Hydrology'] = train_df['Elevation'] - train_df['Vertical_Distance_To_Hydrology']

# Drop 'Hillshade' columns along with 'Id', 'Cover_Type', and 'Aspect'
X = train_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)


y_original = train_df['Cover_Type']

# Map target to 0-6 for sklearn compatibility
# Original: 1-7, Transformed: 0-6
y_transformed = y_original - 1

n_splits = 3 # As per task description for 3-Fold CV

# --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
# StratifiedKFold requires that each class have at least n_splits samples in each fold.
# If a class has fewer samples than n_splits (e.g., class 5 with 1 sample and n_splits=3),
# StratifiedKFold will raise a ValueError.
# To address this, we identify such problematic classes and exclude their samples
# from the dataset used for cross-validation splitting.
# The CV performance will then reflect the model's accuracy on the remaining, properly stratified data.

# Find classes with counts less than n_splits
class_counts = y_transformed.value_counts()
problematic_classes = class_counts[class_counts < n_splits].index.tolist()

if problematic_classes:
    # Collect all indices of samples belonging to problematic classes
    indices_to_exclude_from_cv = y_transformed[y_transformed.isin(problematic_classes)].index.tolist()

    # Create CV subsets by dropping these indices
    X_cv = X.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    y_cv = y_transformed.drop(indices_to_exclude_from_cv).reset_index(drop=True)

    # Print a summary of excluded samples for transparency
    excluded_counts = y_transformed[y_transformed.isin(problematic_classes)].value_counts()
    print(f"Warning: The following classes have fewer than {n_splits} samples and will be excluded from cross-validation:")
    for cls, count in excluded_counts.items():
        # Map back to original class number for clarity in message
        print(f"  - Original Cover_Type: {cls + 1} (transformed: {cls}) has {count} sample(s).")
    print(f"Total {len(indices_to_exclude_from_cv)} samples excluded from CV. CV will be performed on remaining {len(X_cv)} samples.")
else:
    # No problematic classes (all have >= n_splits samples)
    X_cv = X
    y_cv = y_transformed
    print(f"All classes have at least {n_splits} samples. Cross-validation will be performed on all {len(X_cv)} samples.")


# --- Ensemble Plan Implementation ---
# 1. Run Multiple Cross-Validation Iterations: Repeat the entire CV procedure.
# 2. Vary Random Seeds: Use a different random_state for StratifiedKFold in each repetition.
# 3. Collect Final Validation Scores: Store the average accuracy from each 3-fold CV run.
# 4. Meta-Average Performance: Calculate the mean of these collected scores.

n_ensemble_runs = 5 # Number of times to repeat the 3-Fold Cross-Validation

all_ensemble_run_scores = [] # List to store the final_validation_score from each ensemble run

print(f"\nStarting {n_ensemble_runs} repeated 3-Fold Cross-Validation runs...")

for ensemble_run in range(n_ensemble_runs):
    # Vary random_state for StratifiedKFold for each ensemble run
    current_skf_random_state = 42 + ensemble_run
    print(f"\n--- Ensemble Run {ensemble_run + 1}/{n_ensemble_runs} (StratifiedKFold Random State: {current_skf_random_state}) ---")

    # Initialize StratifiedKFold with a varying random_state
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_skf_random_state)

    # Model initialization (random_state for the model remains constant as per original solution)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Store fold performances for the current ensemble run
    fold_accuracies = []

    # Perform 3-Fold Cross-Validation for the current ensemble run
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
        print(f"  Fold {fold+1} Accuracy: {accuracy:.4f}")

    # Calculate average performance across all folds for the current 3-Fold CV run
    current_ensemble_run_score = np.mean(fold_accuracies)
    all_ensemble_run_scores.append(current_ensemble_run_score)
    print(f"Average accuracy for Ensemble Run {ensemble_run + 1}: {current_ensemble_run_score:.4f}")

# Calculate the meta-average performance from all ensemble runs
final_validation_performance = np.mean(all_ensemble_run_scores)

# Print the final meta-average validation performance as required
print(f'\nFinal Validation Performance: {final_validation_performance}')

# The problem description states: "There is not submission and no test dataset".
# Therefore, no further steps for test data prediction or submission file generation are needed.
