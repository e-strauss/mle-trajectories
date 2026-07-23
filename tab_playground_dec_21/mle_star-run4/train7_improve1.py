
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





import numpy as np

# Create a composite 'Hillshade' feature by averaging the three existing 'Hillshade' columns
train_df['Hillshade_composite'] = (train_df['Hillshade_9am'] + train_df['Hillshade_Noon'] + train_df['Hillshade_3pm']) / 3

# Create 'Elevation_at_Hydrology'
train_df['Elevation_at_Hydrology'] = train_df['Elevation'] - train_df['Vertical_Distance_To_Hydrology']

# Create 'Total_Distance_To_Hydrology' as Euclidean distance
train_df['Total_Distance_To_Hydrology'] = np.sqrt(train_df['Horizontal_Distance_To_Hydrology']**2 + train_df['Vertical_Distance_To_Hydrology']**2)

# Create 'Slope_squared' to capture non-linear effects
train_df['Slope_squared'] = train_df['Slope']**2

# Drop original 'Hillshade' columns, original hydrology distance columns, 'Id' and 'Cover_Type'
X = train_df.drop([
    'Id',
    'Cover_Type',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology'
], axis=1)





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


# Initialize StratifiedKFold
# Ensure shuffle=True and random_state for reproducibility
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Model initialization
# Using RandomForestClassifier as a robust and commonly effective classifier for tabular data.
# n_jobs=-1 uses all available CPU cores.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Store fold performances
fold_accuracies = []

# Perform 3-Fold Cross-Validation
print("\nStarting 3-Fold Cross-Validation...")
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
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

# Calculate average performance across all folds
final_validation_score = np.mean(fold_accuracies)

# Print the final validation performance as required
print(f'Final Validation Performance: {final_validation_score}')

# The problem description states: "There is not submission and no test dataset".
# Therefore, no further steps for test data prediction or submission file generation are needed.
