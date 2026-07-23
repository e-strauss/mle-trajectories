
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Define the path to the training data
train_file_path = "./input/train.csv"

# Check if the training data file exists
if not os.path.exists(train_file_path):
    print(f"Error: The training data file '{train_file_path}' was not found. Please ensure it is in the './input/' directory.")
    # For this task, we assume the file will be present for execution.
    # If not, the script will naturally terminate with a pandas FileNotFoundError.

# Load the training data
train_df = pd.read_csv(train_file_path)

# Separate features (X) and original target (y_original)
X = train_df.drop(["Id", "Cover_Type"], axis=1)
y_original = train_df["Cover_Type"]

# Transform the target variable from 1-7 to 0-6 for model training
y_transformed = y_original - 1

# Identify and handle the single sample for Cover_Type 5 (which is 4 after transformation)
# The problem statement notes that class 5 has only one sample, which is problematic for cross-validation.
class_5_indices = y_original[y_original == 5].index.tolist()

use_reduced_data_for_cv = False
if len(class_5_indices) == 1:
    # Extract the single class 5 sample
    class_5_X = X.loc[class_5_indices]
    class_5_y = y_transformed.loc[class_5_indices]

    # Create a dataset without the class 5 sample for StratifiedKFold
    X_reduced = X.drop(class_5_indices)
    y_reduced = y_transformed.drop(class_5_indices)
    use_reduced_data_for_cv = True
    print(f"Detected single sample for Cover_Type 5 at original index {class_5_indices[0]}. Handling it specially for Cross-Validation.")
elif len(class_5_indices) == 0:
    print("Warning: Cover_Type 5 not found in the dataset. Proceeding with standard StratifiedKFold on available classes.")
    X_reduced = X
    y_reduced = y_transformed
else:
    print(f"Warning: Cover_Type 5 has {len(class_5_indices)} samples, not 1. Proceeding with standard StratifiedKFold on the full dataset.")
    X_reduced = X
    y_reduced = y_transformed

# Initialize Stratified K-Fold Cross-Validation
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store accuracy scores for each fold
validation_scores = []

# Initialize Out-Of-Fold predictions array. NaNs indicate samples not in any validation set.
oof_preds_transformed = np.full(len(X), np.nan)

# Initialize the classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

if use_reduced_data_for_cv:
    # Perform CV on the reduced dataset (excluding the single class 5 sample)
    for fold, (train_index, val_index) in enumerate(skf.split(X_reduced, y_reduced)):
        # Split data for the current fold
        X_train_fold, X_val_fold = X_reduced.iloc[train_index], X_reduced.iloc[val_index]
        y_train_fold, y_val_fold = y_reduced.iloc[train_index], y_reduced.iloc[val_index]

        # Add the single class 5 sample back to the training set for this fold
        # This ensures the model learns from it, but it's never in the validation set.
        # To ensure a clean, continuous index for the training data (a common practice
        # to avoid subtle issues with non-contiguous indices), we reset the index
        # before and after concatenating. This acts as a robustness improvement.
        X_train_fold = pd.concat([X_train_fold.reset_index(drop=True), class_5_X.reset_index(drop=True)]).reset_index(drop=True)
        y_train_fold = pd.concat([y_train_fold.reset_index(drop=True), class_5_y.reset_index(drop=True)]).reset_index(drop=True)

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Make predictions on the validation set (predictions are in 0-6 space)
        val_preds_0_6 = model.predict(X_val_fold)

        # Transform predictions and true labels back to 1-7 space for scoring, as per instructions
        val_preds_1_7 = val_preds_0_6 + 1
        y_val_fold_1_7 = y_val_fold + 1

        # Calculate and store the accuracy for the current fold
        fold_accuracy = accuracy_score(y_val_fold_1_7, val_preds_1_7)
        validation_scores.append(fold_accuracy)

        # Store OOF predictions
        # Map validation indices from the reduced dataset back to the original full dataset indices
        original_val_indices = y_reduced.iloc[val_index].index
        oof_preds_transformed[original_val_indices] = val_preds_0_6

else:
    # If no special handling for class 5 is needed, perform standard StratifiedKFold on the full dataset
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_transformed)):
        # Split data for the current fold
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y_transformed.iloc[train_index], y_transformed.iloc[val_index]

        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Make predictions on the validation set (predictions are in 0-6 space)
        val_preds_0_6 = model.predict(X_val_fold)

        # Transform predictions and true labels back to 1-7 space for scoring
        val_preds_1_7 = val_preds_0_6 + 1
        y_val_fold_1_7 = y_val_fold + 1

        # Calculate and store the accuracy for the current fold
        fold_accuracy = accuracy_score(y_val_fold_1_7, val_preds_1_7)
        validation_scores.append(fold_accuracy)

        # Store OOF predictions
        oof_preds_transformed[val_index] = val_preds_0_6

# Calculate the final validation performance as the average accuracy across all folds
final_validation_score = np.mean(validation_scores)

# Print the final validation performance in the specified format
print(f"Final Validation Performance: {final_validation_score}")

# The problem statement indicates there is no test dataset or submission required.
# Thus, no code for making predictions on a test set is included.
