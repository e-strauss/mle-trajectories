
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os # Import the os module for path manipulation

# Load the training data
try:
    # Construct the path relative to the current script's directory
    script_dir = os.path.dirname(__file__)
    train_file_path = os.path.join(script_dir, "input", "train.csv")
    train_df = pd.read_csv(train_file_path)
except FileNotFoundError:
    print(f"Error: train.csv not found. Make sure the 'input' directory and 'train.csv' are in the correct path relative to the script: {train_file_path}")
    # If the file is not found, the script cannot proceed.
    # The subsequent lines will raise an UnboundLocalError if train_df is accessed without being defined.
    # Re-raise the FileNotFoundError to stop execution gracefully.
    raise

# Separate features (X) and target (y)
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
X = train_df.drop(columns=['Id', 'Cover_Type'])
y = train_df['Cover_Type']

# Initialize the model
# RandomForestClassifier is a good choice for this type of tabular data and multi-class classification.
# Using a fixed random_state for reproducibility.
# n_jobs=-1 uses all available processors for parallel processing, speeding up training.
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Set up 3-Fold Cross-Validation
# StratifiedKFold is crucial for classification tasks, especially with potentially imbalanced classes,
# to ensure that each fold has approximately the same percentage of samples of each target class as the complete set.
# Shuffle set to True for random splits, and random_state for reproducibility.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform cross-validation
# The 'accuracy' scoring metric is appropriate for this multi-class classification problem.
# n_jobs=-1 uses all available processors for parallel processing during cross-validation.
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

# Calculate and print the final validation performance
final_validation_score = np.mean(cv_scores)
print(f'Cross-validation scores for each fold: {cv_scores}')
print(f'Final Validation Performance: {final_validation_score}')

# Note: The problem description mentions a test set (565892 observations) for which predictions
# need to be made, but also states "There is not submission and no test dataset" for *this* task.
# Therefore, the script focuses on cross-validation performance on the training data only.
# If a test set were provided and predictions were required, the model would be trained
# on the entire training dataset (X, y) and then used to predict on the test features.
