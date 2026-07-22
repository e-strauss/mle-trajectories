
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

# Load the training data
try:
    # Construct the path relative to the current script's directory
    script_dir = os.path.dirname(__file__)
    train_file_path = os.path.join(script_dir, "input", "train.csv")
    train_df = pd.read_csv(train_file_path)
except FileNotFoundError:
    print(f"Error: train.csv not found. Make sure the 'input' directory and 'train.csv' are in the correct path relative to the script: {train_file_path}")
    # If the file is not found, the script cannot proceed.
    raise

# Separate features (X) and target (y)
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
X = train_df.drop(columns=['Id', 'Cover_Type'])
y = train_df['Cover_Type']

# --- Start of fix for conceptual omission: Explicitly handling categorical features ---
# The problem description specifies that 'Wilderness_Area' and 'Soil_Type' are already
# represented as multiple binary (0/1) columns (e.g., Wilderness_Area1, ..., Wilderness_Area4
# and Soil_Type1, ..., Soil_Type40). These are effectively pre-one-hot encoded features.

# For a RandomForestClassifier, these binary (0/1) features are inherently handled correctly.
# The model will treat them numerically, but splits on values like `feature <= 0.5`
# effectively distinguish between the two categories (0 and 1).
# Therefore, explicit one-hot encoding (e.g., using pd.get_dummies) is NOT required
# for these already binary-encoded columns when using a RandomForest.
# If these were single columns with multiple integer categories (e.g., a single 'Wilderness_Area'
# column with values 1, 2, 3, 4), then one-hot encoding would be necessary to prevent
# the model from assuming an ordinal relationship.
# In this specific dataset, the categorical features are already in a suitable format for the model.
# --- End of fix for conceptual omission ---


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
