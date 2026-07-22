
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load the training data
# Assuming train.csv is in an 'input' directory relative to the script
train_file_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_file_path)

# Separate features and target
X = train_df.drop(["Id", "Cover_Type"], axis=1)
y = train_df["Cover_Type"]

# Initialize a list to store fold-wise accuracy scores
fold_accuracies = []

# Define 3-Fold Stratified Cross-Validation
# StratifiedKFold is used because the target variable (Cover_Type) is categorical
# and we want to maintain the proportion of each class in each fold.
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Starting 3-Fold Cross-Validation...")

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold + 1}/3 ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize the model
    # RandomForestClassifier is a good choice for this type of dataset
    # n_estimators, max_features, min_samples_leaf are common parameters to tune
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

# Calculate the average accuracy across all folds
final_validation_score = np.mean(fold_accuracies)

# Print the final validation performance
print(f'Final Validation Performance: {final_validation_score}')

