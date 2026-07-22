
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
# The training data is available in the './input' directory.
train_df = pd.read_csv("./input/train.csv")

# 2. Prepare Data
# Features (X): All columns except 'Id' and 'Cover_Type'
# Target (y): 'Cover_Type'
# LightGBM and scikit-learn metrics typically expect 0-indexed class labels for multiclass.
# The 'Cover_Type' values are 1-7, so we subtract 1 to make them 0-6.
X = train_df.drop(['Id', 'Cover_Type'], axis=1)
y = train_df['Cover_Type'] - 1  # Adjust target labels from 1-7 to 0-6

# Define the number of classes (7 in this problem)
n_classes = len(y.unique())

# 3. Evaluation Metric
# Accuracy is a suitable metric for multi-class classification problems,
# especially when class imbalance is not severe, or for a simple initial solution.

# 4. Implement 3-Fold Cross-Validation
# Use StratifiedKFold to ensure that each fold has approximately the same percentage
# of samples of each target class as the complete set.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []

print(f"Starting {n_splits}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize LightGBM Classifier
    # For multi-class classification, set objective='multiclass' and num_class=n_classes.
    # We use a fixed random_state for reproducibility of the model training.
    model_lgbm = LGBMClassifier(objective='multiclass', num_class=n_classes, random_state=42)

    # Train the model on the training data for the current fold
    model_lgbm.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred_val = model_lgbm.predict(X_val)

    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_val, y_pred_val)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

# 5. Calculate Final Performance
# Average the accuracies across all folds to get the final validation score.
final_validation_score = np.mean(fold_accuracies)

# 6. Print the result
print("\nCross-Validation Complete.")
print(f'Final Validation Performance: {final_validation_score}')
