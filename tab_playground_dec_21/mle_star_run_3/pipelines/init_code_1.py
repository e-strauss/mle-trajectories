
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    print("Error: train.csv not found in the ./input directory. Please ensure the data is correctly placed.")
    train_df = None # Explicitly set to None to ensure subsequent errors are AttributeError

if train_df is None:
    # If train_df could not be loaded, we cannot proceed.
    # The script will naturally terminate with an AttributeError
    # when trying to use train_df in the next lines if not handled further.
    print("Cannot proceed without train.csv. Please place the file in the './input' directory.")
else:
    # Separate features (X) and target (y)
    X = train_df.drop(["Id", "Cover_Type"], axis=1)
    y = train_df["Cover_Type"]

    # Initialize a list to store fold-wise accuracy scores
    fold_accuracies = []

    # Define the number of splits for 3-Fold Cross-Validation
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Starting {n_splits}-Fold Cross-Validation...")

    # Iterate through each fold
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split data into training and validation sets for the current fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize a RandomForestClassifier model
        # n_estimators is chosen as 100 as a good balance for initial exploration,
        # and random_state for reproducibility. n_jobs=-1 utilizes all available CPU cores.
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        # Train the model
        print("Training model...")
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Make predictions on the validation set
        print("Making predictions on the validation set...")
        y_pred = model.predict(X_val)

        # Evaluate the model's performance for the current fold
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    # Calculate the average accuracy across all folds
    final_validation_score = np.mean(fold_accuracies)
    print(f"\nFinal Validation Performance: {final_validation_score:.4f}")

    # The problem description also mentioned a test set for prediction,
    # but since there is no submission and no test dataset provided in the task for prediction,
    # we only perform the cross-validation and report its performance.

