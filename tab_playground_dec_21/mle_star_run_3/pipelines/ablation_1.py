
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def run_analysis():
    # Load the training data
    try:
        train_df = pd.read_csv("./input/train.csv")
    except FileNotFoundError:
        print("Error: train.csv not found. Please ensure the file is in the ./input directory.")
        return # Exit the function if the file is not found

    # Separate features (X) and target (y)
    # Drop the 'Id' column as it's not a feature
    X = train_df.drop(["Id", "Cover_Type"], axis=1)
    y = train_df["Cover_Type"]

    # Initialize a list to store fold-wise accuracies
    fold_accuracies = []

    # Initialize StratifiedKFold for 3-Fold CV
    # We use a fixed random_state for reproducibility
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Starting 3-Fold Cross-Validation...")

    # Iterate over each fold
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} ---")

        # Split data into training and validation sets for the current fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize the classifier (using RandomForestClassifier as a robust baseline)
        # Parameters can be tuned for better performance
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate and store the accuracy for the current fold
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

    # Calculate the average validation performance across all folds
    final_validation_score = np.mean(fold_accuracies)

    # Print the final validation performance
    print(f"\nFinal Validation Performance: {final_validation_score:.4f}")

if __name__ == "__main__":
    run_analysis()
