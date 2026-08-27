
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Define the path to the dataset
DATA_DIR = './input'
TRAIN_FILE = 'train.csv'

def main():
    """
    Main function to load data, train a RandomForestClassifier with 3-Fold Cross-Validation,
    and print the final validation performance.
    """
    
    # Load the training data
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))

    # Separate features (X) and target (y)
    # Drop 'Id' column as it's not a feature
    X = train_df.drop(['Id', 'Cover_Type'], axis=1)
    y = train_df['Cover_Type']

    # Initialize the model
    # Using a RandomForestClassifier with a reasonable number of estimators
    # and a random_state for reproducibility.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Initialize StratifiedKFold for 3-Fold Cross-Validation
    # StratifiedKFold ensures that each fold has approximately the same percentage of samples of each target class.
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    validation_scores = []

    print(f"Starting {n_splits}-Fold Cross-Validation...")

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        validation_scores.append(accuracy)
        print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

    # Calculate and print the average validation performance
    final_validation_score = np.mean(validation_scores)
    print(f"\nFinal Validation Performance: {final_validation_score:.4f}")

if __name__ == "__main__":
    main()
