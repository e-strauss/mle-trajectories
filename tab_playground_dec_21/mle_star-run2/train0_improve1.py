
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train_and_evaluate_model():
    """
    Trains a RandomForestClassifier model using 3-Fold Stratified Cross-Validation
    and evaluates its performance.
    """
    input_dir = "./input"
    train_file_path = os.path.join(input_dir, "train.csv")

    try:
        train_df = pd.read_csv(train_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{train_file_path}' was not found. "
                                "Please ensure 'train.csv' is in the './input' directory.")

    # Drop the 'Id' column as it's not a feature
    train_df = train_df.drop("Id", axis=1)

    # Separate features (X) and target (y)
    X = train_df.drop("Cover_Type", axis=1)
    y = train_df["Cover_Type"]

    # Initialize the model
    # Using a RandomForestClassifier, a robust choice for this type of dataset
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Initialize StratifiedKFold for 3-Fold cross-validation
    # StratifiedKFold ensures that each fold has approximately the same percentage
    # of samples of each target class as the complete set.
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []

    print(f"Starting {n_splits}-Fold Stratified Cross-Validation...")

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate accuracy for the current fold
        fold_accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

    # Calculate the mean accuracy across all folds
    final_validation_score = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\nFinal Validation Performance: {final_validation_score:.4f}")

if __name__ == "__main__":
    train_and_evaluate_model()
