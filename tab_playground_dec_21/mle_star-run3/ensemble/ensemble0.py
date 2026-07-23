
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

def train_and_evaluate():
    # Define the path to the input directory
    input_dir = "./input"
    train_file = os.path.join(input_dir, "train.csv")

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found. Please ensure 'train.csv' is in this directory.")
        return

    # Load the training data
    try:
        train_df = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"Error: '{train_file}' not found. Please ensure the training data is in the './input/' directory.")
        return
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Separate features (X) and target (y)
    # Drop 'Id' column as it's not a feature
    X = train_df.drop(["Id", "Cover_Type"], axis=1)
    y = train_df["Cover_Type"]

    # LightGBM's 'multiclass' objective expects target labels to be 0-indexed (0 to num_class - 1).
    # The problem statement indicates Cover_Type is 1 to 7. We need to subtract 1 from y.
    y = y - 1

    # Initialize StratifiedKFold for 3-Fold Cross-Validation
    # StratifiedKFold ensures that each fold has approximately the same percentage of samples of each target class
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # List to store accuracy scores for each fold
    fold_accuracies = []

    # Iterate over each fold
    print(f"Starting {n_splits}-Fold Cross-Validation...")
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Split data into training and validation sets for the current fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Initialize the LightGBM Classifier
        # Using default parameters as a strong baseline, could be tuned further
        # objective='multiclass' for multi-class classification
        # num_class=7 for the 7 different cover types
        lgbm_clf = lgb.LGBMClassifier(objective='multiclass', num_class=7, random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31)

        # --- FIX START ---
        # The error "ValueError: y contains previously unseen labels" occurs when LightGBM's `eval_set`
        # contains target labels that were not present in the `y_train` for that specific fold.
        # This can happen even with StratifiedKFold if a very rare class is present, or due to how
        # LightGBM internally handles the evaluation set's labels.
        # To prevent this, we filter the validation set used for early stopping to only include
        # labels that are also present in the training set for the current fold.
        # The full `X_val` and `y_val` are still used for the final accuracy calculation for the fold.

        # Get unique labels in the training set
        train_labels = set(y_train.unique())
        # Get unique labels in the validation set
        val_labels = set(y_val.unique())

        # Identify labels in y_val that are not present in y_train
        unseen_val_labels_in_train = val_labels - train_labels

        X_val_eval = X_val.copy()
        y_val_eval = y_val.copy()

        if unseen_val_labels_in_train:
            print(f"  Warning: Fold {fold + 1}: Validation set for early stopping evaluation contains labels "
                  f"not present in the training set: {sorted(list(unseen_val_labels_in_train))}. "
                  f"Filtering these samples from eval_set to prevent ValueError during LightGBM fit.")
            # Filter out rows from X_val_eval and y_val_eval that correspond to unseen labels
            rows_to_keep_for_eval = ~y_val_eval.isin(list(unseen_val_labels_in_train))
            X_val_eval = X_val_eval[rows_to_keep_for_eval]
            y_val_eval = y_val_eval[rows_to_keep_for_eval]
            print(f"  Filtered {len(y_val) - len(y_val_eval)} samples from the early stopping evaluation set.")
        # --- FIX END ---

        # Train the model
        print("Training model...")
        lgbm_clf.fit(X_train, y_train,
                      eval_set=[(X_val_eval, y_val_eval)], # Use potentially filtered validation set for early stopping
                      eval_metric='multi_logloss',
                      callbacks=[lgb.early_stopping(100, verbose=False)]) # Early stopping to prevent overfitting

        # Make predictions on the FULL original validation set for the current fold
        print("Making predictions on the validation set...")
        y_pred = lgbm_clf.predict(X_val)

        # Calculate accuracy for the current fold
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    # Calculate the average accuracy across all folds
    final_validation_score = np.mean(fold_accuracies)
    print(f"\nFinal Validation Performance: {final_validation_score:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
