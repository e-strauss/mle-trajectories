
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# Define the path to the input directory
INPUT_DIR = "./input"

def train_predict():
    """
    Trains an XGBoost model on the forest cover type dataset,
    performs 3-fold stratified cross-validation, and prints the average accuracy.
    """
    try:
        # Load the training data
        train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    except FileNotFoundError:
        print(f"Error: train.csv not found in {INPUT_DIR}. Please make sure the file is in the correct directory.")
        raise FileNotFoundError(f"train.csv not found in {INPUT_DIR}")

    # Prepare the data
    # Drop 'Id' column as it's not a feature
    train_df = train_df.drop('Id', axis=1)

    # Separate features (X) and target (y)
    X = train_df.drop('Cover_Type', axis=1)
    y = train_df['Cover_Type']

    # XGBoost expects target labels to be 0-indexed for multi-class classification
    # So, we convert 1-7 to 0-6
    y_mapped = y - 1

    # Convert the target variable to a categorical type for XGBoost (important for multi-class)
    y_mapped = y_mapped.astype('category')

    # Initialize StratifiedKFold for 3-fold cross-validation
    # We use a fixed random_state for reproducibility
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    fold_accuracies = []

    # Iterate through each fold
    for fold, (train_index, val_index) in enumerate(skf.split(X, y_mapped)):
        print(f"--- Fold {fold + 1} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_mapped.iloc[train_index], y_mapped.iloc[val_index]

        # Initialize XGBoost classifier
        # 'objective': 'multi:softmax' for multi-class classification
        # 'num_class': 7 because we have 7 cover types (0-6)
        # 'eval_metric': 'mlogloss' is a common metric for multi-class problems
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=7,
            eval_metric='mlogloss',
            use_label_encoder=False, # Suppress the warning
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1 # Use all available cores
        )

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Calculate accuracy for the current fold
        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)
        print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

    # Calculate and print the average accuracy across all folds
    final_avg_accuracy = np.mean(fold_accuracies)
    print(f"\nFinal Validation Performance: {final_avg_accuracy:.4f}")

if __name__ == "__main__":
    train_predict()
