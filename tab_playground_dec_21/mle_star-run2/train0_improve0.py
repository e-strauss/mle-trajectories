
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Load the training data
# The problem statement specifies not to use try-except for unintended behavior,
# so we assume the file path is correct.
script_dir = os.path.dirname(__file__)
train_file_path = os.path.join(script_dir, "input", "train.csv")
train_df = pd.read_csv(train_file_path)

# Separate features (X) and target (y)
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
X = train_df.drop(columns=['Id', 'Cover_Type'])
y = train_df['Cover_Type']

# --- Integration of Reference Solution Insight ---
# The reference solution clarified that 'Wilderness_Area' and 'Soil_Type' features
# are already represented as multiple binary (0/1) columns (e.g., Wilderness_Area1,...,
# Soil_Type1,...). These are effectively pre-one-hot encoded features.
# For a RandomForestClassifier, such binary features are handled correctly without
# requiring additional explicit one-hot encoding. The model treats them numerically,
# and splits on values like `feature <= 0.5` effectively distinguish between the categories.
# Therefore, no explicit preprocessing for these categorical features is needed beyond
# what's already done by reading the CSV.
# --- End of Reference Solution Insight Integration ---


# Initialize multiple models for ensembling
# Model 1 (based on the original base solution)

model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Model 2 and ensembling logic have been removed as per the improvement plan.
# Ablation study showed that Model 1 alone performs better than the ensemble.

# Set up 3-Fold Cross-Validation
# StratifiedKFold ensures each fold has a similar distribution of target classes.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store Model 1's accuracy for each fold
# Renamed from 'ensemble_cv_scores' to reflect evaluation of a single model.
model1_cv_scores = []

# Perform K-Fold Cross-Validation using only Model 1
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    # Split data for the current fold
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Train Model 1
    model1.fit(X_train_fold, y_train_fold)

    # Get predictions from Model 1 on the validation set
    # No ensembling, direct predictions from Model 1 are used.
    model1_preds = model1.predict(X_val_fold)

    # Calculate accuracy for the current fold
    fold_accuracy = accuracy_score(y_val_fold, model1_preds)
    model1_cv_scores.append(fold_accuracy)

# Calculate and print the final validation performance for Model 1
# Keeping 'final_validation_score' variable name for consistency with original script's output.
final_validation_score = np.mean(model1_cv_scores)
print(f'Cross-validation scores for Model 1 (each fold): {model1_cv_scores}')
print(f'Final Validation Performance (Model 1 only): {final_validation_score}')


