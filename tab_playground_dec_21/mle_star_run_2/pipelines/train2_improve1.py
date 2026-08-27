
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


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Separate features (X) and target (y)
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
X = train_df.drop(columns=['Id', 'Cover_Type']).copy()
y = train_df['Cover_Type']

# --- Implement Improvement Plan ---

# 1. Derive Elevation_Relative_To_Hydrology
X['Elevation_Relative_To_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']

# 2. Enhance feature interactions by creating ratios and products among horizontal distance features
horizontal_distance_features = [
    'Horizontal_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Horizontal_Distance_To_Fire_Points'
]

# Create product features
for i in range(len(horizontal_distance_features)):
    for j in range(i + 1, len(horizontal_distance_features)):
        feat1 = horizontal_distance_features[i]
        feat2 = horizontal_distance_features[j]
        X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]

# Create ratio features (add a small epsilon to denominator to prevent division by zero)
epsilon = 1e-6
for i in range(len(horizontal_distance_features)):
    for j in range(len(horizontal_distance_features)):
        if i != j: # Ensure we don't divide a feature by itself
            feat1 = horizontal_distance_features[i]
            feat2 = horizontal_distance_features[j]
            # Create two ratio features for each pair (feat1/feat2 and feat2/feat1)
            X[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + epsilon)


# 3. Apply np.log1p transformations to highly skewed distance features before scaling
# These are typically non-negative and often highly skewed.
skewed_distance_features_for_log1p = [
    'Horizontal_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Horizontal_Distance_To_Fire_Points'
]

for col in skewed_distance_features_for_log1p:
    if col in X.columns and (X[col] >= 0).all(): # Ensure column exists and values are non-negative
        X[f'{col}_log1p'] = np.log1p(X[col])

# 4. Scale all numerical features, including the new ones, using StandardScaler
# Identify all numerical columns. This includes original numerical features and all newly engineered features.
numerical_cols = X.select_dtypes(include=np.number).columns

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# --- End of Improvement Plan Implementation ---


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

# Model 2 (additional model as required, with varied parameters for diversity)
# Using more estimators and a slightly constrained max_depth to encourage different decision boundaries.
model2 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=43, n_jobs=-1)


# Set up 3-Fold Cross-Validation
# StratifiedKFold ensures each fold has a similar distribution of target classes.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store ensemble accuracy for each fold
ensemble_cv_scores = []

# Perform K-Fold Cross-Validation and Ensemble
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    # Split data for the current fold
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Train Model 1 and get probability predictions on the validation set
    model1.fit(X_train_fold, y_train_fold)
    proba1 = model1.predict_proba(X_val_fold)

    # Train Model 2 and get probability predictions on the validation set
    model2.fit(X_train_fold, y_train_fold)
    proba2 = model2.predict_proba(X_val_fold)

    # Ensemble: Average the predicted probabilities (soft voting)
    # This combines the "confidence" of each model for each class.
    ensemble_proba = (proba1 + proba2) / 2

    # Determine the final class prediction by picking the class with the highest average probability.
    # np.argmax gives the 0-indexed column index of the max probability.
    # model1.classes_ maps these 0-indexed indices back to the original class labels (e.g., 1-7).
    ensemble_preds = model1.classes_[np.argmax(ensemble_proba, axis=1)]

    # Calculate accuracy for the current fold
    fold_accuracy = accuracy_score(y_val_fold, ensemble_preds)
    ensemble_cv_scores.append(fold_accuracy)

# Calculate and print the final validation performance
final_validation_score = np.mean(ensemble_cv_scores)
print(f'Cross-validation scores for each fold: {ensemble_cv_scores}')
print(f'Final Validation Performance: {final_validation_score}')

