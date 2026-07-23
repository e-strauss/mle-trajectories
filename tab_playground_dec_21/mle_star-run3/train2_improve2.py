
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training data (assuming train.csv is in the ./input directory as per instructions)
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop(["Id", "Cover_Type"], axis=1)
y = train_df["Cover_Type"]

# Feature Engineering (from base solution)

import numpy as np

X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']


import numpy as np

# Retain the existing sine and cosine transformations for cyclic information
X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))

# Additionally, introduce the original Aspect feature, normalized to a 0-1 range, for linear relationships.
# Assuming 'Aspect' values are typically between 0 and 360 degrees,
# normalization is done by dividing by the maximum possible value (360).
X['Aspect_normalized'] = X['Aspect'] / 360.0


# Introduce Relative_Elevation_to_Hydrology
X['Relative_Elevation_to_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']

# Create a Hillshade_Variation feature (e.g., standard deviation of hillshade values)
X['Hillshade_Variation'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1)

# Add an interaction term combining Elevation with Horizontal_Distance_To_Fire_Points
X['Elevation_x_Fire_Distance'] = X['Elevation'] * X['Horizontal_Distance_To_Fire_Points']


# Define numerical features for scaling.
# Wilderness_Area and Soil_Type columns are already binary (one-hot encoded) and generally
# do not require scaling when used with tree-based models, and it's best to keep them as is.
numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points',
                      'Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope']

# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Model Training and Evaluation using 3-Fold Cross-Validation with Ensemble
# We will use the manual StratifiedKFold loop from the reference solution
# to allow for training multiple models and ensembling their predictions.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

ensemble_accuracies = []

print(f"Starting {n_splits}-Fold Cross-Validation with Ensemble...")

# Iterate through each fold
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Model 1: A RandomForestClassifier with a different n_estimators for diversity in ensemble.
    # This serves as one of the 'additional models' and contributes to the ensemble diversity.
    model1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    print("Training Model 1 (RandomForestClassifier with n_estimators=50)...")
    model1.fit(X_train, y_train)
    print("Model 1 training complete.")

    # Model 2: The RandomForestClassifier with n_estimators=100 as specified in the reference solution.
    # This is the second 'additional model' and the one explicitly from the reference.
    model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Model 2 (RandomForestClassifier with n_estimators=100, from reference)...")
    model2.fit(X_train, y_train)
    print("Model 2 training complete.")

    # Get probabilities from both models for ensembling
    print("Making predictions on the validation set...")
    proba1 = model1.predict_proba(X_val)
    proba2 = model2.predict_proba(X_val)

    # Ensemble: Average probabilities from both models
    # This simple averaging strategy for `predict_proba` is effective for classification ensembles.
    avg_proba = (proba1 + proba2) / 2

    # Get final predictions by taking the class with the highest average probability
    y_pred_ensemble = np.argmax(avg_proba, axis=1)

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble Accuracy: {accuracy:.4f}")

# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")
