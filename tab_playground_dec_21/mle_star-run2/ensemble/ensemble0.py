

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
from sklearn.preprocessing import MinMaxScaler

# Separate features (X) and target (y)
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
# Create a copy of train_df to work on feature engineering without modifying the original train_df
X = train_df.drop(columns=['Id', 'Cover_Type']).copy()
y = train_df['Cover_Type']

# --- Implement Hillshade Features (9am, Noon, 3pm) ---
# Convert degrees to radians for trigonometric functions
aspect_rad = np.deg2rad(X['Aspect'])
slope_rad = np.deg2rad(X['Slope'])

# Solar parameters (approximated for a typical day/region, common in Kaggle solutions for this dataset)
# Solar altitude (angle of sun above horizon), often approximated or fixed
solar_altitude_rad = np.deg2rad(60) # A common approximation for solar altitude (e.g., mid-day, mid-season)

# Solar azimuth (angle of sun relative to North, clockwise)
solar_azimuth_9am_rad = np.deg2rad(105) # ~9 AM
solar_azimuth_noon_rad = np.deg2rad(180) # ~12 PM (Noon)
solar_azimuth_3pm_rad = np.deg2rad(255) # ~3 PM

# Calculate Hillshade using the standard formula (values usually scaled 0-255)
# Formula: HS = 255 * (sin(Solar_Altitude) * cos(Slope) + cos(Solar_Altitude) * sin(Slope) * cos(Solar_Azimuth - Aspect))
X['Hillshade_9am'] = (
    np.sin(solar_altitude_rad) * np.cos(slope_rad)
    + np.cos(solar_altitude_rad) * np.sin(slope_rad) * np.cos(solar_azimuth_9am_rad - aspect_rad)
) * 255

X['Hillshade_Noon'] = (
    np.sin(solar_altitude_rad) * np.cos(slope_rad)
    + np.cos(solar_altitude_rad) * np.sin(slope_rad) * np.cos(solar_azimuth_noon_rad - aspect_rad)
) * 255

X['Hillshade_3pm'] = (
    np.sin(solar_altitude_rad) * np.cos(slope_rad)
    + np.cos(solar_altitude_rad) * np.sin(slope_rad) * np.cos(solar_azimuth_3pm_rad - aspect_rad)
) * 255

# Clip hillshade values to ensure they are within the standard [0, 255] range
X['Hillshade_9am'] = X['Hillshade_9am'].clip(0, 255)
X['Hillshade_Noon'] = X['Hillshade_Noon'].clip(0, 255)
X['Hillshade_3pm'] = X['Hillshade_3pm'].clip(0, 255)

# --- Implement Distance/Accessibility Features ---
# Euclidean Distance to Hydrology
X['Euclidean_Distance_To_Hydrology'] = np.sqrt(
    X['Horizontal_Distance_To_Hydrology']**2 +
    X['Vertical_Distance_To_Hydrology']**2
)

# Accessibility Index to water (e.g., sum of horizontal and absolute vertical distances)
X['Accessibility_Index_To_Hydrology'] = (
    X['Horizontal_Distance_To_Hydrology'] +
    np.abs(X['Vertical_Distance_To_Hydrology'])
)

# --- Scale numerical features in X using MinMaxScaler ---
# Identify all numerical features in X, including the newly created ones.
# This approach will scale integer-encoded categorical features (like Wilderness_Area or Soil_Type
# if they are present as numbers in X and not already one-hot encoded) as part of the numerical features.
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

scaler = MinMaxScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


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

# Model 3 (new model with diverse parameters as per ensemble plan)
model3 = RandomForestClassifier(n_estimators=120, max_depth=10, max_features='sqrt', random_state=44, n_jobs=-1)


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

    # Train Model 3 and get probability predictions on the validation set (new as per ensemble plan)
    model3.fit(X_train_fold, y_train_fold)
    proba3 = model3.predict_proba(X_val_fold)

    # Ensemble: Average the predicted probabilities from three models (soft voting with equal weights)
    ensemble_proba = (proba1 + proba2 + proba3) / 3
    
    # Optional: Weighted soft voting example (commented out as per plan's optional nature and output requirements)
    # weights = [0.4, 0.3, 0.3] # Example weights; these would typically be tuned
    # ensemble_proba = (weights[0] * proba1 + weights[1] * proba2 + weights[2] * proba3) / sum(weights)


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
