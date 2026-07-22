
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
from sklearn.preprocessing import MinMaxScaler

# --- Feature Engineering ---

# Calculate Hillshade at 9 AM, Noon, and 3 PM
# Solar altitude is typically assumed to be 45 degrees for these calculations
solar_altitude_rad = np.deg2rad(45)

# Convert Aspect and Slope to radians for trigonometric functions
aspect_rad = np.deg2rad(train_df['Aspect'])
slope_rad = np.deg2rad(train_df['Slope'])

# Hillshade at 9 AM (azimuth 315 degrees)
azimuth_9am_rad = np.deg2rad(315)
train_df['Hillshade_9am'] = 255 * (
    (np.cos(solar_altitude_rad) * np.sin(slope_rad)) +
    (np.sin(solar_altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_9am_rad - aspect_rad))
)
# Clip values to be within [0, 255], which is the standard range for hillshade
train_df['Hillshade_9am'] = np.clip(train_df['Hillshade_9am'], 0, 255)

# Hillshade at Noon (azimuth 180 degrees)
azimuth_noon_rad = np.deg2rad(180)
train_df['Hillshade_Noon'] = 255 * (
    (np.cos(solar_altitude_rad) * np.sin(slope_rad)) +
    (np.sin(solar_altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_noon_rad - aspect_rad))
)
train_df['Hillshade_Noon'] = np.clip(train_df['Hillshade_Noon'], 0, 255)

# Hillshade at 3 PM (azimuth 225 degrees)
azimuth_3pm_rad = np.deg2rad(225)
train_df['Hillshade_3pm'] = 255 * (
    (np.cos(solar_altitude_rad) * np.sin(slope_rad)) +
    (np.sin(solar_altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_3pm_rad - aspect_rad))
)
train_df['Hillshade_3pm'] = np.clip(train_df['Hillshade_3pm'], 0, 255)

# Derive Euclidean_Distance_To_Hydrology from its horizontal and vertical components
train_df['Euclidean_Distance_To_Hydrology'] = np.sqrt(
    train_df['Horizontal_Distance_To_Hydrology']**2 +
    train_df['Vertical_Distance_To_Hydrology']**2
)

# Combine relevant proximity features into a single "Accessibility_Index"
# Assuming 'Horizontal_Distance_To_Roadways' and 'Horizontal_Distance_To_Fire_Points'
# are the features representing proximity to roadways and fire points.
train_df['Accessibility_Index'] = (
    train_df['Horizontal_Distance_To_Roadways'] +
    train_df['Horizontal_Distance_To_Fire_Points'] +
    train_df['Euclidean_Distance_To_Hydrology']
)

# --- Separate features (X) and target (y) ---
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
X = train_df.drop(columns=['Id', 'Cover_Type'])
y = train_df['Cover_Type']

# --- Normalize all continuous numerical features using MinMaxScaler ---
# Identify all numerical columns in X.
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# Exclude one-hot encoded 'Wilderness_Area' and 'Soil_Type' columns from scaling,
# as they are typically binary and already in a suitable scale (0 or 1).
categorical_features_to_exclude = [col for col in numerical_cols if 'Wilderness_Area' in col or 'Soil_Type' in col]
features_to_scale = [col for col in numerical_cols if col not in categorical_features_to_exclude]

scaler = MinMaxScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])


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

