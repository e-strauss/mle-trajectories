
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

# Feature Engineering
# Calculate the Euclidean distance to hydrology features
X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
# Sum of hillshade indices as a general indicator of light exposure
X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
# Interaction term between elevation and slope, often relevant in terrain analysis
X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']

# Enhance Aspect by applying sine and cosine transformations for cyclical features
X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))

# Introduce Relative_Elevation_to_Hydrology to capture vertical proximity to water
X['Relative_Elevation_to_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']

# Create a Hillshade_Variation feature (standard deviation of hillshade values)
# This captures the variability of light exposure throughout the day.
# For rows where all three hillshade values are identical, std will be 0, which is valid.
X['Hillshade_Variation'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1)

# Add an interaction term combining Elevation with Horizontal_Distance_To_Fire_Points
# This can indicate areas prone to wildfires at certain elevations.
X['Elevation_x_Fire_Distance'] = X['Elevation'] * X['Horizontal_Distance_To_Fire_Points']

# Introduce Hydrology_Fire_Risk: ratio of distance to fire points to distance to hydrology.
# A smaller denominator (closer to water) implies lower risk relative to fire proximity.
# Added 1 to the denominator to prevent division by zero, which could create infinite values.
X['Hydrology_Fire_Risk'] = X['Horizontal_Distance_To_Fire_Points'] / (X['Horizontal_Distance_To_Hydrology'] + 1)

# Add Average_Hillshade: the mean of the three hillshade indices.
X['Average_Hillshade'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)

# Add Elevation_Wetness_Index: interaction between Elevation and Horizontal_Distance_To_Hydrology.
# High elevation and close to water might indicate specific cover types.
X['Elevation_Wetness_Index'] = X['Elevation'] * X['Horizontal_Distance_To_Hydrology']

# Drop the original 'Aspect' column as it has been transformed into 'Aspect_sin' and 'Aspect_cos'.
# Keeping both could introduce multicollinearity and the transformed features are more suitable for cyclical data.
X = X.drop('Aspect', axis=1)

# Define numerical features for scaling.
# This list includes all continuous features, excluding the binary Wilderness_Area and Soil_Type columns.
# It also includes all newly engineered continuous features.
numerical_features = [
    'Elevation',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Hydrology_Distance',
    'Hillshade_Index',
    'Elevation_x_Slope',
    'Aspect_sin',
    'Aspect_cos',
    'Relative_Elevation_to_Hydrology',
    'Hillshade_Variation',
    'Elevation_x_Fire_Distance',
    'Hydrology_Fire_Risk',
    'Average_Hillshade',
    'Elevation_Wetness_Index'
]

# Ensure that all features in numerical_features actually exist in X's columns
numerical_features = [f for f in numerical_features if f in X.columns]

# --- Handle potential NaN and Inf values before scaling and training ---
# Check for NaNs across the entire DataFrame
if X.isnull().sum().sum() > 0:
    print("NaN values detected in X. Imputing with median for affected numerical columns.")
    # Impute NaNs with the median of their respective numerical columns
    for col in numerical_features: # Only impute numerical features
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

# Check for Infs specifically in numerical features and replace them for robust training
for col in numerical_features:
    if np.isinf(X[col]).any():
        print(f"Infinity values detected in column '{col}'. Replacing with NaN for imputation.")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        # Re-impute after replacing inf with NaN, in case the max_finite was originally inf
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())


# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Model Training and Evaluation using 3-Fold Cross-Validation with Ensemble
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

    # Model 1: RandomForestClassifier with a different n_estimators for ensemble diversity
    model1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    print("Training Model 1 (RandomForestClassifier with n_estimators=50)...")
    model1.fit(X_train, y_train)
    print("Model 1 training complete.")

    # Model 2: RandomForestClassifier with n_estimators=100 as specified in the reference solution
    model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Model 2 (RandomForestClassifier with n_estimators=100)...")
    model2.fit(X_train, y_train)
    print("Model 2 training complete.")

    # Get probabilities from both models for ensembling
    print("Making predictions on the validation set...")
    proba1 = model1.predict_proba(X_val)
    proba2 = model2.predict_proba(X_val)

    # Ensemble: Average probabilities from both models
    avg_proba = (proba1 + proba2) / 2

    # Get final predictions by taking the class with the highest average probability
    # Add 1 to convert from 0-indexed predictions (from np.argmax) to 1-indexed Cover_Type labels
    y_pred_ensemble = np.argmax(avg_proba, axis=1) + 1

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble Accuracy: {accuracy:.4f}")

# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")
