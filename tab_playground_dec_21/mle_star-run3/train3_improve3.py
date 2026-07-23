
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop(["Id", "Cover_Type"], axis=1)
y = train_df["Cover_Type"]

# --- Feature Engineering ---

# Ensure distance columns are non-negative before log1p and other calculations.
# This is crucial for preventing 'invalid value' or 'divide by zero' runtime warnings
# when using np.log1p or during division in Hydrology_Gradient, as negative distances are nonsensical.
X['Horizontal_Distance_To_Hydrology'] = X['Horizontal_Distance_To_Hydrology'].clip(lower=0)
X['Vertical_Distance_To_Hydrology'] = X['Vertical_Distance_To_Hydrology'].clip(lower=0)
X['Horizontal_Distance_To_Roadways'] = X['Horizontal_Distance_To_Roadways'].clip(lower=0)
X['Horizontal_Distance_To_Fire_Points'] = X['Horizontal_Distance_To_Fire_Points'].clip(lower=0)

# Original feature engineering from the prompt
X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']

# Additional feature engineering from the prompt
X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))

X['Relative_Elevation_to_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']

# Create a Hillshade_Variation feature (standard deviation of hillshade values)
# Fill NaN values with 0 if, for example, all hillshade values are identical in a row (std would be 0, which is correctly NaN by default for 0 variance).
X['Hillshade_Variation'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1).fillna(0)

X['Elevation_x_Fire_Distance'] = X['Elevation'] * X['Horizontal_Distance_To_Fire_Points']

# Apply np.log1p to specified distance features to handle skewness.
# Clipping above ensures non-negative inputs, preventing issues with log1p.
X['Horizontal_Distance_To_Hydrology_log'] = np.log1p(X['Horizontal_Distance_To_Hydrology'])
X['Horizontal_Distance_To_Roadways_log'] = np.log1p(X['Horizontal_Distance_To_Roadways'])
X['Horizontal_Distance_To_Fire_Points_log'] = np.log1p(X['Horizontal_Distance_To_Fire_Points'])

# Introduce 'Hydrology_Gradient'
epsilon = 1e-6 # Small constant to prevent division by zero
X['Hydrology_Gradient'] = X['Vertical_Distance_To_Hydrology'] / (X['Horizontal_Distance_To_Hydrology'] + epsilon)

# Create 'Hillshade_Daily_Range'
X['Hillshade_Daily_Range'] = np.abs(X['Hillshade_9am'] - X['Hillshade_3pm'])

# Define numerical features for scaling.
# This list is updated to include all original and newly engineered continuous numerical features.
numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points',
                      'Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope',
                      'Aspect_sin', 'Aspect_cos', 'Relative_Elevation_to_Hydrology',
                      'Hillshade_Variation', 'Elevation_x_Fire_Distance',
                      'Horizontal_Distance_To_Hydrology_log', 'Horizontal_Distance_To_Roadways_log',
                      'Horizontal_Distance_To_Fire_Points_log', 'Hydrology_Gradient',
                      'Hillshade_Daily_Range']

# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# --- Robustness checks after scaling ---
# Check for NaN values and impute if necessary (e.g., if a feature had zero variance and resulted in NaN after scaling)
if X.isnull().any().any():
    print("Warning: NaN values found in features after scaling. Imputing with mean.")
    X.fillna(X.mean(numeric_only=True), inplace=True) # Use numeric_only=True to average only numerical columns

# Check for infinity values and replace if necessary.
# This handles cases where StandardScaler might produce inf due to problematic inputs (e.g., zero variance for an inf column).
if np.isinf(X).any().any():
    print("Warning: Inf values found in features after scaling. Replacing with capped values.")
    for col in X.columns:
        if np.isinf(X[col]).any():
            # Get finite max/min values for capping.
            finite_values = X[col][np.isfinite(X[col])]
            if not finite_values.empty:
                max_finite = finite_values.max()
                min_finite = finite_values.min()
                # Replace positive inf with a value slightly larger than the max finite value,
                # and negative inf with a value slightly smaller than the min finite value.
                X[col] = X[col].replace(np.inf, max_finite + 1)
                X[col] = X[col].replace(-np.inf, min_finite - 1)
            else:
                # Fallback: if all values in the column are inf, replace with 0.
                X[col] = X[col].replace([np.inf, -np.inf], 0)

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

    # Model 1: A RandomForestClassifier with a different n_estimators for diversity in ensemble.
    model1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    print("Training Model 1 (RandomForestClassifier with n_estimators=50)...")
    model1.fit(X_train, y_train)
    print("Model 1 training complete.")

    # Model 2: The RandomForestClassifier with n_estimators=100 as specified in the reference solution.
    model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Model 2 (RandomForestClassifier with n_estimators=100, from reference)...")
    model2.fit(X_train, y_train)
    print("Model 2 training complete.")

    # Get probabilities from both models for ensembling
    print("Making predictions on the validation set...")
    proba1 = model1.predict_proba(X_val)
    proba2 = model2.predict_proba(X_val)

    # Ensemble: Average probabilities from both models
    avg_proba = (proba1 + proba2) / 2

    # Get final predictions by taking the class with the highest average probability.
    # Add 1 because Cover_Type labels are 1-indexed (1 to 7), while np.argmax returns 0-indexed class labels.
    y_pred_ensemble = np.argmax(avg_proba, axis=1) + 1

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble Accuracy: {accuracy:.4f}")

# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")
