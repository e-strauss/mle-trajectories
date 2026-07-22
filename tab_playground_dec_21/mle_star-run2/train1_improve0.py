
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

# Create a copy to perform feature engineering and scaling without modifying the original train_df
train_df_processed = train_df.copy()

# --- 1. Extensive Feature Engineering ---

# 1.1 Distance-to-hydrology metrics
train_df_processed['Distance_To_Hydrology'] = np.sqrt(
    train_df_processed['Horizontal_Distance_To_Hydrology']**2 +
    train_df_processed['Vertical_Distance_To_Hydrology']**2
)
train_df_processed['Hydro_Elevation_Diff'] = train_df_processed['Elevation'] - train_df_processed['Vertical_Distance_To_Hydrology']
train_df_processed['Hydro_Horizontal_Interaction'] = train_df_processed['Horizontal_Distance_To_Hydrology'] * train_df_processed['Vertical_Distance_To_Hydrology']
train_df_processed['Hydro_Difference'] = train_df_processed['Horizontal_Distance_To_Hydrology'] - train_df_processed['Vertical_Distance_To_Hydrology']
train_df_processed['Hydro_Sum'] = train_df_processed['Horizontal_Distance_To_Hydrology'] + train_df_processed['Vertical_Distance_To_Hydrology']

# 1.2 Elevation interactions
train_df_processed['Elevation_Road_Diff'] = train_df_processed['Elevation'] - train_df_processed['Horizontal_Distance_To_Roadways']
train_df_processed['Elevation_x_Slope'] = train_df_processed['Elevation'] * train_df_processed['Slope']
train_df_processed['Elevation_x_Hydro_Dist'] = train_df_processed['Elevation'] * train_df_processed['Distance_To_Hydrology']
# Assuming Horizontal_Distance_To_Fire_Points is a feature
if 'Horizontal_Distance_To_Fire_Points' in train_df_processed.columns:
    train_df_processed['Elevation_x_Fire_Dist'] = train_df_processed['Elevation'] * train_df_processed['Horizontal_Distance_To_Fire_Points']

# 1.3 Hillshade sums/means
train_df_processed['Hillshade_Mean'] = (
    train_df_processed['Hillshade_9am'] +
    train_df_processed['Hillshade_Noon'] +
    train_df_processed['Hillshade_3pm']
) / 3
train_df_processed['Hillshade_Sum'] = (
    train_df_processed['Hillshade_9am'] +
    train_df_processed['Hillshade_Noon'] +
    train_df_processed['Hillshade_3pm']
)
train_df_processed['Hillshade_Diff_9_Noon'] = train_df_processed['Hillshade_9am'] - train_df_processed['Hillshade_Noon']
train_df_processed['Hillshade_Diff_Noon_3pm'] = train_df_processed['Hillshade_Noon'] - train_df_processed['Hillshade_3pm']
train_df_processed['Hillshade_Diff_9_3pm'] = train_df_processed['Hillshade_9am'] - train_df_processed['Hillshade_3pm']
train_df_processed['Hillshade_Night'] = (train_df_processed['Hillshade_9am'] + train_df_processed['Hillshade_3pm']) / 2 - train_df_processed['Hillshade_Noon']

# 1.4 Trigonometric transformations for Aspect and Slope
# Convert degrees to radians for trigonometric functions
train_df_processed['Aspect_rad'] = np.deg2rad(train_df_processed['Aspect'])
train_df_processed['Slope_rad'] = np.deg2rad(train_df_processed['Slope'])

train_df_processed['Aspect_sin'] = np.sin(train_df_processed['Aspect_rad'])
train_df_processed['Aspect_cos'] = np.cos(train_df_processed['Aspect_rad'])
train_df_processed['Slope_sin'] = np.sin(train_df_processed['Slope_rad'])
train_df_processed['Slope_cos'] = np.cos(train_df_processed['Slope_rad'])

# --- 2. Scale all numerical features (original and newly engineered) ---

# Identify original numerical features to scale (excluding 'Id' and 'Cover_Type' and any one-hot encoded categories)
original_numeric_cols = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'
]
if 'Horizontal_Distance_To_Fire_Points' in train_df_processed.columns:
    original_numeric_cols.append('Horizontal_Distance_To_Fire_Points')

# List all newly created numerical features
engineered_features = [
    'Distance_To_Hydrology', 'Hydro_Elevation_Diff', 'Hydro_Horizontal_Interaction',
    'Hydro_Difference', 'Hydro_Sum', 'Elevation_Road_Diff', 'Elevation_x_Slope',
    'Elevation_x_Hydro_Dist',
    'Hillshade_Mean', 'Hillshade_Sum', 'Hillshade_Diff_9_Noon',
    'Hillshade_Diff_Noon_3pm', 'Hillshade_Diff_9_3pm', 'Hillshade_Night',
    'Aspect_rad', 'Slope_rad', 'Aspect_sin', 'Aspect_cos', 'Slope_sin', 'Slope_cos'
]
if 'Elevation_x_Fire_Dist' in train_df_processed.columns:
    engineered_features.append('Elevation_x_Fire_Dist')


# Combine for the full list of columns to scale, ensuring they exist in the dataframe
features_to_scale = [col for col in (original_numeric_cols + engineered_features) if col in train_df_processed.columns]

# Initialize StandardScaler
scaler = StandardScaler()

# Apply scaling to the selected numerical features
train_df_processed[features_to_scale] = scaler.fit_transform(train_df_processed[features_to_scale])

# --- Separate features (X) and target (y) ---
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
# Drop temporary radian columns from X, as their sin/cos transformations are used instead.
columns_to_drop_for_X = ['Id', 'Cover_Type', 'Aspect_rad', 'Slope_rad']
columns_to_drop_for_X = [col for col in columns_to_drop_for_X if col in train_df_processed.columns]

X = train_df_processed.drop(columns=columns_to_drop_for_X)
y = train_df_processed['Cover_Type']


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

