
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

# Feature Engineering
# 1. Enhance Aspect feature by converting it into its sine and cosine components
train_df['Aspect_sin'] = np.sin(np.deg2rad(train_df['Aspect']))
train_df['Aspect_cos'] = np.cos(np.deg2rad(train_df['Aspect']))

# 2. Create a comprehensive Hydrology_Distance feature using Euclidean distance
train_df['Hydrology_Distance'] = np.sqrt(
    train_df['Horizontal_Distance_To_Hydrology']**2 + 
    train_df['Vertical_Distance_To_Hydrology']**2
)

# 3. Engineer key interaction features involving Elevation
train_df['Elevation_Slope'] = train_df['Elevation'] * train_df['Slope']
train_df['Elevation_HDTH'] = train_df['Elevation'] * train_df['Horizontal_Distance_To_Hydrology']

# Separate features (X) and target (y)
# 'Id' column is not a feature, so drop it.
# 'Cover_Type' is the target variable.
# The original 'Aspect' feature is replaced by its sine and cosine components, so it can be dropped.
columns_to_drop_from_X = ['Id', 'Cover_Type', 'Aspect']
X = train_df.drop(columns=columns_to_drop_from_X)
y = train_df['Cover_Type']

# Scale all numerical features using MinMaxScaler
# Identify numerical columns in X for scaling
numerical_features = X.select_dtypes(include=np.number).columns

scaler = MinMaxScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])


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

