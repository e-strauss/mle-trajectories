
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the training data
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    train_df = pd.read_csv("train.csv")


# Drop the 'Id' column as it's not a feature
train_df = train_df.drop('Id', axis=1)

# Separate features (X) and target (y)
X = train_df.drop('Cover_Type', axis=1)
y = train_df['Cover_Type']

# Feature Engineering
# Create new features based on existing ones, similar to many Kaggle solutions for this dataset
X['Horizontal_Distance_To_Hydrology_sq'] = X['Horizontal_Distance_To_Hydrology']**2
X['Vertical_Distance_To_Hydrology_sq'] = X['Vertical_Distance_To_Hydrology']**2
X['Distance_To_Hydrology'] = np.sqrt(X['Horizontal_Distance_To_Hydrology_sq'] + X['Vertical_Distance_To_Hydrology_sq'])
X['Elevation_Vertical_Hydrology'] = X['Elevation'] + X['Vertical_Distance_To_Hydrology']
X['Elevation_Minus_Vertical_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']
X['Hillshade_9am_Noon_3pm_sum'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
X['Hillshade_9am_Noon_3pm_mean'] = (X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']) / 3

# Interaction features with Elevation
X['Elevation_HD_Roadways'] = X['Elevation'] + X['Horizontal_Distance_To_Roadways']
X['Elevation_HD_Firepoints'] = X['Elevation'] + X['Horizontal_Distance_To_Fire_Points']

# Aspect and Slope related features
X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))
X['Slope_sin'] = np.sin(np.deg2rad(X['Slope']))
X['Slope_cos'] = np.cos(np.deg2rad(X['Slope']))

# Initialize StandardScaler
scaler = StandardScaler()

# Columns to scale (numerical features)
numerical_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                  'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology_sq',
                  'Vertical_Distance_To_Hydrology_sq', 'Distance_To_Hydrology',
                  'Elevation_Vertical_Hydrology', 'Elevation_Minus_Vertical_Hydrology',
                  'Hillshade_9am_Noon_3pm_sum', 'Hillshade_9am_Noon_3pm_mean',
                  'Elevation_HD_Roadways', 'Elevation_HD_Firepoints',
                  'Aspect_sin', 'Aspect_cos', 'Slope_sin', 'Slope_cos']

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# 3-Fold Stratified Cross-Validation
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Model Definitions
# RandomForestClassifier with optimized parameters
rf_clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                min_samples_leaf=5, random_state=42, n_jobs=-1,
                                class_weight='balanced') # Added class_weight

# XGBClassifier with optimized parameters
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=7,  # Ensure num_class is set for multi-class
                        n_estimators=500, learning_rate=0.05, max_depth=8,
                        subsample=0.7, colsample_bytree=0.7, gamma=0.1,
                        random_state=42, use_label_encoder=False, eval_metric='mlogloss',
                        n_jobs=-1)


# Ensemble using VotingClassifier
# Using 'hard' voting for predictions
ensemble_clf = VotingClassifier(estimators=[('rf', rf_clf), ('xgb', xgb_clf)],
                                voting='hard', n_jobs=-1, weights=[0.5, 0.5]) # Adjusted weights


validation_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    print(f"--- Fold {fold+1}/{n_splits} ---")

    # Train the ensemble model
    ensemble_clf.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = ensemble_clf.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    validation_scores.append(accuracy)
    print(f"Validation Accuracy for Fold {fold+1}: {accuracy:.4f}")

# Calculate and print the final mean validation accuracy
final_validation_score = np.mean(validation_scores)
print(f"Final Validation Performance: {final_validation_score:.4f}")
