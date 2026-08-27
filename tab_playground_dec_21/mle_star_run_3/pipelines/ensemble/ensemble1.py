
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from scipy import stats # For majority voting

# Load the training data (assuming train.csv is in the ./input directory as per instructions)
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop(["Id", "Cover_Type"], axis=1)
y = train_df["Cover_Type"]

# Feature Engineering (from base solution, adapted for consistency)
X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']
X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))
X['Relative_Elevation_to_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']
X['Hillshade_Variation'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1)
X['Elevation_x_Fire_Distance'] = X['Elevation'] * X['Horizontal_Distance_To_Fire_Points']

# Define numerical features for scaling.
# Wilderness_Area and Soil_Type columns are already binary (one-hot encoded) and generally
# do not require scaling when used with tree-based models, and it's best to keep them as is.
numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points',
                      'Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope',
                      'Aspect_sin', 'Aspect_cos', 'Relative_Elevation_to_Hydrology',
                      'Hillshade_Variation', 'Elevation_x_Fire_Distance']

# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Model Training and Evaluation using 3-Fold Cross-Validation with Ensemble
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

ensemble_accuracies = []

print(f"Starting {n_splits}-Fold Cross-Validation with Ensemble (LightGBM, KNN, GNB Majority Vote)...")

# Iterate through each fold
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Model 1: LightGBM Classifier (as specified in the ensemble plan, replacing original RF)
    # Using n_estimators=100 to align with typical settings for tree models in such problems
    model_lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, objective='multiclass', num_class=len(np.unique(y)))
    print("Training Model_LGBM (LightGBMClassifier)...")
    model_lgbm.fit(X_train, y_train)
    print("Model_LGBM training complete.")

    # Model 2: KNeighborsClassifier (as specified in the ensemble plan)
    model_knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    print("Training Model_KNN (KNeighborsClassifier with n_neighbors=7)...")
    model_knn.fit(X_train, y_train)
    print("Model_KNN training complete.")

    # Model 3: Gaussian Naive Bayes Classifier (as specified in the ensemble plan)
    model_gnb = GaussianNB()
    print("Training Model_GNB (GaussianNB Classifier)...")
    model_gnb.fit(X_train, y_train)
    print("Model_GNB training complete.")

    # Get hard predictions from all three models on the validation set
    print("Making predictions on the validation set...")
    pred_lgbm = model_lgbm.predict(X_val)
    pred_knn = model_knn.predict(X_val)
    pred_gnb = model_gnb.predict(X_val)

    # Combine predictions for majority voting
    # Stack predictions horizontally (each column is a model's predictions for a sample)
    predictions_stacked = np.vstack([pred_lgbm, pred_knn, pred_gnb])

    # Implement simple majority voting across the models for each sample
    # stats.mode returns the mode and its count. We need the mode itself [0].
    # axis=0 calculates mode for each column (i.e., for each sample).
    # The fix: stats.mode(predictions_stacked, axis=0)[0] returns an array of modes, one for each sample.
    # The original code had an extra [0] at the end, which was selecting only the first mode from the array.
    y_pred_ensemble = stats.mode(predictions_stacked, axis=0)[0]

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble (Majority Vote) Accuracy: {accuracy:.4f}")

# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")
