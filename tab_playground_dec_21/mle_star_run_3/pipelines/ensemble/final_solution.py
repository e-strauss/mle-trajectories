
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from scipy import stats # For majority voting
import os # For creating directories

# Load the training data (assuming train.csv is in the ./input directory as per instructions)
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X_train_full = train_df.drop(["Id", "Cover_Type"], axis=1)
y_train_full = train_df["Cover_Type"]

# Feature Engineering (from base solution, adapted for consistency)
def apply_feature_engineering(df):
    df['Hydrology_Distance'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    df['Hillshade_Index'] = df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']
    df['Elevation_x_Slope'] = df['Elevation'] * df['Slope']
    df['Aspect_sin'] = np.sin(np.deg2rad(df['Aspect']))
    df['Aspect_cos'] = np.cos(np.deg2rad(df['Aspect']))
    df['Relative_Elevation_to_Hydrology'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['Hillshade_Variation'] = df[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1).fillna(0) # .fillna(0) handles cases where std is NaN for a single row, if any.
    df['Elevation_x_Fire_Distance'] = df['Elevation'] * df['Horizontal_Distance_To_Fire_Points']
    return df

X_train_full = apply_feature_engineering(X_train_full.copy())

# Define numerical features for scaling.
# This list must include all original numerical features AND all newly engineered numerical features.
numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                      'Horizontal_Distance_To_Fire_Points',
                      'Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope',
                      'Aspect_sin', 'Aspect_cos', 'Relative_Elevation_to_Hydrology',
                      'Hillshade_Variation', 'Elevation_x_Fire_Distance']

# Apply Standard Scaling to numerical features
scaler = StandardScaler()
X_train_full[numerical_features] = scaler.fit_transform(X_train_full[numerical_features])

# Model Training and Evaluation using 3-Fold Cross-Validation with Ensemble
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

ensemble_accuracies = []

print(f"Starting {n_splits}-Fold Cross-Validation with Ensemble (LightGBM, KNN, GNB Majority Vote)...")

# Iterate through each fold
for fold, (train_index, val_index) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

    # Model 1: LightGBM Classifier (as specified in the ensemble plan, replacing original RF)
    model_lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, objective='multiclass', num_class=len(np.unique(y_train_full)))
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
    predictions_stacked = np.vstack([pred_lgbm, pred_knn, pred_gnb])
    y_pred_ensemble = stats.mode(predictions_stacked, axis=0)[0].flatten()

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble (Majority Vote) Accuracy: {accuracy:.4f}")

# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")

print("\n--- Training final models on full training data ---")

# Retrain models on the full training dataset
final_model_lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, objective='multiclass', num_class=len(np.unique(y_train_full)))
print("Training final Model_LGBM on full data...")
final_model_lgbm.fit(X_train_full, y_train_full)
print("Final Model_LGBM training complete.")

final_model_knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
print("Training final Model_KNN on full data...")
final_model_knn.fit(X_train_full, y_train_full)
print("Final Model_KNN training complete.")

final_model_gnb = GaussianNB()
print("Training final Model_GNB on full data...")
final_model_gnb.fit(X_train_full, y_train_full)
print("Final Model_GNB training complete.")

# --- Handling Test Data and Submission ---
# The original error was FileNotFoundError for './input/test.csv'.
# Given the potentially contradictory instructions ("predict on test set" vs "no test dataset"),
# this block is now wrapped in a try-except to gracefully handle a missing test.csv
# while still allowing for test predictions if the file is present.
try:
    # Load test data
    test_df = pd.read_csv("./input/test.csv")
    test_ids = test_df["Id"]
    X_test = test_df.drop("Id", axis=1)

    # Apply the same feature engineering to the test data
    X_test = apply_feature_engineering(X_test.copy())

    # Apply the same scaling (using the scaler fitted on training data) to numerical features in test data
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Make predictions on the test data with the final models
    print("Making predictions on the test set...")
    pred_lgbm_test = final_model_lgbm.predict(X_test)
    pred_knn_test = final_model_knn.predict(X_test)
    pred_gnb_test = final_model_gnb.predict(X_test)

    # Ensemble test predictions using majority voting
    predictions_stacked_test = np.vstack([pred_lgbm_test, pred_knn_test, pred_gnb_test])
    final_test_predictions = stats.mode(predictions_stacked_test, axis=0)[0].flatten()

    # Create submission file
    submission_df = pd.DataFrame({'Id': test_ids, 'Cover_Type': final_test_predictions})

    # Ensure the ./final directory exists
    os.makedirs('./final', exist_ok=True)

    submission_path = "./final/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created successfully at {submission_path}")

except FileNotFoundError:
    print("\nWarning: 'test.csv' not found in './input' directory.")
    print("Skipping test data prediction and submission file generation.")
    print("Please ensure 'test.csv' is present in the correct path if test predictions are required.")

