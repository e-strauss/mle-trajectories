
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
X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']

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

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier # Import LGBMClassifier

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Model 1: A RandomForestClassifier with n_estimators=50
    model1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    print("Configured Model 1 (RandomForestClassifier with n_estimators=50).")

    # Model 2: The RandomForestClassifier with n_estimators=100 (from reference)
    model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Configured Model 2 (RandomForestClassifier with n_estimators=100, from reference).")

    # Model 3: A diverse LGBMClassifier with slightly improved default parameters
    model3 = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)
    print("Configured Model 3 (LGBMClassifier with n_estimators=300, learning_rate=0.05).")

    # Ensemble: Combine models using sklearn.ensemble.VotingClassifier
    # Using 'soft' voting to average predicted probabilities and equal weights
    print("Initializing VotingClassifier with three models...")
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf50', model1),
            ('rf100', model2),
            ('lgbm', model3)
        ],
        voting='soft', # Use soft voting for probability averaging
        weights=[1, 1, 1], # Initially equal weights for all three models
        n_jobs=-1 # Allow parallel fitting of base estimators if possible
    )

    # Train the ensemble model
    print("Training the ensemble model (VotingClassifier)...")
    ensemble_model.fit(X_train, y_train)
    print("Ensemble model training complete.")

    # Make predictions on the validation set using the ensemble model
    print("Making predictions on the validation set with the ensemble model...")
    y_pred_ensemble = ensemble_model.predict(X_val)

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble Accuracy: {accuracy:.4f}")


# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")
