
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb # Assuming lightgbm is imported as 'lgb'

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize Model 1: LGBMClassifier (introduced for robust tree-based learning)
    # n_jobs=-1 enables utilization of all available CPU cores for faster training.
    lgbm_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    print("Initializing LGBMClassifier (Model 1)...")

    # Initialize Model 2: RandomForestClassifier with n_estimators=100 (from reference)
    # This remains as a strong and diverse tree-based component of the ensemble.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Initializing RandomForestClassifier (n_estimators=100, Model 2)...")

    # Initialize Model 3: LogisticRegression (replaces RandomForestClassifier n_estimators=50)
    # Provides crucial algorithmic diversity as a fast and distinct linear classifier.
    # Default solver 'lbfgs' is generally efficient and robust, and does not use 'n_jobs'.
    lr_model = LogisticRegression(random_state=42)
    print("Initializing LogisticRegression (Model 3)...")

    # Ensemble: VotingClassifier to combine the three distinct models
    # 'soft' voting is used to average predicted probabilities, enhancing robustness.
    # Equal weights are applied to each model as specified in the plan.
    # n_jobs=-1 parallelizes the fitting and prediction of base estimators where supported.
    print("Setting up VotingClassifier with LGBM, RandomForest, and LogisticRegression...")
    ensemble_model = VotingClassifier(
        estimators=[
            ('lgbm', lgbm_model),
            ('rf', rf_model),
            ('lr', lr_model)
        ],
        voting='soft',
        weights=[1, 1, 1], # Equal weights as per the improvement plan
        n_jobs=-1
    )

    # Train the ensemble model. This step internally trains all constituent models.
    print("Training the VotingClassifier ensemble...")
    ensemble_model.fit(X_train, y_train)
    print("Ensemble training complete.")

    # Get final predictions from the ensemble on the validation set
    # The .predict() method for 'soft' voting automatically uses the averaged probabilities.
    print("Making predictions on the validation set with the ensemble...")
    y_pred_ensemble = ensemble_model.predict(X_val)

    # Evaluate the ensemble model's performance for the current fold
    accuracy = accuracy_score(y_val, y_pred_ensemble)
    ensemble_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Ensemble Accuracy: {accuracy:.4f}")


# Calculate the average accuracy across all folds for the ensemble
final_validation_score = np.mean(ensemble_accuracies)
print(f"\nFinal Validation Performance: {final_validation_score:.4f}")
