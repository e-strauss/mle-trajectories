

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import copy

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X_base = train_df.drop(["Id", "Cover_Type"], axis=1)
y_base = train_df["Cover_Type"]

# Define the core experiment function to encapsulate the training and evaluation logic
def run_experiment(X_initial, y, use_feature_engineering, use_scaling, use_ensemble):
    X = X_initial.copy() # Work on a copy to avoid side effects across experiments

    # Define numerical features, initially without engineered ones
    current_numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                                  'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                                  'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                                  'Horizontal_Distance_To_Fire_Points']

    if use_feature_engineering:
        # Apply Feature Engineering
        X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
        X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
        X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']
        # Add new engineered features to the numerical features list
        current_numerical_features.extend(['Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope'])

    # Filter `current_numerical_features` to ensure only columns present in `X` are selected
    # This is crucial if feature engineering is skipped, as new features won't exist.
    current_numerical_features = [col for col in current_numerical_features if col in X.columns]

    if use_scaling:
        # Apply Standard Scaling to numerical features
        scaler = StandardScaler()
        X[current_numerical_features] = scaler.fit_transform(X[current_numerical_features])

    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    # Iterate through each fold for cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if use_ensemble:
            # Train and predict with Model 1
            model1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model1.fit(X_train, y_train)
            proba1 = model1.predict_proba(X_val)

            # Train and predict with Model 2 (from original solution)
            model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model2.fit(X_train, y_train)
            proba2 = model2.predict_proba(X_val)

            # Ensemble: Average probabilities
            avg_proba = (proba1 + proba2) / 2
            y_pred = np.argmax(avg_proba, axis=1)
        else: # Use a single model (Model 2: RandomForestClassifier with n_estimators=100)
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# --- Perform Ablation Study ---

print("Starting Ablation Study...\n")

# Store results for comparison
results = {}

# Baseline: Original Solution (all features, scaling, and ensemble)
baseline_accuracy = run_experiment(X_base, y_base,
                                   use_feature_engineering=True,
                                   use_scaling=True,
                                   use_ensemble=True)
results["Baseline"] = baseline_accuracy
print(f"Baseline (Original Solution) Accuracy: {baseline_accuracy:.4f}\n")

# Ablation 1: Disable Feature Engineering
no_fe_accuracy = run_experiment(X_base, y_base,
                                use_feature_engineering=False,
                                use_scaling=True,
                                use_ensemble=True)
results["No Feature Engineering"] = no_fe_accuracy
print(f"Ablation: No Feature Engineering Accuracy: {no_fe_accuracy:.4f} (Change from Baseline: {no_fe_accuracy - baseline_accuracy:.4f})\n")

# Ablation 2: Disable Standard Scaling
no_scaling_accuracy = run_experiment(X_base, y_base,
                                     use_feature_engineering=True,
                                     use_scaling=False,
                                     use_ensemble=True)
results["No Standard Scaling"] = no_scaling_accuracy
print(f"Ablation: No Standard Scaling Accuracy: {no_scaling_accuracy:.4f} (Change from Baseline: {no_scaling_accuracy - baseline_accuracy:.4f})\n")

# Ablation 3: Disable Ensemble (use only Model 2 with n_estimators=100)
no_ensemble_accuracy = run_experiment(X_base, y_base,
                                      use_feature_engineering=True,
                                      use_scaling=True,
                                      use_ensemble=False)
results["No Ensemble (Single Model)"] = no_ensemble_accuracy
print(f"Ablation: No Ensemble (Single Model) Accuracy: {no_ensemble_accuracy:.4f} (Change from Baseline: {no_ensemble_accuracy - baseline_accuracy:.4f})\n")

# Determine which part contributes the most
print("\n--- Ablation Study Summary ---")
performance_drops = {}
for name, acc in results.items():
    if name != "Baseline":
        drop = baseline_accuracy - acc
        performance_drops[name] = drop
        print(f"Removing '{name}' resulted in a performance drop of: {drop:.4f}")

if performance_drops:
    # Find the ablation that caused the largest performance drop
    most_contributing_part = max(performance_drops, key=performance_drops.get)
    print(f"\nThe part that contributes the most to the overall performance is: '{most_contributing_part}'")
else:
    print("\nNo ablations performed or recorded.")

