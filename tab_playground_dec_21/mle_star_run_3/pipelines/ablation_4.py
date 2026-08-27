
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import copy

# Load the training data (assuming train.csv is in the ./input directory as per instructions)
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X_orig = train_df.drop(["Id", "Cover_Type"], axis=1)
y_orig = train_df["Cover_Type"]

def run_ablation_experiment(X_input, y_input,
                            enable_standard_scaling=True,
                            enable_initial_fe=True, # Controls Hydrology_Distance, Hillshade_Index, Elevation_x_Slope
                            model1_n_estimators=50,
                            model2_n_estimators=100):
    
    # Create a deep copy to ensure modifications do not affect other experiments
    X = X_input.copy() 
    
    # Define a base list of numerical features that are always present and might be scaled
    # Wilderness_Area and Soil_Type columns are already binary (one-hot encoded) and generally
    # do not require scaling when used with tree-based models, and it's best to keep them as is.
    base_numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                               'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                               'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                               'Horizontal_Distance_To_Fire_Points']
    
    current_numerical_features_to_scale = list(base_numerical_features)

    # --- Feature Engineering ---
    # These are the original engineered features from the provided solution
    # The flags below control their inclusion.
    
    # Group 1: Initial Feature Engineering (controlled by enable_initial_fe)
    if enable_initial_fe:
        X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
        X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
        X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']
        current_numerical_features_to_scale.extend(['Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope'])

    # Group 2: Other Feature Engineering (always enabled for these ablations as they are not the target)
    X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
    X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))
    X['Relative_Elevation_to_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']
    X['Hillshade_Variation'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1)
    X['Elevation_x_Fire_Distance'] = X['Elevation'] * X['Horizontal_Distance_To_Fire_Points']

    # Filter numerical features to only include those that actually exist in X
    # This handles cases where some FE is disabled
    features_for_scaler = [f for f in current_numerical_features_to_scale if f in X.columns]

    # --- Standard Scaling ---
    if enable_standard_scaling:
        scaler = StandardScaler()
        X[features_for_scaler] = scaler.fit_transform(X[features_for_scaler])

    # --- Model Training and Evaluation using 3-Fold Cross-Validation with Ensemble ---
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ensemble_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y_input)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y_input.iloc[train_index], y_input.iloc[val_index]

        # Model 1
        model1 = RandomForestClassifier(n_estimators=model1_n_estimators, random_state=42, n_jobs=-1)
        model1.fit(X_train, y_train)

        # Model 2
        model2 = RandomForestClassifier(n_estimators=model2_n_estimators, random_state=42, n_jobs=-1)
        model2.fit(X_train, y_train)

        # Get probabilities from both models for ensembling
        proba1 = model1.predict_proba(X_val)
        proba2 = model2.predict_proba(X_val)

        # Ensemble: Average probabilities from both models
        avg_proba = (proba1 + proba2) / 2

        # Get final predictions by taking the class with the highest average probability
        # IMPORTANT: Add 1 to `np.argmax` output because Cover_Type labels are 1-indexed (1-7),
        # while `np.argmax` returns 0-indexed class indices. This fixes a common bug.
        y_pred_ensemble = np.argmax(avg_proba, axis=1) + 1

        # Evaluate the ensemble model's performance for the current fold
        accuracy = accuracy_score(y_val, y_pred_ensemble)
        ensemble_accuracies.append(accuracy)

    return np.mean(ensemble_accuracies)

# Dictionary to store results
ablation_results = {}

print("--- Starting Ablation Study ---")

# --- Baseline Run (Full solution as provided) ---
print("Running Baseline (Full solution with all default settings)...")
baseline_accuracy = run_ablation_experiment(X_orig, y_orig)
ablation_results["Baseline (Full solution)"] = baseline_accuracy
print(f"Baseline Average Validation Accuracy: {baseline_accuracy:.4f}\n")

# --- Ablation 1: No Standard Scaling ---
# Disables StandardScaler application to numerical features.
print("Running Ablation 1: No Standard Scaling...")
no_scaling_accuracy = run_ablation_experiment(X_orig, y_orig, enable_standard_scaling=False)
ablation_results["No Standard Scaling"] = no_scaling_accuracy
print(f"Ablation 1 Average Validation Accuracy (No Standard Scaling): {no_scaling_accuracy:.4f}\n")

# --- Ablation 2: Remove Initial Feature Engineering ---
# This disables the creation of 'Hydrology_Distance', 'Hillshade_Index', and 'Elevation_x_Slope'.
print("Running Ablation 2: Remove Initial Feature Engineering (Hydrology_Distance, Hillshade_Index, Elevation_x_Slope)...")
no_initial_fe_accuracy = run_ablation_experiment(X_orig, y_orig, enable_initial_fe=False)
ablation_results["No Initial Feature Engineering"] = no_initial_fe_accuracy
print(f"Ablation 2 Average Validation Accuracy (No Initial Feature Engineering): {no_initial_fe_accuracy:.4f}\n")

# --- Ablation 3: Increase n_estimators for both ensemble models to 200 ---
# This tests the impact of increasing the complexity/power of the base learners in the ensemble.
print("Running Ablation 3: Ensemble n_estimators = 200 for both models...")
increased_estimators_accuracy = run_ablation_experiment(X_orig, y_orig,
                                                        model1_n_estimators=200,
                                                        model2_n_estimators=200)
ablation_results["Ensemble n_estimators = 200"] = increased_estimators_accuracy
print(f"Ablation 3 Average Validation Accuracy (Ensemble n_estimators = 200): {increased_estimators_accuracy:.4f}\n")

# --- Final Analysis ---
print("\n--- Ablation Study Results Summary ---")
for name, acc in ablation_results.items():
    print(f"- {name}: {acc:.4f}")

contributions = {}
print("\n--- Performance Impact from Baseline ---")
for name, acc in ablation_results.items():
    if name != "Baseline (Full solution)":
        performance_change = acc - baseline_accuracy # Positive means improvement, negative means drop
        contributions[name] = performance_change
        print(f"  '{name}': Change from Baseline: {performance_change:+.4f}")

if contributions:
    # Find the modification that caused the largest positive impact (highest accuracy)
    most_beneficial_change_part = max(contributions, key=contributions.get)
    max_performance_gain = contributions[most_beneficial_change_part]

    # Find the modification that caused the largest negative impact (lowest accuracy)
    # This indicates the most crucial positive component that was removed, or a detrimental modification.
    most_detrimental_change_part = min(contributions, key=contributions.get)
    max_performance_drop = contributions[most_detrimental_change_part]

    print("\n--- Conclusion ---")
    if max_performance_gain > 0:
        print(f"The modification that improved performance the most is: '{most_beneficial_change_part}' with a performance gain of {max_performance_gain:.4f}.")
        print(f"This indicates that either removing a detrimental part or making a beneficial adjustment contributed the most.")
    elif max_performance_drop < 0:
        print(f"The part that contributes the most positively to the overall performance (if removed or changed, caused the largest drop) is related to: '{most_detrimental_change_part}', leading to a performance drop of {-max_performance_drop:.4f}.")
        print(f"This implies that the original '{most_detrimental_change_part}' (before modification/removal) was highly beneficial.")
    else:
        print("No significant performance change detected across ablations, or all changes were neutral.")
else:
    print("No ablation results to analyze.")
