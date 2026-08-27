
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
X_original = train_df.drop(["Id", "Cover_Type"], axis=1)
y_original = train_df["Cover_Type"]

# Store results
ablation_results = {}

def run_experiment(X_data, y_data, description, 
                   apply_aspect_features=True,
                   apply_elevation_x_fire_distance=True, 
                   model1_n_estimators_override=None):

    print(f"\n--- Running Experiment: {description} ---")
    X = X_data.copy()
    y = y_data.copy()

    # --- Feature Engineering (from base solution, or conditional for ablation) ---
    # These core engineered features are always included in the baseline for this study
    # unless they are explicitly targeted for ablation.
    X['Hydrology_Distance'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2)
    X['Hillshade_Index'] = X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']
    X['Elevation_x_Slope'] = X['Elevation'] * X['Slope']

    # Conditional Feature Engineering for ablation
    if apply_aspect_features:
        X['Aspect_sin'] = np.sin(np.deg2rad(X['Aspect']))
        X['Aspect_cos'] = np.cos(np.deg2rad(X['Aspect']))

    # These features are also part of the base solution, not targeted for this ablation study
    X['Relative_Elevation_to_Hydrology'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']
    X['Hillshade_Variation'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].std(axis=1)

    if apply_elevation_x_fire_distance:
        X['Elevation_x_Fire_Distance'] = X['Elevation'] * X['Horizontal_Distance_To_Fire_Points']

    # --- Define numerical features for scaling dynamically ---
    # All features that are not 'Wilderness_Area' or 'Soil_Type' (which are one-hot encoded binary)
    # will be considered numerical and scaled. This correctly includes all original and engineered numerical features.
    numerical_features_to_scale = [col for col in X.columns if not col.startswith(('Wilderness_Area', 'Soil_Type'))]
    
    # Apply Standard Scaling to numerical features
    scaler = StandardScaler()
    X[numerical_features_to_scale] = scaler.fit_transform(X[numerical_features_to_scale])

    # --- Model Training and Evaluation using 3-Fold Cross-Validation with Ensemble ---
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    ensemble_accuracies = []

    # Use override for model1_n_estimators if provided, otherwise default to 50
    model1_n_estimators = model1_n_estimators_override if model1_n_estimators_override is not None else 50
    model2_n_estimators = 100 # This remains 100 as per original script

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model1 = RandomForestClassifier(n_estimators=model1_n_estimators, random_state=42, n_jobs=-1)
        model1.fit(X_train, y_train)

        model2 = RandomForestClassifier(n_estimators=model2_n_estimators, random_state=42, n_jobs=-1)
        model2.fit(X_train, y_train)

        proba1 = model1.predict_proba(X_val)
        proba2 = model2.predict_proba(X_val)

        avg_proba = (proba1 + proba2) / 2
        # FIX for previous ablation study observation: np.argmax returns 0-indexed classes.
        # Assuming Cover_Type labels are 1-indexed (common in Kaggle), add 1 to align with actual labels.
        y_pred_ensemble = np.argmax(avg_proba, axis=1) + 1

        accuracy = accuracy_score(y_val, y_pred_ensemble)
        ensemble_accuracies.append(accuracy)

    final_validation_score = np.mean(ensemble_accuracies)
    print(f"Average Validation Accuracy: {final_validation_score:.4f}")
    return final_validation_score

# --- Ablation Study Execution ---

# 1. Corrected Baseline
#    This baseline incorporates fixes identified from previous studies:
#    - Fixed ensemble prediction (y_pred + 1 for 1-indexed labels).
#    - Fixed scaling: all engineered numerical features are now correctly scaled.
baseline_score = run_experiment(X_original, y_original, "Corrected Baseline (All Features, Full Ensemble, Fixed Bugs)")
ablation_results["Corrected Baseline"] = baseline_score

# 2. Ablation: Remove Aspect Transformations (Aspect_sin, Aspect_cos)
#    Evaluates the contribution of transforming 'Aspect' into sine and cosine components.
score_no_aspect = run_experiment(X_original, y_original, "Ablation: No Aspect Transformations (sin/cos)",
                                 apply_aspect_features=False)
ablation_results["No Aspect Transformations"] = score_no_aspect

# 3. Ablation: Remove Interaction Term (Elevation_x_Fire_Distance)
#    Evaluates the impact of a specific interaction feature combining Elevation and Fire Points.
score_no_elev_x_fire = run_experiment(X_original, y_original, "Ablation: No Elevation_x_Fire_Distance",
                                      apply_elevation_x_fire_distance=False)
ablation_results["No Elevation_x_Fire_Distance"] = score_no_elev_x_fire

# 4. Ablation: Reduced Ensemble Diversity (Model 1 n_estimators = 100)
#    Tests if having different `n_estimators` (50 vs 100) in the ensemble contributes to performance.
#    Here, both models will have `n_estimators=100`.
score_less_diversity = run_experiment(X_original, y_original, "Ablation: Reduced Ensemble Diversity (model1_n_estimators=100)",
                                      model1_n_estimators_override=100)
ablation_results["Reduced Ensemble Diversity"] = score_less_diversity


# --- Print all results and determine the most contributing part ---
print("\n--- Ablation Study Results ---")
for key, score in ablation_results.items():
    print(f"{key}: {score:.4f}")

# Calculate performance drops from the corrected baseline
# A positive drop means removal hurt performance (component contributed positively)
# A negative drop means removal improved performance (component contributed negatively)
performance_drops = {}
for key, score in ablation_results.items():
    if key != "Corrected Baseline":
        performance_drops[key] = baseline_score - score

print("\n--- Performance Impact (Drop from Corrected Baseline) ---")
for key, drop in performance_drops.items():
    print(f"Removing/Modifying '{key}' resulted in a performance drop of: {drop:.4f}")

if not performance_drops:
    print("\nNo ablations performed to determine contributions.")
else:
    # Find the ablation that caused the largest positive drop (i.e., its removal hurt performance the most)
    # This indicates the removed component contributed positively and most significantly.
    max_positive_contributor = None
    max_positive_drop_value = -np.inf

    # Find the ablation that caused the largest negative drop (i.e., its removal improved performance the most)
    # This indicates the removed component contributed negatively and most significantly.
    max_negative_contributor = None
    max_negative_drop_value = np.inf

    for component, drop in performance_drops.items():
        if drop > max_positive_drop_value:
            max_positive_drop_value = drop
            max_positive_contributor = component
        if drop < max_negative_drop_value: # drop is negative, meaning a gain from removal
            max_negative_drop_value = drop
            max_negative_contributor = component
    
    # Determine the most contributing part based on magnitude of impact
    # Use a small threshold (e.g., 0.0001) to consider a change as significant.
    if abs(max_positive_drop_value) > abs(max_negative_drop_value):
        if max_positive_drop_value > 0.0001: 
            print(f"\nThe part that contributes the most positively to the overall performance (largest drop when removed) is: '{max_positive_contributor}' with an estimated contribution of {max_positive_drop_value:.4f}.")
        else:
            print("\nBased on the current ablations, no single part showed a significantly larger positive or negative contribution.")
    else: 
        if max_negative_drop_value < -0.0001: 
            print(f"\nThe part that was most detrimental to performance (largest gain when removed) is: '{max_negative_contributor}' with an estimated detrimental impact of {-max_negative_drop_value:.4f}.")
        else:
            print("\nBased on the current ablations, no single part showed a significantly larger positive or negative contribution.")

