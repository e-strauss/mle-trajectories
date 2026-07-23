
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to run the model with a given feature set
def run_ablation_model(X_data, y_data):
    # Dynamically define numerical features present in the current X_data
    # This list includes all potential numerical features (original and engineered)
    all_potential_numerical_features = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        # Engineered features from original solution:
        'Hydrology_Distance', 'Hillshade_Index', 'Elevation_x_Slope',
        'Aspect_sin', 'Aspect_cos', # Aspect transformations
        'Relative_Elevation_to_Hydrology', 'Hillshade_Variation', 'Elevation_x_Fire_Distance',
        # New engineered features from plan_implement_agent_1 context:
        'Northness', 'Eastness'
    ]
    
    # Filter to only include columns actually present in the current X_data dataframe
    numerical_features_to_scale = [col for col in all_potential_numerical_features if col in X_data.columns]

    # Apply Standard Scaling to numerical features
    scaler = StandardScaler()
    # Only scale if there are numerical features identified for scaling
    if numerical_features_to_scale:
        X_data[numerical_features_to_scale] = scaler.fit_transform(X_data[numerical_features_to_scale])

    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    ensemble_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_data, y_data)):
        X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
        y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]

        # Model 1: RandomForestClassifier with n_estimators=50
        model1 = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model1.fit(X_train, y_train)

        # Model 2: RandomForestClassifier with n_estimators=100
        model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model2.fit(X_train, y_train)

        # Get probabilities from both models for ensembling
        proba1 = model1.predict_proba(X_val)
        proba2 = model2.predict_proba(X_val)

        # Ensemble: Average probabilities
        avg_proba = (proba1 + proba2) / 2
        
        # Get final predictions by taking the class with the highest average probability
        # FIX: Ensure predictions are 1-indexed to match y_val (Cover_Type labels are 1-7)
        y_pred_ensemble = np.argmax(avg_proba, axis=1) + 1

        accuracy = accuracy_score(y_val, y_pred_ensemble)
        ensemble_accuracies.append(accuracy)

    return np.mean(ensemble_accuracies)

# Load the training data
train_df = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_df.drop(["Id", "Cover_Type"], axis=1)
y = train_df["Cover_Type"]

# Store original features to easily create copies for ablations
X_original_raw_features = X.copy()

# --- Baseline Model (Full Feature Engineering including new ones) ---
# Create a helper function for Feature Engineering to avoid code repetition
def apply_feature_engineering(df_X, include_northness_eastness=True, include_relative_elevation=True, include_hillshade_variation=True):
    X_fe = df_X.copy()

    # Original Feature Engineering (from base solution)
    X_fe['Hydrology_Distance'] = np.sqrt(X_fe['Horizontal_Distance_To_Hydrology']**2 + X_fe['Vertical_Distance_To_Hydrology']**2)
    X_fe['Hillshade_Index'] = X_fe['Hillshade_9am'] + X_fe['Hillshade_Noon'] + X_fe['Hillshade_3pm']
    X_fe['Elevation_x_Slope'] = X_fe['Elevation'] * X_fe['Slope']

    # Aspect transformations (from original script, also used for Northness/Eastness)
    X_fe['Aspect_rad'] = np.deg2rad(X_fe['Aspect'])
    X_fe['Aspect_sin'] = np.sin(X_fe['Aspect_rad'])
    X_fe['Aspect_cos'] = np.cos(X_fe['Aspect_rad'])
    
    # New Aspect/Slope features (Northness/Eastness) from plan_implement_agent_1 context
    if include_northness_eastness:
        X_fe['Slope_rad'] = np.deg2rad(X_fe['Slope'])
        X_fe['Northness'] = X_fe['Aspect_cos'] * np.sin(X_fe['Slope_rad'])
        X_fe['Eastness'] = X_fe['Aspect_sin'] * np.sin(X_fe['Slope_rad'])

    # Other Feature Engineering (from original train.py)
    if include_relative_elevation:
        X_fe['Relative_Elevation_to_Hydrology'] = X_fe['Elevation'] - X_fe['Vertical_Distance_To_Hydrology']
    
    if include_hillshade_variation:
        # Check if all columns exist before calculating std
        hillshade_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
        if all(col in X_fe.columns for col in hillshade_cols):
            X_fe['Hillshade_Variation'] = X_fe[hillshade_cols].std(axis=1)
        else:
            # Handle cases where columns might be missing if they were ablated (not in this study, but for robustness)
            X_fe['Hillshade_Variation'] = 0 # Or some other default/drop

    X_fe['Elevation_x_Fire_Distance'] = X_fe['Elevation'] * X_fe['Horizontal_Distance_To_Fire_Points']
    
    return X_fe

print("Running Baseline Model (with all features and ensemble fix)...")
X_baseline = apply_feature_engineering(X_original_raw_features)
baseline_accuracy = run_ablation_model(X_baseline.copy(), y)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}\n")

ablation_results = {}
ablation_results['Baseline'] = baseline_accuracy
performance_changes = {}

# --- Ablation 1: Remove Northness and Eastness features ---
print("Running Ablation: No Northness and Eastness features...")
X_ablation_1 = apply_feature_engineering(X_original_raw_features, include_northness_eastness=False)
accuracy_ablation_1 = run_ablation_model(X_ablation_1.copy(), y)
print(f"Ablation (No Northness/Eastness) Accuracy: {accuracy_ablation_1:.4f}")
performance_change_1 = accuracy_ablation_1 - baseline_accuracy
performance_changes['No Northness and Eastness'] = performance_change_1
print(f"Performance Change: {performance_change_1:.4f}\n")

# --- Ablation 2: Remove Relative_Elevation_to_Hydrology feature ---
print("Running Ablation: No Relative_Elevation_to_Hydrology feature...")
X_ablation_2 = apply_feature_engineering(X_original_raw_features, include_relative_elevation=False)
accuracy_ablation_2 = run_ablation_model(X_ablation_2.copy(), y)
print(f"Ablation (No Relative_Elevation_to_Hydrology) Accuracy: {accuracy_ablation_2:.4f}")
performance_change_2 = accuracy_ablation_2 - baseline_accuracy
performance_changes['No Relative_Elevation_to_Hydrology'] = performance_change_2
print(f"Performance Change: {performance_change_2:.4f}\n")

# --- Ablation 3: Remove Hillshade_Variation feature ---
print("Running Ablation: No Hillshade_Variation feature...")
X_ablation_3 = apply_feature_engineering(X_original_raw_features, include_hillshade_variation=False)
accuracy_ablation_3 = run_ablation_model(X_ablation_3.copy(), y)
print(f"Ablation (No Hillshade_Variation) Accuracy: {accuracy_ablation_3:.4f}")
performance_change_3 = accuracy_ablation_3 - baseline_accuracy
performance_changes['No Hillshade_Variation'] = performance_change_3
print(f"Performance Change: {performance_change_3:.4f}\n")


# Determine the part that contributes the most to overall performance
# This is defined as the part whose removal causes the largest absolute change in accuracy.
# If removal causes a drop, it means the feature was positive.
# If removal causes a gain, it means the feature was detrimental.

most_impactful_part = None
max_abs_impact = 0.0

for part, change in performance_changes.items():
    if abs(change) > max_abs_impact:
        max_abs_impact = abs(change)
        most_impactful_part = part

if most_impactful_part:
    impact_value = performance_changes[most_impactful_part]
    if impact_value < 0:
        print(f"The part that contributes the most to the overall performance (largest positive contribution if kept) is: '{most_impactful_part}' (removing it caused a drop of {-impact_value:.4f}).")
    elif impact_value > 0:
        print(f"The part that contributes the most to the overall performance (largest negative contribution, i.e., detrimental) is: '{most_impactful_part}' (removing it caused a gain of {impact_value:.4f}).")
    else:
        print(f"The part that contributes the most to the overall performance (no significant impact) is: '{most_impactful_part}'.")
else:
    print("No ablated parts showed a significant impact on performance.")

