
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Define a function to run a specific configuration for ablation study
def run_ablation_scenario(
    scenario_name,
    use_hillshade_fe=True,
    use_distance_fe=True,
    use_accessibility_fe=True,
    use_minmax_scaler=True,
    base_train_df=None # Pass the original DataFrame to avoid reloading for each scenario
):
    # Start with a fresh copy of the base DataFrame for each scenario to ensure isolation
    current_df = base_train_df.copy()

    # --- Feature Engineering ---
    # Apply Hillshade feature engineering if enabled
    if use_hillshade_fe:
        solar_altitude_rad = np.deg2rad(45) # Typical solar altitude
        aspect_rad = np.deg2rad(current_df['Aspect'])
        slope_rad = np.deg2rad(current_df['Slope'])

        # Hillshade at 9 AM (azimuth 315 degrees)
        azimuth_9am_rad = np.deg2rad(315)
        current_df['Hillshade_9am'] = 255 * (
            (np.cos(solar_altitude_rad) * np.sin(slope_rad)) +
            (np.sin(solar_altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_9am_rad - aspect_rad))
        )
        current_df['Hillshade_9am'] = np.clip(current_df['Hillshade_9am'], 0, 255)

        # Hillshade at Noon (azimuth 180 degrees)
        azimuth_noon_rad = np.deg2rad(180)
        current_df['Hillshade_Noon'] = 255 * (
            (np.cos(solar_altitude_rad) * np.sin(slope_rad)) +
            (np.sin(solar_altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_noon_rad - aspect_rad))
        )
        current_df['Hillshade_Noon'] = np.clip(current_df['Hillshade_Noon'], 0, 255)

        # Hillshade at 3 PM (azimuth 225 degrees)
        azimuth_3pm_rad = np.deg2rad(225)
        current_df['Hillshade_3pm'] = 255 * (
            (np.cos(solar_altitude_rad) * np.sin(slope_rad)) +
            (np.sin(solar_altitude_rad) * np.cos(slope_rad) * np.cos(azimuth_3pm_rad - aspect_rad))
        )
        current_df['Hillshade_3pm'] = np.clip(current_df['Hillshade_3pm'], 0, 255)

    # Apply Euclidean Distance to Hydrology feature engineering if enabled
    if use_distance_fe:
        current_df['Euclidean_Distance_To_Hydrology'] = np.sqrt(
            current_df['Horizontal_Distance_To_Hydrology']**2 +
            current_df['Vertical_Distance_To_Hydrology']**2
        )

    # Apply Accessibility Index feature engineering if enabled
    if use_accessibility_fe:
        # Determine Euclidean Distance component for Accessibility Index
        # If use_distance_fe is False, Euclidean_Distance_To_Hydrology will not be in current_df, so use 0.
        euclidean_dist_val = current_df['Euclidean_Distance_To_Hydrology'] if 'Euclidean_Distance_To_Hydrology' in current_df.columns else 0
        current_df['Accessibility_Index'] = (
            current_df['Horizontal_Distance_To_Roadways'] +
            current_df['Horizontal_Distance_To_Fire_Points'] +
            euclidean_dist_val
        )

    # Separate features (X) and target (y)
    X = current_df.drop(columns=['Id', 'Cover_Type'])
    y = current_df['Cover_Type']

    # Identify numerical features to scale, excluding pre-encoded categorical features
    features_to_scale = []
    if use_minmax_scaler:
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features_to_exclude = [col for col in numerical_cols if 'Wilderness_Area' in col or 'Soil_Type' in col]
        features_to_scale = [col for col in numerical_cols if col not in categorical_features_to_exclude]
        features_to_scale = [f for f in features_to_scale if f in X.columns] # Ensure features exist in current X

    # Initialize models for ensembling
    model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model2 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=43, n_jobs=-1)

    # Set up 3-Fold Stratified Cross-Validation
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    scaler = MinMaxScaler() # Initialize scaler

    # Perform K-Fold Cross-Validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Apply scaling within the fold if enabled and features to scale exist
        if use_minmax_scaler and features_to_scale:
            X_train_scaled = X_train_fold.copy()
            X_val_scaled = X_val_fold.copy()

            scaler.fit(X_train_scaled[features_to_scale]) # Fit on training data only
            X_train_scaled[features_to_scale] = scaler.transform(X_train_scaled[features_to_scale])
            X_val_scaled[features_to_scale] = scaler.transform(X_val_scaled[features_to_scale])

            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
        else:
            X_train_final = X_train_fold
            X_val_final = X_val_fold

        # Train models and get probability predictions
        model1.fit(X_train_final, y_train_fold)
        proba1 = model1.predict_proba(X_val_final)

        model2.fit(X_train_final, y_train_fold)
        proba2 = model2.predict_proba(X_val_final)

        # Ensemble: Average predicted probabilities (soft voting)
        ensemble_proba = (proba1 + proba2) / 2
        ensemble_preds = model1.classes_[np.argmax(ensemble_proba, axis=1)]

        # Calculate accuracy for the current fold
        fold_accuracy = accuracy_score(y_val_fold, ensemble_preds)
        cv_scores.append(fold_accuracy)

    # Return the mean cross-validation score for the scenario
    final_score = np.mean(cv_scores)
    return final_score

# --- Main script execution for ablation study ---

# Load the training data once
script_dir = os.path.dirname(__file__)
train_file_path = os.path.join(script_dir, "input", "train.csv")
original_train_df = pd.read_csv(train_file_path)

results = {}

# 1. Original Solution Baseline (as provided in the initial prompt)
original_X = original_train_df.drop(columns=['Id', 'Cover_Type'])
original_y = original_train_df['Cover_Type']
original_model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
original_model2 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=43, n_jobs=-1)
original_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
original_cv_scores = []
for fold, (train_index, val_index) in enumerate(original_skf.split(original_X, original_y)):
    X_train_fold, X_val_fold = original_X.iloc[train_index], original_X.iloc[val_index]
    y_train_fold, y_val_fold = original_y.iloc[train_index], original_y.iloc[val_index]

    original_model1.fit(X_train_fold, y_train_fold)
    proba1 = original_model1.predict_proba(X_val_fold)
    original_model2.fit(X_train_fold, y_train_fold)
    proba2 = original_model2.predict_proba(X_val_fold)

    ensemble_proba = (proba1 + proba2) / 2
    ensemble_preds = original_model1.classes_[np.argmax(ensemble_proba, axis=1)]
    original_cv_scores.append(accuracy_score(y_val_fold, ensemble_preds))
results["Original Solution Baseline"] = np.mean(original_cv_scores)
print(f'Performance of Original Solution Baseline: {results["Original Solution Baseline"]:.6f}')


# 2. Full Integrated Solution (incorporating all proposed FE and Scaling) - This is the base for ablation
results["Full Integrated Solution (all FE + Scaling)"] = run_ablation_scenario(
    "Full Integrated Solution (all FE + Scaling)",
    use_hillshade_fe=True,
    use_distance_fe=True,
    use_accessibility_fe=True,
    use_minmax_scaler=True,
    base_train_df=original_train_df
)
print(f'Performance of Full Integrated Solution (all FE + Scaling): {results["Full Integrated Solution (all FE + Scaling)"]:.6f}')


# 3. Ablation 1: No Hillshade Features (from the Full Integrated Solution)
results["Ablation: No Hillshade Features"] = run_ablation_scenario(
    "Ablation: No Hillshade Features",
    use_hillshade_fe=False, # Disabled
    use_distance_fe=True,
    use_accessibility_fe=True,
    use_minmax_scaler=True,
    base_train_df=original_train_df
)
print(f'Performance of Ablation: No Hillshade Features: {results["Ablation: No Hillshade Features"]:.6f}')


# 4. Ablation 2: No Distance/Accessibility Features (from the Full Integrated Solution)
results["Ablation: No Distance/Accessibility Features"] = run_ablation_scenario(
    "Ablation: No Distance/Accessibility Features",
    use_hillshade_fe=True,
    use_distance_fe=False, # Disabled
    use_accessibility_fe=False, # Disabled
    use_minmax_scaler=True,
    base_train_df=original_train_df
)
print(f'Performance of Ablation: No Distance/Accessibility Features: {results["Ablation: No Distance/Accessibility Features"]:.6f}')


# 5. Ablation 3: No Scaling (from the Full Integrated Solution)
results["Ablation: No Scaling"] = run_ablation_scenario(
    "Ablation: No Scaling",
    use_hillshade_fe=True,
    use_distance_fe=True,
    use_accessibility_fe=True,
    use_minmax_scaler=False, # Disabled
    base_train_df=original_train_df
)
print(f'Performance of Ablation: No Scaling: {results["Ablation: No Scaling"]:.6f}')


# --- Determine the most contributing part ---
full_integrated_score = results["Full Integrated Solution (all FE + Scaling)"]

# Calculate the impact of each ablated component by comparing its absence to the full solution
impact_hillshade = full_integrated_score - results["Ablation: No Hillshade Features"]
impact_distance_accessibility = full_integrated_score - results["Ablation: No Distance/Accessibility Features"]
impact_minmax_scaler = full_integrated_score - results["Ablation: No Scaling"]

impacts = {
    "Hillshade Features": impact_hillshade,
    "Distance and Accessibility Features": impact_distance_accessibility,
    "MinMaxScaler": impact_minmax_scaler
}

# The component whose removal causes the largest performance drop (i.e., has the largest positive impact value)
# is considered the most positively contributing part.
most_contributing_part_name = max(impacts, key=impacts.get)
most_contributing_impact = impacts[most_contributing_part_name]

# Print out what part of the code contributes the most to the overall performance
if most_contributing_impact > 0:
    print(f"\nThe '{most_contributing_part_name}' contributes the most to the overall performance, as its removal caused the largest performance drop.")
elif most_contributing_impact < 0:
    print(f"\nNo single component had a positive impact upon removal. The '{most_contributing_part_name}' had the least negative impact (or contributed least negatively).")
else:
    print(f"\nNo single component had a significant impact on the overall performance in this ablation study.")
