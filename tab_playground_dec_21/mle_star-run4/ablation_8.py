
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Helper function to run the model and get average accuracy ---
def run_model(df_features, df_target, n_splits=3, random_state=42, n_estimators=100):
    y_original = df_target
    y_transformed = y_original - 1 # Map target to 0-6 for sklearn compatibility

    # Handle classes with fewer samples than n_splits for StratifiedKFold
    class_counts = y_transformed.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        # Exclude samples belonging to problematic classes from CV
        indices_to_exclude_from_cv = y_transformed[y_transformed.isin(problematic_classes)].index.tolist()
        X_cv = df_features.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_transformed.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    else:
        X_cv = df_features
        y_cv = y_transformed

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    fold_accuracies = []
    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# Load data
try:
    train_df_original = pd.read_csv('./input/train.csv')
except FileNotFoundError:
    print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
    raise FileNotFoundError("train.csv not found in the './input' directory.")

# --- Baseline run ---
print("--- Running Baseline Model ---")
train_df_baseline = train_df_original.copy()

# Apply all feature engineering steps from the original solution for baseline
train_df_baseline['Hillshade_composite'] = (train_df_baseline['Hillshade_9am'] + train_df_baseline['Hillshade_Noon'] + train_df_baseline['Hillshade_3pm']) / 3
train_df_baseline['Elevation_at_Hydrology'] = train_df_baseline['Elevation'] - train_df_baseline['Vertical_Distance_To_Hydrology']
train_df_baseline['Horizontal_Distance_To_Roadways_log'] = np.log1p(np.clip(train_df_baseline['Horizontal_Distance_To_Roadways'], 0, None)) # New column name for transformed version
train_df_baseline['Hydro_Road_Interaction'] = train_df_baseline['Horizontal_Distance_To_Hydrology'] * train_df_baseline['Horizontal_Distance_To_Roadways_log'] # Use transformed version
train_df_baseline['Elevation_x_Fire_Points'] = train_df_baseline['Elevation'] * train_df_baseline['Horizontal_Distance_To_Fire_Points']

X_baseline = train_df_baseline.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect', 'Horizontal_Distance_To_Roadways'], axis=1) # Drop original Horizontal_Distance_To_Roadways
y_baseline = train_df_baseline['Cover_Type']

baseline_score = run_model(X_baseline, y_baseline)
print(f'Final Validation Performance: {baseline_score:.4f}\n') # Changed for parsing

# --- Ablation 1: Remove log1p transformation on 'Horizontal_Distance_To_Roadways' ---
print("--- Running Ablation 1: Remove log1p on Horizontal_Distance_To_Roadways ---")
train_df_ablation1 = train_df_original.copy()

train_df_ablation1['Hillshade_composite'] = (train_df_ablation1['Hillshade_9am'] + train_df_ablation1['Hillshade_Noon'] + train_df_ablation1['Hillshade_3pm']) / 3
train_df_ablation1['Elevation_at_Hydrology'] = train_df_ablation1['Elevation'] - train_df_ablation1['Vertical_Distance_To_Hydrology']
# Skip log1p transformation for 'Horizontal_Distance_To_Roadways', use original
# Interaction features (using original 'Horizontal_Distance_To_Roadways' value for Hydro_Road_Interaction)
train_df_ablation1['Hydro_Road_Interaction'] = train_df_ablation1['Horizontal_Distance_To_Hydrology'] * train_df_ablation1['Horizontal_Distance_To_Roadways']
train_df_ablation1['Elevation_x_Fire_Points'] = train_df_ablation1['Elevation'] * train_df_ablation1['Horizontal_Distance_To_Fire_Points']

# The 'Horizontal_Distance_To_Roadways' is not transformed, so the original column is used and not dropped/replaced.
# The 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect' are dropped as in baseline.
X_ablation1 = train_df_ablation1.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)
y_ablation1 = train_df_ablation1['Cover_Type']

ablation1_score = run_model(X_ablation1, y_ablation1)
print(f'Ablation 1 Final Validation Performance (No log1p on Roadways): {ablation1_score:.4f}')
print(f'Change from Baseline: {ablation1_score - baseline_score:.4f}\n')

# --- Ablation 2: Remove 'Elevation_x_Fire_Points' interaction feature ---
print("--- Running Ablation 2: Remove Elevation_x_Fire_Points ---")
train_df_ablation2 = train_df_original.copy()

train_df_ablation2['Hillshade_composite'] = (train_df_ablation2['Hillshade_9am'] + train_df_ablation2['Hillshade_Noon'] + train_df_ablation2['Hillshade_3pm']) / 3
train_df_ablation2['Elevation_at_Hydrology'] = train_df_ablation2['Elevation'] - train_df_ablation2['Vertical_Distance_To_Hydrology']
train_df_ablation2['Horizontal_Distance_To_Roadways_log'] = np.log1p(np.clip(train_df_ablation2['Horizontal_Distance_To_Roadways'], 0, None))

# Introduce novel interaction features (only Hydro_Road_Interaction)
train_df_ablation2['Hydro_Road_Interaction'] = train_df_ablation2['Horizontal_Distance_To_Hydrology'] * train_df_ablation2['Horizontal_Distance_To_Roadways_log']
# Skip 'Elevation_x_Fire_Points' creation

# FIX: 'Elevation_x_Fire_Points' was never created in this ablation, so it should not be in the drop list.
X_ablation2 = train_df_ablation2.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect', 'Horizontal_Distance_To_Roadways'], axis=1)
y_ablation2 = train_df_ablation2['Cover_Type']

ablation2_score = run_model(X_ablation2, y_ablation2)
print(f'Ablation 2 Final Validation Performance (No Elevation_x_Fire_Points): {ablation2_score:.4f}')
print(f'Change from Baseline: {ablation2_score - baseline_score:.4f}\n')

# --- Ablation 3: Remove 'Hydro_Road_Interaction' interaction feature ---
print("--- Running Ablation 3: Remove Hydro_Road_Interaction ---")
train_df_ablation3 = train_df_original.copy()

train_df_ablation3['Hillshade_composite'] = (train_df_ablation3['Hillshade_9am'] + train_df_ablation3['Hillshade_Noon'] + train_df_ablation3['Hillshade_3pm']) / 3
train_df_ablation3['Elevation_at_Hydrology'] = train_df_ablation3['Elevation'] - train_df_ablation3['Vertical_Distance_To_Hydrology']
train_df_ablation3['Horizontal_Distance_To_Roadways_log'] = np.log1p(np.clip(train_df_ablation3['Horizontal_Distance_To_Roadways'], 0, None))

# Introduce novel interaction features (only Elevation_x_Fire_Points)
# Skip 'Hydro_Road_Interaction' creation
train_df_ablation3['Elevation_x_Fire_Points'] = train_df_ablation3['Elevation'] * train_df_ablation3['Horizontal_Distance_To_Fire_Points']

# FIX: 'Hydro_Road_Interaction' was never created in this ablation, so it should not be in the drop list.
X_ablation3 = train_df_ablation3.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect', 'Horizontal_Distance_To_Roadways'], axis=1)
y_ablation3 = train_df_ablation3['Cover_Type']

ablation3_score = run_model(X_ablation3, y_ablation3)
print(f'Ablation 3 Final Validation Performance (No Hydro_Road_Interaction): {ablation3_score:.4f}')
print(f'Change from Baseline: {ablation3_score - baseline_score:.4f}\n')

# --- Determine the most contributing part ---
ablation_results = {
    "Baseline": baseline_score,
    "Ablation 1 (No log1p on Horizontal_Distance_To_Roadways)": ablation1_score,
    "Ablation 2 (No Elevation_x_Fire_Points)": ablation2_score,
    "Ablation 3 (No Hydro_Road_Interaction)": ablation3_score,
}

print("\n--- Ablation Study Summary ---")
for name, score in ablation_results.items():
    if name == "Baseline":
        print(f"{name}: {score:.4f}")
    else:
        print(f"{name}: {score:.4f} (Change from Baseline: {score - baseline_score:.4f})")

# Calculate performance drop (baseline_score - ablation_score) for each ablation.
# A positive value means removing the feature caused a drop, thus it was contributing positively.
# A negative value means removing the feature caused an increase, thus it was detrimental or neutral.
performance_drops = {
    "log1p transformation on 'Horizontal_Distance_To_Roadways'": baseline_score - ablation1_score,
    "'Elevation_x_Fire_Points' interaction feature": baseline_score - ablation2_score,
    "'Hydro_Road_Interaction' interaction feature": baseline_score - ablation3_score,
}

# Find the feature whose removal caused the largest positive performance drop.
# This identifies the most positively contributing feature.
most_contributing_feature = None
max_drop = -float('inf')

for feature, drop in performance_drops.items():
    if drop > max_drop:
        max_drop = drop
        most_contributing_feature = feature

if most_contributing_feature is not None and max_drop > 0:
    print(f"\nThe part of the code that contributes the most to the overall performance is: {most_contributing_feature} (Removing it caused a drop of {max_drop:.4f}).")
else:
    # If max_drop is non-positive, it means all removals either improved performance or had no impact.
    print("\nNone of the ablated parts were found to contribute positively to the model's overall performance. Their removal either improved performance or had a negligible effect.")
