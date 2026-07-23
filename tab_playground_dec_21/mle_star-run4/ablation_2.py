
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def run_experiment(df_path, feature_modifications, n_splits=3, random_state=42):
    """
    Runs a single experiment with specified feature modifications and returns the average accuracy.
    """
    try:
        train_df_base = pd.read_csv(df_path)
    except FileNotFoundError:
        print(f"Error: {df_path} not found. Please ensure the file is present.")
        raise FileNotFoundError(f"{df_path} not found.")

    train_df = train_df_base.copy() # Make a fresh copy for each experiment

    # Apply feature engineering based on modifications
    if feature_modifications.get('create_hillshade_composite', True):
        train_df['Hillshade_composite'] = (train_df['Hillshade_9am'] + train_df['Hillshade_Noon'] + train_df['Hillshade_3pm']) / 3

    # Define features to drop from the final X
    features_to_drop = ['Id', 'Cover_Type']

    # Handle original Hillshade columns based on 'drop_original_hillshade' flag
    if feature_modifications.get('drop_original_hillshade', True):
        features_to_drop.extend(['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])
    
    # Handle specific ablations for other features
    if feature_modifications.get('remove_hydrology_features', False):
        features_to_drop.extend(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])

    if feature_modifications.get('remove_roadways_feature', False):
        features_to_drop.append('Horizontal_Distance_To_Roadways')

    # Identify all columns that should be in X
    X_cols = [col for col in train_df.columns if col not in features_to_drop]
    
    X = train_df[X_cols]
    
    y_original = train_df_base['Cover_Type']
    y_transformed = y_original - 1

    # --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
    class_counts = y_transformed.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_transformed[y_transformed.isin(problematic_classes)].index.tolist()
        X_cv = X.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_transformed.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    else:
        X_cv = X
        y_cv = y_transformed


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)


# Define base path for data
DATA_PATH = './input/train.csv'

# --- Baseline Experiment ---
# Configuration mimicking the original script's feature engineering
baseline_modifications = {
    'create_hillshade_composite': True,
    'drop_original_hillshade': True,  # Original script drops these and uses composite
    'remove_hydrology_features': False,
    'remove_roadways_feature': False
}
baseline_score = run_experiment(DATA_PATH, baseline_modifications)
print(f'Baseline Performance: {baseline_score:.4f}')

# --- Ablation 1: Remove Hydrology Features (Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology) ---
ablation1_modifications = baseline_modifications.copy()
ablation1_modifications['remove_hydrology_features'] = True
ablation1_score = run_experiment(DATA_PATH, ablation1_modifications)
print(f'Ablation 1 Performance (Removed Hydrology features): {ablation1_score:.4f} (Change: {ablation1_score - baseline_score:.4f})')

# --- Ablation 2: Remove Horizontal_Distance_To_Roadways ---
ablation2_modifications = baseline_modifications.copy()
ablation2_modifications['remove_roadways_feature'] = True
ablation2_score = run_experiment(DATA_PATH, ablation2_modifications)
print(f'Ablation 2 Performance (Removed Horizontal_Distance_To_Roadways): {ablation2_score:.4f} (Change: {ablation2_score - baseline_score:.4f})')

# --- Ablation 3: Use Original Hillshade Features instead of Composite ---
ablation3_modifications = baseline_modifications.copy()
ablation3_modifications['create_hillshade_composite'] = False # Do not create the composite
ablation3_modifications['drop_original_hillshade'] = False    # Do NOT drop the originals (so they are used in X)
ablation3_score = run_experiment(DATA_PATH, ablation3_modifications)
print(f'Ablation 3 Performance (Used Original Hillshade instead of Composite): {ablation3_score:.4f} (Change: {ablation3_score - baseline_score:.4f})')

# Determine the most contributing part based on performance drops
results = {
    "Baseline": baseline_score,
    "Removed Hydrology features": ablation1_score,
    "Removed Horizontal_Distance_To_Roadways": ablation2_score,
    "Used Original Hillshade instead of Composite": ablation3_score
}

performance_drops = {}
if baseline_score > ablation1_score:
    performance_drops["Hydrology Features"] = baseline_score - ablation1_score
if baseline_score > ablation2_score:
    performance_drops["Horizontal_Distance_To_Roadways"] = baseline_score - ablation2_score
if baseline_score > ablation3_score:
    performance_drops["Hillshade Composite Feature (vs. Originals)"] = baseline_score - ablation3_score

if performance_drops:
    most_contributing_part = max(performance_drops, key=performance_drops.get)
    print(f'The part of the code that contributes the most to the overall performance is: "{most_contributing_part}"')
else:
    print('No significant performance drop observed from any ablation. All ablated parts contribute minimally or negatively.')
