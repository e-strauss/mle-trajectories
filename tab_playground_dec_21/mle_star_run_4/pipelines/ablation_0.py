
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Helper function to run a model with specific configurations ---
def run_model_and_evaluate(X_full, y_full, n_splits, model_params, feature_cols=None):
    """
    Runs a RandomForestClassifier with specified parameters and features,
    performing StratifiedKFold cross-validation and handling problematic classes.
    """
    if feature_cols is not None:
        X_current = X_full[feature_cols]
    else:
        X_current = X_full

    # Handle classes with fewer samples than n_splits for StratifiedKFold
    class_counts = y_full.value_counts()
    problematic_classes = class_counts[class_counts < n_splits].index.tolist()

    if problematic_classes:
        indices_to_exclude_from_cv = y_full[y_full.isin(problematic_classes)].index.tolist()
        X_cv = X_current.drop(indices_to_exclude_from_cv).reset_index(drop=True)
        y_cv = y_full.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    else:
        X_cv = X_current
        y_cv = y_full

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = RandomForestClassifier(**model_params)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)

# --- Main execution block for ablation study ---
if __name__ == "__main__":
    # Load data
    try:
        train_df = pd.read_csv('./input/train.csv')
    except FileNotFoundError:
        print("Error: train.csv not found in the './input' directory. Please ensure the file is present.")
        exit()

    # Separate features and target
    X_initial = train_df.drop(['Id', 'Cover_Type'], axis=1)
    y_initial = train_df['Cover_Type']

    # Map target to 0-6 for sklearn compatibility
    y_transformed = y_initial - 1

    n_splits = 3 # As per task description for 3-Fold CV

    print("Running Ablation Study...\n")

    # Define common model parameters
    base_model_params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }

    # Identify Soil_Type columns for ablation
    soil_type_cols = [col for col in X_initial.columns if 'Soil_Type' in col]
    # Identify non-Soil_Type columns
    non_soil_type_cols = [col for col in X_initial.columns if 'Soil_Type' not in col]


    # 1. Baseline Solution
    print("--- Baseline Solution (Original) ---")
    original_performance = run_model_and_evaluate(
        X_full=X_initial,
        y_full=y_transformed,
        n_splits=n_splits,
        model_params=base_model_params,
        feature_cols=X_initial.columns.tolist()
    )
    print(f'Original Solution Performance: {original_performance:.4f}\n')

    # 2. Ablation 1: Reduced n_estimators
    print("--- Ablation 1: RandomForestClassifier with n_estimators=50 ---")
    ablation1_model_params = base_model_params.copy()
    ablation1_model_params['n_estimators'] = 50
    ablation1_performance = run_model_and_evaluate(
        X_full=X_initial,
        y_full=y_transformed,
        n_splits=n_splits,
        model_params=ablation1_model_params,
        feature_cols=X_initial.columns.tolist()
    )
    print(f'Ablation 1 Performance (n_estimators=50): {ablation1_performance:.4f}\n')

    # 3. Ablation 2: Exclude all Soil_Type features
    print("--- Ablation 2: Exclude all Soil_Type features ---")
    ablation2_performance = run_model_and_evaluate(
        X_full=X_initial,
        y_full=y_transformed,
        n_splits=n_splits,
        model_params=base_model_params,
        feature_cols=non_soil_type_cols # Use features without Soil_Type columns
    )
    print(f'Ablation 2 Performance (No Soil_Type features): {ablation2_performance:.4f}\n')

    # Compare and conclude
    print("\n--- Ablation Study Summary ---")
    print(f"Baseline Performance: {original_performance:.4f}")
    print(f"Ablation 1 (Reduced n_estimators=50) Performance: {ablation1_performance:.4f}")
    print(f"Ablation 2 (No Soil_Type features) Performance: {ablation2_performance:.4f}")

    # Calculate performance drops
    drop_from_ablation1 = original_performance - ablation1_performance
    drop_from_ablation2 = original_performance - ablation2_performance

    # Determine which part contributes the most (largest drop indicates highest contribution)
    max_drop = 0
    most_contributing_part = "None of the ablated parts showed a significant positive contribution (or performance improved in some ablations)."

    # Set a small threshold to consider a drop significant
    SIGNIFICANT_DROP_THRESHOLD = 0.0001

    if drop_from_ablation1 > max_drop:
        max_drop = drop_from_ablation1
        most_contributing_part = "the default number of estimators (n_estimators=100) in the RandomForestClassifier"

    if drop_from_ablation2 > max_drop:
        max_drop = drop_from_ablation2
        most_contributing_part = "the 'Soil_Type' features"

    print("\nAnalysis of Contributions:")
    if max_drop > SIGNIFICANT_DROP_THRESHOLD:
        print(f"The part of the code that contributes the most to the overall performance is: {most_contributing_part}.")
        print(f"This is inferred because its ablation resulted in the largest performance drop of {max_drop:.4f}.")
    else:
        print(most_contributing_part) # This will print the default message if no significant drop occurred.

