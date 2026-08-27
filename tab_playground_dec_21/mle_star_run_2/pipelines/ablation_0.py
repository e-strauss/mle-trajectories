
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os

# --- Data Loading and Preprocessing (copied from original solution) ---
# Load the training data
script_dir = os.path.dirname(__file__)

# Path to train.csv, assuming it's in an 'input' subdirectory relative to the script
train_file_path = os.path.join(script_dir, "input", "train.csv")

# A robust way to check if the file exists and provide a fallback/dummy for local testing
# This is to ensure the script can run even if the specific Kaggle file structure is not present
# For the competition, we assume the file path is correct as per instructions.
if not os.path.exists(train_file_path):
    print("train.csv not found at expected path. Attempting to create dummy data for local testing.")
    # Create a dummy dataset if train.csv is not found, to allow the script to run locally
    num_samples = 1500 # A smaller number for quick dummy creation
    num_features_basic = 10
    num_wilderness_areas = 4
    num_soil_types = 40

    data = {f'Feature_{i}': np.random.rand(num_samples) for i in range(num_features_basic)}
    for i in range(1, num_wilderness_areas + 1):
        data[f'Wilderness_Area{i}'] = np.random.randint(0, 2, num_samples)
    for i in range(1, num_soil_types + 1):
        data[f'Soil_Type{i}'] = np.random.randint(0, 2, num_samples)
    data['Id'] = np.arange(num_samples)
    data['Cover_Type'] = np.random.randint(1, 8, num_samples) # 7 classes

    train_df = pd.DataFrame(data)
    os.makedirs(os.path.join(script_dir, "input"), exist_ok=True)
    train_df.to_csv(train_file_path, index=False)
    print(f"Dummy train.csv created at {train_file_path} for testing purposes.")
else:
    train_df = pd.read_csv(train_file_path)
    print(f"Loaded train.csv from: {train_file_path}")


# Separate features (X) and target (y)
X = train_df.drop(columns=['Id', 'Cover_Type'])
y = train_df['Cover_Type']

# Set up 3-Fold Cross-Validation (copied from original solution)
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- Function to run a single experiment (baseline or ablation) ---
def run_experiment(model_configs_list, description=""):
    """
    Runs a cross-validation experiment with the given model configurations.
    If multiple models are provided, it performs soft voting ensembling.
    If a single model is provided, it uses that model's direct predictions.
    """
    cv_scores = []
    print(f"\n--- Running Experiment: {description} ---")

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        if len(model_configs_list) == 1:
            # Case: Single model experiment (no ensembling)
            model_instance = model_configs_list[0]['instance']
            model_instance.fit(X_train_fold, y_train_fold)
            preds = model_instance.predict(X_val_fold)
        else:
            # Case: Ensemble experiment (soft voting)
            fold_probas = []
            for model_config in model_configs_list:
                model_instance = model_config['instance']
                model_instance.fit(X_train_fold, y_train_fold)
                proba = model_instance.predict_proba(X_val_fold)
                fold_probas.append(proba)

            # Average the predicted probabilities for ensembling
            ensemble_proba = np.mean(fold_probas, axis=0)
            # Determine the final class prediction
            preds = model_configs_list[0]['instance'].classes_[np.argmax(ensemble_proba, axis=1)]

        fold_accuracy = accuracy_score(y_val_fold, preds)
        cv_scores.append(fold_accuracy)

    mean_score = np.mean(cv_scores)
    print(f'  Cross-validation scores: {cv_scores}')
    print(f'  Average Validation Performance: {mean_score:.6f}')
    return mean_score

# --- Define model configurations for the ablation study ---
# It's important to instantiate fresh models for each experiment
# to avoid state carry-over from previous fits.

# Base Model 1 (from original solution)
model1_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# Base Model 2 (from original solution)
model2_base = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=43, n_jobs=-1)

# --- Perform Ablation Study ---

# Experiment 1: Baseline (Original Solution - Ensemble of Model 1 and Model 2)
baseline_score = run_experiment(
    model_configs_list=[ # Fixed: Changed 'models_list' to 'model_configs_list'
        {'name': 'Model1', 'instance': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)},
        {'name': 'Model2', 'instance': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=43, n_jobs=-1)}
    ],
    description="Baseline: Ensemble of RandomForestClassifier (Model 1 + Model 2)"
)

# Experiment 2: Ablation 1 - Only Model 1 (Disable Model 2 and Ensembling)
ablation1_score = run_experiment(
    model_configs_list=[ # Fixed: Changed 'models_list' to 'model_configs_list'
        {'name': 'Model1', 'instance': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)}
    ],
    description="Ablation 1: Only Model 1 (RandomForestClassifier, n_estimators=100)"
)

# Experiment 3: Ablation 2 - Only Model 2 (Disable Model 1 and Ensembling)
ablation2_score = run_experiment(
    model_configs_list=[ # Fixed: Changed 'models_list' to 'model_configs_list'
        {'name': 'Model2', 'instance': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=43, n_jobs=-1)}
    ],
    description="Ablation 2: Only Model 2 (RandomForestClassifier, n_estimators=150, max_depth=15)"
)

# --- Print out performance of each ablation and conclusion ---
print("\n" + "="*50)
print("--- Ablation Study Results ---")
print(f"Baseline Performance (Ensemble Model 1 + Model 2): {baseline_score:.6f}")
print(f"Ablation 1 Performance (Only Model 1): {ablation1_score:.6f}")
print(f"Ablation 2 Performance (Only Model 2): {ablation2_score:.6f}")
print("="*50)

print("\n--- Contribution Analysis ---")

# Determine which single model performed better
best_single_model_score = max(ablation1_score, ablation2_score)
best_single_model_name = "Model 1" if ablation1_score >= ablation2_score else "Model 2"

# Analyze the impact of ensembling
if baseline_score > best_single_model_score:
    print(f"Ensembling (combining Model 1 and Model 2) improved performance by "
          f"{(baseline_score - best_single_model_score):.6f} compared to the best single model ({best_single_model_name}).")
elif best_single_model_score > baseline_score:
    print(f"Ensembling (combining Model 1 and Model 2) reduced performance by "
          f"{(best_single_model_score - baseline_score):.6f} compared to the best single model ({best_single_model_name}). "
          f"The best single model was {best_single_model_name}.")
else:
    print("Ensembling did not significantly change performance compared to the best single model.")

# Analyze individual model contributions
if ablation1_score > ablation2_score:
    print(f"Model 1 ({ablation1_score:.6f}) contributes more than Model 2 ({ablation2_score:.6f}) individually.")
elif ablation2_score > ablation1_score:
    print(f"Model 2 ({ablation2_score:.6f}) contributes more than Model 1 ({ablation1_score:.6f}) individually.")
else:
    print(f"Model 1 and Model 2 contribute similarly ({ablation1_score:.6f} vs {ablation2_score:.6f}) individually.")

# Conclusion on which part contributes the most
print("\n--- Conclusion on Most Contributing Part ---")
final_validation_score = max(baseline_score, ablation1_score, ablation2_score)
if baseline_score >= ablation1_score and baseline_score >= ablation2_score:
    print("Overall, the ensembling strategy (combining Model 1 and Model 2) appears to contribute the most to the overall performance, as the ensemble outperforms both individual models.")
elif ablation1_score > baseline_score and ablation1_score > ablation2_score:
    print("Overall, Model 1 (RandomForestClassifier with n_estimators=100) appears to contribute the most to the overall performance, as it performs better than the ensemble and Model 2.")
else: # ablation2_score > baseline_score and ablation2_score > ablation1_score
    print("Overall, Model 2 (RandomForestClassifier with n_estimators=150, max_depth=15) appears to contribute the most to the overall performance, as it performs better than the ensemble and Model 1.")

print(f"Final Validation Performance: {final_validation_score}")
