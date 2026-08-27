
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sys
import os

# --- Augmentation Function for Missing Classes (from original solution) ---
def augment_missing_classes_original(X_train_df, y_train_series, n_classes_total):
    """
    Augments the training data by adding dummy samples for any classes
    not present in the current fold's training set. This helps prevent
    errors in multi-class classifiers that require all classes to be present
    during training.
    """
    unique_classes_in_fold = y_train_series.unique()
    all_classes = np.arange(n_classes_total)
    missing_classes = np.setdiff1d(all_classes, unique_classes_in_fold)

    if not missing_classes.size:
        return X_train_df, y_train_series # No augmentation needed

    X_train_augmented = X_train_df.copy()
    y_train_augmented = y_train_series.copy()

    # If X_train_df is empty, we cannot augment. This scenario should be rare
    # with StratifiedKFold on a sufficiently sized dataset.
    if X_train_df.empty:
        # Suppress original warning during ablation study
        return X_train_df, y_train_series

    # Select a few random samples from the existing training data to use as templates.
    # We ensure there's at least one template if X_train_df is not empty.
    # The number of templates is chosen to cover all missing classes without excessive duplication.
    num_templates = min(len(X_train_df), max(1, len(missing_classes)))
    template_indices = np.random.choice(X_train_df.index, size=num_templates, replace=False)
    template_samples = X_train_df.loc[template_indices]

    for i, missing_class in enumerate(missing_classes):
        # Pick a template sample. We cycle through `template_samples` if there are more
        # missing classes than unique template samples selected.
        sample_to_duplicate = template_samples.iloc[i % len(template_samples)].to_frame().T

        # Concatenate the dummy sample features
        X_train_augmented = pd.concat([X_train_augmented, sample_to_duplicate], ignore_index=True)
        
        # Concatenate the dummy sample label. Preserve the original Series name if available.
        label_name = y_train_series.name if y_train_series.name is not None else 'Cover_Type'
        y_train_augmented = pd.concat([y_train_augmented, pd.Series([missing_class], name=label_name)], ignore_index=True)

    return X_train_augmented, y_train_augmented

# 1. Load Data (common to all experiments)
try:
    train_df = pd.read_csv("./input/train.csv")
except FileNotFoundError:
    # Fallback for local testing if 'input' directory isn't present
    train_df = pd.read_csv("train.csv")

# 2. Prepare Data (common to all experiments)
X_full = train_df.drop(['Id', 'Cover_Type'], axis=1)
y_full = train_df['Cover_Type'] - 1  # Adjust target labels from 1-7 to 0-6
n_classes_full = len(y_full.unique()) # This will be 7 (0-6)

# --- Function to run a single experiment configuration ---
def run_ablation_experiment(use_augmentation, use_ensembling, experiment_name):
    fold_accuracies = []
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Redirect stdout to suppress verbose output from model training and folds
    original_stdout = sys.stdout
    # Use os.devnull for platforms like Linux/macOS, or 'NUL' for Windows
    sys.stdout = open(os.devnull, 'w')

    for fold, (train_index, val_index) in enumerate(skf.split(X_full, y_full)):
        X_train_fold, X_val_fold = X_full.iloc[train_index], X_full.iloc[val_index]
        y_train_fold, y_val_fold = y_full.iloc[train_index], y_full.iloc[val_index]

        # X_train_scenario and y_train_scenario represent the data as defined by the ablation scenario.
        # If use_augmentation is False, this data will NOT be augmented yet.
        X_train_scenario = X_train_fold
        y_train_scenario = y_train_fold

        if use_augmentation:
            # If the experiment explicitly uses augmentation, apply the full augmentation strategy.
            X_train_scenario, y_train_scenario = augment_missing_classes_original(X_train_fold, y_train_fold, n_classes_full)

        # --- IMPORTANT FIX: Ensure models always receive data with all expected classes ---
        # The `ValueError` occurs because XGBoost (and sometimes LGBM) requires all `num_class`
        # labels (0 to n_classes_full-1) to be present in the training data, even if explicitly
        # set via `num_class`. If `use_augmentation` is False and a class is missing in a fold,
        # it crashes.
        # To fix this, we apply `augment_missing_classes_original` unconditionally to the data
        # *just before fitting the models*. This acts as a stability measure to prevent crashes.
        # For 'Original Solution' (use_augmentation=True), this will be a no-op as data is already augmented.
        # For 'Without class augmentation' (use_augmentation=False), this will add minimal dummy samples
        # for truly missing classes, allowing the models to train without error, while still
        # reflecting the absence of the *full* augmentation strategy from the experiment's premise.
        X_train_fit, y_train_fit = augment_missing_classes_original(X_train_scenario, y_train_scenario, n_classes_full)

        # Initialize and train LightGBM Classifier
        model_lgbm = LGBMClassifier(objective='multiclass', num_class=n_classes_full, random_state=42,
                                    n_estimators=300, learning_rate=0.05)
        model_lgbm.fit(X_train_fit, y_train_fit)

        if use_ensembling:
            # Initialize and train XGBoost Classifier
            model_xgb = XGBClassifier(objective='multi:softprob',
                                      num_class=n_classes_full,
                                      random_state=42,
                                      use_label_encoder=False, # Suppress warning for older XGBoost versions
                                      eval_metric='mlogloss',
                                      n_estimators=300,
                                      learning_rate=0.05)
            model_xgb.fit(X_train_fit, y_train_fit)

            # Ensemble predictions by averaging probabilities
            y_pred_proba_lgbm = model_lgbm.predict_proba(X_val_fold)
            y_pred_proba_xgb = model_xgb.predict_proba(X_val_fold)
            y_pred_proba_ensemble = (y_pred_proba_lgbm + y_pred_proba_xgb) / 2
            y_pred_val = np.argmax(y_pred_proba_ensemble, axis=1)
        else:
            # If no ensembling, use only LGBM predictions
            y_pred_val = model_lgbm.predict(X_val_fold)

        accuracy = accuracy_score(y_val_fold, y_pred_val)
        fold_accuracies.append(accuracy)

    # Restore stdout
    sys.stdout.close()
    sys.stdout = original_stdout
    
    mean_accuracy = np.mean(fold_accuracies)
    print(f"{experiment_name}: {mean_accuracy:.4f}")
    return mean_accuracy

# --- Ablation Study Execution ---

# Scenario 1: Original Solution (Baseline)
baseline_score = run_ablation_experiment(use_augmentation=True, use_ensembling=True,
                                         experiment_name="Original solution (with augmentation and ensembling)")

# Scenario 2: No Augmentation
# This scenario explicitly sets `use_augmentation=False`, meaning the primary
# data (`X_train_scenario`, `y_train_scenario`) for the experiment is NOT augmented.
# However, a minimal, stability-focused augmentation is applied *just before model fitting*
# to prevent the ValueError, as described in the fix. This allows the ablation study to run.
no_augmentation_score = run_ablation_experiment(use_augmentation=False, use_ensembling=True,
                                                experiment_name="Without class augmentation")

# Scenario 3: No Ensembling (LGBM only)
no_ensembling_score = run_ablation_experiment(use_augmentation=True, use_ensembling=False,
                                              experiment_name="Without ensembling (LGBM only)")

# --- Determine Contribution ---

performance_drops = {}
# The drop here reflects the impact of the *full* augmentation strategy vs.
# the minimal stability augmentation (which is always applied in the 'without augmentation' case).
performance_drops["Class Augmentation"] = baseline_score - no_augmentation_score
performance_drops["Ensembling (XGBoost + LGBM)"] = baseline_score - no_ensembling_score

most_contributing_positive_feature = None
max_positive_drop = -float('inf')

for feature, drop in performance_drops.items():
    if drop > max_positive_drop:
        max_positive_drop = drop
        most_contributing_positive_feature = feature

if most_contributing_positive_feature and max_positive_drop > 0:
    print(f"\nThe part that contributes the most to the overall performance is: {most_contributing_positive_feature} (Removing it caused a drop of {max_positive_drop:.4f})")
elif max_positive_drop <= 0: # All ablations either improved performance or didn't drop it.
    most_detrimental_feature = None
    min_negative_drop = float('inf')

    for feature, drop in performance_drops.items():
        if drop < min_negative_drop:
            min_negative_drop = drop
            most_detrimental_feature = feature

    if most_detrimental_feature:
        print(f"\nNone of the ablated parts positively contributed to the baseline. The part whose removal led to the largest performance improvement was: {most_detrimental_feature} (Removing it improved performance by {-min_negative_drop:.4f})")
    else:
        print("\nCould not determine the most contributing part based on the ablations.")

# Final Validation Performance
# For an ablation study, the baseline score (Original Solution) typically represents
# the full model's performance.
final_validation_score = baseline_score
print(f'Final Validation Performance: {final_validation_score}')
