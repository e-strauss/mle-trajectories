
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# All the provided input data is stored in "./input" directory.
# Load data
train_df = pd.read_csv('./input/train.csv')

# Separate features and target
X = train_df.drop(['Id', 'Cover_Type'], axis=1)
y_original = train_df['Cover_Type']

# Map target to 0-6 for sklearn compatibility
# Original: 1-7, Transformed: 0-6
y_transformed = y_original - 1

n_splits = 3 # As per task description for 3-Fold CV
n_classes = len(y_transformed.unique()) # This will be 7 for the full dataset (0-6)

# --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
# This section identifies all classes with fewer than n_splits samples.
# These samples will be handled specifically for the two models:
# - For Model 1 (Base-style): They are excluded from CV altogether.
# - For Model 2 (Reference-style): They are added to the training set of each fold, but never to the validation set.

class_counts = y_transformed.value_counts()
problematic_classes_list = class_counts[class_counts < n_splits].index.tolist()

if problematic_classes_list:
    # Identify indices of problematic samples in the original dataset
    indices_problematic = y_transformed[y_transformed.isin(problematic_classes_list)].index
    
    # Store problematic samples
    X_problematic_samples = X.loc[indices_problematic].reset_index(drop=True)
    y_problematic_samples = y_transformed.loc[indices_problematic].reset_index(drop=True)

    # Create dataset for CV-eligible samples (all samples MINUS problematic ones)
    X_cv_eligible = X.drop(indices_problematic).reset_index(drop=True)
    y_cv_eligible = y_transformed.drop(indices_problematic).reset_index(drop=True)
else:
    # No problematic classes (all have >= n_splits samples)
    X_problematic_samples = pd.DataFrame() # Empty if no problematic samples
    y_problematic_samples = pd.Series(dtype=int) # Empty if no problematic samples
    X_cv_eligible = X
    y_cv_eligible = y_transformed


# Initialize StratifiedKFold for the eligible samples
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF (Out-Of-Fold) prediction storage for ensembling
# Store probabilities for each model across all original samples
oof_preds_proba_model1 = np.full((len(X), n_classes), np.nan)
oof_preds_proba_model2 = np.full((len(X), n_classes), np.nan)
oof_true_labels = np.full(len(X), np.nan) # Store true labels for samples included in any validation fold

# Model 1: Base-style RandomForest (excludes problematic samples from CV altogether)
# This model uses X_cv_eligible and y_cv_eligible for both training and validation splits within CV.
model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Model 2: Reference-style RandomForest (trains on problematic samples, but validates only on eligible)
# This model uses X_cv_eligible and y_cv_eligible for validation splits, but adds problematic samples to training set.
model2 = RandomForestClassifier(n_estimators=100, random_state=43, n_jobs=-1) # Different random_state for variety

# Perform 3-Fold Cross-Validation
for fold, (train_idx_eligible, val_idx_eligible) in enumerate(skf.split(X_cv_eligible, y_cv_eligible)):
    # Get original indices from the full dataset for validation samples
    # These are the indices of X_cv_eligible.iloc[val_idx_eligible] within the original X.
    # We need to map back to the original full dataframe's index using y_transformed's index for these
    original_val_indices_iloc = y_cv_eligible.iloc[val_idx_eligible].index # These are indices in the original X, y_transformed
    
    # --- Model 1 (Base-style) Training and Prediction ---
    X_train_model1 = X_cv_eligible.iloc[train_idx_eligible]
    y_train_model1 = y_cv_eligible.iloc[train_idx_eligible]
    X_val_model1 = X_cv_eligible.iloc[val_idx_eligible]

    model1.fit(X_train_model1, y_train_model1)
    probas_val_model1_raw = model1.predict_proba(X_val_model1)
    
    # Create a temporary array to store probabilities with full n_classes columns
    probas_val_model1_full = np.zeros((len(X_val_model1), n_classes), dtype=float)
    # Map the probabilities from the model's learned classes to the full n_classes columns
    for i, class_label in enumerate(model1.classes_):
        probas_val_model1_full[:, class_label] = probas_val_model1_raw[:, i]
    
    oof_preds_proba_model1[original_val_indices_iloc] = probas_val_model1_full

    # --- Model 2 (Reference-style) Training and Prediction ---
    # Add problematic samples to the training set for Model 2
    X_train_model2_base = X_cv_eligible.iloc[train_idx_eligible].reset_index(drop=True)
    y_train_model2_base = y_cv_eligible.iloc[train_idx_eligible].reset_index(drop=True)
    
    # Concatenate problematic samples to training set, ensuring clean indices
    X_train_model2 = pd.concat([X_train_model2_base, X_problematic_samples]).reset_index(drop=True)
    y_train_model2 = pd.concat([y_train_model2_base, y_problematic_samples]).reset_index(drop=True)

    X_val_model2 = X_cv_eligible.iloc[val_idx_eligible] # Validation set is the same as Model 1

    model2.fit(X_train_model2, y_train_model2)
    probas_val_model2_raw = model2.predict_proba(X_val_model2)
    
    # Create a temporary array to store probabilities with full n_classes columns
    probas_val_model2_full = np.zeros((len(X_val_model2), n_classes), dtype=float)
    # Map the probabilities from the model's learned classes to the full n_classes columns
    for i, class_label in enumerate(model2.classes_):
        probas_val_model2_full[:, class_label] = probas_val_model2_raw[:, i]

    oof_preds_proba_model2[original_val_indices_iloc] = probas_val_model2_full

    # Store true labels for these validation samples (only need to do it once as it's the same for both models' validation set)
    oof_true_labels[original_val_indices_iloc] = y_transformed.loc[original_val_indices_iloc]


# --- Ensembling and Final Validation Performance Calculation ---

# Identify samples that were part of any validation fold (i.e., not NaN in oof_true_labels)
valid_indices_for_scoring = ~np.isnan(oof_true_labels)

# Ensure there are samples to score before proceeding
if valid_indices_for_scoring.any():
    # Get combined probabilities for the valid samples by averaging predictions from both models
    combined_probas = (oof_preds_proba_model1[valid_indices_for_scoring] + oof_preds_proba_model2[valid_indices_for_scoring]) / 2

    # Get ensemble predictions by taking argmax of averaged probabilities (0-6 space)
    ensemble_preds_0_6 = np.argmax(combined_probas, axis=1)

    # Get true labels for the valid samples (0-6 space)
    y_true_0_6 = oof_true_labels[valid_indices_for_scoring]

    # Transform predictions and true labels back to 1-7 space for accuracy calculation
    final_validation_score = accuracy_score(y_true_0_6 + 1, ensemble_preds_0_6 + 1)
else:
    # If no samples were eligible for cross-validation (e.g., all classes problematic)
    final_validation_score = 0.0

# Print the final validation performance as required
print(f'Final Validation Performance: {final_validation_score}')
