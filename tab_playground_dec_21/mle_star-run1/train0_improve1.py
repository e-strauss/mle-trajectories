
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb # Added to fix NameError: name 'lgb' is not defined
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Augmentation Function for Missing Classes ---
def augment_missing_classes(X_train_df, y_train_series, n_classes_total):
    """
    Augments the training data by adding dummy samples for any classes
    not present in the current fold's training set. This helps prevent
    errors in multi-class classifiers that require all classes to be present
    during training.
    """
    unique_classes_in_fold = y_train_series.unique()
    # Adjust n_classes_total to be 0-indexed if y is 0-indexed
    all_classes = np.arange(n_classes_total)
    missing_classes = np.setdiff1d(all_classes, unique_classes_in_fold)

    if not missing_classes.size:
        return X_train_df, y_train_series # No augmentation needed

    X_train_augmented = X_train_df.copy()
    y_train_augmented = y_train_series.copy()

    # If X_train_df is empty, we cannot augment. This scenario should be rare
    # with StratifiedKFold on a sufficiently sized dataset.
    if X_train_df.empty:
        print("Warning: X_train_df is empty, cannot perform class augmentation.")
        return X_train_df, y_train_series

    # Select a few random samples from the existing training data to use as templates.
    # We ensure there's at least one template if X_train_df is not empty.
    # The number of templates is chosen to cover all missing classes without excessive duplication.
    num_templates = min(len(X_train_df), max(1, len(missing_classes)))
    # Use a fixed random_state for reproducibility in selecting templates
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


# 1. Load Data
# The training data is available in the './input' directory.
train_df = pd.read_csv("./input/train.csv")

# 2. Prepare Data
# Features (X): All columns except 'Id' and 'Cover_Type'
# Target (y): 'Cover_Type'
# LightGBM and scikit-learn metrics typically expect 0-indexed class labels for multiclass.
# The 'Cover_Type' values are 1-7, so we subtract 1 to make them 0-6.
X = train_df.drop(['Id', 'Cover_Type'], axis=1)
y = train_df['Cover_Type'] - 1  # Adjust target labels from 1-7 to 0-6

# Define the number of classes (7 in this problem, after adjustment to 0-6)
# It's important to use the actual number of unique classes found in y, not just the max value,
# especially after subtracting 1.
n_classes = y.nunique() 

# 3. Evaluation Metric
# Accuracy is a suitable metric for multi-class classification problems,
# especially when class imbalance is not severe, or for a simple initial solution.

# 4. Implement 3-Fold Cross-Validation
# Use StratifiedKFold to ensure that each fold has approximately the same percentage
# of samples of each target class as the complete set.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []

print(f"Starting {n_splits}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")

    # Split data into training and validation sets for the current fold
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # --- Apply Dummy Sample Augmentation to ensure all classes are present in the training fold ---
    X_train_augmented, y_train_augmented = augment_missing_classes(X_train_fold, y_train_fold, n_classes)

    # Initialize LightGBM Classifier (from base solution)
    model_lgbm = LGBMClassifier(objective='multiclass', num_class=n_classes, random_state=42,
                                n_estimators=1000, # Increased initial n_estimators
                                learning_rate=0.01,  # Decreased learning_rate
                                class_weight='balanced') # Enabled class_weight='balanced'

    # Initialize XGBoost Classifier (additional model)
    model_xgb = XGBClassifier(objective='multi:softprob',
                              num_class=n_classes,
                              random_state=42,
                              use_label_encoder=False, # Suppress warning for deprecated use_label_encoder
                              eval_metric='mlogloss', # Specify evaluation metric for multi-class classification
                              n_estimators=300,        # Consistent number of estimators with previous LGBM
                              learning_rate=0.05)      # Consistent learning rate with previous LGBM

    # Train both models for the current fold
    # Implemented early stopping for LGBM using the validation fold.
    model_lgbm.fit(X_train_augmented, y_train_augmented,
                    eval_set=[(X_val_fold, y_val_fold)], # Use validation set for early stopping
                    eval_metric='multi_logloss',        # Metric for multi-class classification
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]) # Early stopping callback

    model_xgb.fit(X_train_augmented, y_train_augmented) # XGBoost training without changes

    # Make predictions (probabilities) on the validation set for both models
    y_pred_proba_lgbm = model_lgbm.predict_proba(X_val_fold)
    y_pred_proba_xgb = model_xgb.predict_proba(X_val_fold)

    # Ensemble predictions by averaging the predicted probabilities from both models.
    y_pred_proba_ensemble = (y_pred_proba_lgbm + y_pred_proba_xgb) / 2

    # Get the final predicted class by taking the argmax (class with highest averaged probability)
    y_pred_val_ensemble = np.argmax(y_pred_proba_ensemble, axis=1)

    # Calculate accuracy for the current fold using the ensembled predictions
    accuracy = accuracy_score(y_val_fold, y_pred_val_ensemble)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

# 5. Calculate Final Performance
# Average the accuracies across all folds to get the final validation score.
final_validation_score = np.mean(fold_accuracies)

# 6. Print the result
print("\nCross-Validation Complete.")
print(f'Final Validation Performance: {final_validation_score}')
