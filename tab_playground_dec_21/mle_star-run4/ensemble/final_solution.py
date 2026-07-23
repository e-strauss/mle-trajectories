
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Create the './input' directory if it doesn't exist
os.makedirs('./input', exist_ok=True)

# Create a dummy train.csv and test.csv for demonstration purposes if they don't exist
# In a real scenario, these files would be pre-provided.
train_file_path = './input/train.csv'
test_file_path = './input/test.csv'

if not os.path.exists(train_file_path):
    print("Creating a dummy train.csv for demonstration.")
    # Based on the head -2 train.csv provided in the problem description
    train_data = {
        'Id': [0, 1, 2, 3, 4],
        'Elevation': [3189, 2596, 2697, 2800, 2750],
        'Aspect': [40, 120, 10, 50, 90],
        'Slope': [8, 15, 20, 12, 5],
        'Horizontal_Distance_To_Hydrology': [30, 200, 100, 50, 150],
        'Vertical_Distance_To_Hydrology': [13, 50, 20, 10, 30],
        'Horizontal_Distance_To_Roadways': [3270, 1500, 2000, 1000, 2500],
        'Hillshade_9am': [206, 220, 210, 200, 230],
        'Hillshade_Noon': [234, 235, 240, 225, 245],
        'Hillshade_3pm': [193, 170, 180, 190, 160],
        'Horizontal_Distance_To_Fire_Points': [4873, 1000, 500, 2000, 1500],
        'Wilderness_Area1': [1, 0, 1, 0, 1],
        'Wilderness_Area2': [0, 1, 0, 1, 0],
        'Wilderness_Area3': [0, 0, 0, 0, 0],
        'Wilderness_Area4': [0, 0, 0, 0, 0],
        'Soil_Type1': [0, 0, 0, 0, 0], 'Soil_Type2': [0, 0, 0, 0, 0], 'Soil_Type3': [0, 0, 0, 0, 0], 'Soil_Type4': [0, 0, 0, 0, 0], 'Soil_Type5': [0, 0, 0, 0, 0],
        'Soil_Type6': [0, 0, 0, 0, 0], 'Soil_Type7': [0, 0, 0, 0, 0], 'Soil_Type8': [0, 0, 0, 0, 0], 'Soil_Type9': [0, 0, 0, 0, 0], 'Soil_Type10': [0, 0, 0, 0, 0],
        'Soil_Type11': [0, 0, 0, 0, 0], 'Soil_Type12': [0, 0, 0, 0, 0], 'Soil_Type13': [0, 0, 0, 0, 0], 'Soil_Type14': [0, 0, 0, 0, 0], 'Soil_Type15': [0, 0, 0, 0, 0],
        'Soil_Type16': [0, 0, 0, 0, 0], 'Soil_Type17': [0, 0, 0, 0, 0], 'Soil_Type18': [0, 0, 0, 0, 0], 'Soil_Type19': [0, 0, 0, 0, 0], 'Soil_Type20': [0, 0, 0, 0, 0],
        'Soil_Type21': [0, 0, 0, 0, 0], 'Soil_Type22': [0, 0, 0, 0, 0], 'Soil_Type23': [0, 0, 0, 0, 0], 'Soil_Type24': [0, 0, 0, 0, 0], 'Soil_Type25': [0, 0, 0, 0, 0],
        'Soil_Type26': [0, 0, 0, 0, 0], 'Soil_Type27': [0, 0, 0, 0, 0], 'Soil_Type28': [0, 0, 0, 0, 0], 'Soil_Type29': [1, 0, 0, 0, 0], 'Soil_Type30': [0, 1, 0, 0, 0],
        'Soil_Type31': [0, 0, 1, 0, 0], 'Soil_Type32': [0, 0, 0, 1, 0], 'Soil_Type33': [0, 0, 0, 0, 1], 'Soil_Type34': [0, 0, 0, 0, 0], 'Soil_Type35': [0, 0, 0, 0, 0],
        'Soil_Type36': [0, 0, 0, 0, 0], 'Soil_Type37': [0, 0, 0, 0, 0], 'Soil_Type38': [0, 0, 0, 0, 0], 'Soil_Type39': [0, 0, 0, 0, 0], 'Soil_Type40': [0, 0, 0, 0, 0],
        'Cover_Type': [1, 2, 3, 4, 5] # Class 5 is present here for demonstration, which has only 1 sample, as described
    }
    pd.DataFrame(train_data).to_csv(train_file_path, index=False)

if not os.path.exists(test_file_path):
    print("Creating a dummy test.csv for demonstration.")
    # Create a test.csv with similar structure to train.csv but without 'Cover_Type'
    # and with 565892 observations as per the problem description.
    # For a dummy, we'll use fewer rows to avoid large file creation.
    num_test_rows = 10 # Using a smaller number of rows for dummy test data
    test_data = {
        'Id': np.arange(15120, 15120 + num_test_rows),
        'Elevation': np.random.randint(2000, 4000, num_test_rows),
        'Aspect': np.random.randint(0, 360, num_test_rows),
        'Slope': np.random.randint(0, 40, num_test_rows),
        'Horizontal_Distance_To_Hydrology': np.random.randint(0, 1000, num_test_rows),
        'Vertical_Distance_To_Hydrology': np.random.randint(-150, 250, num_test_rows),
        'Horizontal_Distance_To_Roadways': np.random.randint(0, 7000, num_test_rows),
        'Hillshade_9am': np.random.randint(0, 255, num_test_rows),
        'Hillshade_Noon': np.random.randint(0, 255, num_test_rows),
        'Hillshade_3pm': np.random.randint(0, 255, num_test_rows),
        'Horizontal_Distance_To_Fire_Points': np.random.randint(0, 7000, num_test_rows),
        'Wilderness_Area1': np.random.randint(0, 2, num_test_rows),
        'Wilderness_Area2': np.random.randint(0, 2, num_test_rows),
        'Wilderness_Area3': np.random.randint(0, 2, num_test_rows),
        'Wilderness_Area4': np.random.randint(0, 2, num_test_rows),
    }
    # Add dummy Soil_Type columns
    for i in range(1, 41):
        test_data[f'Soil_Type{i}'] = np.zeros(num_test_rows, dtype=int)
    # Set one random soil type to 1 for each row for realism
    for i in range(num_test_rows):
        test_data[f'Soil_Type{np.random.randint(1, 41)}'][i] = 1

    pd.DataFrame(test_data).to_csv(test_file_path, index=False)


# All the provided input data is stored in "./input" directory.
# Load data
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv') # Load test data

# Store test_ids for submission before dropping 'Id' from X_test
test_ids = test_df['Id']

# --- Feature Engineering for Training Data ---
train_df['Hillshade_composite'] = (train_df['Hillshade_9am'] + train_df['Hillshade_Noon'] + train_df['Hillshade_3pm']) / 3
train_df['Elevation_at_Hydrology'] = train_df['Elevation'] - train_df['Vertical_Distance_To_Hydrology']

# Drop 'Hillshade' columns along with 'Id', 'Cover_Type', and 'Aspect' for training features
X = train_df.drop(['Id', 'Cover_Type', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)
y_original = train_df['Cover_Type']
y_transformed = y_original - 1 # Map target to 0-6 for sklearn compatibility


# --- Feature Engineering for Test Data ---
test_df['Hillshade_composite'] = (test_df['Hillshade_9am'] + test_df['Hillshade_Noon'] + test_df['Hillshade_3pm']) / 3
test_df['Elevation_at_Hydrology'] = test_df['Elevation'] - test_df['Vertical_Distance_To_Hydrology']

# Drop 'Hillshade' columns along with 'Id' and 'Aspect' for test features
X_test = test_df.drop(['Id', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Aspect'], axis=1)

# Ensure columns in X_test match the columns in X. This is important for consistent feature sets.
# Reindex X_test to match X's columns, filling missing columns with 0 and dropping extra ones.
# This is crucial if, for instance, a dummy test.csv did not have all Soil_Type columns or had extras.
missing_cols_in_test = set(X.columns) - set(X_test.columns)
for c in missing_cols_in_test:
    X_test[c] = 0
# Ensure the order of columns in X_test is the same as in X
X_test = X_test[X.columns]


n_splits = 3 # As per task description for 3-Fold CV

# --- Handle classes with fewer samples than n_splits for StratifiedKFold ---
class_counts = y_transformed.value_counts()
problematic_classes = class_counts[class_counts < n_splits].index.tolist()

if problematic_classes:
    indices_to_exclude_from_cv = y_transformed[y_transformed.isin(problematic_classes)].index.tolist()
    X_cv = X.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    y_cv = y_transformed.drop(indices_to_exclude_from_cv).reset_index(drop=True)
    excluded_counts = y_transformed[y_transformed.isin(problematic_classes)].value_counts()
    print(f"Warning: The following classes have fewer than {n_splits} samples and will be excluded from cross-validation:")
    for cls, count in excluded_counts.items():
        print(f"  - Original Cover_Type: {cls + 1} (transformed: {cls}) has {count} sample(s).")
    print(f"Total {len(indices_to_exclude_from_cv)} samples excluded from CV. CV will be performed on remaining {len(X_cv)} samples.")
else:
    X_cv = X
    y_cv = y_transformed
    print(f"All classes have at least {n_splits} samples. Cross-validation will be performed on all {len(X_cv)} samples.")


# --- Ensemble Plan Implementation for Cross-Validation ---
n_ensemble_runs = 3 # Number of outer repetitions of 3-Fold CV
n_sub_models = 5    # Number of RandomForestClassifier instances per fold (e.g., 5 as per plan)
skf_random_state_base = 42 # Base random state for StratifiedKFold splits
submodel_random_state_base = 100 # Base random state for individual RandomForest models

overall_ensemble_run_scores = [] # To store average accuracies from each outer ensemble run

print(f"\nStarting {n_ensemble_runs} Ensemble Runs with {n_sub_models} sub-models per fold for Cross-Validation...")

for ensemble_run in range(n_ensemble_runs):
    print(f"\n--- Ensemble Run {ensemble_run + 1}/{n_ensemble_runs} ---")
    current_skf_random_state = skf_random_state_base + ensemble_run
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_skf_random_state)
    print(f"StratifiedKFold random_state for this run: {current_skf_random_state}")

    current_run_fold_accuracies = [] # Store fold accuracies for the current ensemble run

    print(f"Starting {n_splits}-Fold Cross-Validation for Ensemble Run {ensemble_run + 1}...")
    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
        y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]

        fold_sub_model_probabilities = []

        print(f"  Training {n_sub_models} sub-models for Fold {fold+1}...")
        for i in range(n_sub_models):
            current_model_random_state = submodel_random_state_base + i
            sub_model = RandomForestClassifier(n_estimators=100, random_state=current_model_random_state, n_jobs=-1)
            sub_model.fit(X_train, y_train)
            fold_sub_model_probabilities.append(sub_model.predict_proba(X_val))

        averaged_probabilities = np.mean(np.array(fold_sub_model_probabilities), axis=0)
        y_pred_val_ensemble = np.argmax(averaged_probabilities, axis=1)

        accuracy = accuracy_score(y_val, y_pred_val_ensemble)
        current_run_fold_accuracies.append(accuracy)
        print(f"  Fold {fold+1} Ensemble Accuracy: {accuracy:.4f}")

    current_ensemble_run_score = np.mean(current_run_fold_accuracies)
    overall_ensemble_run_scores.append(current_ensemble_run_score)
    print(f"Average Accuracy for Ensemble Run {ensemble_run + 1}: {current_ensemble_run_score:.4f}")

final_validation_performance = np.mean(overall_ensemble_run_scores)
print(f'Final Validation Performance: {final_validation_performance}')

# --- Train final ensemble model on full training data and predict on test data ---
print("\nTraining final ensemble model on the full training dataset and predicting on test data...")

final_ensemble_probabilities = []
for i in range(n_sub_models):
    current_model_random_state = submodel_random_state_base + i
    final_model = RandomForestClassifier(n_estimators=100, random_state=current_model_random_state, n_jobs=-1)
    
    # Train on the full X and y_transformed (original, not CV-restricted)
    final_model.fit(X, y_transformed) # Use X and y_transformed which are the full training data
    
    # Predict probabilities on the preprocessed test data
    final_ensemble_probabilities.append(final_model.predict_proba(X_test))

# Average the predicted probabilities from the final ensemble models
averaged_test_probabilities = np.mean(np.array(final_ensemble_probabilities), axis=0)

# Get the final predictions by selecting the class with the highest average probability
y_pred_test_transformed = np.argmax(averaged_test_probabilities, axis=1)

# Transform predictions back to original 1-7 range
y_pred_test_original = y_pred_test_transformed + 1

# --- Create submission file ---
print("Creating submission file...")
submission_df = pd.DataFrame({'Id': test_ids, 'Cover_Type': y_pred_test_original})

# Create the './final' directory if it doesn't exist
os.makedirs('./final', exist_ok=True)

# Save the submission file
submission_file_path = './final/submission.csv'
submission_df.to_csv(submission_file_path, index=False)
print(f"Submission file saved to {submission_file_path}")
