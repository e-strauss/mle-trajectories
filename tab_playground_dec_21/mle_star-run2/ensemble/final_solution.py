
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Define file paths
# Assuming 'train.csv' and 'test.csv' are in an 'input' subdirectory
script_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
train_file_path = os.path.join(script_dir, 'input', 'train.csv')
# The original problem description states "The test set contains only the features." and "You must predict the Cover_Type for every row in the test set (565892 observations)."
# However, the provided `head -2 train.csv` output implies that the 'test.csv' might not be available or its structure is not fully defined in the snippet.
# Given the task description "There is not submission and no test dataset", I will modify this to use the train.csv for demonstration of the CV and assume
# if a test.csv were available, the prediction logic would apply.
# Let's assume there is a test.csv for the purpose of generating the submission file as requested by the initial problem statement.
# If test.csv does not exist, this part will raise a FileNotFoundError for test_file_path.
test_file_path = os.path.join(script_dir, 'input', 'test.csv')


# Load the data
try:
    train_df = pd.read_csv(train_file_path)
    # Check if 'test.csv' exists before trying to load it
    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path)
    else:
        print(f"Warning: {test_file_path} not found. Proceeding with training data only for CV demonstration.")
        test_df = None # Set to None if not found
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please ensure 'train.csv' and 'test.csv' (if applicable) are in the '{os.path.join(script_dir, 'input')}' directory.")
    # Exit gracefully if essential files are missing, or handle appropriately.
    # For this exercise, we'll create dummy test_df if not found, to allow the script to run through CV logic.
    train_df = pd.read_csv(train_file_path) # Re-load train_df in case it was part of the error message
    print("Creating a dummy test_df for demonstration purposes as test.csv was not found.")
    # Create a dummy test_df using a copy of train_df features if test.csv is truly missing.
    # In a real scenario, this would likely stop execution or require different handling.
    test_df = train_df.drop(columns=['Cover_Type']).copy()
    test_df['Id'] = test_df['Id'] + len(train_df)


# Separate target variable
X = train_df.drop(['Id', 'Cover_Type'], axis=1)
y = train_df['Cover_Type']
train_ids = train_df['Id']

if test_df is not None:
    test_ids = test_df['Id']
    X_test = test_df.drop('Id', axis=1)
else:
    test_ids = pd.Series([]) # Empty series if no test_df
    X_test = pd.DataFrame() # Empty DataFrame if no test_df

# Combine for preprocessing
if not X_test.empty:
    combined_df = pd.concat([X, X_test], ignore_index=True)
else:
    combined_df = X.copy()

# Feature Engineering
# Distance to Hydrology
combined_df['Euclidean_Distance_To_Hydrology'] = np.sqrt(
    combined_df['Horizontal_Distance_To_Hydrology']**2 + combined_df['Vertical_Distance_To_Hydrology']**2
)
combined_df['Manhattan_Distance_To_Hydrology'] = (
    combined_df['Horizontal_Distance_To_Hydrology'].abs() + combined_df['Vertical_Distance_To_Hydrology'].abs()
)

# Hillshade features
combined_df['Hillshade_9am_Noon_diff'] = combined_df['Hillshade_Noon'] - combined_df['Hillshade_9am']
combined_df['Hillshade_Noon_3pm_diff'] = combined_df['Hillshade_Noon'] - combined_df['Hillshade_3pm']
combined_df['Hillshade_mean'] = (combined_df['Hillshade_9am'] + combined_df['Hillshade_Noon'] + combined_df['Hillshade_3pm']) / 3

# Interactions with Elevation
combined_df['Elevation_Hydro_Interaction'] = combined_df['Elevation'] + combined_df['Vertical_Distance_To_Hydrology']
combined_df['Elevation_Road_Interaction'] = combined_df['Elevation'] + combined_df['Horizontal_Distance_To_Roadways']
combined_df['Elevation_Fire_Interaction'] = combined_df['Elevation'] + combined_df['Horizontal_Distance_To_Fire_Points']

# Aspect and Slope transformations (to handle circularity for Aspect)
combined_df['Aspect_sin'] = np.sin(np.deg2rad(combined_df['Aspect']))
combined_df['Aspect_cos'] = np.cos(np.deg2rad(combined_df['Aspect']))
combined_df['Slope_sin'] = np.sin(np.deg2rad(combined_df['Slope']))
combined_df['Slope_cos'] = np.cos(np.deg2rad(combined_df['Slope']))

# Numerical features to scale
numerical_cols = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Euclidean_Distance_To_Hydrology', 'Manhattan_Distance_To_Hydrology',
    'Hillshade_9am_Noon_diff', 'Hillshade_Noon_3pm_diff', 'Hillshade_mean',
    'Elevation_Hydro_Interaction', 'Elevation_Road_Interaction', 'Elevation_Fire_Interaction'
]

# Ensure only existing numerical columns are scaled
numerical_cols = [col for col in numerical_cols if col in combined_df.columns]

# Apply MinMaxScaler
scaler = MinMaxScaler()
combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])


# Separate back into training and test sets
X_processed = combined_df.iloc[:len(X)]
X_test_processed = combined_df.iloc[len(X):]

# 3-Fold Stratified Cross-Validation
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_processed), len(y.unique())))
final_test_preds = np.zeros((len(X_test_processed), len(y.unique())))
validation_scores = []

# Ensemble of three RandomForestClassifiers
# We'll train three separate models and combine their predictions later
# or use VotingClassifier if they are different enough.
# Given the description "ensemble of three RandomForestClassifier models with hard voting",
# we will instantiate three RF models with slightly different random_states.

for fold, (train_index, val_index) in enumerate(skf.split(X_processed, y)):
    print(f"--- Fold {fold+1}/{n_splits} ---")
    X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize three RandomForestClassifiers for the ensemble
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
    clf3 = RandomForestClassifier(n_estimators=100, random_state=789, n_jobs=-1)

    # Train the models
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)

    # Get predictions for validation set
    val_preds_1 = clf1.predict(X_val)
    val_preds_2 = clf2.predict(X_val)
    val_preds_3 = clf3.predict(X_val)

    # Hard voting for validation predictions
    # Convert predictions to a DataFrame to easily apply mode (most frequent)
    val_ensemble_preds_df = pd.DataFrame({
        'pred1': val_preds_1,
        'pred2': val_preds_2,
        'pred3': val_preds_3
    })
    val_ensemble_preds = val_ensemble_preds_df.mode(axis=1)[0].astype(int)

    fold_accuracy = accuracy_score(y_val, val_ensemble_preds)
    validation_scores.append(fold_accuracy)
    print(f"Fold {fold+1} Validation Accuracy: {fold_accuracy:.4f}")

    # Store OOF predictions (probabilities for soft voting, but we are doing hard)
    # For hard voting, we just need the final prediction, but for consistency with probability-based OOF,
    # we can store the one-hot encoded version of the hard vote.
    oof_preds[val_index, val_ensemble_preds - 1] = 1 # -1 because Cover_Type is 1-7, but array index is 0-6

    if not X_test_processed.empty:
        # Get predictions for test set
        test_preds_1 = clf1.predict(X_test_processed)
        test_preds_2 = clf2.predict(X_test_processed)
        test_preds_3 = clf3.predict(X_test_processed)

        # Hard voting for test predictions
        test_ensemble_preds_df = pd.DataFrame({
            'pred1': test_preds_1,
            'pred2': test_preds_2,
            'pred3': test_preds_3
        })
        test_ensemble_preds = test_ensemble_preds_df.mode(axis=1)[0].astype(int)

        # Accumulate test predictions (for averaging or voting across folds)
        # For hard voting, we can directly accumulate the votes
        for i, pred_class in enumerate(test_ensemble_preds):
            final_test_preds[i, pred_class - 1] += 1 # Accumulate vote for the predicted class


final_validation_score = np.mean(validation_scores)
print(f'Final Validation Performance: {final_validation_score}')

# Generate final predictions for the test set
if not X_test_processed.empty:
    # For hard voting across folds, we choose the class with the most votes across all folds
    # In this specific implementation where we accumulate votes, `final_test_preds` holds counts.
    # We take argmax to get the class index (0-6) and add 1 for Cover_Type (1-7).
    final_test_predictions_classes = np.argmax(final_test_preds, axis=1) + 1

    # Create submission file
    submission_df = pd.DataFrame({'Id': test_ids, 'Cover_Type': final_test_predictions_classes})
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully.")
else:
    print("No test data processed, therefore no submission file generated.")
