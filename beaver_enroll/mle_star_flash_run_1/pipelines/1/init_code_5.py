
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Define paths for input data.
INPUT_DIR = os.path.abspath("./input")
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")

# Ensure the directory for training data exists.
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

# --- Mock Data Generation ---
def generate_mock_data():
    """
    Generates a mock gold_enrollment_train.csv if it doesn't exist.
    This ensures the script is self-contained and runnable for demonstration purposes,
    mimicking the expected data structure.
    """
    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
        print(f"Generating mock data: {GOLD_ENROLLMENT_TRAIN_FILE}")
        num_rows = 1000
        # Use a range of terms to enable meaningful time-based validation splits.
        terms = ['201801', '201805', '201809', '201901', '201905', '201909',
                 '202001', '202005', '202009', '202101', '202105', '202109',
                 '202201', '202205', '202209', '202301', '202305', '202309']
        subjects = [f'SUBJ{i:03d}' for i in range(50)] # Simulate 50 unique subjects

        data = {
            'TERM_CODE': np.random.choice(terms, num_rows),
            'SUBJECT_ID_SORT': np.random.choice(subjects, num_rows),
            'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], num_rows, p=[0.3, 0.7]) # 30% high enrollment
        }
        mock_gold_df = pd.DataFrame(data)
        # Convert TERM_CODE to integer for correct chronological sorting.
        mock_gold_df['TERM_CODE'] = mock_gold_df['TERM_CODE'].astype(int)
        mock_gold_df = mock_gold_df.sort_values(by='TERM_CODE').reset_index(drop=True)
        mock_gold_df.to_csv(GOLD_ENROLLMENT_TRAIN_FILE, index=False)
        print("Mock gold_enrollment_train.csv created.")
    else:
        print(f"Mock data {GOLD_ENROLLMENT_TRAIN_FILE} already exists. Skipping generation.")


# --- Data Loading and Feature Engineering ---
def load_data():
    """
    Loads the gold enrollment data and engineers relevant features.
    In a real scenario, more complex features would be derived from other tables.
    """
    print(f"Attempting to load gold enrollment data from {GOLD_ENROLLMENT_TRAIN_FILE}...")

    if not os.path.exists(GOLD_ENROLLMENT_TRAIN_FILE):
        print(f"Error: Gold enrollment file not found at {GOLD_ENROLLMENT_TRAIN_FILE}.")
        return pd.DataFrame() # Return empty DataFrame if file is missing.

    gold_df = pd.read_csv(GOLD_ENROLLMENT_TRAIN_FILE)

    if gold_df.empty:
        print("Loaded gold enrollment data is empty. Please check the CSV file.")
        return gold_df

    # Set random seed for reproducibility of synthetic features.
    np.random.seed(42)

    # Feature Engineering: Derive features from TERM_CODE.
    gold_df['TERM_CODE_STR'] = gold_df['TERM_CODE'].astype(str)
    gold_df['TERM_YEAR'] = gold_df['TERM_CODE_STR'].str[:4].astype(int)
    gold_df['TERM_MONTH'] = gold_df['TERM_CODE_STR'].str[4:].astype(int) # 01=Spring, 05=Summer, 09=Fall

    # Simulate additional numerical features, as other tables are not available for this task.
    gold_df['feature_avg_gpa_prereq'] = np.random.normal(3.0, 0.5, len(gold_df))
    gold_df['feature_course_capacity'] = np.random.randint(10, 200, len(gold_df))
    gold_df['feature_prev_enrollment_ratio'] = np.random.uniform(0.1, 1.5, len(gold_df))

    # Ensure 'TERM_CODE' remains integer for time-based splitting.
    gold_df['TERM_CODE'] = gold_df['TERM_CODE'].astype(int)

    return gold_df

def create_features_labels(df):
    """
    Extracts features (X) and target labels (y) from the processed DataFrame,
    and performs one-hot encoding for categorical features suitable for Logistic Regression.
    """
    # Define the set of numerical and categorical features to be used in the model.
    numerical_features = [
        'TERM_YEAR',
        'feature_avg_gpa_prereq',
        'feature_course_capacity',
        'feature_prev_enrollment_ratio'
    ]
    categorical_features = [
        'SUBJECT_ID_SORT',
        'TERM_MONTH'
    ]

    all_features = numerical_features + categorical_features

    # Verify all required feature columns exist.
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns in DataFrame: {missing_cols}")
        return pd.DataFrame(), pd.Series() # Return empty if critical features are missing.

    # Convert categorical features to 'category' dtype for get_dummies
    for col in categorical_features:
        df[col] = df[col].astype('category')

    # Perform one-hot encoding for categorical features
    X_categorical = pd.get_dummies(df[categorical_features], prefix=categorical_features, drop_first=True) # drop_first to avoid multicollinearity
    X_numerical = df[numerical_features]

    # Combine numerical and one-hot encoded categorical features
    X = pd.concat([X_numerical, X_categorical], axis=1)

    # Convert target to binary (1/0)
    y = df['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    return X, y

# --- Main Script Logic ---
if __name__ == "__main__":
    final_validation_score = 0.0 # Initialize validation score.

    generate_mock_data() # Ensure mock data is present if actual data is not.

    print("Starting training process with Logistic Regression...")

    # Load and process training data.
    train_df = load_data()

    if train_df.empty:
        print("Training data is empty after loading. Cannot proceed with model training.")
    else:
        # Extract features and labels.
        X_raw, y = create_features_labels(train_df)

        if X_raw.empty or y.empty:
            print("Features or labels are empty after creation. Cannot proceed with model training.")
        else:
            # Time-based validation split: Use the latest terms for validation.
            # This follows the task's recommendation for a time-based validation slice.
            unique_terms_num = sorted(train_df['TERM_CODE'].unique())

            X_train, y_train, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
            performed_random_split = False

            if len(unique_terms_num) > 1:
                # Use the latest 20% of unique terms for the validation set.
                split_point = int(len(unique_terms_num) * 0.8)
                train_terms_num = unique_terms_num[:split_point]
                val_terms_num = unique_terms_num[split_point:]

                # Create masks based on original train_df TERM_CODE, then apply to X_raw and y
                train_indices = train_df['TERM_CODE'].isin(train_terms_num)
                val_indices = train_df['TERM_CODE'].isin(val_terms_num)

                X_train_full = X_raw[train_indices]
                y_train_full = y[train_indices]
                X_val_full = X_raw[val_indices]
                y_val_full = y[val_indices]

                # Fallback to a random split if the time-based split results in an empty set.
                if X_val_full.empty or y_val_full.empty or X_train_full.empty or y_train_full.empty:
                    print("Time-based validation split resulted in an empty train or validation set. Falling back to random split.")
                    performed_random_split = True
                else:
                    X_train, y_train = X_train_full, y_train_full
                    X_val, y_val = X_val_full, y_val_full
            else:
                print("Not enough unique terms for a time-based split (need more than 1). Falling back to random split.")
                performed_random_split = True

            if performed_random_split:
                # Perform a stratified random split as a fallback or if time-based split is not feasible.
                # Ensure enough samples per class for stratification.
                if len(y.unique()) > 1 and y.value_counts().min() > 1:
                    X_train, X_val, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
                else:
                    print("Not enough classes or samples per class for stratified split. Using non-stratified random split.")
                    X_train, X_val, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)

            if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
                print("Train or validation sets are empty after all splitting attempts. Cannot proceed with model training.")
            else:
                # Scale numerical features. It's good practice for Logistic Regression.
                numerical_features_to_scale = [col for col in X_train.columns if 'feature_' in col or col == 'TERM_YEAR']
                scaler = StandardScaler()
                X_train[numerical_features_to_scale] = scaler.fit_transform(X_train[numerical_features_to_scale])
                X_val[numerical_features_to_scale] = scaler.transform(X_val[numerical_features_to_scale])

                # Initialize and train the Logistic Regression Classifier
                # 'balanced' class_weight adjusts weights inversely proportional to class frequencies
                model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
                print("Training Logistic Regression model...")
                model.fit(X_train, y_train)

                # Predict on the validation set
                y_pred = model.predict(X_val)

                # Calculate Macro F1 Score
                macro_f1 = f1_score(y_val, y_pred, average='macro')
                print(f"Validation Macro F1 Score: {macro_f1:.4f}")
                final_validation_score = macro_f1

    # Print the final validation performance in the required format.
    print(f"Final Validation Performance: {final_validation_score}")
