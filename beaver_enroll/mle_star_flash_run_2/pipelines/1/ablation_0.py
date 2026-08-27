
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Suppress warnings to keep output clean, especially for small datasets causing specific warnings
warnings.filterwarnings('ignore')

# --- Dummy Data Setup (for self-contained execution) ---
# Create necessary directories and dummy CSV files if they don't exist
def setup_dummy_data():
    os.makedirs("./input/table_splits/train", exist_ok=True)

    gold_enrollment_train_data = """TERM_CODE,SUBJECT_ID_SORT,HIGH_ENROLLMENT
202301,CS101,Y
202301,MA201,N
202301,PH101,Y
202302,CS101,N
202302,MA201,Y
202302,PH101,N
202303,CS101,Y
202303,MA201,N
202303,PH101,Y
"""
    with open("./input/gold_enrollment_train.csv", "w") as f:
        f.write(gold_enrollment_train_data)

    subject_summary_data = """TERM_CODE,SUBJECT_ID_SORT,AVG_GRADE,ENROLLMENT_CAP,ACTUAL_ENROLLMENT,SOME_OTHER_NUMERIC_FEATURE
202301,CS101,3.5,50,45,10
202301,MA201,2.8,40,20,5
202301,PH101,3.2,60,55,12
202302,CS101,3.0,50,30,8
202202,MA201,3.1,40,35,9
202302,PH101,2.9,60,40,7
202303,CS101,3.7,50,48,11
202303,MA201,2.5,40,25,6
202303,PH101,3.4,60,50,13
"""
    with open("./input/table_splits/train/subject_summary.csv", "w") as f:
        f.write(subject_summary_data)

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Modified Function to load data and engineer features for ablation ---
def load_data_ablated(data_dir, gold_file, include_encoded_identifiers=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Configurable to exclude encoded TERM_CODE and SUBJECT_ID_SORT.
    """
    gold_df = pd.read_csv(gold_file)
    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})
    
    final_feature_cols = []

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        # Identifier columns used for merging and to be excluded from direct feature selection
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        # Merge gold_df (which now contains numerical 'HIGH_ENROLLMENT') with subject_summary_df to get features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        # Select numerical features for the model.
        # Exclude original identifiers and the target column.
        feature_cols_candidate = [
            col for col in df.columns 
            if col not in identifier_cols + ['HIGH_ENROLLMENT']
        ]
        
        # Filter for truly numeric columns from the candidate list.
        numeric_feature_cols = df[feature_cols_candidate].select_dtypes(include=np.number).columns.tolist()
        
        # Conditionally encode TERM_CODE and SUBJECT_ID_SORT to use them as numerical features
        if include_encoded_identifiers:
            le_term = LabelEncoder()
            df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
            if 'TERM_CODE_ENCODED' not in numeric_feature_cols:
                numeric_feature_cols.append('TERM_CODE_ENCODED')

            le_subject = LabelEncoder()
            df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
            if 'SUBJECT_ID_SORT_ENCODED' not in numeric_feature_cols:
                numeric_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(numeric_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found after merge, add a dummy feature.
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv or due to exclusion. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge (if some gold entries don't have summary data)
        # or from missing values in the subject_summary. Simple imputation with 0.
        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        
        # Store feature columns for later use
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        
        # Even if include_encoded_identifiers is False, we might need these for DUMMY_FEATURE
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        # Determine final features based on include_encoded_identifiers
        if include_encoded_identifiers:
            df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        else:
            # If identifiers are explicitly excluded, and summary not found, rely only on dummy.
            df._feature_cols = ['DUMMY_FEATURE']
            if 'DUMMY_FEATURE' not in df.columns: # Ensure it exists if df._feature_cols is only DUMMY_FEATURE
                 df['DUMMY_FEATURE'] = 0

        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Main script as a function for ablation ---
def run_ablation_experiment(
    experiment_name,
    n_estimators_rf=100,
    include_encoded_identifiers=True,
    print_details=True
):
    """
    Runs a single training and validation experiment with specified ablation settings.
    """
    if print_details:
        print(f"\n--- Running Experiment: {experiment_name} ---")
        print(f"  Configuration: n_estimators_rf={n_estimators_rf}, include_encoded_identifiers={include_encoded_identifiers}")

    df = load_data_ablated(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, include_encoded_identifiers)

    if df.empty:
        if print_details:
            print("Loaded DataFrame is empty. Cannot proceed with training.")
        return 0.0

    # Ensure TERM_CODE is treated as string for consistent sorting behavior
    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    # Determine validation set: latest TERM_CODE
    unique_terms = df['TERM_CODE'].unique()
    
    # Ensure there are enough terms to create both training and validation sets
    if len(unique_terms) < 2:
        if print_details:
            print("Not enough unique terms for a time-based validation split (at least 2 required).")
        return 0.0

    # Use the last term for validation to simulate a future term, as per problem description.
    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    # Retrieve feature columns determined during data loading
    feature_cols = getattr(df, '_feature_cols', [])
    
    # Fallback to identify feature columns if not properly set (should not happen with robust load_data)
    if not feature_cols: 
        warnings.warn("Feature columns not correctly identified by load_data_ablated. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        if print_details:
            print("No usable features available for training after all fallback attempts.")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    # Check for empty training data
    if X_train.empty or y_train.empty:
        if print_details:
            print("Training set is empty. Cannot train a model.")
        return 0.0

    # Check if target variable in training set has only one class
    if len(y_train.unique()) < 2:
        if print_details:
            print(f"Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    model = RandomForestClassifier(random_state=42, n_estimators=n_estimators_rf)
    model.fit(X_train, y_train)

    # --- F1 Score Calculation with robustness checks ---
    final_validation_score = 0.0 # Default score if any issue prevents calculation
    
    # 1. Check if validation set is empty
    if y_val.empty:
        if print_details:
            print("Validation set is empty. F1 score cannot be calculated.")
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        # 2. Handle cases where the validation set contains only one class
        if len(unique_y_val) < 2:
            if print_details:
                print(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            # If true labels are single-class: F1 is 1.0 if all predictions match this class, else 0.0.
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            # Standard macro F1 calculation for multi-class validation set
            # zero_division=0 ensures that if a class has no true instances or no predicted instances,
            # its F1 score contribution is 0, preventing division by zero warnings/errors.
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    if print_details:
        print(f"Validation F1 Score: {final_validation_score}")
    return final_validation_score

# --- Ablation Study Orchestration ---
if __name__ == "__main__":
    setup_dummy_data() # Ensure dummy data is available for execution

    results = {}

    # 1. Base run (original solution settings)
    results['Original Solution'] = run_ablation_experiment(
        experiment_name="Original Solution",
        n_estimators_rf=100,
        include_encoded_identifiers=True
    )

    # 2. Ablation: Exclude encoded TERM_CODE and SUBJECT_ID_SORT from features
    results['Ablation: Exclude Encoded Identifiers (Feature Engineering)'] = run_ablation_experiment(
        experiment_name="Ablation: Exclude Encoded Identifiers",
        n_estimators_rf=100,
        include_encoded_identifiers=False
    )

    # 3. Ablation: Reduce n_estimators for RandomForestClassifier
    results['Ablation: Reduce n_estimators (10) (Model Hyperparameter)'] = run_ablation_experiment(
        experiment_name="Ablation: Reduce n_estimators (10)",
        n_estimators_rf=10,
        include_encoded_identifiers=True
    )

    print("\n=== Ablation Study Results ===")
    for experiment, score in results.items():
        print(f"- {experiment}: {score:.4f}")

    # Determine which part contributes the most
    original_score = results['Original Solution']
    
    performance_impacts = {}
    for experiment, score in results.items():
        if experiment != 'Original Solution':
            impact = original_score - score
            performance_impacts[experiment] = impact
    
    if not performance_impacts:
        print("\nNo ablations performed to compare against the original solution.")
    else:
        most_impactful_ablation = max(performance_impacts, key=performance_impacts.get)
        highest_impact_value = performance_impacts[most_impactful_ablation]

        print(f"\n--- Conclusion ---")
        if highest_impact_value > 0:
            print(f"The most impactful part, when ablated, leading to the largest performance drop, is: '{most_impactful_ablation}'.")
            print(f"Removing/modifying it decreased the F1 score by {highest_impact_value:.4f}.")
            print(f"This indicates that the aspect related to '{most_impactful_ablation.replace('Ablation: ', '').split('(')[0].strip()}' contributes significantly to the overall performance.")
        else:
            print("None of the ablations significantly decreased performance, or some even improved it (unlikely for simple ablations).")
            print("This might suggest the ablated parts are not critical or the dummy data is too simple to show strong effects.")

