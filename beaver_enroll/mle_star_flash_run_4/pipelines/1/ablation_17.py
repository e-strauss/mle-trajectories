
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Define file paths
INPUT_DIR = './input'
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, 'table_splits/train')
GOLD_ENROLLMENT_PATH = os.path.join(INPUT_DIR, 'gold_enrollment_train.csv')

# --- 1. Load Data ---
def load_data():
    """Loads all necessary CSVs into pandas DataFrames."""
    try:
        course_summary = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'course_summary.csv'))
        offering = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'offering.csv'))
        schedule = pd.read_csv(os.path.join(TRAIN_DATA_DIR, 'schedule.csv'))
        gold_enrollment = pd.read_csv(GOLD_ENROLLMENT_PATH)
        return course_summary, offering, schedule, gold_enrollment
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the input files are in the correct directory.")
        return None, None, None, None

# --- 2. Feature Engineering ---
def create_features(course_summary, offering, schedule, gold_enrollment):
    """Merges tables and engineers features for the model."""

    # Pre-process schedule data: count meetings per offering
    schedule_agg = schedule.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).size().reset_index(name='MEETING_COUNT')

    # Merge dataframes
    # Start with gold_enrollment to ensure we have the target for all rows
    df = gold_enrollment.copy()
    
    # Merge with course_summary
    df = pd.merge(df, course_summary, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Merge with offering data
    # offering might have multiple rows per course, aggregate it first
    offering_agg = offering.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg({
        'ENROLLMENT_CAPACITY': 'sum',
        'INSTRUCTOR_COUNT': 'mean' # mean instructors per offering of a course
    }).reset_index()
    df = pd.merge(df, offering_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Merge with aggregated schedule data
    df = pd.merge(df, schedule_agg, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # --- Feature Creation ---

    # Time-based features
    df['TERM_CODE'] = df['TERM_CODE'].astype(int)
    df = df.sort_values(by=['SUBJECT_ID_SORT', 'TERM_CODE'])
    
    # Lag features: previous term's enrollment for the same course
    # The original data has ENROLLMENT_COUNT. Let's use that.
    # Note: 'ENROLLMENT_COUNT' from course_summary is the actual enrollment, not the target label.
    # The target is 'HIGH_ENROLLMENT', which is a derived quartile-based flag.
    for lag in [1, 2]:
        df[f'LAG_ENROLLMENT_{lag}'] = df.groupby('SUBJECT_ID_SORT')['ENROLLMENT_COUNT'].shift(lag)

    # Rolling average enrollment
    df['ROLLING_ENROLLMENT_3_TERMS'] = df.groupby('SUBJECT_ID_SORT')['ENROLLMENT_COUNT'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Term-based features
    df['TERM_YEAR'] = df['TERM_CODE'] // 10
    df['TERM_SEASON'] = df['TERM_CODE'] % 10 # Assuming Fall=4, Spring=2, Summer=3 etc.

    # Interaction features
    df['CAPACITY_PER_MEETING'] = df['ENROLLMENT_CAPACITY'] / df['MEETING_COUNT']
    
    # Handle missing values
    # Fill NaNs created by lag/rolling features with 0 or a median/mean if appropriate.
    # For enrollment, 0 is a reasonable fill if a course didn't exist before.
    df.fillna({
        'LAG_ENROLLMENT_1': 0,
        'LAG_ENROLLMENT_2': 0,
        'ROLLING_ENROLLMENT_3_TERMS': 0,
        'MEETING_COUNT': 0,
        'INSTRUCTOR_COUNT': 1, # Assume at least one instructor
        'CAPACITY_PER_MEETING': 0
    }, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)


    # --- Target and Categorical Encoding ---

    # Encode target variable
    le_target = LabelEncoder()
    df['HIGH_ENROLLMENT'] = le_target.fit_transform(df['HIGH_ENROLLMENT'])
    
    # Select feature columns
    # Keep 'SUBJECT_ID_SORT' for now if needed, but it should not be a feature itself.
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove keys from categorical features
    if 'SUBJECT_ID_SORT' in categorical_cols:
        categorical_cols.remove('SUBJECT_ID_SORT')
        
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True, drop_first=True)

    return df

# --- 3. Main Execution ---
if __name__ == "__main__":
    
    # Load data
    course_summary, offering, schedule, gold_enrollment = load_data()

    if course_summary is not None:
        # Engineer features
        # The KeyError 'SUBJECT_ID_SORT' was likely due to it being dropped or becoming an index.
        # The key is preserved by using reset_index() after aggregations and ensuring it's in the merge keys.
        featured_data = create_features(course_summary, offering, schedule, gold_enrollment)

        # --- Time-based Validation Split ---
        # Sort terms and select the latest year (or equivalent) for validation
        unique_terms = sorted(featured_data['TERM_CODE'].unique())
        
        # Hold out the last 3 terms for validation, which is roughly one academic year
        validation_terms = unique_terms[-3:] 
        train_terms = [t for t in unique_terms if t not in validation_terms]
        
        # Ensure we have a training set. If there are too few terms, adjust.
        if not train_terms:
            # Fallback for very few terms: use the last term for validation
            validation_terms = unique_terms[-1:]
            train_terms = unique_terms[:-1]

        train_df = featured_data[featured_data['TERM_CODE'].isin(train_terms)]
        val_df = featured_data[featured_data['TERM_CODE'].isin(validation_terms)]

        # --- Model Training ---
        
        # Define features (X) and target (y)
        # Drop non-feature columns
        cols_to_drop = ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT', 'ENROLLMENT_COUNT']
        
        # Make sure all columns to be dropped exist in the dataframe
        X_train_cols = [col for col in train_df.columns if col not in cols_to_drop]
        
        X_train = train_df[X_train_cols]
        y_train = train_df['HIGH_ENROLLMENT']
        
        X_val = val_df.reindex(columns=X_train_cols, fill_value=0)
        y_val = val_df['HIGH_ENROLLMENT']
        
        # Align columns after one-hot encoding, in case some categories only appear in one split
        train_cols = X_train.columns
        val_cols = X_val.columns

        missing_in_val = set(train_cols) - set(val_cols)
        for c in missing_in_val:
            X_val[c] = 0
            
        missing_in_train = set(val_cols) - set(train_cols)
        for c in missing_in_train:
            X_train[c] = 0
        
        X_val = X_val[train_cols] # Ensure order is the same

        # Initialize and train the model
        # RandomForest is a good baseline. Class weight is used for imbalanced datasets.
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        model.fit(X_train, y_train)

        # --- Evaluation ---
        
        # Predict on validation set
        y_pred = model.predict(X_val)

        # Calculate macro F1 score
        # Check if validation set is not empty
        if not y_val.empty:
            final_validation_score = f1_score(y_val, y_pred, average='macro')
            print(f"Final Validation Performance: {final_validation_score}")
        else:
            print("Validation set is empty, cannot calculate performance.")
            print(f"Final Validation Performance: 0.0")
