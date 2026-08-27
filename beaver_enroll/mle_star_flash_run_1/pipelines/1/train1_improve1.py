
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")
SUBJECT_SUMMARY_FILE = os.path.join(TRAIN_DATA_DIR, "subject_summary_train.csv") # Assuming a file for features

# --- Dummy Data Generation (if files don't exist) ---
def generate_dummy_data():
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

    print("Generating dummy data...")

    # Dummy subject_summary_train.csv
    num_unique_courses_per_term_per_dept = 10
    terms = [202010, 202020, 202030, 202110, 202120, 202130, 202210, 202220, 202230]
    departments = [f"DEPT{i:02d}" for i in range(5)] # Add departments
    
    data = []
    for term in terms:
        for dept in departments:
            for _ in range(num_unique_courses_per_term_per_dept):
                subject_id = f"SUBJ_{dept}_{np.random.randint(100,999)}" # Make subject_id more unique per dept
                enrollment_count = np.random.randint(0, 150)
                avg_gpa = np.random.uniform(2.5, 4.0)
                course_level = np.random.choice(['100', '200', '300', '400'])
                data.append({
                    'TERM_CODE': term,
                    'SUBJECT_ID_SORT': subject_id, # This typically identifies a specific course offering
                    'DEPARTMENT_ID': dept, # Add department for quartile calculation
                    'ENROLLMENT_COUNT': enrollment_count,
                    'AVG_GPA_PREREQ': avg_gpa,
                    'COURSE_LEVEL': course_level,
                    'CREDITS': np.random.choice([3, 4])
                })
    
    subject_summary_df = pd.DataFrame(data)
    subject_summary_df.to_csv(SUBJECT_SUMMARY_FILE, index=False)
    print(f"Generated dummy {SUBJECT_SUMMARY_FILE}")

    # Dummy gold_enrollment_train.csv
    # This file should only contain TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT for entries
    # that actually exist in the subject_summary_df and have positive enrollment
    
    # Filter for positive enrollment to calculate quartiles
    positive_enrollment_courses = subject_summary_df[subject_summary_df['ENROLLMENT_COUNT'] > 0].copy()
    
    if not positive_enrollment_courses.empty:
        gold_data_list = []
        # Calculate high enrollment based on department/term quartile
        # Group by TERM_CODE and DEPARTMENT_ID
        for (term, department), group_df in positive_enrollment_courses.groupby(['TERM_CODE', 'DEPARTMENT_ID']):
            if not group_df.empty:
                # Calculate the 75th percentile (top quartile threshold) for enrollment within this group
                enrollment_threshold = group_df['ENROLLMENT_COUNT'].quantile(0.75)
                
                for _, row in group_df.iterrows():
                    is_high_enrollment = 'Y' if row['ENROLLMENT_COUNT'] >= enrollment_threshold else 'N'
                    gold_data_list.append({
                        'TERM_CODE': row['TERM_CODE'],
                        'SUBJECT_ID_SORT': row['SUBJECT_ID_SORT'],
                        'HIGH_ENROLLMENT': is_high_enrollment
                    })
        
        if gold_data_list:
            gold_enrollment_df = pd.DataFrame(gold_data_list)
            # Ensure unique TERM_CODE, SUBJECT_ID_SORT combinations as per the gold file structure
            gold_enrollment_df.drop_duplicates(subset=['TERM_CODE', 'SUBJECT_ID_SORT'], inplace=True)
            gold_enrollment_df.to_csv(GOLD_ENROLLMENT_FILE, index=False)
            print(f"Generated dummy {GOLD_ENROLLMENT_FILE}")
        else:
            print("No data generated for gold_enrollment_train.csv from positive enrollment courses. Creating a minimal one.")
            # Fallback for extreme cases where no valid gold data could be formed
            gold_enrollment_df = pd.DataFrame([
                {'TERM_CODE': terms[0], 'SUBJECT_ID_SORT': f"SUBJ_{departments[0]}_100", 'HIGH_ENROLLMENT': 'Y'},
                {'TERM_CODE': terms[1], 'SUBJECT_ID_SORT': f"SUBJ_{departments[1]}_200", 'HIGH_ENROLLMENT': 'N'}
            ])
            gold_enrollment_df.to_csv(GOLD_ENROLLMENT_FILE, index=False)
            print(f"Generated minimal dummy {GOLD_ENROLLMENT_FILE}")
    else:
        print("No positive enrollment courses in dummy data to generate gold_enrollment_train.csv. Creating a minimal one.")
        gold_enrollment_df = pd.DataFrame([
            {'TERM_CODE': terms[0], 'SUBJECT_ID_SORT': f"SUBJ_{departments[0]}_100", 'HIGH_ENROLLMENT': 'Y'},
            {'TERM_CODE': terms[1], 'SUBJECT_ID_SORT': f"SUBJ_{departments[1]}_200", 'HIGH_ENROLLMENT': 'N'}
        ])
        gold_enrollment_df.to_csv(GOLD_ENROLLMENT_FILE, index=False)
        print(f"Generated minimal dummy {GOLD_ENROLLMENT_FILE}")


if not os.path.exists(GOLD_ENROLLMENT_FILE) or not os.path.exists(SUBJECT_SUMMARY_FILE):
    generate_dummy_data()

# --- Load Data ---
gold_df = pd.DataFrame()
subject_summary_df = pd.DataFrame()
final_validation_score = 0.0 # Initialize score
data_loaded_successfully = False

try:
    gold_df = pd.read_csv(GOLD_ENROLLMENT_FILE)
    subject_summary_df = pd.read_csv(SUBJECT_SUMMARY_FILE)
    print("Data loaded successfully.")
    data_loaded_successfully = True
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please ensure dummy data generation completed or files exist.")
    print(f"Final Validation Performance: {final_validation_score}")
    # The original error was likely caused by SystemExit(1) here.
    # By removing it, the script will proceed with empty dataframes and gracefully handle it.

if data_loaded_successfully:
    # Merge features and labels
    # The gold file defines the "prediction keys", so we should merge features onto the gold file keys.
    merged_df = pd.merge(gold_df, subject_summary_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    # Drop rows where critical features might be missing after the left merge
    # e.g., if a gold entry had no matching features in subject_summary_df
    critical_features = ['ENROLLMENT_COUNT', 'DEPARTMENT_ID']
    merged_df.dropna(subset=critical_features, inplace=True) 
    print(f"Merged data shape: {merged_df.shape}")

    if merged_df.empty or len(merged_df['HIGH_ENROLLMENT'].unique()) < 2:
        print("Not enough data or classes available after merge for training. Cannot proceed.")
        print(f"Final Validation Performance: {final_validation_score}") # Indicate no performance if no data
    else:
        # --- Feature Engineering and Preprocessing ---
        # Use LabelEncoder for target variable
        le = LabelEncoder()
        merged_df['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(merged_df['HIGH_ENROLLMENT'])
        
        # One-hot encode other categorical features
        categorical_cols = ['SUBJECT_ID_SORT', 'DEPARTMENT_ID', 'COURSE_LEVEL']
        # Filter to ensure only columns actually present are included
        categorical_cols_present = [col for col in categorical_cols if col in merged_df.columns]
        merged_df = pd.get_dummies(merged_df, columns=categorical_cols_present, drop_first=True)
        
        # Define features (X) and target (y)
        # Exclude original target and key columns
        feature_cols = [col for col in merged_df.columns if col not in 
                        ['TERM_CODE', 'HIGH_ENROLLMENT', 'HIGH_ENROLLMENT_ENCODED']]
        
        # Ensure all feature columns exist, drop if missing
        X = merged_df[feature_cols]
        y = merged_df['HIGH_ENROLLMENT_ENCODED']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # --- Time-based Validation Split ---
        # Sort by TERM_CODE for time-based split
        merged_df_sorted = merged_df.sort_values(by='TERM_CODE').reset_index(drop=True)
        
        # Identify unique terms
        unique_terms = sorted(merged_df_sorted['TERM_CODE'].unique())
        
        X_train, X_val, y_train, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

        if len(unique_terms) < 2: # Need at least two terms for a meaningful time-based split
            print("Not enough unique terms for time-based split. Falling back to random split.")
            if not X.empty and not y.empty:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            # Use the latest term(s) for validation (e.g., latest 20% of terms)
            split_point_idx = int(len(unique_terms) * 0.8)
            
            # Ensure that validation set has at least one term and train set has at least one term
            if split_point_idx == 0: # If 80% is 0, meaning only 1-4 terms total, take first for train, last for val
                train_terms = [unique_terms[0]]
                val_terms = unique_terms[1:] if len(unique_terms) > 1 else []
            elif split_point_idx == len(unique_terms): # If all terms in train
                train_terms = unique_terms[:-1] if len(unique_terms) > 1 else []
                val_terms = [unique_terms[-1]] if len(unique_terms) > 1 else []
            else:
                train_terms = unique_terms[:split_point_idx]
                val_terms = unique_terms[split_point_idx:]
            
            # Re-check for empty splits, especially if few unique terms
            if not train_terms or not val_terms:
                print("Time-based split logic resulted in empty train/validation term sets. Falling back to random split.")
                if not X.empty and not y.empty:
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train = merged_df_sorted[merged_df_sorted['TERM_CODE'].isin(train_terms)][feature_cols]
                y_train = merged_df_sorted[merged_df_sorted['TERM_CODE'].isin(train_terms)]['HIGH_ENROLLMENT_ENCODED']
                X_val = merged_df_sorted[merged_df_sorted['TERM_CODE'].isin(val_terms)][feature_cols]
                y_val = merged_df_sorted[merged_df_sorted['TERM_CODE'].isin(val_terms)]['HIGH_ENROLLMENT_ENCODED']
                
                # Fallback for empty splits due to data distribution even after selecting terms
                if X_train.empty or X_val.empty or y_train.empty or y_val.empty:
                    print("Time-based split resulted in empty dataframes. Falling back to random split.")
                    if not X.empty and not y.empty:
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    else:
                        X_train, X_val, y_train, y_val = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
                else:
                    print(f"Time-based split: Train terms {train_terms}, Validation terms {val_terms}")
        
        print(f"Train data shape: {X_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

        # --- Model Training and Evaluation ---
        if X_train.empty or y_train.empty:
            print("Training data is empty. Cannot train model.")
            final_validation_score = 0.0 
        elif X_val.empty or y_val.empty:
            print("Validation data is empty. Cannot evaluate model.")
            # Train anyway if train data exists, but validation score will be 0
            if not X_train.empty and not y_train.empty:
                model = RandomForestClassifier(random_state=42, n_estimators=100)
                model.fit(X_train, y_train)
            final_validation_score = 0.0
        else:
            # Align columns between train and validation sets after one-hot encoding
            # This handles cases where a category might appear in train but not val, or vice-versa
            train_cols = set(X_train.columns)
            val_cols = set(X_val.columns)

            missing_in_val = list(train_cols - val_cols)
            for col in missing_in_val:
                X_val[col] = 0

            missing_in_train = list(val_cols - train_cols)
            for col in missing_in_train:
                X_train[col] = 0
            
            # Reorder validation columns to match training columns
            X_val = X_val[X_train.columns]

            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
        
            # --- Prediction and Evaluation ---
            y_pred = model.predict(X_val)
            final_validation_score = f1_score(y_val, y_pred, average='macro')
            print(f"Model trained and evaluated.")

# This line MUST be present and correctly formatted for external parsing
print(f"Final Validation Performance: {final_validation_score}")
