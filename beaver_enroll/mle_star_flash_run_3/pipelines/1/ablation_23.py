
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration for data paths ---
# All data is expected to be in the 'input' directory relative to the script execution.
BASE_DATA_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(BASE_DATA_DIR, "table_splits/train")
GOLD_TRAIN_LABELS = os.path.join(BASE_DATA_DIR, "eval/gold_enrollment_train.csv")

# Function to encapsulate the core logic for reuse and modification
def run_ablation_scenario(
    include_temporal_features=True,
    numerical_imputation_strategy='zero', # 'zero' or 'mean'
    validation_years_strategy='last_two_years' # 'last_two_years' or 'last_year'
):
    
    train_main_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    try:
        # Load the main subject summary data
        subject_summary_path = os.path.join(TRAIN_DATA_DIR, "subject_summary.csv")
        train_main_df = pd.read_csv(subject_summary_path)
    except FileNotFoundError:
        # Fallback to dummy data if real data loading fails
        train_main_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501, 202501, 202507, 202601, 202601],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201', 'CS-101', 'MA-201', 'PH-101', 'BI-301'],
            'CREDITS_ATTEMPTED_COUNT': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3, 3, 4, 3, 4],
            'WAITLIST_COUNT': [0, 5, 0, 0, 2, 1, 0, 0, 3, 0, 0, 5, 0, 1],
            'MAX_ENROLL': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65, 100, 50, 120, 70],
            'ENROLL_COUNT': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50, 80, 45, 110, 60],
            'PRE_ENROLL_COUNT': [70, 40, 100, 60, 85, 50, 90, 50, 75, 40, 70, 40, 100, 50],
            'ROOM_CAPACITY': [100, 50, 120, 80, 100, 60, 110, 70, 90, 65, 100, 50, 120, 70],
            'INSTRUCTOR_COUNT': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'SECTION_COUNT': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'GRADES_COUNT': [80, 45, 110, 70, 95, 55, 100, 60, 85, 50, 80, 45, 110, 60],
            'DROP_COUNT': [5, 2, 8, 3, 6, 1, 7, 2, 5, 0, 5, 2, 8, 3],
            'COURSE_LEVEL': [100, 200, 100, 100, 100, 200, 100, 300, 100, 200, 100, 200, 100, 300],
            'ACADEMIC_ORGANIZATION_COUNT': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'SCH_CREDITS': [3, 4, 3, 3, 3, 4, 3, 4, 3, 3, 3, 4, 3, 4],
        })

    except Exception:
        return 0.0 # Indicate failure

    try:
        # Load the gold enrollment labels
        labels_df = pd.read_csv(GOLD_TRAIN_LABELS)
    except FileNotFoundError:
        labels_df = pd.DataFrame({
            'TERM_CODE': [202301, 202301, 202301, 202307, 202307, 202401, 202401, 202407, 202407, 202501, 202501, 202507, 202601, 202601],
            'SUBJECT_ID_SORT': ['CS-101', 'MA-201', 'CS-102', 'PH-101', 'CS-101', 'MA-201', 'CS-103', 'BI-301', 'PH-101', 'CH-201', 'CS-101', 'MA-201', 'PH-101', 'BI-301'],
            'HIGH_ENROLLMENT': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] 
        })
    except Exception:
        return 0.0 # Indicate failure

    # Merge subject summary with labels
    train_main_df = pd.merge(train_main_df, labels_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

    if 'HIGH_ENROLLMENT' not in train_main_df.columns:
        return 0.0
    
    train_main_df.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    
    if train_main_df.empty:
        return 0.0

    # Ensure TERM_CODE is numeric for sorting and feature extraction
    train_main_df['TERM_CODE'] = pd.to_numeric(train_main_df['TERM_CODE'], errors='coerce').fillna(0).astype(int)

    # --- Feature Engineering ---

    # Encode SUBJECT_ID_SORT using LabelEncoder
    if 'SUBJECT_ID_SORT' in train_main_df.columns:
        le_subject_id = LabelEncoder()
        train_main_df['SUBJECT_ID_SORT_ENCODED'] = le_subject_id.fit_transform(train_main_df['SUBJECT_ID_SORT'])
    else:
        train_main_df['SUBJECT_ID_SORT_ENCODED'] = 0

    # Ablation 1: Include/Exclude derived temporal features
    if include_temporal_features:
        train_main_df['TERM_YEAR'] = train_main_df['TERM_CODE'].astype(str).str[:4].astype(int)
        train_main_df['TERM_SEMESTER'] = train_main_df['TERM_CODE'].astype(str).str[4:].astype(int)
    # If not included, these columns won't be in the DataFrame and thus not in features

    numerical_features_list = [
        'CREDITS_ATTEMPTED_COUNT', 'WAITLIST_COUNT', 'MAX_ENROLL', 'ENROLL_COUNT',
        'PRE_ENROLL_COUNT', 'ROOM_CAPACITY', 'INSTRUCTOR_COUNT',
        'SECTION_COUNT', 'GRADES_COUNT', 'DROP_COUNT', 'COURSE_LEVEL',
        'ACADEMIC_ORGANIZATION_COUNT', 'SCH_CREDITS'
    ]

    # Calculate means for imputation (only if strategy is 'mean')
    numerical_means = {}
    if numerical_imputation_strategy == 'mean':
        for col in numerical_features_list:
            if col in train_main_df.columns:
                numerical_means[col] = train_main_df[col].mean()
            else:
                numerical_means[col] = 0 # Default mean if column doesn't exist

    for col in numerical_features_list:
        if col in train_main_df.columns:
            train_main_df[col] = pd.to_numeric(train_main_df[col], errors='coerce')
            # Ablation 2: Numerical Imputation Strategy
            if numerical_imputation_strategy == 'zero':
                train_main_df[col] = train_main_df[col].fillna(0)
            elif numerical_imputation_strategy == 'mean':
                train_main_df[col] = train_main_df[col].fillna(numerical_means.get(col, 0))
        else:
            train_main_df[col] = 0 # If column missing, fill with 0

    # Define columns to drop
    features_to_drop = ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']
    if not include_temporal_features:
        # If temporal features are not included, ensure the derived columns are not present as features
        features_to_drop.extend(['TERM_YEAR', 'TERM_SEMESTER'])
    
    # Filter features_to_drop to only include columns actually present in the DataFrame
    features_to_drop_present = [col for col in features_to_drop if col in train_main_df.columns]
    
    # Define features (X) and target (y)
    X_full = train_main_df.drop(columns=features_to_drop_present)
    y_full = train_main_df['HIGH_ENROLLMENT']

    # --- Time-based validation split ---
    # For splitting, always derive TERM_YEAR for consistent time-based split logic
    temp_train_df_for_splitting = train_main_df.copy()
    temp_train_df_for_splitting['TERM_YEAR_SPLIT'] = temp_train_df_for_splitting['TERM_CODE'].astype(str).str[:4].astype(int)
    temp_train_df_for_splitting = temp_train_df_for_splitting.sort_values(by=['TERM_YEAR_SPLIT', 'TERM_CODE'])
    unique_years_for_split = sorted(temp_train_df_for_splitting['TERM_YEAR_SPLIT'].unique())

    X_train_val, y_train_val, X_val, y_val = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()

    if len(unique_years_for_split) >= 2:
        # Ablation 3: Validation years strategy
        if validation_years_strategy == 'last_two_years':
            validation_years_to_use = unique_years_for_split[-2:]
        elif validation_years_strategy == 'last_year':
            validation_years_to_use = unique_years_for_split[-1:]
        else: # Default for safety
            validation_years_to_use = unique_years_for_split[-2:]

        train_mask = ~temp_train_df_for_splitting['TERM_YEAR_SPLIT'].isin(validation_years_to_use)
        val_mask = temp_train_df_for_splitting['TERM_YEAR_SPLIT'].isin(validation_years_to_use)

        # Use .loc with original indices to get correct rows from X_full and y_full
        X_train_val = X_full.loc[temp_train_df_for_splitting[train_mask].index]
        y_train_val = y_full.loc[temp_train_df_for_splitting[train_mask].index]
        X_val = X_full.loc[temp_train_df_for_splitting[val_mask].index]
        y_val = y_full.loc[temp_train_df_for_splitting[val_mask].index]

    elif len(temp_train_df_for_splitting) > 10 and len(X_full.columns) > 0: # Fallback to random split if not enough years
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    else:
        return 0.0

    # Ensure feature columns are consistent between training and validation sets.
    all_feature_cols = X_full.columns.tolist()

    X_train_val = X_train_val[all_feature_cols]
    X_val = X_val[all_feature_cols]

    if X_train_val.empty or y_train_val.empty:
        return 0.0

    # --- Model Training ---
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_val, y_train_val)

    # --- Evaluation ---
    if X_val.empty or y_val.empty:
        final_validation_score = 0.0
    else:
        y_pred = model.predict(X_val)
        final_validation_score = f1_score(y_val, y_pred, average='macro')
    
    return final_validation_score

def main():
    results = {}

    # BASELINE: Original logic
    baseline_score = run_ablation_scenario(
        include_temporal_features=True,
        numerical_imputation_strategy='zero',
        validation_years_strategy='last_two_years'
    )
    results["Baseline"] = baseline_score
    print(f"Final Validation Performance: {baseline_score}")

    # ABLATION 1: Remove derived temporal features (TERM_YEAR, TERM_SEMESTER)
    ablation1_score = run_ablation_scenario(
        include_temporal_features=False,
        numerical_imputation_strategy='zero',
        validation_years_strategy='last_two_years'
    )
    results["Ablation 1: Remove Derived Temporal Features"] = ablation1_score
    print(f"Final Validation Performance: {ablation1_score}")

    # ABLATION 2: Numerical Imputation Strategy (Change from 'zero' to 'mean')
    ablation2_score = run_ablation_scenario(
        include_temporal_features=True,
        numerical_imputation_strategy='mean',
        validation_years_strategy='last_two_years'
    )
    results["Ablation 2: Numerical Imputation (Mean)"] = ablation2_score
    print(f"Final Validation Performance: {ablation2_score}")

    # ABLATION 3: Time-based Validation Split (Use only last year)
    ablation3_score = run_ablation_scenario(
        include_temporal_features=True,
        numerical_imputation_strategy='zero',
        validation_years_strategy='last_year'
    )
    results["Ablation 3: Validation Split (Last Year Only)"] = ablation3_score
    print(f"Final Validation Performance: {ablation3_score}")

    # Determine the most impactful change
    max_impact_value = -1.0
    most_impactful_parts_list = []

    # Calculate impacts relative to baseline
    impacts = []
    impacts.append(("Derived Temporal Features Inclusion (original: True)", abs(results["Ablation 1: Remove Derived Temporal Features"] - baseline_score)))
    impacts.append(("Numerical Imputation Strategy (original: Zero)", abs(results["Ablation 2: Numerical Imputation (Mean)"] - baseline_score)))
    impacts.append(("Validation Split Strategy (original: Last Two Years)", abs(results["Ablation 3: Validation Split (Last Year Only)"] - baseline_score)))

    # Find the most impactful part(s)
    for part, impact in impacts:
        if impact > max_impact_value:
            max_impact_value = impact
            most_impactful_parts_list = [part]
        elif impact == max_impact_value and impact > 0:
            most_impactful_parts_list.append(part)

    if max_impact_value == 0:
        print("Most impactful part: No specific part showed a significant impact on performance (all changes were 0.0).")
    else:
        parts_str = " and ".join(most_impactful_parts_list)
        print(f"Most impactful part: '{parts_str}' with an absolute F1-score change of {max_impact_value:.4f}.")


if __name__ == "__main__":
    main()

