
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features (modified for ablation) ---
def load_data_ablation(data_dir, gold_file, ablation_no_interaction_features=False, ablation_fillna_strategy='zero'):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Configurable for ablation study.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        # Merge gold_df (which now contains numerical 'HIGH_ENROLLMENT') with subject_summary_df to get features.
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        # Initialize the list of feature columns
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
        current_feature_cols.extend(existing_numeric_cols)

        # 2. Target Encode TERM_CODE and SUBJECT_ID_SORT
        term_code_te_col_name = 'TERM_CODE_TARGET_ENCODED'
        df[term_code_te_col_name] = df.groupby('TERM_CODE')['HIGH_ENROLLMENT'].transform('mean')
        current_feature_cols.append(term_code_te_col_name)

        subject_id_sort_te_col_name = 'SUBJECT_ID_SORT_TARGET_ENCODED'
        df[subject_id_sort_te_col_name] = df.groupby('SUBJECT_ID_SORT')['HIGH_ENROLLMENT'].transform('mean')
        current_feature_cols.append(subject_id_sort_te_col_name)
        
        # Identify candidate columns for advanced feature engineering from subject_summary
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        
        # 3. Polynomial Features (e.g., squared terms for key numerical features)
        # Select up to 3 features from `numeric_summary_cols` for squaring to avoid feature explosion.
        poly_features_to_square = numeric_summary_cols[:3] 
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        if not ablation_no_interaction_features:
            relevant_numeric_for_interaction = numeric_summary_cols[:3] 

            for num_col in relevant_numeric_for_interaction:
                # Interaction with Target Encoded TERM_CODE
                new_col_name_term = f"{num_col}_x_TERM_TE"
                df[new_col_name_term] = df[num_col] * df[term_code_te_col_name]
                current_feature_cols.append(new_col_name_term)

                # Interaction with Target Encoded SUBJECT_ID_SORT
                new_col_name_subject = f"{num_col}_x_SUBJECT_TE"
                df[new_col_name_subject] = df[num_col] * df[subject_id_sort_te_col_name]
                current_feature_cols.append(new_col_name_subject)

        # 5. Statistical Aggregate Features (mean, standard deviation)
        key_num_cols_for_agg = numeric_summary_cols[:5] 

        if key_num_cols_for_agg: 
            for col in key_num_cols_for_agg:
                mean_by_term_col_name = f"{col}_MEAN_BY_TERM"
                std_by_term_col_name = f"{col}_STD_BY_TERM"
                df[mean_by_term_col_name] = df.groupby('TERM_CODE')[col].transform('mean')
                df[std_by_term_col_name] = df.groupby('TERM_CODE')[col].transform('std')
                current_feature_cols.extend([mean_by_term_col_name, std_by_term_col_name])
                
                mean_by_subject_col_name = f"{col}_MEAN_BY_SUBJECT"
                std_by_subject_col_name = f"{col}_STD_BY_SUBJECT"
                df[mean_by_subject_col_name] = df.groupby('SUBJECT_ID_SORT')[col].transform('mean')
                df[std_by_subject_col_name] = df.groupby('SUBJECT_ID_SORT')[col].transform('std')
                current_feature_cols.extend([mean_by_subject_col_name, std_by_subject_col_name])

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(current_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found after all engineering, add a dummy feature.
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge or missing values in summary data.
        # This applies to all original and newly engineered features.
        if ablation_fillna_strategy == 'zero':
            df[final_feature_cols] = df[final_feature_cols].fillna(0)
        elif ablation_fillna_strategy == 'mean':
            for col in final_feature_cols:
                # Ensure the column exists and is numeric before attempting mean imputation
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mean())
                elif col in df.columns: # If not numeric, fill with 0 or a placeholder
                    df[col] = df[col].fillna(0)
        
        # Store feature columns for later use
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        warnings.warn(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0) # Minimal features still fillna 0
        
    return df

# --- Main script (modified for ablation) ---
def run_training_and_validation_ablation(rf_min_samples_leaf=1, ablation_no_interaction_features=False, ablation_fillna_strategy='zero'):
    df = load_data_ablation(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE, 
                            ablation_no_interaction_features=ablation_no_interaction_features, 
                            ablation_fillna_strategy=ablation_fillna_strategy)

    if df.empty:
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
    if not feature_cols: 
        warnings.warn("Feature columns not correctly identified by load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        return 0.0

    if len(y_train.unique()) < 2:
        pass
    
    model = RandomForestClassifier(random_state=42, n_estimators=100, min_samples_leaf=rf_min_samples_leaf)
    model.fit(X_train, y_train)

    final_validation_score = 0.0 
    
    if y_val.empty:
        pass
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            warnings.warn(f"Validation set 'HIGH_ENROLLMENT' has only one class: {unique_y_val}. Macro F1 calculation adjusted for this edge case.")
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    return final_validation_score

# Store results
results = {}

# Scenario 1: Baseline
print("Baseline F1 Score: {:.4f}".format(run_training_and_validation_ablation()))
results['Baseline'] = run_training_and_validation_ablation()

# Scenario 2: Ablation: Change NaN Imputation to Mean
print("Ablation: NaN Imputation to Mean F1 Score: {:.4f}".format(run_training_and_validation_ablation(ablation_fillna_strategy='mean')))
results['Ablation: NaN Imputation to Mean'] = run_training_and_validation_ablation(ablation_fillna_strategy='mean')

# Scenario 3: Ablation: No Interaction Features
print("Ablation: No Interaction Features F1 Score: {:.4f}".format(run_training_and_validation_ablation(ablation_no_interaction_features=True)))
results['Ablation: No Interaction Features'] = run_training_and_validation_ablation(ablation_no_interaction_features=True)

# Scenario 4: Ablation: RF min_samples_leaf=20
print("Ablation: RF min_samples_leaf=20 F1 Score: {:.4f}".format(run_training_and_validation_ablation(rf_min_samples_leaf=20)))
results['Ablation: RF min_samples_leaf=20'] = run_training_and_validation_ablation(rf_min_samples_leaf=20)


# Determine the most impactful part
most_impactful_part = "None of the ablated components had a significant impact, or the dataset is too simple to show strong effects."
largest_drop = 0.0
impact_summary = {}

for name, score in results.items():
    if name != 'Baseline':
        drop = results['Baseline'] - score
        impact_summary[name] = drop
        if drop > largest_drop:
            largest_drop = drop
            most_impactful_part = name

if largest_drop > 0:
    print(f"\nConclusion: The most impactful change was '{most_impactful_part}', which caused a performance drop of {largest_drop:.4f}.")
elif largest_drop == 0 and results['Baseline'] == 1.0:
    print("\nConclusion: None of the ablated components caused a performance drop from the perfect baseline (1.0000), suggesting the dataset might be too simple or the validation strategy not challenging enough to differentiate their impact.")
else:
    # This case handles scenarios where baseline is not 1.0 and no drop occurred, or even improvements.
    largest_change = 0.0
    most_impactful_element = "None"
    
    for name, drop in impact_summary.items():
        if abs(drop) > largest_change:
            largest_change = abs(drop)
            most_impactful_element = name
            
    if largest_change > 0:
        if impact_summary[most_impactful_element] > 0:
            print(f"\nConclusion: The most impactful change was '{most_impactful_element}', which caused a performance drop of {impact_summary[most_impactful_element]:.4f}.")
        else:
            print(f"\nConclusion: The most impactful change was '{most_impactful_element}', which caused a performance increase of {-impact_summary[most_impactful_element]:.4f}.")
    else:
        print("\nConclusion: None of the ablated components caused a performance change.")

