

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings

# Suppress all warnings for cleaner output during the ablation study
warnings.filterwarnings('ignore')

# Define paths (assuming they are relative to the script execution)
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# --- Function to load data and engineer features ---
# Modified to accept flags for ablation study
def load_data(data_dir, gold_file, 
              include_existing_numeric_summary_cols=True, # Ablation control for direct numeric summary features
              include_polynomial_features=True, # Control for previous ablations (kept for completeness)
              include_interaction_features=True # Control for previous ablations (kept for completeness)
              ):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Ablation flags allow selective inclusion of feature groups.
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
        
        current_feature_cols = []

        # 1. Add existing numerical columns from the merged DataFrame (excluding identifiers and target)
        if include_existing_numeric_summary_cols: # Ablation point for direct numeric summary features
            existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + ['HIGH_ENROLLMENT']]
            current_feature_cols.extend(existing_numeric_cols)

        # 2. Encode TERM_CODE and SUBJECT_ID_SORT and add them as numerical features
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        current_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        # Identify candidate columns for advanced feature engineering from subject_summary
        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        if include_polynomial_features:
            # 3. Polynomial Features (e.g., squared terms for key numerical features)
            poly_features_to_square = numeric_summary_cols[:3]
            for col in poly_features_to_square:
                new_col_name = f"{col}_SQUARED"
                df[new_col_name] = df[col] ** 2
                current_feature_cols.append(new_col_name)
            
        if include_interaction_features:
            # 4. Interaction Features (product of two distinct features)
            if 'TERM_CODE_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 1:
                col_to_interact = numeric_summary_cols[0] 
                new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
                current_feature_cols.append(new_col_name)

            if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) >= 2:
                col_to_interact = numeric_summary_cols[1] 
                new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                current_feature_cols.append(new_col_name)
            elif 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and len(numeric_summary_cols) == 1:
                 col_to_interact = numeric_summary_cols[0] 
                 new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
                 df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
                 current_feature_cols.append(new_col_name)

            if len(numeric_summary_cols) >= 2:
                col1 = numeric_summary_cols[0]
                col2 = numeric_summary_cols[1]
                new_col_name = f"{col1}_x_{col2}"
                df[new_col_name] = df[col1] * df[col2]
                current_feature_cols.append(new_col_name)

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(current_feature_cols))

        # Remove columns from final_feature_cols that might have been dropped or are non-existent after merge
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found, ensure encoded identifiers are present, or add dummy.
        if not final_feature_cols and ('TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns):
            final_feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
        elif not final_feature_cols:
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge or missing values in summary data.
        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        
        # Store feature columns for later use
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        # Fallback to the minimal features if subject_summary.csv is not found
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Experiment Runner ---
def run_experiment(name, 
                   rf_n_estimators=100, 
                   rf_random_state=42, 
                   rf_max_depth=None, # Ablation 1 control: Max depth of individual trees
                   rf_bootstrap=True, # Ablation 2 control: Whether bootstrap samples are used
                   include_existing_numeric_summary_cols=True, # Ablation 3 control: Direct numeric features from subject_summary
                   include_polynomial_features=True, 
                   include_interaction_features=True
                   ):
    
    df = load_data(TRAIN_DATA_DIR, GOLD_ENROLLMENT_TRAIN_FILE,
                   include_existing_numeric_summary_cols=include_existing_numeric_summary_cols,
                   include_polynomial_features=include_polynomial_features,
                   include_interaction_features=include_interaction_features
                   )

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
        pass # Training set target has only one class; handle gracefully.
    
    # Model Training with ablation parameters
    model = RandomForestClassifier(random_state=rf_random_state, 
                                   n_estimators=rf_n_estimators,
                                   max_depth=rf_max_depth,
                                   bootstrap=rf_bootstrap)
    model.fit(X_train, y_train)

    final_validation_score = 0.0 
    
    if y_val.empty:
        pass # Validation set is empty.
    else:
        y_pred = model.predict(X_val)
        
        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)
    
    return final_validation_score

# --- Main Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Baseline: Original Solution
    results['Baseline'] = run_experiment("Baseline (Original Solution)")

    # Ablation 1: Constrain Random Forest max_depth to 5 (default is None)
    results['Ablation: RF max_depth=5'] = run_experiment("Ablation: RF max_depth=5", rf_max_depth=5)

    # Ablation 2: Disable bootstrapping in Random Forest (default is True)
    results['Ablation: RF bootstrap=False'] = run_experiment("Ablation: RF bootstrap=False", rf_bootstrap=False)

    # Ablation 3: Exclude existing_numeric_cols (direct numerical features from subject_summary.csv)
    # This keeps encoded identifiers and advanced features (poly/interaction) which might still use other derived numeric summary features.
    results['Ablation: No direct numeric summary features'] = run_experiment("Ablation: No direct numeric summary features", 
                                                                         include_existing_numeric_summary_cols=False)

    print("--- Ablation Study Summary ---")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    baseline_score = results.get('Baseline', 0.0)
    
    impact_scores = {}
    for name, score in results.items():
        if name != 'Baseline':
            # Impact is the difference from baseline. A larger drop in score indicates greater impact of the ablated part.
            impact_scores[name] = baseline_score - score

    if impact_scores:
        most_impactful_ablation = max(impact_scores, key=impact_scores.get)
        max_impact = impact_scores[most_impactful_ablation]
        
        if max_impact > 0:
            print(f"\nThe part that contributed most to the overall performance based on this study is: '{most_impactful_ablation.replace('Ablation: ', '')}' (reduced performance by {max_impact:.4f} when removed/modified).")
        else:
            print("\nBased on this study, no single ablated component showed a negative impact on performance, or all components performed identically. This might indicate the dataset is too simple or other factors are dominant.")
    else:
        print("\nNo ablations were performed to determine impact.")

