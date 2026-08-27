
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression # For ablation 2

# Define paths (assuming they are relative to the script execution)
# These paths are placeholders and expect the data to be present in the specified structure.
TRAIN_DATA_DIR = "./input/table_splits/train"
GOLD_ENROLLMENT_TRAIN_FILE = "./input/gold_enrollment_train.csv"

# Suppress warnings for cleaner output during the ablation study
warnings.filterwarnings('ignore')

# --- Function to load data and engineer features (modified for ablation) ---
def load_data_ablation(data_dir, gold_file, use_proxy_for_feature_selection=True):
    """
    Loads gold labels and merges with subject summary data for feature engineering.
    Handles missing subject_summary.csv by falling back to minimal features.
    Includes a flag to control the proxy model for feature selection in advanced FE.
    """
    gold_df = pd.read_csv(gold_file)

    subject_summary_path = os.path.join(data_dir, 'subject_summary.csv')
    
    df = gold_df.copy()

    # Convert HIGH_ENROLLMENT to numerical early, as it's the target.
    df['HIGH_ENROLLMENT'] = df['HIGH_ENROLLMENT'].map({'Y': 1, 'N': 0})

    target_col = 'HIGH_ENROLLMENT' 

    try:
        subject_summary_df_full = pd.read_csv(subject_summary_path)
        
        identifier_cols = ['TERM_CODE', 'SUBJECT_ID_SORT'] 
        
        df = pd.merge(df, subject_summary_df_full, on=identifier_cols, how='left')
        
        current_feature_cols = []

        existing_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        existing_numeric_cols = [col for col in existing_numeric_cols if col not in identifier_cols + [target_col]]
        current_feature_cols.extend(existing_numeric_cols)

        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        current_feature_cols.append('TERM_CODE_ENCODED')

        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        current_feature_cols.append('SUBJECT_ID_SORT_ENCODED')

        summary_cols_added = [col for col in subject_summary_df_full.columns if col not in identifier_cols]
        numeric_summary_cols = df[summary_cols_added].select_dtypes(include=np.number).columns.tolist()
        
        # --- Advanced Feature Engineering ---
        poly_features_to_square = []
        numeric_for_interaction_1 = None
        numeric_for_interaction_2 = None

        if use_proxy_for_feature_selection:
            features_for_proxy_model = [col for col in current_feature_cols if col != target_col and col in df.columns]

            if target_col in df.columns and len(features_for_proxy_model) > 0:
                X_proxy = df[features_for_proxy_model].fillna(0) # Fill NaNs for proxy model training
                y_proxy = df[target_col]

                # Only train proxy if target has variance
                if len(y_proxy.unique()) > 1:
                    proxy_model = lgb.LGBMRegressor(objective='regression_l1',
                                                    n_estimators=100,
                                                    learning_rate=0.1,
                                                    max_depth=5,
                                                    num_leaves=31,
                                                    min_child_samples=20,
                                                    subsample=0.8,
                                                    colsample_bytree=0.8,
                                                    random_state=42,
                                                    n_jobs=-1)

                    proxy_model.fit(X_proxy, y_proxy)

                    feature_importances = pd.Series(proxy_model.feature_importances_, index=X_proxy.columns)

                    TOP_N_IMPORTANT_NUMERICAL_FEATURES = 5
                    
                    important_numeric_features = feature_importances[
                        feature_importances.index.isin(numeric_summary_cols)
                    ].sort_values(ascending=False)

                    top_selected_numeric_features = important_numeric_features.head(TOP_N_IMPORTANT_NUMERICAL_FEATURES).index.tolist()

                    poly_features_to_square = [col for col in top_selected_numeric_features[:3] if col in df.columns]

                    if len(top_selected_numeric_features) >= 1:
                        numeric_for_interaction_1 = top_selected_numeric_features[0]
                    elif len(numeric_summary_cols) >= 1:
                        numeric_for_interaction_1 = numeric_summary_cols[0]

                    if len(top_selected_numeric_features) >= 2:
                        numeric_for_interaction_2 = top_selected_numeric_features[1]
                    elif len(numeric_summary_cols) >= 2:
                        numeric_for_interaction_2 = numeric_summary_cols[1]
                    elif len(numeric_summary_cols) == 1:
                        numeric_for_interaction_2 = numeric_summary_cols[0]
                else: # Fallback if target has no variance
                    poly_features_to_square = numeric_summary_cols[:3]
                    if len(numeric_summary_cols) >= 1: numeric_for_interaction_1 = numeric_summary_cols[0]
                    if len(numeric_summary_cols) >= 2: numeric_for_interaction_2 = numeric_summary_cols[1]
                    elif len(numeric_summary_cols) == 1: numeric_for_interaction_2 = numeric_summary_cols[0]
            else: # Fallback if no features for proxy model
                poly_features_to_square = numeric_summary_cols[:3]
                if len(numeric_summary_cols) >= 1: numeric_for_interaction_1 = numeric_summary_cols[0]
                if len(numeric_summary_cols) >= 2: numeric_for_interaction_2 = numeric_summary_cols[1]
                elif len(numeric_summary_cols) == 1: numeric_for_interaction_2 = numeric_summary_cols[0]

        else: # Fallback to original advanced feature engineering logic without proxy model
            poly_features_to_square = numeric_summary_cols[:3]
            if len(numeric_summary_cols) >= 1:
                numeric_for_interaction_1 = numeric_summary_cols[0] 
            if len(numeric_summary_cols) >= 2:
                numeric_for_interaction_2 = numeric_summary_cols[1] 
            elif len(numeric_summary_cols) == 1:
                numeric_for_interaction_2 = numeric_summary_cols[0] 

        # 3. Polynomial Features (e.g., squared terms)
        for col in poly_features_to_square:
            new_col_name = f"{col}_SQUARED"
            df[new_col_name] = df[col] ** 2
            current_feature_cols.append(new_col_name)
        
        # 4. Interaction Features (product of two distinct features)
        if 'TERM_CODE_ENCODED' in current_feature_cols and numeric_for_interaction_1 is not None and \
           numeric_for_interaction_1 in df.columns:
            col_to_interact = numeric_for_interaction_1 
            new_col_name = f"{col_to_interact}_x_TERM_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['TERM_CODE_ENCODED']
            current_feature_cols.append(new_col_name)

        if 'SUBJECT_ID_SORT_ENCODED' in current_feature_cols and numeric_for_interaction_2 is not None and \
           numeric_for_interaction_2 in df.columns:
            col_to_interact = numeric_for_interaction_2 
            new_col_name = f"{col_to_interact}_x_SUBJECT_ENCODED"
            df[new_col_name] = df[col_to_interact] * df['SUBJECT_ID_SORT_ENCODED']
            current_feature_cols.append(new_col_name)

        if numeric_for_interaction_1 is not None and numeric_for_interaction_2 is not None and \
           numeric_for_interaction_1 != numeric_for_interaction_2 and \
           numeric_for_interaction_1 in df.columns and numeric_for_interaction_2 in df.columns:
            col1 = numeric_for_interaction_1
            col2 = numeric_for_interaction_2
            new_col_name = f"{col1}_x_{col2}"
            df[new_col_name] = df[col1] * df[col2]
            current_feature_cols.append(new_col_name)

        # Final list of feature columns, ensuring no duplicates
        final_feature_cols = list(set(current_feature_cols))
        final_feature_cols = [col for col in final_feature_cols if col in df.columns]

        # Fallback: If no numeric features are found, add a dummy feature.
        if not final_feature_cols:
            warnings.warn("No numeric features found after merging with subject_summary.csv and feature engineering. Adding a dummy feature.")
            df['DUMMY_FEATURE'] = 0 
            final_feature_cols = ['DUMMY_FEATURE']

        # Fill any NaNs that might result from the left merge or missing values.
        df[final_feature_cols] = df[final_feature_cols].fillna(0)
        
        df._feature_cols = final_feature_cols

    except FileNotFoundError:
        # Fallback to minimal features if subject_summary.csv is not found
        print(f"Warning: {subject_summary_path} not found. Using minimal features from gold_df.")
        
        le_term = LabelEncoder()
        df['TERM_CODE_ENCODED'] = le_term.fit_transform(df['TERM_CODE'])
        
        le_subject = LabelEncoder()
        df['SUBJECT_ID_SORT_ENCODED'] = le_subject.fit_transform(df['SUBJECT_ID_SORT'])
        
        df['DUMMY_FEATURE'] = df['TERM_CODE_ENCODED'] % 5 + df['SUBJECT_ID_SORT_ENCODED'] % 7
        
        df._feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED', 'DUMMY_FEATURE']
        df[df._feature_cols] = df[df._feature_cols].fillna(0)

    return df

# --- Generic function to run a single ablation scenario ---
def run_ablation_scenario(
    load_data_func,
    model_class,
    model_params,
    scenario_name,
    data_dir=TRAIN_DATA_DIR,
    gold_file=GOLD_ENROLLMENT_TRAIN_FILE
):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    df = load_data_func(data_dir, gold_file)

    if df.empty:
        print(f"{scenario_name}: Loaded DataFrame is empty. Cannot proceed with training.")
        print(f"{scenario_name}: Final Validation Performance: 0.0")
        return 0.0

    df['TERM_CODE'] = df['TERM_CODE'].astype(str)
    df = df.sort_values('TERM_CODE').reset_index(drop=True)

    unique_terms = df['TERM_CODE'].unique()
    
    if len(unique_terms) < 2:
        print(f"{scenario_name}: Not enough unique terms for a time-based validation split (at least 2 required).")
        print(f"{scenario_name}: Final Validation Performance: 0.0")
        return 0.0

    validation_term = unique_terms[-1] 
    
    train_df = df[df['TERM_CODE'] != validation_term]
    val_df = df[df['TERM_CODE'] == validation_term]

    feature_cols = getattr(df, '_feature_cols', [])
    
    if not feature_cols: 
        warnings.warn(f"{scenario_name}: Feature columns not correctly identified by load_data. Attempting dynamic identification as fallback.")
        candidate_cols = [col for col in df.columns if col not in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']]
        feature_cols = df[candidate_cols].select_dtypes(include=np.number).columns.tolist()
        if not feature_cols: 
            if 'TERM_CODE_ENCODED' in df.columns and 'SUBJECT_ID_SORT_ENCODED' in df.columns:
                feature_cols = ['TERM_CODE_ENCODED', 'SUBJECT_ID_SORT_ENCODED']
            else:
                df['DUMMY_FEATURE'] = 0
                feature_cols = ['DUMMY_FEATURE']

    if not feature_cols:
        print(f"{scenario_name}: No usable features available for training after all fallback attempts.")
        print(f"{scenario_name}: Final Validation Performance: 0.0")
        return 0.0

    X_train = train_df[feature_cols]
    y_train = train_df['HIGH_ENROLLMENT']
    X_val = val_df[feature_cols]
    y_val = val_df['HIGH_ENROLLMENT']

    if X_train.empty or y_train.empty:
        print(f"{scenario_name}: Training set is empty. Cannot train a model.")
        print(f"{scenario_name}: Final Validation Performance: 0.0")
        return 0.0

    if len(y_train.unique()) < 2:
        print(f"{scenario_name}: Training set target 'HIGH_ENROLLMENT' has only one class: {y_train.unique()}. This might lead to trivial predictions.")
    
    # Model Training
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    final_validation_score = 0.0
    
    if y_val.empty:
        print(f"{scenario_name}: Validation set is empty. F1 score cannot be calculated.")
    else:
        try:
            y_pred = model.predict(X_val)
        except Exception as e:
            print(f"{scenario_name}: Error during prediction: {e}. Defaulting F1 to 0.0.")
            return 0.0

        unique_y_val = np.unique(y_val)
        unique_y_pred = np.unique(y_pred)

        if len(unique_y_val) < 2:
            if len(unique_y_pred) == 1 and unique_y_pred[0] == unique_y_val[0]:
                final_validation_score = 1.0
            else:
                final_validation_score = 0.0
        else:
            final_validation_score = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"{scenario_name}: Final Validation Performance: {final_validation_score:.4f}")
    return final_validation_score

# --- Main Ablation Study Execution ---
if __name__ == "__main__":
    results = {}

    # Define wrappers for load_data_ablation to control the proxy feature selection
    def load_data_with_proxy_fe(*args, **kwargs):
        return load_data_ablation(*args, use_proxy_for_feature_selection=True, **kwargs)

    def load_data_without_proxy_fe(*args, **kwargs):
        return load_data_ablation(*args, use_proxy_for_feature_selection=False, **kwargs)

    # Baseline Scenario: RandomForest with Proxy Model for Feature Selection
    results['Baseline (RF, Proxy FE)'] = run_ablation_scenario(
        load_data_func=load_data_with_proxy_fe,
        model_class=RandomForestClassifier,
        model_params={'random_state': 42, 'n_estimators': 100},
        scenario_name="Baseline (RandomForest, with Proxy Feature Selection)"
    )

    # Ablation 1: Disable Proxy Model for Feature Selection
    # The advanced feature engineering will revert to selecting features by simpler heuristics (e.g., first few columns).
    results['Ablation 1 (RF, No Proxy FE)'] = run_ablation_scenario(
        load_data_func=load_data_without_proxy_fe,
        model_class=RandomForestClassifier,
        model_params={'random_state': 42, 'n_estimators': 100},
        scenario_name="Ablation 1 (RandomForest, NO Proxy Feature Selection)"
    )

    # Ablation 2: Change Model to Logistic Regression
    # Tests if a simpler, linear model can achieve similar performance, retaining the proxy FE.
    results['Ablation 2 (Logistic Regression, Proxy FE)'] = run_ablation_scenario(
        load_data_func=load_data_with_proxy_fe,
        model_class=LogisticRegression,
        model_params={'random_state': 42, 'solver': 'liblinear'}, # 'liblinear' solver recommended for small datasets or L1/L2 penalties.
        scenario_name="Ablation 2 (Logistic Regression, with Proxy Feature Selection)"
    )

    # Ablation 3: Random Forest with reduced max_features
    # Tests the impact of feature sub-sampling during tree construction.
    results['Ablation 3 (RF, max_features=0.3, Proxy FE)'] = run_ablation_scenario(
        load_data_func=load_data_with_proxy_fe,
        model_class=RandomForestClassifier,
        model_params={'random_state': 42, 'n_estimators': 100, 'max_features': 0.3},
        scenario_name="Ablation 3 (RandomForest, max_features=0.3, with Proxy Feature Selection)"
    )

    print("\n--- Ablation Study Results Summary ---")
    for scenario, score in results.items():
        print(f"{scenario}: F1 Score = {score:.4f}")

    baseline_score = results['Baseline (RF, Proxy FE)']
    
    if all(score == 0.0 for score in results.values()):
        print("\nAll F1 scores are 0.0. This suggests a problem with the data or the evaluation setup, making it impossible to determine relative impact.")
        print("It's highly likely that the validation set is always empty, or contains only one class that is never predicted correctly.")
    elif all(score == 1.0 for score in results.values()):
        print("\nAll F1 scores are 1.0. This suggests the dataset is too simple or the task is trivially easy for the model, making it impossible to determine relative impact.")
    else:
        impacts = {}
        for scenario, score in results.items():
            if scenario == 'Baseline (RF, Proxy FE)':
                continue
            impacts[scenario] = baseline_score - score
        
        if not impacts or max(impacts.values()) <= 0:
            print("\nNo ablation led to a performance drop, or all ablations performed equally well or better than the baseline.")
            print("This could mean the ablated parts are not critical, or the current data does not expose their impact.")
        else:
            most_impactful_scenario = max(impacts, key=impacts.get)
            largest_drop = impacts[most_impactful_scenario]
            print(f"\nThe part that contributes the most to the overall performance is likely related to: '{most_impactful_scenario}', as its ablation led to the largest performance drop of {largest_drop:.4f}.")

