
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    import subprocess
    import sys
    print("Installing required packages: pandas, numpy, scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    print("Packages installed successfully.")
    # Re-import after installation
    import pandas
    import numpy
    import sklearn


# Define paths
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

def run_experiment(
    use_enrollment_rate_aggs=True,
    use_historical_trends=True,
    rf_min_samples_leaf=1 # Default RandomForestClassifier setting
):
    print(f"\n--- Running Experiment: "
          f"EnrollmentRateAggs={use_enrollment_rate_aggs}, "
          f"HistoricalTrends={use_historical_trends}, "
          f"RF_min_samples_leaf={rf_min_samples_leaf} ---")

    # --- 1. Load Gold Labels ---
    gold_enrollment_train = pd.DataFrame()
    
    # Generate robust dummy data for testing if files are not found or are problematic
    gold_enrollment_train_dummy = pd.DataFrame({
        'TERM_CODE': [202001, 202001, 202002, 202002, 202101, 202101, 202102, 202102, 202201, 202201, 202202, 202202, 202301, 202301, 202302, 202302,
                      202001, 202001, 202002, 202002, 202101, 202101, 202102, 202102, 202201, 202201, 202202, 202202, 202301, 202301, 202302, 202302],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH',
                            'HIST', 'MATH', 'HIST', 'CHEM', 'MATH', 'BIO', 'HIST', 'CHEM', 'MATH', 'BIO', 'HIST', 'CHEM', 'MATH', 'BIO', 'HIST', 'CHEM'],
        'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N',
                            'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y']
    })
    offerings_df_dummy = pd.DataFrame({
        'TERM_CODE': [202001, 202001, 202002, 202002, 202101, 202101, 202102, 202102, 202201, 202201, 202202, 202202, 202301, 202301, 202302, 202302,
                      202001, 202001, 202002, 202002, 202101, 202101, 202102, 202102, 202201, 202201, 202202, 202202, 202301, 202301, 202302, 202302,
                      202001, 202001, 202002, 202002, 202101, 202101, 202102, 202102, 202201, 202201, 202202, 202202, 202301, 202301, 202302, 202302],
        'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH',
                            'HIST', 'MATH', 'HIST', 'CHEM', 'MATH', 'BIO', 'HIST', 'CHEM', 'MATH', 'BIO', 'HIST', 'CHEM', 'MATH', 'BIO', 'HIST', 'CHEM',
                            'CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH'], # More entries for different subjects
        'ACTUAL_ENROLLMENT': [50, 40, 60, 30, 45, 35, 55, 25, 48, 38, 58, 28, 52, 42, 62, 32,
                              20, 70, 25, 65, 22, 68, 28, 62, 24, 66, 30, 60, 26, 64, 32, 58,
                              55, 45, 65, 35, 50, 40, 60, 30, 53, 43, 63, 33, 57, 47, 67, 37],
        'CAPACITY': [60, 50, 70, 40, 55, 45, 65, 35, 58, 48, 68, 38, 62, 52, 72, 42,
                     30, 80, 35, 75, 32, 78, 38, 72, 34, 76, 40, 70, 36, 74, 42, 68,
                     65, 55, 75, 45, 60, 50, 70, 40, 63, 53, 73, 43, 67, 57, 77, 47]
    })
    terms_df_dummy = pd.DataFrame({
        'TERM_CODE': [202001, 202002, 202101, 202102, 202201, 202202, 202301, 202302],
        'YEAR': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023]
    })

    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading gold_enrollment_train.csv: {e}. Using dummy gold_enrollment_train data.")
        gold_enrollment_train = gold_enrollment_train_dummy

    # Helper function to load a table if it exists - modified to use dummy data if paths aren't valid
    def load_table_if_exists_abl(directory, filename, dummy_df=None):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}. Returning dummy DataFrame if available.")
                return dummy_df if dummy_df is not None else pd.DataFrame()
        else:
            print(f"Warning: {filename} not found at {filepath}. Using dummy data if provided.")
            return dummy_df if dummy_df is not None else pd.DataFrame()

    # Load potential feature tables
    terms_df = load_table_if_exists_abl(TRAIN_DATA_DIR, 'terms.csv', terms_df_dummy)
    offerings_df = load_table_if_exists_abl(TRAIN_DATA_DIR, 'offerings.csv', offerings_df_dummy)

    # Create a base dataframe for merging features, starting with gold labels
    data = gold_enrollment_train.copy()

    # Add features from offerings_df if available and has required columns
    if not offerings_df.empty:
        if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
            offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
            offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

            # Calculate enrollment_rate at the individual offering level
            offerings_df['enrollment_rate'] = offerings_df['ACTUAL_ENROLLMENT'] / offerings_df['CAPACITY'].replace(0, np.nan)

            agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                avg_enrollment=('ACTUAL_ENROLLMENT', 'mean'),
                max_capacity=('CAPACITY', 'max'),
                num_offerings=('TERM_CODE', 'count'),
                sum_capacity=('CAPACITY', 'sum')
            ).reset_index()

            # Conditionally add enrollment rate aggregations
            if use_enrollment_rate_aggs:
                rate_aggs = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                    avg_enrollment_rate=('enrollment_rate', 'mean'),
                    max_enrollment_rate=('enrollment_rate', 'max'),
                    min_enrollment_rate=('enrollment_rate', 'min')
                ).reset_index()
                agg_features = pd.merge(agg_features, rate_aggs, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

            # Conditionally add historical trend features
            if use_historical_trends:
                temp_agg_features_for_hist = agg_features.copy()
                temp_agg_features_for_hist['TERM_CODE_numeric'] = pd.to_numeric(temp_agg_features_for_hist['TERM_CODE'])
                temp_agg_features_for_hist = temp_agg_features_for_hist.sort_values(by=['SUBJECT_ID_SORT', 'TERM_CODE_numeric'])
                temp_agg_features_for_hist['avg_enrollment_pct_change'] = temp_agg_features_for_hist.groupby('SUBJECT_ID_SORT')['avg_enrollment'].pct_change()
                temp_agg_features_for_hist['max_capacity_pct_change'] = temp_agg_features_for_hist.groupby('SUBJECT_ID_SORT')['max_capacity'].pct_change()
                agg_features = pd.merge(agg_features, temp_agg_features_for_hist[['TERM_CODE', 'SUBJECT_ID_SORT', 'avg_enrollment_pct_change', 'max_capacity_pct_change']],
                                        on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')

            data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
        else:
            print("Warning: offerings_df missing expected columns for aggregation (ACTUAL_ENROLLMENT, CAPACITY). Skipping merge.")
    else:
        print("Warning: offerings_df is empty. Proceeding with limited features.")

    # Add features from terms_df if available and has required columns
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')
        else:
            print("Warning: terms_df missing expected columns (YEAR). Skipping merge.")

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    le_subject = LabelEncoder()
    data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])

    features = ['TERM_YEAR', 'TERM_SEMESTER', 'SUBJECT_ID_SORT_encoded']

    if 'avg_enrollment' in data.columns: features.append('avg_enrollment')
    if 'max_capacity' in data.columns: features.append('max_capacity')
    if 'num_offerings' in data.columns: features.append('num_offerings')
    if 'sum_capacity' in data.columns: features.append('sum_capacity')
    if 'YEAR' in data.columns: features.append('YEAR')

    # Dynamically add ablated features
    if use_enrollment_rate_aggs:
        if 'avg_enrollment_rate' in data.columns: features.append('avg_enrollment_rate')
        if 'max_enrollment_rate' in data.columns: features.append('max_enrollment_rate')
        if 'min_enrollment_rate' in data.columns: features.append('min_enrollment_rate')
    if use_historical_trends:
        if 'avg_enrollment_pct_change' in data.columns: features.append('avg_enrollment_pct_change')
        if 'max_capacity_pct_change' in data.columns: features.append('max_capacity_pct_change')

    target = 'HIGH_ENROLLMENT_TARGET'

    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        pass # Optional: print(f"Dropped {initial_rows - data.shape[0]} rows due to NaN in features or target.")

    if data.empty or len(data[target].unique()) < 2:
        print("Error: No data remaining or target has only one class after feature engineering and NaN removal. Cannot train model.")
        return 0.0
    else:
        X = data[features]
        y = data[target]

        # --- 4. Data Splitting (Time-based validation) ---
        train_df, val_df = pd.DataFrame(), pd.DataFrame()
        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_train_year = sorted_years[-1]

            train_df = data[data['TERM_YEAR'] < latest_train_year]
            val_df = data[data['TERM_YEAR'] == latest_train_year]

            if val_df.empty and len(sorted_years) > 1:
                second_latest_train_year = sorted_years[-2]
                train_df = data[data['TERM_YEAR'] < second_latest_train_year]
                val_df = data[data['TERM_YEAR'] == second_latest_train_year]
            elif val_df.empty: # Only one or two years and time-based split created empty validation
                train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
        else: # 'TERM_YEAR' not available or only one year of data
            train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)

        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print("Error: Training or validation set is empty, or target has only one class. Cannot proceed with model training.")
            return 0.0
        else:
            # --- 5. Model Training ---
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', min_samples_leaf=rf_min_samples_leaf)
            model.fit(X_train, y_train)

            # --- 6. Evaluation ---
            val_predictions = model.predict(X_val)
            final_validation_score = f1_score(y_val, val_predictions, average='macro')
            return final_validation_score

# --- Ablation Study Execution ---
results = {}

# Baseline
baseline_score = run_experiment()
results['Baseline'] = baseline_score

# Ablation 1: No Enrollment Rate Aggregations
ablation1_score = run_experiment(use_enrollment_rate_aggs=False)
results['No Enrollment Rate Aggregations'] = ablation1_score

# Ablation 2: No Historical Trend Features
ablation2_score = run_experiment(use_historical_trends=False)
results['No Historical Trend Features'] = ablation2_score

# Ablation 3: RandomForest with min_samples_leaf=5
ablation3_score = run_experiment(rf_min_samples_leaf=5)
results['RF with min_samples_leaf=5'] = ablation3_score

print("\n--- Ablation Study Results ---")
for experiment, score in results.items():
    print(f"{experiment}: Macro F1 Score = {score:.4f}")

# Determine the most contributing part
if baseline_score == 0.0 and all(s == 0.0 for s in results.values()):
    print("\nConclusion: All experiments, including the baseline, resulted in a Macro F1 Score of 0.0. This indicates a potential issue with the dataset or setup (e.g., insufficient dummy data, single-class validation set) preventing meaningful model training and evaluation. The results are inconclusive.")
else:
    most_impactful_change_desc = "None of the ablated components had a significant impact on performance."
    max_impact_diff = 0 # Store absolute difference for "most impactful"

    # Identify the configuration with the best performance (could be baseline or an ablation)
    best_score_overall = baseline_score
    best_config_name = 'Baseline'
    for config, score in results.items():
        if score > best_score_overall:
            best_score_overall = score
            best_config_name = config

    if best_config_name == 'Baseline':
        # If baseline is best or tied, look for components that, when removed, caused a significant drop
        impactful_positive_contributions = []
        for config, score in results.items():
            if config == 'Baseline':
                continue
            drop = baseline_score - score
            if drop > 0.01: # Consider a drop of more than 0.01 as significant
                impactful_positive_contributions.append(f"'{config}' (removal led to a drop of {drop:.4f})")
                if drop > max_impact_diff:
                    max_impact_diff = drop
                    most_impactful_change_desc = f"Keeping the component modified in '{config}' (its removal led to the largest drop of {drop:.4f})"
        
        if impactful_positive_contributions:
            print(f"\nConclusion: The 'Baseline' configuration (Macro F1 Score = {baseline_score:.4f}) yielded the best performance.")
            print(f"The most impactful positive contributions come from components whose removal caused a significant performance decrease: {', '.join(impactful_positive_contributions)}.")
        else:
            print(f"\nConclusion: The 'Baseline' configuration (Macro F1 Score = {baseline_score:.4f}) yielded the best performance. No single ablated component showed a significant positive or negative impact on performance compared to the baseline.")
    else:
        # An ablation improved the score, meaning the removed/modified component in baseline was detrimental
        improvement = best_score_overall - baseline_score
        if improvement > max_impact_diff:
            max_impact_diff = improvement
            most_impactful_change_desc = f"Removing/modifying the component in '{best_config_name}' (resulted in the largest improvement of {improvement:.4f})"
            
        print(f"\nConclusion: The most impactful change was '{best_config_name}' (Macro F1 Score = {best_score_overall:.4f}). This means the component that was removed or modified in this experiment was detrimental to the baseline's performance, or its modification led to an improvement.")
        if best_config_name == 'No Enrollment Rate Aggregations':
            print("Specifically, removing 'Enrollment Rate Aggregations' improved the model, suggesting they might be noisy or unhelpful in the baseline setup.")
        elif best_config_name == 'No Historical Trend Features':
            print("Specifically, removing 'Historical Trend Features' improved the model, suggesting they might be noisy or unhelpful in the baseline setup.")
        elif best_config_name == 'RF with min_samples_leaf=5':
            print("Specifically, increasing 'min_samples_leaf' to 5 improved the model, suggesting the default value (1) might lead to overfitting on this dataset.")

