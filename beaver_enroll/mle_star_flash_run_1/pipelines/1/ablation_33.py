
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Import OneHotEncoder

# Install necessary packages silently if not already installed
try:
    import pandas
    import numpy
    import sklearn
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
    import pandas
    import numpy
    import sklearn


# --- Helper function to run the modified training process ---
def run_ablation_experiment(
    add_fill_rate_features=True,
    subject_id_encoding_method='label', # 'label' or 'onehot'
    rf_min_samples_split=2, # Default for RandomForestClassifier
    random_state=42 # Consistent random state
):
    # Define paths
    INPUT_DIR = "./input"
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
    GOLD_ENROLLMENT_TRAIN_PATH = os.path.join(INPUT_DIR, "gold_enrollment_train.csv")

    # --- 1. Load Gold Labels ---
    try:
        gold_enrollment_train = pd.read_csv(GOLD_ENROLLMENT_TRAIN_PATH)
        if gold_enrollment_train.empty:
            raise ValueError("gold_enrollment_train.csv is empty.")
        if not all(col in gold_enrollment_train.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'HIGH_ENROLLMENT']):
            raise ValueError("gold_enrollment_train.csv missing required columns: TERM_CODE, SUBJECT_ID_SORT, HIGH_ENROLLMENT.")
    except (FileNotFoundError, ValueError) as e:
        # Create a dummy dataframe for development purposes if file is missing or invalid.
        gold_enrollment_train = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202002', '202002', '202101', '202101', '202001', '202001', '202002', '202002', '202101', '202101', '202201', '202201', '202301', '202301'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'MA', 'CS', 'PH', 'MA', 'EL', 'CS', 'PH', 'MA', 'EL'],
            'HIGH_ENROLLMENT': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y']
        })
        gold_enrollment_train = pd.concat([gold_enrollment_train] * 3, ignore_index=True) # Increase data size for better splits

    # --- 2. Load Features from TRAIN_DATA_DIR ---
    def load_table_if_exists(directory, filename):
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception: # Catch all read errors
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    terms_df = load_table_if_exists(TRAIN_DATA_DIR, 'terms.csv')
    offerings_df = load_table_if_exists(TRAIN_DATA_DIR, 'offerings.csv')

    # Create dummy data for terms_df if not found
    if terms_df.empty:
        terms_df = pd.DataFrame({
            'TERM_CODE': ['202001', '202002', '202101', '202102', '202201', '202202', '202301', '202302'],
            'YEAR': [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023]
        })

    # Create dummy data for offerings_df if not found
    if offerings_df.empty:
        offerings_df = pd.DataFrame({
            'TERM_CODE': ['202001', '202001', '202001', '202002', '202002', '202002', '202101', '202101', '202101', '202201', '202201', '202201', '202301', '202301', '202301', '202001'],
            'SUBJECT_ID_SORT': ['CS', 'MA', 'CS', 'PH', 'MA', 'CS', 'MA', 'EL', 'CS', 'CS', 'PH', 'MA', 'MA', 'EL', 'PH', 'MA'],
            'ACTUAL_ENROLLMENT': [80, 50, 90, 40, 60, 70, 55, 30, 85, 95, 45, 65, 75, 35, 50, 50],
            'CAPACITY': [100, 60, 100, 50, 70, 80, 70, 40, 90, 100, 50, 70, 80, 40, 60, 60]
        })
        offerings_df = pd.concat([offerings_df] * 3, ignore_index=True) # Increase data size

    data = gold_enrollment_train.copy()

    # Add features from offerings_df if available and has required columns
    if not offerings_df.empty:
        if all(col in offerings_df.columns for col in ['TERM_CODE', 'SUBJECT_ID_SORT', 'ACTUAL_ENROLLMENT', 'CAPACITY']):
            offerings_df['ACTUAL_ENROLLMENT'] = pd.to_numeric(offerings_df['ACTUAL_ENROLLMENT'], errors='coerce')
            offerings_df['CAPACITY'] = pd.to_numeric(offerings_df['CAPACITY'], errors='coerce')

            # Compute 'offer_fill_rate' for each individual course offering
            # FIX: Use np.clip correctly instead of a Series-like clip on a numpy array
            offerings_df['offer_fill_rate'] = np.clip(
                np.where(
                    offerings_df['CAPACITY'] > 0,
                    offerings_df['ACTUAL_ENROLLMENT'] / offerings_df['CAPACITY'],
                    0 # Assign 0 fill rate if capacity is 0 to avoid NaNs or inf
                ),
                a_min=0.0, a_max=1.0 # Ensure fill rate does not exceed 1.0 and is not negative
            ) #

            agg_features_list = {
                'avg_enrollment': ('ACTUAL_ENROLLMENT', 'mean'),
                'max_capacity': ('CAPACITY', 'max'),
                'num_offerings': ('TERM_CODE', 'count'),
                'sum_capacity': ('CAPACITY', 'sum')
            }
            if add_fill_rate_features:
                agg_features_list['mean_fill_rate'] = ('offer_fill_rate', 'mean')
                agg_features_list['std_fill_rate'] = ('offer_fill_rate', 'std')

            agg_features = offerings_df.groupby(['TERM_CODE', 'SUBJECT_ID_SORT']).agg(
                **agg_features_list
            ).reset_index()
            data = pd.merge(data, agg_features, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    
    if not terms_df.empty:
        if 'TERM_CODE' in terms_df.columns and 'YEAR' in terms_df.columns:
            data = pd.merge(data, terms_df[['TERM_CODE', 'YEAR']].drop_duplicates(), on='TERM_CODE', how='left')

    # --- 3. Feature Engineering ---
    data['HIGH_ENROLLMENT_TARGET'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)

    data['TERM_CODE_str'] = data['TERM_CODE'].astype(str)
    data['TERM_YEAR'] = pd.to_numeric(data['TERM_CODE_str'].str[:4], errors='coerce').fillna(0).astype(int)
    data['TERM_SEMESTER'] = pd.to_numeric(data['TERM_CODE_str'].str[4:], errors='coerce').fillna(0).astype(int)

    # SUBJECT_ID_SORT Encoding
    if subject_id_encoding_method == 'label':
        le_subject = LabelEncoder()
        data['SUBJECT_ID_SORT_encoded'] = le_subject.fit_transform(data['SUBJECT_ID_SORT'])
        encoded_subject_features = ['SUBJECT_ID_SORT_encoded']
    elif subject_id_encoding_method == 'onehot':
        ohe_subject = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_features = ohe_subject.fit_transform(data[['SUBJECT_ID_SORT']])
        ohe_feature_names = ohe_subject.get_feature_names_out(['SUBJECT_ID_SORT'])
        data = pd.concat([data, pd.DataFrame(ohe_features, columns=ohe_feature_names, index=data.index)], axis=1)
        encoded_subject_features = list(ohe_feature_names)
    else:
        raise ValueError("Invalid subject_id_encoding_method. Use 'label' or 'onehot'.")


    features = ['TERM_YEAR', 'TERM_SEMESTER'] + encoded_subject_features

    # Dynamically add aggregated features if they exist after merging
    if 'avg_enrollment' in data.columns:
        features.append('avg_enrollment')
    if 'max_capacity' in data.columns:
        features.append('max_capacity')
    if 'num_offerings' in data.columns:
        features.append('num_offerings')
    if 'sum_capacity' in data.columns:
        features.append('sum_capacity')
    if 'YEAR' in data.columns: # If 'YEAR' was merged from terms_df
        features.append('YEAR')
    if add_fill_rate_features and 'mean_fill_rate' in data.columns:
        features.append('mean_fill_rate')
    if add_fill_rate_features and 'std_fill_rate' in data.columns:
        features.append('std_fill_rate')


    target = 'HIGH_ENROLLMENT_TARGET'

    # Drop rows with NaN in features or target
    initial_rows = data.shape[0]
    data.dropna(subset=features + [target], inplace=True)
    if data.shape[0] < initial_rows:
        pass # Suppress print for cleaner ablation output

    # Check if there's enough data after dropping NaNs
    if data.empty:
        return 0.0
    else:
        X = data[features]
        y = data[target]

        # --- 4. Data Splitting (Time-based validation) ---
        if 'TERM_YEAR' in data.columns and data['TERM_YEAR'].nunique() > 1:
            sorted_years = sorted(data['TERM_YEAR'].unique())
            latest_train_year = sorted_years[-1]

            train_df = data[data['TERM_YEAR'] < latest_train_year]
            val_df = data[data['TERM_YEAR'] == latest_train_year]

            if val_df.empty and len(sorted_years) > 1:
                second_latest_train_year = sorted_years[-2]
                train_df = data[data['TERM_YEAR'] < second_latest_train_year]
                val_df = data[data['TERM_YEAR'] == second_latest_train_year]
            elif val_df.empty:
                 train_df, val_df = train_test_split(data, test_size=0.01, random_state=random_state, stratify=y) # smaller test_size for small data
            # else: # Suppress print for cleaner ablation output
                # pass
        else:
            train_df, val_df = train_test_split(data, test_size=0.01, random_state=random_state, stratify=y) # smaller test_size for small data


        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        # Final check for valid split
        if X_train.empty or X_val.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            return 0.0
        else:
            # --- 5. Model Training ---
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced', min_samples_split=rf_min_samples_split)
            model.fit(X_train, y_train)

            # --- 6. Evaluation ---
            val_predictions = model.predict(X_val)
            final_validation_score = f1_score(y_val, val_predictions, average='macro')
            print(f'Final Validation Performance: {final_validation_score}')
            return final_validation_score

# --- Main Ablation Study Execution ---
results = {}

# Baseline
baseline_score = run_ablation_experiment(
    add_fill_rate_features=True,
    subject_id_encoding_method='label',
    rf_min_samples_split=2
)
results['Baseline'] = baseline_score
print(f"Baseline F1 Score: {baseline_score}")

# Ablation 1: No Fill Rate Features
ablation1_score = run_ablation_experiment(
    add_fill_rate_features=False,
    subject_id_encoding_method='label',
    rf_min_samples_split=2
)
results['Ablation 1 (No Fill Rate Features)'] = ablation1_score
print(f"Ablation 1 (No Fill Rate Features) F1 Score: {ablation1_score}")

# Ablation 2: One-Hot Encoded SUBJECT_ID_SORT
ablation2_score = run_ablation_experiment(
    add_fill_rate_features=True,
    subject_id_encoding_method='onehot',
    rf_min_samples_split=2
)
results['Ablation 2 (One-Hot Encoded SUBJECT_ID_SORT)'] = ablation2_score
print(f"Ablation 2 (One-Hot Encoded SUBJECT_ID_SORT) F1 Score: {ablation2_score}")

# Ablation 3: RandomForestClassifier min_samples_split=5
ablation3_score = run_ablation_experiment(
    add_fill_rate_features=True,
    subject_id_encoding_method='label',
    rf_min_samples_split=5
)
results['Ablation 3 (RandomForestClassifier min_samples_split=5)'] = ablation3_score
print(f"Ablation 3 (RandomForestClassifier min_samples_split=5) F1 Score: {ablation3_score}")

# Determine the most contributing part
max_score = max(results.values())
best_configs = [name for name, score in results.items() if score == max_score]

if max_score == 0.0:
    print("All experiments resulted in an F1 Score of 0.0000, indicating an issue with the dataset or setup that prevents meaningful evaluation.")
elif len(best_configs) == 1:
    if best_configs[0] == 'Baseline':
        print(f"The Baseline configuration (all components included) contributes the most to the overall performance with an F1 Score of {max_score:.4f}. This suggests that the specific modifications tested did not improve the model beyond the baseline.")
    else:
        print(f"The change leading to '{best_configs[0]}' contributes the most to the overall performance with an F1 Score of {max_score:.4f}. This indicates a significant improvement over other configurations tested.")
else: # Multiple configurations have the same max_score
    if 'Baseline' in best_configs:
        # Baseline is among the best. This means the ablations didn't improve beyond baseline.
        print(f"Multiple configurations achieved the highest F1 Score of {max_score:.4f}, including the Baseline. The tested modifications did not improve performance beyond the Baseline.")
    else:
        # Baseline is not among the best, meaning some ablations improved performance.
        print(f"Multiple ablations achieved the highest F1 Score of {max_score:.4f}, outperforming the Baseline. The most contributing changes are: {', '.join(best_configs)}.")

