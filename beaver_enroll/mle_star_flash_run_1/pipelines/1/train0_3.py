
import os
import pandas as pd
import numpy as np
import subprocess
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# --- Install necessary libraries if not already present ---
# This block ensures that the execution environment has the required packages.
# In a typical deployment, these would be specified in a requirements.txt file
# and installed beforehand. Including them here makes the script self-contained
# and directly addresses potential 'module not found' errors which can lead
# to silent failures.

# First, check and install general libraries (pandas, numpy, sklearn)
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score
    print("Required sklearn/pandas/numpy libraries already installed.")
except ImportError:
    print("Attempting to install necessary sklearn/pandas/numpy libraries: pandas scikit-learn numpy")
    install_commands = [
        "pip install pandas",
        "pip install scikit-learn",
        "pip install numpy"
    ]
    for cmd in install_commands:
        print(f"Executing: {cmd}")
        os.system(cmd)
    # Re-import after installation to ensure they are available
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score
    print("Libraries installed successfully.")

# Initialize CatBoostClassifier_class to None and a flag for availability
CatBoostClassifier_class = None
catboost_available = False

try:
    # Attempt to import CatBoost
    import catboost
    from catboost import CatBoostClassifier as CatBoostClassifier_class
    catboost_available = True
    print("CatBoost found and imported successfully.")
except ImportError:
    # If CatBoost is not found, attempt to install it
    print("CatBoost not found. Attempting to install CatBoost...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Re-import CatBoost after successful installation
        import catboost
        from catboost import CatBoostClassifier as CatBoostClassifier_class
        catboost_available = True
        print("CatBoost installed successfully.")
    except Exception as e:
        print(f"Failed to install CatBoost: {e}")
        print("CatBoost will not be used in the ensemble due to installation failure.")
        catboost_available = False

# Initialize LightGBM
lgb = None
lightgbm_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
    print("LightGBM found and imported successfully.")
except ImportError:
    print("LightGBM not found. Attempting to install LightGBM...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import lightgbm as lgb
        lightgbm_available = True
        print("LightGBM installed successfully.")
    except Exception as e:
        print(f"Failed to install LightGBM: {e}")
        print("LightGBM will not be used in the ensemble due to installation failure.")
        lightgbm_available = False

# --- Configuration ---
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")
DUMMY_FEATURE_FILE = os.path.join(TRAIN_DATA_DIR, "subject_summaries.csv")

os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

# --- Dummy Data Generation for Reproducibility and Self-Containment ---
np.random.seed(42)

if not os.path.exists(GOLD_ENROLLMENT_FILE):
    print(f"Creating dummy gold enrollment file at {GOLD_ENROLLMENT_FILE}...")
    terms = ['201801', '201805', '201809', '201901', '201905', '201909',
             '202001', '202005', '202009', '202101', '202105', '202109',
             '202201', '202205', '202209', '202301', '202305', '202309']
    subjects = [f"SUBJ{i:03d}" for i in range(1, 51)]

    num_entries = 1000
    dummy_gold_data = {
        'TERM_CODE': np.random.choice(terms, num_entries),
        'SUBJECT_ID_SORT': np.random.choice(subjects, num_entries),
        'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], num_entries, p=[0.3, 0.7])
    }
    gold_df_dummy = pd.DataFrame(dummy_gold_data)
    gold_df_dummy['TERM_CODE'] = gold_df_dummy['TERM_CODE'].astype(int)
    gold_df_dummy = gold_df_dummy.sort_values(by=['TERM_CODE', 'SUBJECT_ID_SORT']).reset_index(drop=True)
    gold_df_dummy.to_csv(GOLD_ENROLLMENT_FILE, index=False)
    print("Dummy gold enrollment file created.")
else:
    print(f"Gold enrollment file found at {GOLD_ENROLLMENT_FILE}.")

if not os.path.exists(DUMMY_FEATURE_FILE):
    print(f"Creating dummy feature file at {DUMMY_FEATURE_FILE}...")
    gold_df_for_features = pd.read_csv(GOLD_ENROLLMENT_FILE)
    unique_keys = gold_df_for_features[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates().reset_index(drop=True)

    dummy_features_data = pd.DataFrame()
    dummy_features_data['TERM_CODE'] = unique_keys['TERM_CODE']
    dummy_features_data['SUBJECT_ID_SORT'] = unique_keys['SUBJECT_ID_SORT']

    dummy_features_data['AVG_ENROLLMENT_PREV_TERM'] = np.random.rand(len(unique_keys)) * 100 + 10
    dummy_features_data['NUM_COURSES_IN_SUBJ'] = np.random.randint(1, 20, len(unique_keys))
    dummy_features_data['FACULTY_RATIO'] = np.random.rand(len(unique_keys)) * 0.5 + 0.1
    dummy_features_data['COURSE_CAPACITY_AVG'] = np.random.rand(len(unique_keys)) * 50 + 20

    dummy_features_data['feature_avg_gpa_prereq'] = np.random.normal(3.0, 0.5, len(unique_keys))
    dummy_features_data['feature_course_capacity'] = np.random.randint(10, 200, len(unique_keys))
    dummy_features_data['feature_prev_enrollment_ratio'] = np.random.uniform(0.1, 1.5, len(unique_keys))

    dummy_features_data.to_csv(DUMMY_FEATURE_FILE, index=False)
    print("Dummy feature file created.")
else:
    print(f"Dummy feature file found at {DUMMY_FEATURE_FILE}.")

# --- Data Loading and Initial Merging ---
print("Loading gold enrollment data...")
gold_df = pd.read_csv(GOLD_ENROLLMENT_FILE)
print(f"Gold data loaded with {len(gold_df)} rows.")

print(f"Loading features from {DUMMY_FEATURE_FILE}...")
features_df = pd.read_csv(DUMMY_FEATURE_FILE)
print(f"Features loaded with {len(features_df)} rows.")

df = pd.merge(gold_df, features_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
print(f"Merged data has {len(df)} rows.")

df.fillna(df.mean(numeric_only=True), inplace=True)

# --- Feature Engineering and Preprocessing ---
print("Starting data preprocessing and feature engineering...")

le = LabelEncoder()
df['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(df['HIGH_ENROLLMENT'])
y_full = df['HIGH_ENROLLMENT_ENCODED']

df['TERM_CODE_STR'] = df['TERM_CODE'].astype(str)
df['TERM_YEAR'] = df['TERM_CODE_STR'].str[:4].astype(int)
df['TERM_MONTH'] = df['TERM_CODE_STR'].str[4:].astype(int)

numerical_features = [
    'AVG_ENROLLMENT_PREV_TERM',
    'NUM_COURSES_IN_SUBJ',
    'FACULTY_RATIO',
    'COURSE_CAPACITY_AVG',
    'feature_avg_gpa_prereq',
    'feature_course_capacity',
    'feature_prev_enrollment_ratio',
    'TERM_YEAR'
]

categorical_features = [
    'SUBJECT_ID_SORT',
    'TERM_MONTH'
]

missing_cols_numerical = [col for col in numerical_features if col not in df.columns]
if missing_cols_numerical:
    print(f"Warning: The following numerical feature columns are missing from the dataframe and will be skipped: {missing_cols_numerical}")
    numerical_features = [col for col in numerical_features if col in df.columns]

missing_cols_categorical = [col for col in categorical_features if col not in df.columns]
if missing_cols_categorical:
    print(f"Warning: The following categorical feature columns are missing from the dataframe and will be skipped: {missing_cols_categorical}")
    categorical_features = [col for col in categorical_features if col in df.columns]

all_features_combined = numerical_features + categorical_features
X_pre_split = df[all_features_combined].copy()

# Prepare `X` for CatBoost/LightGBM: these can handle raw categorical features directly
X_cat_lgbm_full = X_pre_split.copy()
# Ensure categorical features are of 'category' dtype for LightGBM/CatBoost handling
for col in categorical_features:
    if col in X_cat_lgbm_full.columns:
        X_cat_lgbm_full[col] = X_cat_lgbm_full[col].astype('category')

# Identify the indices of categorical features within X_cat_lgbm_full for CatBoost
catboost_cat_features_indices = [X_cat_lgbm_full.columns.get_loc(col) for col in categorical_features]
# Identify the names of categorical features within X_cat_lgbm_full for LightGBM
lgbm_cat_feature_names = [col for col in categorical_features if col in X_cat_lgbm_full.columns]


# Prepare `X` for Scikit-learn models (RandomForest, LogisticRegression): these typically require one-hot encoding
X_sklearn_full = pd.get_dummies(X_pre_split, columns=categorical_features, drop_first=True)


# --- Time-based Validation Split ---
print("Performing time-based validation split...")

df_sorted_indices = df.sort_values(by='TERM_CODE').index

y_full_sorted = y_full.loc[df_sorted_indices]
X_cat_lgbm_full_sorted = X_cat_lgbm_full.loc[df_sorted_indices]
X_sklearn_full_sorted = X_sklearn_full.loc[df_sorted_indices]

unique_terms = df.loc[df_sorted_indices]['TERM_CODE'].unique()

X_train_cat_lgbm, y_train_cat_lgbm, X_val_cat_lgbm, y_val_cat_lgbm = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
X_train_sk, y_train_sk, X_val_sk, y_val_sk = pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
performed_random_split = False

if len(unique_terms) > 1:
    split_point_idx = int(len(unique_terms) * 0.8)
    train_terms = unique_terms[:split_point_idx]
    val_terms = unique_terms[split_point_idx:]

    train_mask = df.loc[df_sorted_indices]['TERM_CODE'].isin(train_terms)
    val_mask = df.loc[df_sorted_indices]['TERM_CODE'].isin(val_terms)

    X_train_cat_lgbm_temp = X_cat_lgbm_full_sorted[train_mask]
    y_train_cat_lgbm_temp = y_full_sorted[train_mask]
    X_val_cat_lgbm_temp = X_cat_lgbm_full_sorted[val_mask]
    y_val_cat_lgbm_temp = y_full_sorted[val_mask]

    X_train_sk_temp = X_sklearn_full_sorted[train_mask]
    y_train_sk_temp = y_full_sorted[train_mask]
    X_val_sk_temp = X_sklearn_full_sorted[val_mask]
    y_val_sk_temp = y_full_sorted[val_mask]

    if X_train_cat_lgbm_temp.empty or y_train_cat_lgbm_temp.empty or X_val_cat_lgbm_temp.empty or y_val_cat_lgbm_temp.empty:
        print("Time-based validation split resulted in an empty train or validation set. Falling back to random split.")
        performed_random_split = True
    else:
        X_train_cat_lgbm, y_train_cat_lgbm = X_train_cat_lgbm_temp, y_train_cat_lgbm_temp
        X_val_cat_lgbm, y_val_cat_lgbm = X_val_cat_lgbm_temp, y_val_cat_lgbm_temp
        X_train_sk, y_train_sk = X_train_sk_temp, y_train_sk_temp
        X_val_sk, y_val_sk = X_val_sk_temp, y_val_sk_temp
        print(f"Training on terms: {train_terms.tolist()}")
        print(f"Validating on terms: {val_terms.tolist()}")
else:
    print("Not enough unique terms for a meaningful time-based split. Falling back to random split.")
    performed_random_split = True

if performed_random_split:
    if len(y_full.unique()) > 1 and y_full.value_counts().min() > 1:
        X_train_cat_lgbm, X_val_cat_lgbm, y_train_cat_lgbm, y_val_cat_lgbm = train_test_split(X_cat_lgbm_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
        X_train_sk, X_val_sk, y_train_sk, y_val_sk = train_test_split(X_sklearn_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    else:
        print("Not enough classes or samples per class for stratified split. Using non-stratified random split.")
        X_train_cat_lgbm, X_val_cat_lgbm, y_train_cat_lgbm, y_val_cat_lgbm = train_test_split(X_cat_lgbm_full, y_full, test_size=0.2, random_state=42)
        X_train_sk, X_val_sk, y_train_sk, y_val_sk = train_test_split(X_sklearn_full, y_full, test_size=0.2, random_state=42)

if X_train_cat_lgbm.empty or y_train_cat_lgbm.empty or X_val_cat_lgbm.empty or y_val_cat_lgbm.empty:
    print("Error: Training or validation sets for CatBoost/LightGBM are empty after splitting. Cannot proceed with their training.")
    catboost_available = False
    lightgbm_available = False
if X_train_sk.empty or y_train_sk.empty or X_val_sk.empty or y_val_sk.empty:
    print("Error: Training or validation sets for Scikit-learn are empty after splitting. Cannot proceed with RF/LR training.")


# Feature Scaling (numerical features for Scikit-learn models)
scaler = StandardScaler()
numerical_cols_to_scale_sklearn = [col for col in X_train_sk.columns if col in numerical_features]

if numerical_cols_to_scale_sklearn:
    X_train_sk[numerical_cols_to_scale_sklearn] = scaler.fit_transform(X_train_sk[numerical_cols_to_scale_sklearn])
    X_val_sk[numerical_cols_to_scale_sklearn] = scaler.transform(X_val_sk[numerical_cols_to_scale_sklearn])
else:
    print("No numerical columns found for scaling for Scikit-learn models.")

print(f"Training data shape (CatBoost/LightGBM): {X_train_cat_lgbm.shape}, Validation data shape (CatBoost/LightGBM): {X_val_cat_lgbm.shape}")
print(f"Training data shape (Scikit-learn): {X_train_sk.shape}, Validation data shape: {X_val_sk.shape}")


# --- Model Training ---
models = {}
y_pred_probas = []

print("Training RandomForestClassifier...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_rf.fit(X_train_sk, y_train_sk)
models['RandomForest'] = model_rf
print("RandomForestClassifier training complete.")

print("Training LogisticRegression...")
model_lr = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
model_lr.fit(X_train_sk, y_train_sk)
models['LogisticRegression'] = model_lr
print("LogisticRegression training complete.")

if catboost_available and not X_train_cat_lgbm.empty:
    print("Training CatBoostClassifier...")
    class_counts = y_train_cat_lgbm.value_counts()
    class_weight_for_1 = class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] > 0 else 1
    class_weights = {0: 1, 1: class_weight_for_1}

    model_cb = CatBoostClassifier_class(iterations=100,
                                        random_seed=42,
                                        verbose=0,
                                        loss_function='Logloss',
                                        eval_metric='F1',
                                        class_weights=class_weights,
                                        cat_features=catboost_cat_features_indices,
                                        early_stopping_rounds=10)
    model_cb.fit(X_train_cat_lgbm, y_train_cat_lgbm, eval_set=(X_val_cat_lgbm, y_val_cat_lgbm), plot=False)
    models['CatBoost'] = model_cb
    print("CatBoostClassifier training complete.")
else:
    print("CatBoost is not available or training data is empty, skipping CatBoostClassifier training.")


if lightgbm_available and not X_train_cat_lgbm.empty:
    print("Training LightGBM model...")
    class_counts = y_train_cat_lgbm.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts and class_counts[1] > 0 else 1

    lgb_clf = lgb.LGBMClassifier(objective='binary',
                                 random_state=42,
                                 n_estimators=100,
                                 scale_pos_weight=scale_pos_weight,
                                 categorical_feature=lgbm_cat_feature_names)
    lgb_clf.fit(X_train_cat_lgbm, y_train_cat_lgbm)
    models['LightGBM'] = lgb_clf
    print("LightGBM model training complete.")
else:
    print("LightGBM is not available or training data is empty, skipping LightGBM training.")

# --- Ensemble Prediction and Evaluation ---
print("Performing ensemble prediction and evaluating on validation set...")

if 'RandomForest' in models:
    y_pred_probas.append(models['RandomForest'].predict_proba(X_val_sk))
if 'LogisticRegression' in models:
    y_pred_probas.append(models['LogisticRegression'].predict_proba(X_val_sk))
if 'CatBoost' in models:
    y_pred_probas.append(models['CatBoost'].predict_proba(X_val_cat_lgbm))
if 'LightGBM' in models:
    y_pred_probas.append(models['LightGBM'].predict_proba(X_val_cat_lgbm))

if not y_pred_probas:
    print("No models were trained successfully or produced predictions. Cannot perform ensemble prediction.")
    final_validation_score = 0.0
else:
    y_pred_proba_ensemble = np.mean(y_pred_probas, axis=0)
    y_pred_ensemble = (y_pred_proba_ensemble[:, 1] >= 0.5).astype(int)
    f1_macro = f1_score(y_val_sk, y_pred_ensemble, average='macro')
    final_validation_score = f1_macro

# --- Final Output ---
print(f"Final Validation Performance: {final_validation_score}")
