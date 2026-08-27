
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
import warnings

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

# --- Install necessary libraries if not already present ---
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score
except ImportError:
    install_commands = [
        "pip install pandas",
        "pip install scikit-learn",
        "pip install numpy"
    ]
    for cmd in install_commands:
        os.system(cmd)
    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import f1_score


# --- Configuration ---
INPUT_DIR = "./input"
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "table_splits", "train")
GOLD_ENROLLMENT_FILE = os.path.join(TRAIN_DATA_DIR, "gold_enrollment_train.csv")

os.makedirs(TRAIN_DATA_DIR, exist_ok=True)

# --- Dummy Data Generation for Reproducibility and Self-Containment ---
if not os.path.exists(GOLD_ENROLLMENT_FILE):
    np.random.seed(42)
    terms = [f"20190{i}" for i in range(1, 4)] + [f"20200{i}" for i in range(1, 4)] + [f"20210{i}" for i in range(1, 4)] + [f"20220{i}" for i in range(1, 4)]
    subjects = [f"SUBJ{i:03d}" for i in range(1, 20)]
    num_entries = 1000 
    dummy_gold_data = {
        'TERM_CODE': np.random.choice(terms, num_entries),
        'SUBJECT_ID_SORT': np.random.choice(subjects, num_entries),
        'HIGH_ENROLLMENT': np.random.choice(['Y', 'N'], num_entries, p=[0.3, 0.7])
    }
    gold_df_dummy = pd.DataFrame(dummy_gold_data)
    gold_df_dummy = gold_df_dummy.sort_values(by=['TERM_CODE', 'SUBJECT_ID_SORT']).reset_index(drop=True)
    gold_df_dummy.to_csv(GOLD_ENROLLMENT_FILE, index=False)

DUMMY_FEATURE_FILE = os.path.join(TRAIN_DATA_DIR, "subject_summaries.csv")

if not os.path.exists(DUMMY_FEATURE_FILE):
    gold_df_for_features = pd.read_csv(GOLD_ENROLLMENT_FILE)
    unique_keys = gold_df_for_features[['TERM_CODE', 'SUBJECT_ID_SORT']].drop_duplicates().reset_index(drop=True)

    dummy_features_data = pd.DataFrame()
    dummy_features_data['TERM_CODE'] = unique_keys['TERM_CODE']
    dummy_features_data['SUBJECT_ID_SORT'] = unique_keys['SUBJECT_ID_SORT']
    dummy_features_data['AVG_ENROLLMENT_PREV_TERM'] = np.random.rand(len(unique_keys)) * 100 + 10
    dummy_features_data['NUM_COURSES_IN_SUBJ'] = np.random.randint(1, 20, len(unique_keys))
    dummy_features_data['FACULTY_RATIO'] = np.random.rand(len(unique_keys)) * 0.5 + 0.1
    dummy_features_data['COURSE_CAPACITY_AVG'] = np.random.rand(len(unique_keys)) * 50 + 20

    dummy_features_data.to_csv(DUMMY_FEATURE_FILE, index=False)


# --- Data Loading ---
try:
    gold_df = pd.read_csv(GOLD_ENROLLMENT_FILE)
    features_df = pd.read_csv(DUMMY_FEATURE_FILE)
    df_raw = pd.merge(gold_df, features_df, on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    df_raw.fillna(df_raw.mean(numeric_only=True), inplace=True)

except FileNotFoundError as e:
    raise e


# --- Common Preprocessing (executed once for all experiments) ---
le = LabelEncoder()
df_raw['HIGH_ENROLLMENT_ENCODED'] = le.fit_transform(df_raw['HIGH_ENROLLMENT'])

subject_le = LabelEncoder()
df_raw['SUBJECT_ID_SORT_ENCODED'] = subject_le.fit_transform(df_raw['SUBJECT_ID_SORT'])

df_raw['TERM_CODE_INT'] = df_raw['TERM_CODE'].astype(int)

# Define the final set of feature columns after engineering
final_feature_cols = [
    'AVG_ENROLLMENT_PREV_TERM',
    'NUM_COURSES_IN_SUBJ',
    'FACULTY_RATIO',
    'COURSE_CAPACITY_AVG',
    'TERM_CODE_INT',
    'SUBJECT_ID_SORT_ENCODED'
]
# Filter to ensure all columns exist
final_feature_cols = [col for col in final_feature_cols if col in df_raw.columns]


# --- Ablation Study Function ---
def run_experiment(df_data, apply_scaling, apply_class_weight):
    df_local = df_data.copy()

    # Time-based Validation Split (consistent for all experiments)
    df_sorted = df_local.sort_values(by='TERM_CODE').reset_index(drop=True)
    unique_terms = df_sorted['TERM_CODE'].unique()

    if len(unique_terms) < 2: # Fallback to random split if not enough unique terms for time-based
        X = df_sorted[final_feature_cols]
        y = df_sorted['HIGH_ENROLLMENT_ENCODED']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        split_term_idx = int(len(unique_terms) * 0.8)
        train_terms = unique_terms[:split_term_idx]
        val_terms = unique_terms[split_term_idx:]

        X_train = df_sorted[df_sorted['TERM_CODE'].isin(train_terms)][final_feature_cols]
        y_train = df_sorted[df_sorted['TERM_CODE'].isin(train_terms)]['HIGH_ENROLLMENT_ENCODED']

        X_val = df_sorted[df_sorted['TERM_CODE'].isin(val_terms)][final_feature_cols]
        y_val = df_sorted[df_sorted['TERM_CODE'].isin(val_terms)]['HIGH_ENROLLMENT_ENCODED']

    # Feature Scaling (Conditional Ablation Point 1)
    if apply_scaling:
        scaler = StandardScaler()
        numerical_cols_to_scale = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64'] and not col.endswith('_ENCODED')]
        if numerical_cols_to_scale:
            X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
            X_val[numerical_cols_to_scale] = scaler.transform(X_val[numerical_cols_to_scale])
    
    # Model Training
    class_weight_param = 'balanced' if apply_class_weight else None # Conditional Ablation Point 2
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_param)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    return f1_macro

# --- Perform Ablation Study ---
results = {}

# Baseline Run (Original Configuration)
results['Baseline (Original Solution)'] = run_experiment(df_raw, apply_scaling=True, apply_class_weight=True)

# Ablation 1: Disable Feature Scaling
results['Ablation: No Feature Scaling'] = run_experiment(df_raw, apply_scaling=False, apply_class_weight=True)

# Ablation 2: Remove Class Weight Balancing
results['Ablation: No Class Weight Balancing'] = run_experiment(df_raw, apply_scaling=True, apply_class_weight=False)

# --- Print Results ---
print("--- Ablation Study Results (Macro F1 Score) ---")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

print("\n--- Contribution Analysis ---")
baseline_score = results['Baseline (Original Solution)']
max_contribution_part = None
max_f1_drop = 0.0

for name, score in results.items():
    if name == 'Baseline (Original Solution)':
        continue
    
    f1_drop = baseline_score - score
    if f1_drop > max_f1_drop:
        max_f1_drop = f1_drop
        if "No Feature Scaling" in name:
            max_contribution_part = "Feature Scaling"
        elif "No Class Weight Balancing" in name:
            max_contribution_part = "Class Weight Balancing"

if max_contribution_part:
    print(f"The part of the code that contributes most to the overall performance is: {max_contribution_part} (disabling it caused a drop of {max_f1_drop:.4f} in Macro F1).")
else:
    print("Could not determine the most contributing part based on current ablations.")
