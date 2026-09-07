
import argparse
import os
import pandas as pd
import numpy as np
import sys
import subprocess
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from io import StringIO

# --- Dependency Installation ---
try:
    # Added 'setuptools' to fix the 'pkg_resources' ModuleNotFoundError
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools", "tabpfn==0.1.0", "lightgbm==4.1.0", "--quiet"])
    from tabpfn import TabPFNClassifier
    from lightgbm import LGBMClassifier
except (ImportError, subprocess.CalledProcessError) as e:
    print(f"Error: Could not install or import dependencies: {e}", file=sys.stderr)
    # Removed sys.exit(1) as per instructions.
    # The program will likely fail on the subsequent imports if installation fails.


# --- Dummy Data Creation ---
def create_dummy_data(temp_dir='./input'):
    """Creates dummy CSV files for a self-contained run."""
    os.makedirs(temp_dir, exist_ok=True)
    
    subject_summary_data = """TERM_CODE,SUBJECT_ID_SORT,TOTAL_ENROLLMENT
202201,CS-101,80
202201,MATH-202,120
202201,PHYS-303,50
202201,CHEM-404,95
202202,CS-101,90
202202,MATH-202,110
202202,PHYS-303,60
202202,CHEM-404,105
202203,CS-101,95
202203,MATH-202,100
202203,PHYS-303,70
202203,CHEM-404,115
202204,CS-101,100
202204,MATH-202,90
202204,PHYS-303,80
202204,CHEM-404,125
"""
    gold_labels_data = """TERM_CODE,SUBJECT_ID_SORT,HIGH_ENROLLMENT
202201,CS-101,N
202201,MATH-202,Y
202201,PHYS-303,N
202201,CHEM-404,Y
202202,CS-101,Y
202202,MATH-202,Y
202202,PHYS-303,N
202202,CHEM-404,Y
202203,CS-101,Y
202203,MATH-202,Y
202203,PHYS-303,N
202203,CHEM-404,Y
202204,CS-101,Y
202204,MATH-202,N
202204,PHYS-303,N
202204,CHEM-404,Y
"""
    course_attributes_data = """SUBJECT_ID_SORT,COURSE_LEVEL,CREDIT_HOURS
CS-101,Undergraduate,3
MATH-202,Undergraduate,4
PHYS-303,Graduate,3
CHEM-404,Graduate,4
"""
    
    with open(os.path.join(temp_dir, 'subject_summary.csv'), 'w') as f: f.write(subject_summary_data)
    with open(os.path.join(temp_dir, 'gold_enrollment_train.csv'), 'w') as f: f.write(gold_labels_data)
    with open(os.path.join(temp_dir, 'course_attributes.csv'), 'w') as f: f.write(course_attributes_data)
    # Create empty files for other potential data sources to avoid errors
    open(os.path.join(temp_dir, 'course_summary.csv'), 'w').close()
    open(os.path.join(temp_dir, 'faculty_summary.csv'), 'w').close()
    
    return temp_dir

# --- Core Logic from Original Script ---

def load_data(train_dir, gold_path):
    """Loads all necessary CSV files."""
    paths = {
        'subject_summary': os.path.join(train_dir, 'subject_summary.csv'),
        'course_attributes': os.path.join(train_dir, 'course_attributes.csv'),
        'course_summary': os.path.join(train_dir, 'course_summary.csv'),
        'faculty_summary': os.path.join(train_dir, 'faculty_summary.csv'),
        'gold_labels': gold_path
    }
    dataframes = {name: pd.read_csv(path) for name, path in paths.items() if os.path.exists(path) and os.path.getsize(path) > 0}
    return dataframes

def feature_engineering(dfs):
    """Merges dataframes and creates features."""
    if 'subject_summary' not in dfs or 'gold_labels' not in dfs:
        raise ValueError("Core data files are missing.")
    
    data = pd.merge(dfs['subject_summary'], dfs['gold_labels'], on=['TERM_CODE', 'SUBJECT_ID_SORT'], how='left')
    if 'course_attributes' in dfs:
         data = pd.merge(data, dfs['course_attributes'], on='SUBJECT_ID_SORT', how='left')

    data['DEPARTMENT'] = data['SUBJECT_ID_SORT'].str.split('-').str[0]
    data.dropna(subset=['HIGH_ENROLLMENT'], inplace=True)
    data['HIGH_ENROLLMENT'] = data['HIGH_ENROLLMENT'].apply(lambda x: 1 if x == 'Y' else 0)
    data['TERM_CODE_INT'] = data['TERM_CODE']
    return data

def build_classifier_pipeline(classifier, numeric_features, categorical_features):
    """Builds a scikit-learn pipeline for preprocessing and classification."""
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)],
        remainder='drop'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

# --- Ablation Experiment Runner ---
def run_experiment(train_dir, use_target_encoding, use_weighted_ensemble):
    """Runs a single experiment configuration."""
    # 1. Load and Engineer Features
    gold_path = os.path.join(train_dir, 'gold_enrollment_train.csv')
    dataframes = load_data(train_dir, gold_path)
    data = feature_engineering(dataframes)
    
    # 2. Data Splitting (Time-based with Holdout)
    data = data.sort_values('TERM_CODE_INT').reset_index(drop=True)
    sorted_terms = sorted(data['TERM_CODE_INT'].unique())
    validation_term = sorted_terms[-1]
    holdout_term = sorted_terms[-2]
    
    val_df = data[data['TERM_CODE_INT'] == validation_term].copy()
    holdout_df = data[data['TERM_CODE_INT'] == holdout_term].copy()
    train_df = data[data['TERM_CODE_INT'] < holdout_term].copy()

    y_train = train_df['HIGH_ENROLLMENT']
    y_val = val_df['HIGH_ENROLLMENT']
    y_holdout = holdout_df['HIGH_ENROLLMENT']
    
    # 3. Feature Processing
    cat_features = [col for col in ['DEPARTMENT', 'COURSE_LEVEL'] if col in data.columns and data[col].dtype == 'object']
    num_features = [col for col in data.select_dtypes(include=np.number).columns if col not in ['HIGH_ENROLLMENT', 'TERM_CODE', 'TERM_CODE_INT']]
    
    models = {}
    
    if use_target_encoding:
        global_mean = y_train.mean()
        for col in cat_features:
            mapping = train_df.groupby(col)['HIGH_ENROLLMENT'].mean()
            train_df[col] = train_df[col].map(mapping).fillna(global_mean)
            val_df[col] = val_df[col].map(mapping).fillna(global_mean)
            holdout_df[col] = holdout_df[col].map(mapping).fillna(global_mean)
        
        features = num_features + cat_features
        X_train, X_val, X_holdout = train_df[features].fillna(0), val_df[features].fillna(0), holdout_df[features].fillna(0)
        
        models['rf'] = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        models['lgbm'] = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        models['rf'].fit(X_train, y_train)
        models['lgbm'].fit(X_train, y_train)
    else: # Use OneHotEncoder pipeline
        rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        lgbm_classifier = LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        models['rf'] = build_classifier_pipeline(rf_classifier, num_features, cat_features)
        models['lgbm'] = build_classifier_pipeline(lgbm_classifier, num_features, cat_features)
        models['rf'].fit(train_df, y_train)
        models['lgbm'].fit(train_df, y_train)
        X_val, X_holdout = val_df, holdout_df
    
    # TabPFN training (always uses raw-ish data)
    tabpfn_cat = [f for f in ['SUBJECT_ID_SORT', 'DEPARTMENT'] if f in train_df.columns]
    tabpfn_num = [f for f in ['TOTAL_ENROLLMENT'] if f in train_df.columns]
    tabpfn_features = tabpfn_num + tabpfn_cat
    
    train_df_tabpfn, val_df_tabpfn, holdout_df_tabpfn = train_df.copy(), val_df.copy(), holdout_df.copy()
    for col in tabpfn_cat:
        codes, uniques = pd.factorize(train_df_tabpfn[col])
        train_df_tabpfn[col] = codes
        mapping = {label: i for i, label in enumerate(uniques)}
        val_df_tabpfn[col] = val_df_tabpfn[col].map(mapping).fillna(-1).astype(int)
        holdout_df_tabpfn[col] = holdout_df_tabpfn[col].map(mapping).fillna(-1).astype(int)

    X_train_tabpfn, X_val_tabpfn, X_holdout_tabpfen = train_df_tabpfn[tabpfn_features].fillna(0), val_df_tabpfn[tabpfn_features].fillna(0), holdout_df_tabpfn[tabpfn_features].fillna(0)

    models['tabpfn'] = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    models['tabpfn'].fit(X_train_tabpfn, y_train, overwrite_warning=True)
    
    # 4. Ensembling and Evaluation
    weights = {'rf': 1.0, 'lgbm': 1.0, 'tabpfn': 1.0}
    if use_weighted_ensemble:
        model_scores = {}
        for name, model in models.items():
            X_h = X_holdout_tabpfen if name == 'tabpfn' else X_holdout
            probas_h = model.predict_proba(X_h)[:, 1]
            score = f1_score(y_holdout, (probas_h >= 0.5).astype(int), average='macro', zero_division=0)
            model_scores[name] = score
        total_score = sum(model_scores.values())
        if total_score > 0:
            weights = {name: score / total_score for name, score in model_scores.items()}

    probas = {}
    probas['rf'] = models['rf'].predict_proba(X_val)[:, 1]
    probas['lgbm'] = models['lgbm'].predict_proba(X_val)[:, 1]
    probas['tabpfn'] = models['tabpfn'].predict_proba(X_val_tabpfn)[:, 1]
    
    total_weight = sum(weights.values())
    proba_ensemble = (probas['rf'] * weights['rf'] + probas['lgbm'] * weights['lgbm'] + probas['tabpfn'] * weights['tabpfn']) / total_weight
    pred_ensemble = (proba_ensemble >= 0.5).astype(int)
    
    return f1_score(y_val, pred_ensemble, average='macro', zero_division=0)

# --- Main Execution Block ---
def main():
    """Main function to run the ablation study."""
    temp_dir = create_dummy_data()
    results = {}
    
    scenarios = {
        "Baseline (Target Encoding + Weighted Ensemble)": {"use_target_encoding": True, "use_weighted_ensemble": True},
        "Ablation: No Target Encoding (uses OneHotEncoder)": {"use_target_encoding": False, "use_weighted_ensemble": True},
        "Ablation: No Weighted Ensemble (uses Simple Average)": {"use_target_encoding": True, "use_weighted_ensemble": False},
    }

    baseline_score = 0.0
    
    for name, kwargs in scenarios.items():
        try:
            score = run_experiment(temp_dir, **kwargs)
            results[name] = score
            if "Baseline" in name:
                baseline_score = score
        except Exception as e:
            results[name] = 0.0
            print(f"Error running scenario '{name}': {e}", file=sys.stderr)
            # traceback.print_exc()

    print(f"Final Validation Performance: {baseline_score}")

    print("\n--- Ablation Study Results ---")
    performance_drops = {}
    for name, score in results.items():
        drop = baseline_score - score
        print(f"{name}: {score:.4f} (Performance Drop: {drop:.4f})")
        if "Ablation" in name:
            component_name = name.split("Ablation: ")[1]
            performance_drops[component_name] = drop
            
    if not performance_drops:
         print("\n--- Conclusion ---\nCould not run ablation scenarios.")
    else:
        # Find the component that caused the biggest drop
        most_impactful_component = max(performance_drops, key=performance_drops.get)
        max_drop = performance_drops[most_impactful_component]

        print("\n--- Conclusion ---")
        if max_drop > 0.001:
            print(f"The component that contributes the most to the overall performance is: '{most_impactful_component}'")
        else:
            print("No single component removal resulted in a significant performance drop.")

if __name__ == '__main__':
    # Ensure all required modules can be imported before running main logic
    try:
        from tabpfn import TabPFNClassifier
        from lightgbm import LGBMClassifier
        main()
    except ImportError as e:
        print(f"Could not run main logic due to missing dependency: {e}", file=sys.stderr)

