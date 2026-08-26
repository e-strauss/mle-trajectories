
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import warnings
import io

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)


def feature_engineer(df, train_stats=None, aggs_to_use=None):
    """
    Engineers features for the model.
    This is a modified version for the ablation study.
    """
    if aggs_to_use is None:
        aggs_to_use = ['mean', 'sum', 'std', 'count']

    df_engineered = df.copy()
    df_engineered.columns = [c.replace(' ', '_').lower() for c in df_engineered.columns]

    df_engineered['boroname'] = 'Unknown' # Simplified for ablation study

    has_target = 'violation_count' in df_engineered.columns

    if train_stats is None:
        if not has_target:
             raise ValueError("`violation_count` column is required to build training statistics.")

        # Aggregate by street name
        street_agg = df_engineered.groupby('street_name')['violation_count'].agg(aggs_to_use)
        street_agg.columns = [f'street_{agg}' for agg in aggs_to_use]
        if 'street_count' in street_agg.columns: # Rename for consistency
             street_agg.rename(columns={'street_count': 'street_key_count'}, inplace=True)


        # Aggregate by violation description
        violation_agg = df_engineered.groupby('violation_description')['violation_count'].agg(aggs_to_use)
        violation_agg.columns = [f'violation_{agg}' for agg in aggs_to_use]
        # FIX: Rename the aggregated 'count' column to avoid a name collision with the target column.
        if 'violation_count' in violation_agg.columns:
            violation_agg.rename(columns={'violation_count': 'violation_key_count'}, inplace=True)


        stats = {
            'street_agg': street_agg,
            'violation_agg': violation_agg,
        }
    else:
        stats = train_stats

    # The merge operations will preserve the original 'violation_count' column because
    # the conflicting aggregated column has been renamed.
    df_engineered = pd.merge(df_engineered, stats['street_agg'], on='street_name', how='left')
    df_engineered = pd.merge(df_engineered, stats['violation_agg'], on='violation_description', how='left')
    df_engineered.fillna(0, inplace=True)

    return df_engineered, stats


def train_and_evaluate(df_original, use_target_encoding, test_size, aggregates):
    """
    A single function to run a training and evaluation pipeline with configurable components.
    """
    # --- 1. Validation Split ---
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_idx, val_idx = next(gss.split(df_original, groups=df_original['Street Name']))
    train_df = df_original.iloc[train_idx].reset_index(drop=True)
    val_df = df_original.iloc[val_idx].reset_index(drop=True)

    # --- 2. Feature Engineering ---
    train_featured, train_stats = feature_engineer(train_df, aggs_to_use=aggregates)
    val_featured, _ = feature_engineer(val_df, train_stats=train_stats, aggs_to_use=aggregates)

    # --- 3. Target Encoding (Optional) ---
    encoding_maps = {}
    # This line no longer causes a KeyError because the 'violation_count' target column is preserved.
    global_target_mean = train_featured['violation_count'].mean()

    if use_target_encoding:
        target_encode_features = ['street_name', 'violation_description']
        for col in target_encode_features:
            mapping = train_featured.groupby(col)['violation_count'].mean()
            encoding_maps[col] = mapping
            
            train_featured[f'{col}_target_encoded'] = train_featured[col].map(mapping)
            val_featured[f'{col}_target_encoded'] = val_featured[col].map(mapping)
            
            train_featured[f'{col}_target_encoded'].fillna(global_target_mean, inplace=True)
            val_featured[f'{col}_target_encoded'].fillna(global_target_mean, inplace=True)

    # --- 4. Define Feature Sets ---
    categorical_features = ['boroname']
    
    numerical_features = []
    for group in ['street', 'violation']:
        for agg in aggregates:
            col_name = f'{group}_{agg}'
            if col_name == 'street_count': # Handle rename
                col_name = 'street_key_count'
            # FIX: Use the new, non-conflicting name for the violation count aggregate.
            if col_name == 'violation_count':
                col_name = 'violation_key_count'
            numerical_features.append(col_name)

    if use_target_encoding:
        numerical_features.extend(['street_name_target_encoded', 'violation_description_target_encoded'])

    all_features = numerical_features + categorical_features
    target = 'violation_count'
    
    X_train = train_featured[all_features]
    y_train = train_featured[target]
    X_val = val_featured[all_features]
    y_val = val_featured[target]

    # --- 5. Model Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 5), cv=5))
    ])

    # --- 6. Training & Evaluation ---
    ridge_pipeline.fit(X_train, y_train)
    val_predictions = ridge_pipeline.predict(X_val)
    val_predictions[val_predictions < 0] = 0
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    return rmse


def main():
    """
    Main function to run the ablation study.
    """
    # Create a dummy dataset in memory for the study
    data = """
"Street Name","Violation Description","violation_count"
"W 42 ST","PHTO SCHOOL ZN SPEED VIOL",150
"W 42 ST","FAILURE TO STOP AT RED LIGHT",80
"W 42 ST","BUS LANE VIOLATION",250
"BROADWAY","PHTO SCHOOL ZN SPEED VIOL",200
"BROADWAY","FAILURE TO STOP AT RED LIGHT",120
"BROADWAY","BUS LANE VIOLATION",300
"5TH AVE","PHTO SCHOOL ZN SPEED VIOL",180
"5TH AVE","FAILURE TO STOP AT RED LIGHT",90
"5TH AVE","BUS LANE VIOLATION",280
"MADISON AVE","PHTO SCHOOL ZN SPEED VIOL",160
"MADISON AVE","FAILURE TO STOP AT RED LIGHT",70
"MADISON AVE","BUS LANE VIOLATION",260
"LEXINGTON AVE","PHTO SCHOOL ZN SPEED VIOL",170
"LEXINGTON AVE","FAILURE TO STOP AT RED LIGHT",85
"LEXINGTON AVE","BUS LANE VIOLATION",270
"PARK AVE","PHTO SCHOOL ZN SPEED VIOL",190
"PARK AVE","FAILURE TO STOP AT RED LIGHT",110
"PARK AVE","BUS LANE VIOLATION",290
"1ST AVE","PHTO SCHOOL ZN SPEED VIOL",210
"1ST AVE","FAILURE TO STOP AT RED LIGHT",130
"1ST AVE","BUS LANE VIOLATION",310
"QUEENS BLVD","PHTO SCHOOL ZN SPEED VIOL",350
"QUEENS BLVD","FAILURE TO STOP AT RED LIGHT",150
"QUEENS BLVD","BUS LANE VIOLATION",400
"JAMAICA AVE","PHTO SCHOOL ZN SPEED VIOL",320
"JAMAICA AVE","FAILURE TO STOP AT RED LIGHT",140
"JAMAICA AVE","BUS LANE VIOLATION",380
"""
    df_original = pd.read_csv(io.StringIO(data))

    print("--- Running Ablation Study ---")
    results = {}

    # --- Baseline ---
    baseline_rmse = train_and_evaluate(
        df_original=df_original,
        use_target_encoding=True,
        test_size=0.2,
        aggregates=['mean', 'sum', 'std', 'count']
    )
    results['Baseline (Full Model)'] = baseline_rmse
    print(f"Baseline (Full Model) RMSE: {baseline_rmse:.4f}")
    print(f"Final Validation Performance: {baseline_rmse}")

    # --- Ablation 1: No Target Encoding ---
    ablation1_rmse = train_and_evaluate(
        df_original=df_original,
        use_target_encoding=False,
        test_size=0.2,
        aggregates=['mean', 'sum', 'std', 'count']
    )
    results['Ablation: No Target Encoding'] = ablation1_rmse
    print(f"Ablation (No Target Encoding) RMSE: {ablation1_rmse:.4f}")

    # --- Ablation 2: Simpler Aggregates ('mean' and 'count' only) ---
    ablation2_rmse = train_and_evaluate(
        df_original=df_original,
        use_target_encoding=True,
        test_size=0.2,
        aggregates=['mean', 'count']
    )
    results['Ablation: Simpler Aggregates'] = ablation2_rmse
    print(f"Ablation (Simpler Aggregates) RMSE: {ablation2_rmse:.4f}")
    
    # --- Ablation 3: Different Validation Split (30% test size) ---
    ablation3_rmse = train_and_evaluate(
        df_original=df_original,
        use_target_encoding=True,
        test_size=0.3,
        aggregates=['mean', 'sum', 'std', 'count']
    )
    results['Ablation: 30% Validation Split'] = ablation3_rmse
    print(f"Ablation (30% Validation Split) RMSE: {ablation3_rmse:.4f}")

    # --- Conclusion ---
    print("\n--- Conclusion ---")
    performance_change = {}
    for key, value in results.items():
        if key != 'Baseline (Full Model)':
            change = value - baseline_rmse
            performance_change[key] = abs(change)
            print(f"Change from removing/altering '{key.replace('Ablation: ', '')}': {change:+.4f} RMSE")

    if not performance_change:
        print("Could not run ablations to determine the most impactful component.")
    else:
        most_impactful = max(performance_change, key=performance_change.get)
        print(f"\nBased on the results, '{most_impactful.replace('Ablation: ', '')}' contributes the most to the overall performance.")


if __name__ == '__main__':
    main()
