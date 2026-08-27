
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

def main():
    data_dir = './input'
    
    # 1. Load Base Table
    base_path = os.path.join(data_dir, 'forest_patches.csv')
    df_base = pd.read_csv(base_path)
    
    # Target mapping (1 -> 0, 2 -> 1)
    y = (df_base['class'] == 2).astype(int).values
    patch_ids = df_base['patch_id'].values
    
    # 2. Process Multi-Row Table: patch_measurements.csv
    pm_path = os.path.join(data_dir, 'patch_measurements.csv')
    if os.path.exists(pm_path):
        pm = pd.read_csv(pm_path)
        
        # Safe feature engineering on measurements based on column presence
        if 'dist_hydrology_h_m' in pm.columns and 'dist_hydrology_v_m' in pm.columns:
            pm['dist_hydrology_3d'] = np.sqrt(pm['dist_hydrology_h_m']**2 + pm['dist_hydrology_v_m']**2)
            pm['dist_hydrology_sum'] = pm['dist_hydrology_h_m'] + pm['dist_hydrology_v_m']
            pm['dist_hydrology_diff'] = pm['dist_hydrology_h_m'] - pm['dist_hydrology_v_m']
            
        if 'aspect_deg' in pm.columns:
            pm['aspect_sin'] = np.sin(np.radians(pm['aspect_deg']))
            pm['aspect_cos'] = np.cos(np.radians(pm['aspect_deg']))
            
        hill_cols = [c for c in ['hillshade_9am', 'hillshade_noon', 'hillshade_3pm'] if c in pm.columns]
        if len(hill_cols) > 0:
            pm['hillshade_mean'] = pm[hill_cols].mean(axis=1)
            pm['hillshade_range'] = pm[hill_cols].max(axis=1) - pm[hill_cols].min(axis=1)
            
        if 'elevation_m' in pm.columns and 'dist_hydrology_v_m' in pm.columns:
            pm['elevation_minus_hyd_v'] = pm['elevation_m'] - pm['dist_hydrology_v_m']
            pm['elevation_plus_hyd_v'] = pm['elevation_m'] + pm['dist_hydrology_v_m']
            
        num_cols = pm.select_dtypes(include=[np.number]).columns.tolist()
        for col_to_remove in ['patch_id', 'obs_no', 'station_id']:
            if col_to_remove in num_cols:
                num_cols.remove(col_to_remove)
                
        dist_cols = [c for c in num_cols if c.startswith('dist_')]
        if len(dist_cols) > 1:
            pm['total_dist_sum'] = pm[dist_cols].sum(axis=1)
            if 'total_dist_sum' not in num_cols:
                num_cols.append('total_dist_sum')
        
        # Group & Aggregate by patch_id
        pm_agg = pm.groupby('patch_id')[num_cols].agg(['mean', 'std', 'min', 'max'])
        pm_agg.columns = [f"pm_{col}_{stat}" for col, stat in pm_agg.columns]
        pm_agg.reset_index(inplace=True)
    else:
        pm_agg = None

    # 3. Process Multi-Row Table: plot_notes.csv
    pn_path = os.path.join(data_dir, 'plot_notes.csv')
    if os.path.exists(pn_path):
        pn = pd.read_csv(pn_path)
        pn_num = pn.select_dtypes(include=[np.number]).columns.tolist()
        if 'parcel_id' in pn_num:
            pn_num.remove('parcel_id')
        if len(pn_num) > 0:
            pn_agg = pn.groupby('parcel_id')[pn_num].agg(['mean', 'std', 'min', 'max', 'count'])
            pn_agg.columns = [f"pn_{col}_{stat}" for col, stat in pn_agg.columns]
            pn_agg.reset_index(inplace=True)
        else:
            pn_agg = pn.groupby('parcel_id').size().to_frame('pn_count').reset_index()
    else:
        pn_agg = None

    # 4. Load 1-to-1 Tables
    parcels = pd.read_csv(os.path.join(data_dir, 'parcels.csv'))
    stands = pd.read_csv(os.path.join(data_dir, 'stands.csv'))
    survey_units = pd.read_csv(os.path.join(data_dir, 'survey_units.csv'))
    soil_registry = pd.read_csv(os.path.join(data_dir, 'soil_registry.csv'))
    
    parcel_land_status = pd.read_csv(os.path.join(data_dir, 'parcel_land_status.csv'))
    parcel_soil_addendum = pd.read_csv(os.path.join(data_dir, 'parcel_soil_addendum.csv'))
    parcel_soil_records = pd.read_csv(os.path.join(data_dir, 'parcel_soil_records.csv'))
    
    stand_land_status = pd.read_csv(os.path.join(data_dir, 'stand_land_status.csv'))
    stand_soil_atlas = pd.read_csv(os.path.join(data_dir, 'stand_soil_atlas.csv'))
    stand_soil_records = pd.read_csv(os.path.join(data_dir, 'stand_soil_records.csv'))
    
    county_soil_atlas = pd.read_csv(os.path.join(data_dir, 'county_soil_atlas.csv'))
    nrcs_soil_map = pd.read_csv(os.path.join(data_dir, 'nrcs_soil_map.csv'))
    sensor_calibration = pd.read_csv(os.path.join(data_dir, 'sensor_calibration.csv'))
    usfs_soil_survey = pd.read_csv(os.path.join(data_dir, 'usfs_soil_survey.csv'))

    # 5. Build Merge Pipeline
    # Merge parcel level
    parcel_df = parcels.merge(parcel_land_status, on='parcel_id', how='left')
    parcel_df = parcel_df.merge(parcel_soil_addendum, on='parcel_id', how='left')
    parcel_df = parcel_df.merge(parcel_soil_records, on='parcel_id', how='left')
    if pn_agg is not None:
        parcel_df = parcel_df.merge(pn_agg, on='parcel_id', how='left')
        
    # Merge stand level
    stand_df = stands.merge(stand_land_status, on='stand_id', how='left')
    stand_df = stand_df.merge(stand_soil_atlas, on='stand_id', how='left')
    stand_df = stand_df.merge(stand_soil_records, on='stand_id', how='left')
    
    # Merge survey unit level
    survey_df = survey_units.merge(county_soil_atlas, on='survey_unit_id', how='left')
    survey_df = survey_df.merge(nrcs_soil_map, on='survey_unit_id', how='left')
    survey_df = survey_df.merge(sensor_calibration, on='survey_unit_id', how='left')
    survey_df = survey_df.merge(usfs_soil_survey, on='survey_unit_id', how='left')

    # Merge all onto base table by patch_id
    full_df = df_base.merge(soil_registry, on='patch_id', how='left')
    full_df = full_df.merge(parcel_df, on='patch_id', how='left')
    full_df = full_df.merge(stand_df, on='patch_id', how='left')
    full_df = full_df.merge(survey_df, on='patch_id', how='left')
    if pm_agg is not None:
        full_df = full_df.merge(pm_agg, on='patch_id', how='left')

    # Assert exact base row count
    assert len(full_df) == 423680, f"row count changed: {len(full_df)}"

    # 6. Feature Selection & Engineering
    id_cols = ['patch_id', 'parcel_id', 'stand_id', 'survey_unit_id', 'class']
    feature_cols = [c for c in full_df.columns if c not in id_cols]
    
    soil_cols = [c for c in feature_cols if c.startswith('Soil_Type')]
    if soil_cols:
        full_df['soil_type_sum'] = full_df[soil_cols].sum(axis=1)
        full_df['soil_type_count'] = (full_df[soil_cols] > 0).sum(axis=1)
        feature_cols.extend(['soil_type_sum', 'soil_type_count'])
        
    wild_cols = [c for c in feature_cols if c.startswith('Wilderness_Area')]
    if wild_cols:
        full_df['wilderness_sum'] = full_df[wild_cols].sum(axis=1)
        feature_cols.append('wilderness_sum')

    feature_cols = list(dict.fromkeys(feature_cols))
    X = full_df[feature_cols].copy()

    # Convert object types to category
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

    # 7. Stratified Group Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=patch_ids)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            learning_rate=0.05,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1500,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    final_validation_score = roc_auc_score(y, oof_preds)
    print(f"Final Validation Performance: {final_validation_score}")

if __name__ == '__main__':
    main()
