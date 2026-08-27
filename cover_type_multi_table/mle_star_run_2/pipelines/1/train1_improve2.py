
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

def get_matching_col(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        if all(k in col_lower for k in keywords):
            return col
    return None

def main():
    input_dir = './input'

    # Load base table
    base_df = pd.read_csv(os.path.join(input_dir, 'forest_patches.csv'))

    # Join 1-to-1 tables with key 'patch_id'
    patch_1to1_files = ['parcels.csv', 'stands.csv', 'survey_units.csv', 'soil_registry.csv']
    for f in patch_1to1_files:
        fpath = os.path.join(input_dir, f)
        if os.path.exists(fpath):
            sub_df = pd.read_csv(fpath)
            base_df = base_df.merge(sub_df, on='patch_id', how='left')


    # Aggregate and join 1-to-many table 'patch_measurements.csv'
    meas_path = os.path.join(input_dir, 'patch_measurements.csv')
    if os.path.exists(meas_path):
        meas_df = pd.read_csv(meas_path)
        meas_num_cols = [c for c in meas_df.columns if c not in ['patch_id', 'obs_no', 'station_id'] and pd.api.types.is_numeric_dtype(meas_df[c])]
        
        # Compute summary statistics and count for standard error calculation
        meas_agg = meas_df.groupby('patch_id')[meas_num_cols].agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Calculate standard error of the mean (std / sqrt(count)) for measurement precision
        sem_df = pd.DataFrame(
            {f'{c}_sem': meas_agg[(c, 'std')] / (meas_agg[(c, 'count')] ** 0.5) for c in meas_num_cols},
            index=meas_agg.index
        )
        
        # Extract summary stats (mean, std, min, max)
        meas_agg_stats = meas_agg.drop(columns='count', level=1)
        meas_agg_stats.columns = [f'{c}_{stat}' for c, stat in meas_agg_stats.columns]
        
        # Compute structural coverage features (sampling metadata)
        coverage_agg = pd.DataFrame(index=meas_agg.index)
        coverage_agg['total_obs_count'] = meas_df.groupby('patch_id').size()
        if 'station_id' in meas_df.columns:
            coverage_agg['distinct_station_count'] = meas_df.groupby('patch_id')['station_id'].nunique()
        else:
            coverage_agg['distinct_station_count'] = 1
        coverage_agg['mean_obs_per_station'] = coverage_agg['total_obs_count'] / coverage_agg['distinct_station_count']
        
        # Merge stats, uncertainty metrics, and coverage metadata
        meas_features = pd.concat([meas_agg_stats, sem_df, coverage_agg], axis=1).reset_index()
        base_df = base_df.merge(meas_features, on='patch_id', how='left')


    # Join tables with key 'survey_unit_id'
    if 'survey_unit_id' in base_df.columns:
        survey_files = ['county_soil_atlas.csv', 'nrcs_soil_map.csv', 'usfs_soil_survey.csv', 'sensor_calibration.csv']
        for f in survey_files:
            fpath = os.path.join(input_dir, f)
            if os.path.exists(fpath):
                sub_df = pd.read_csv(fpath)
                base_df = base_df.merge(sub_df, on='survey_unit_id', how='left')

    # Join tables with key 'parcel_id'
    if 'parcel_id' in base_df.columns:
        parcel_files = ['parcel_land_status.csv', 'parcel_soil_addendum.csv', 'parcel_soil_records.csv']
        for f in parcel_files:
            fpath = os.path.join(input_dir, f)
            if os.path.exists(fpath):
                sub_df = pd.read_csv(fpath)
                base_df = base_df.merge(sub_df, on='parcel_id', how='left')

        # Aggregate 1-to-many parcel table 'plot_notes.csv'
        notes_path = os.path.join(input_dir, 'plot_notes.csv')
        if os.path.exists(notes_path):
            notes_df = pd.read_csv(notes_path)
            notes_num_cols = [c for c in notes_df.columns if c not in ['parcel_id', 'note_no'] and pd.api.types.is_numeric_dtype(notes_df[c])]
            if notes_num_cols:
                notes_agg = notes_df.groupby('parcel_id')[notes_num_cols].agg(['mean', 'std', 'min', 'max', 'count'])
                notes_agg.columns = [f'notes_{c}_{stat}' for c, stat in notes_agg.columns]
                notes_agg = notes_agg.reset_index()
                base_df = base_df.merge(notes_agg, on='parcel_id', how='left')

    # Join tables with key 'stand_id'
    if 'stand_id' in base_df.columns:
        stand_files = ['stand_land_status.csv', 'stand_soil_atlas.csv', 'stand_soil_records.csv']
        for f in stand_files:
            fpath = os.path.join(input_dir, f)
            if os.path.exists(fpath):
                sub_df = pd.read_csv(fpath)
                base_df = base_df.merge(sub_df, on='stand_id', how='left')


    # Spatial & Cartographic Feature Engineering
    h_dist_col = get_matching_col(base_df, ['hydrolog', 'h', 'mean']) or get_matching_col(base_df, ['hydrolog', 'mean'])
    v_dist_col = get_matching_col(base_df, ['hydrolog', 'v', 'mean']) or get_matching_col(base_df, ['hydrolog', 'mean'])
    elev_col = get_matching_col(base_df, ['elevation', 'mean']) or get_matching_col(base_df, ['elev', 'mean'])
    noon_col = get_matching_col(base_df, ['noon', 'mean'])
    am_col = get_matching_col(base_df, ['9am', 'mean'])
    pm_col = get_matching_col(base_df, ['3pm', 'mean'])
    slope_col = get_matching_col(base_df, ['slope', 'mean'])
    aspect_col = get_matching_col(base_df, ['aspect', 'mean'])
    road_col = get_matching_col(base_df, ['road', 'mean'])
    fire_col = get_matching_col(base_df, ['fire', 'mean'])

    if h_dist_col and v_dist_col:
        base_df['euclidean_distance_to_hydrology'] = np.sqrt(base_df[h_dist_col]**2 + base_df[v_dist_col]**2)

    if elev_col and v_dist_col:
        base_df['hydrology_elevation_diff'] = base_df[elev_col] - base_df[v_dist_col]
        base_df['hydrology_elevation_sum'] = base_df[elev_col] + base_df[v_dist_col]

    if am_col and noon_col and pm_col:
        hs_cols = [am_col, noon_col, pm_col]
        base_df['hillshade_mean'] = base_df[hs_cols].mean(axis=1)
        base_df['hillshade_range'] = base_df[hs_cols].max(axis=1) - base_df[hs_cols].min(axis=1)
        base_df['hillshade_std'] = base_df[hs_cols].std(axis=1)
        base_df['hillshade_ratio_pm_am'] = (base_df[pm_col] + 1.0) / (base_df[am_col] + 1.0)

    if noon_col and am_col:
        base_df['hillshade_ratio_noon_9am'] = (base_df[noon_col] + 1.0) / (base_df[am_col] + 1.0)
        base_df['hillshade_diff_noon_9am'] = base_df[noon_col] - base_df[am_col]

    if noon_col and pm_col:
        base_df['hillshade_diff_3pm_noon'] = base_df[pm_col] - base_df[noon_col]

    if aspect_col:
        base_df['aspect_sin'] = np.sin(np.radians(base_df[aspect_col]))
        base_df['aspect_cos'] = np.cos(np.radians(base_df[aspect_col]))

    if slope_col:
        base_df['slope_sin'] = np.sin(np.radians(base_df[slope_col]))
        base_df['slope_cos'] = np.cos(np.radians(base_df[slope_col]))
        if elev_col:
            base_df['elevation_slope_interaction'] = base_df[elev_col] * np.sin(np.radians(base_df[slope_col].fillna(0)))

    if aspect_col and slope_col:
        base_df['aspect_slope_sin_vec'] = base_df['aspect_sin'] * base_df['slope_sin']
        base_df['aspect_slope_cos_vec'] = base_df['aspect_cos'] * base_df['slope_sin']

    if road_col and fire_col:
        base_df['road_fire_dist_sum'] = base_df[road_col] + base_df[fire_col]
        base_df['road_fire_dist_diff'] = base_df[road_col] - base_df[fire_col]

    if road_col and h_dist_col:
        base_df['road_hydro_dist_diff'] = base_df[road_col] - base_df[h_dist_col]

    dist_cols = [c for c in [road_col, fire_col, h_dist_col] if c is not None]
    if dist_cols:
        for c in dist_cols:
            base_df[f'{c}_log1p'] = np.log1p(np.maximum(0, base_df[c]))
        if len(dist_cols) > 1:
            base_df['distance_infrastructure_mean'] = base_df[dist_cols].mean(axis=1)
            base_df['distance_infrastructure_min'] = base_df[dist_cols].min(axis=1)
            base_df['distance_infrastructure_max'] = base_df[dist_cols].max(axis=1)


    # Target definition
    y = (base_df['class'] == 2).astype(int)
    groups = base_df['patch_id']

    # Select feature columns (excluding all identifier columns)
    id_cols = ['patch_id', 'class', 'survey_unit_id', 'parcel_id', 'stand_id', 'station_id', 'obs_no', 'note_no']
    feature_cols = [c for c in base_df.columns if c not in id_cols]

    cat_cols = [c for c in feature_cols if base_df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(base_df[c])]

    # Prepare datasets for CatBoost and LightGBM
    X_cb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].fillna('missing').astype(str)

    X_lgb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype('category')

    # Safety checks
    assert len(X_cb) == 423680, f"row count changed: {len(X_cb)}"

    # 3-Fold Stratified Group Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    cb_oof = np.zeros(len(base_df))
    lgb_oof = np.zeros(len(base_df))

    cb_params = {
        'iterations': 1500,
        'learning_rate': 0.05,
        'depth': 7,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0,
        'thread_count': -1
    }

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_cb, y, groups=groups)):
        # Train CatBoost
        X_tr_cb, y_tr = X_cb.iloc[train_idx], y.iloc[train_idx]
        X_va_cb, y_va = X_cb.iloc[val_idx], y.iloc[val_idx]

        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(
            X_tr_cb, y_tr,
            eval_set=(X_va_cb, y_va),
            cat_features=cat_cols if len(cat_cols) > 0 else None,
            early_stopping_rounds=50,
            verbose=False
        )
        cb_oof[val_idx] = cb_model.predict_proba(X_va_cb)[:, 1]

        # Train LightGBM
        X_tr_lgb = X_lgb.iloc[train_idx]
        X_va_lgb = X_lgb.iloc[val_idx]

        lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
        lgb_model.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_oof[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]

    oof_preds = 0.5 * cb_oof + 0.5 * lgb_oof
    final_auc = roc_auc_score(y, oof_preds)
    print(f"Final Validation Performance: {final_auc:.5f}")

if __name__ == "__main__":
    main()
