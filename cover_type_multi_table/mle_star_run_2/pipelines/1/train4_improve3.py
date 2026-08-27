
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


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
  patch_1to1_files = [
      'parcels.csv',
      'stands.csv',
      'survey_units.csv',
      'soil_registry.csv',
  ]
  for f in patch_1to1_files:
    fpath = os.path.join(input_dir, f)
    if os.path.exists(fpath):
      sub_df = pd.read_csv(fpath)
      base_df = base_df.merge(sub_df, on='patch_id', how='left')

  # Aggregate and join 1-to-many table 'patch_measurements.csv'
  meas_path = os.path.join(input_dir, 'patch_measurements.csv')
  if os.path.exists(meas_path):
    meas_df = pd.read_csv(meas_path)
    meas_num_cols = [
        c
        for c in meas_df.columns
        if c not in ['patch_id', 'obs_no', 'station_id']
        and pd.api.types.is_numeric_dtype(meas_df[c])
    ]
    meas_agg = meas_df.groupby('patch_id')[meas_num_cols].agg(
        ['mean', 'std', 'min', 'max']
    )
    meas_agg.columns = [f'{c}_{stat}' for c, stat in meas_agg.columns]
    meas_agg = meas_agg.reset_index()
    base_df = base_df.merge(meas_agg, on='patch_id', how='left')

  # Helper to process, aggregate, and merge secondary tables safely
  def process_and_merge(df, file_list, key_col):
    for f in file_list:
      fpath = os.path.join(input_dir, f)
      if not os.path.exists(fpath):
        continue
      sub_df = pd.read_csv(fpath)
      if key_col not in sub_df.columns:
        continue

      prefix = f.replace('.csv', '')
      is_one_to_many = sub_df[key_col].duplicated().any()

      if is_one_to_many:
        # 1-to-many table: compute group aggregations, value counts, & missingness flags
        val_cols = [c for c in sub_df.columns if c != key_col]
        num_cols = [
            c for c in val_cols if pd.api.types.is_numeric_dtype(sub_df[c])
        ]
        cat_cols = [
            c for c in val_cols if not pd.api.types.is_numeric_dtype(sub_df[c])
        ]

        agg_parts = []
        if num_cols:
          n_agg = sub_df.groupby(key_col)[num_cols].agg(
              ['mean', 'std', 'min', 'max']
          )
          n_agg.columns = [f'{prefix}_{c}_{stat}' for c, stat in n_agg.columns]
          agg_parts.append(n_agg)

        if cat_cols:
          c_agg = sub_df.groupby(key_col)[cat_cols].agg(['nunique'])
          c_agg.columns = [f'{prefix}_{c}_nunique' for c, stat in c_agg.columns]
          agg_parts.append(c_agg)

        if val_cols:
          null_agg = sub_df.isnull().groupby(sub_df[key_col])[val_cols].sum()
          null_agg.columns = [
              f'{prefix}_{c}_null_cnt' for c in null_agg.columns
          ]
          agg_parts.append(null_agg)

        if agg_parts:
          sub_agg = pd.concat(agg_parts, axis=1).reset_index()
          overlap = [
              c for c in sub_agg.columns if c in df.columns and c != key_col
          ]
          if overlap:
            sub_agg = sub_agg.drop(columns=overlap)
          df = df.merge(sub_agg, on=key_col, how='left')
      else:
        # 1-to-1 table: deduplicate overlapping columns prior to merge to prevent suffixes
        overlap = [
            c for c in sub_df.columns if c in df.columns and c != key_col
        ]
        if overlap:
          sub_df = sub_df.drop(columns=overlap)
        df = df.merge(sub_df, on=key_col, how='left')
    return df

  # Join tables with key 'survey_unit_id'
  if 'survey_unit_id' in base_df.columns:
    survey_files = [
        'county_soil_atlas.csv',
        'nrcs_soil_map.csv',
        'usfs_soil_survey.csv',
        'sensor_calibration.csv',
    ]
    base_df = process_and_merge(base_df, survey_files, 'survey_unit_id')

  # Join tables with key 'parcel_id'
  if 'parcel_id' in base_df.columns:
    parcel_files = [
        'parcel_land_status.csv',
        'parcel_soil_addendum.csv',
        'parcel_soil_records.csv',
        'plot_notes.csv',
    ]
    base_df = process_and_merge(base_df, parcel_files, 'parcel_id')

  # Join tables with key 'stand_id'
  if 'stand_id' in base_df.columns:
    stand_files = [
        'stand_land_status.csv',
        'stand_soil_atlas.csv',
        'stand_soil_records.csv',
    ]
    base_df = process_and_merge(base_df, stand_files, 'stand_id')

  # Spatial hierarchy context features (stand and survey unit level baselines)
  spatial_keys = [
      k for k in ['stand_id', 'survey_unit_id'] if k in base_df.columns
  ]
  for key in spatial_keys:
    num_cols = [
        c
        for c in base_df.columns
        if c != key
        and pd.api.types.is_numeric_dtype(base_df[c])
        and not c.endswith('_id')
    ]
    if num_cols:
      target_cols = num_cols[:20]
      mean_grouped = base_df.groupby(key)[target_cols].transform('mean')
      std_grouped = base_df.groupby(key)[target_cols].transform('std')
      mean_grouped.columns = [f'{key}_ctx_{c}_mean' for c in target_cols]
      std_grouped.columns = [f'{key}_ctx_{c}_std' for c in target_cols]
      base_df = pd.concat([base_df, mean_grouped, std_grouped], axis=1)

  # Column pruning and variance thresholding post-merge
  cols_to_drop = [c for c in base_df.columns if base_df[c].isna().all()]
  for c in base_df.columns:
    if c not in cols_to_drop and pd.api.types.is_numeric_dtype(base_df[c]):
      var_val = base_df[c].var()
      if pd.isna(var_val) or var_val == 0:
        cols_to_drop.append(c)

  if cols_to_drop:
    base_df = base_df.drop(columns=list(set(cols_to_drop)))

  # Spatial & Cartographic Feature Engineering
  h_dist_col = get_matching_col(
      base_df, ['hydrolog', 'h', 'mean']
  ) or get_matching_col(base_df, ['hydrolog', 'mean'])
  v_dist_col = get_matching_col(
      base_df, ['hydrolog', 'v', 'mean']
  ) or get_matching_col(base_df, ['hydrolog', 'mean'])
  elev_col = get_matching_col(
      base_df, ['elevation', 'mean']
  ) or get_matching_col(base_df, ['elev', 'mean'])
  noon_col = get_matching_col(base_df, ['noon', 'mean'])
  am_col = get_matching_col(base_df, ['9am', 'mean'])
  pm_col = get_matching_col(base_df, ['3pm', 'mean'])
  slope_col = get_matching_col(base_df, ['slope', 'mean'])
  aspect_col = get_matching_col(base_df, ['aspect', 'mean'])
  road_col = get_matching_col(base_df, ['road', 'mean'])
  fire_col = get_matching_col(base_df, ['fire', 'mean'])

  eps = 1e-6

  if h_dist_col and v_dist_col:
    base_df['euclidean_distance_to_hydrology'] = np.sqrt(
        base_df[h_dist_col] ** 2 + base_df[v_dist_col] ** 2
    )
    base_df['hydrology_slope_gradient'] = np.arctan2(
        base_df[v_dist_col], base_df[h_dist_col]
    )

  if elev_col and v_dist_col:
    base_df['hydrology_water_elevation'] = (
        base_df[elev_col] - base_df[v_dist_col]
    )
    base_df['hydrology_elevation_diff'] = (
        base_df[elev_col] - base_df[v_dist_col]
    )
    base_df['hydrology_elevation_sum'] = (
        base_df[elev_col] + base_df[v_dist_col]
    )

  if am_col and noon_col and pm_col:
    hs_cols = [am_col, noon_col, pm_col]
    base_df['hillshade_mean'] = base_df[hs_cols].mean(axis=1)
    base_df['hillshade_range'] = base_df[hs_cols].max(axis=1) - base_df[
        hs_cols
    ].min(axis=1)
    base_df['hillshade_std'] = base_df[hs_cols].std(axis=1)
    base_df['hillshade_norm_diff_pm_am'] = (
        base_df[pm_col] - base_df[am_col]
    ) / (base_df[pm_col] + base_df[am_col] + eps)

  if noon_col and am_col:
    base_df['hillshade_norm_diff_noon_9am'] = (
        base_df[noon_col] - base_df[am_col]
    ) / (base_df[noon_col] + base_df[am_col] + eps)
    base_df['hillshade_diff_noon_9am'] = base_df[noon_col] - base_df[am_col]

  if noon_col and pm_col:
    base_df['hillshade_norm_diff_3pm_noon'] = (
        base_df[pm_col] - base_df[noon_col]
    ) / (base_df[pm_col] + base_df[noon_col] + eps)
    base_df['hillshade_diff_3pm_noon'] = base_df[pm_col] - base_df[noon_col]

  if aspect_col:
    base_df['aspect_sin'] = np.sin(np.radians(base_df[aspect_col]))
    base_df['aspect_cos'] = np.cos(np.radians(base_df[aspect_col]))
    base_df['northness'] = base_df['aspect_cos']
    base_df['eastness'] = base_df['aspect_sin']

  if slope_col:
    base_df['slope_sin'] = np.sin(np.radians(base_df[slope_col]))
    base_df['slope_cos'] = np.cos(np.radians(base_df[slope_col]))
    if elev_col:
      base_df['elevation_slope_interaction'] = base_df[elev_col] * np.sin(
          np.radians(base_df[slope_col].fillna(0))
      )

  if aspect_col and slope_col:
    base_df['northness_slope'] = base_df['northness'] * base_df['slope_sin']
    base_df['eastness_slope'] = base_df['eastness'] * base_df['slope_sin']
    base_df['aspect_slope_sin_vec'] = (
        base_df['aspect_sin'] * base_df['slope_sin']
    )
    base_df['aspect_slope_cos_vec'] = (
        base_df['aspect_cos'] * base_df['slope_sin']
    )

  if road_col and fire_col:
    base_df['road_fire_dist_sum'] = base_df[road_col] + base_df[fire_col]
    base_df['road_fire_dist_diff'] = base_df[road_col] - base_df[fire_col]
    base_df['road_fire_norm_diff'] = (
        base_df[road_col] - base_df[fire_col]
    ) / (base_df[road_col] + base_df[fire_col] + eps)

  if road_col and h_dist_col:
    base_df['road_hydro_dist_diff'] = base_df[road_col] - base_df[h_dist_col]
    base_df['road_hydro_norm_diff'] = (
        base_df[road_col] - base_df[h_dist_col]
    ) / (base_df[road_col] + base_df[h_dist_col] + eps)

  if fire_col and h_dist_col:
    base_df['fire_hydro_norm_diff'] = (
        base_df[fire_col] - base_df[h_dist_col]
    ) / (base_df[fire_col] + base_df[h_dist_col] + eps)

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
  id_cols = [
      'patch_id',
      'class',
      'survey_unit_id',
      'parcel_id',
      'stand_id',
      'station_id',
      'obs_no',
      'note_no',
  ]
  feature_cols = [c for c in base_df.columns if c not in id_cols]

  cat_cols = [
      c
      for c in feature_cols
      if base_df[c].dtype == 'object'
      or pd.api.types.is_categorical_dtype(base_df[c])
  ]

  # Prepare datasets for CatBoost and LightGBM
  X_cb = base_df[feature_cols].copy()
  for c in cat_cols:
    X_cb[c] = X_cb[c].fillna('missing').astype(str)

  X_lgb = base_df[feature_cols].copy()
  for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype('category')

  # Safety checks
  assert len(X_cb) == 423680, f'row count changed: {len(X_cb)}'

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
      'thread_count': -1,
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
      'verbose': -1,
  }

  for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_cb, y, groups=groups)):
    # Train CatBoost
    X_tr_cb, y_tr = X_cb.iloc[train_idx], y.iloc[train_idx]
    X_va_cb, y_va = X_cb.iloc[val_idx], y.iloc[val_idx]

    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_va_cb, y_va),
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_oof[val_idx] = cb_model.predict_proba(X_va_cb)[:, 1]

    # Train LightGBM
    X_tr_lgb = X_lgb.iloc[train_idx]
    X_va_lgb = X_lgb.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_oof[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]

  oof_preds = 0.5 * cb_oof + 0.5 * lgb_oof
  final_auc = roc_auc_score(y, oof_preds)
  print(f'Final Validation Performance: {final_auc:.5f}')


if __name__ == '__main__':
  main()
