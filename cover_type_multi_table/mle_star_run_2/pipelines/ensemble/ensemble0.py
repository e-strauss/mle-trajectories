
import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata


def get_matching_col(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        if all(k in col_lower for k in keywords):
            return col
    return None


def _aggregate_and_prep(file_path, join_key, prefix):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    if join_key not in df.columns:
        return None

    num_cols = [c for c in df.columns if c != join_key and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != join_key and not pd.api.types.is_numeric_dtype(df[c])]

    if df[join_key].duplicated().any():
        agg_dfs = []
        if num_cols:
            num_agg = df.groupby(join_key)[num_cols].agg(['mean', 'std', 'min', 'max'])
            num_agg.columns = [f"{prefix}_{c}_{stat}" for c, stat in num_agg.columns]
            agg_dfs.append(num_agg)

        if cat_cols:
            cat_dict = {}
            for c in cat_cols:
                cat_dict[f"{prefix}_{c}_first"] = df.groupby(join_key)[c].first()
                cat_dict[f"{prefix}_{c}_nunique"] = df.groupby(join_key)[c].nunique()
            agg_dfs.append(pd.DataFrame(cat_dict))

        count_s = df.groupby(join_key).size().rename(f"{prefix}_record_count")
        agg_dfs.append(count_s.to_frame())

        res = pd.concat(agg_dfs, axis=1).reset_index()
    else:
        res = df.copy()
        res.columns = [c if c == join_key else f"{prefix}_{c}" for c in res.columns]

    return res


def optimize_weights(preds_matrix, y_true):
    num_models = preds_matrix.shape[1]

    def loss_func(weights):
        w = weights / np.sum(weights)
        blend = np.dot(preds_matrix, w)
        return -roc_auc_score(y_true, blend)

    init_weights = np.ones(num_models) / num_models
    bounds = [(0, 1)] * num_models
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

    res = minimize(
        loss_func,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    best_weights = res.x / np.sum(res.x)
    return best_weights


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
            dup_cols = [c for c in sub_df.columns if c != 'patch_id' and c in base_df.columns]
            if dup_cols:
                sub_df = sub_df.drop(columns=dup_cols)
            base_df = base_df.merge(sub_df, on='patch_id', how='left')

    # Aggregate and join 1-to-many table 'patch_measurements.csv'
    meas_path = os.path.join(input_dir, 'patch_measurements.csv')
    if os.path.exists(meas_path):
        meas_df = pd.read_csv(meas_path)
        meas_num_cols = [c for c in meas_df.columns if c not in ['patch_id', 'obs_no', 'station_id'] and pd.api.types.is_numeric_dtype(meas_df[c])]
        meas_agg = meas_df.groupby('patch_id')[meas_num_cols].agg(['mean', 'std', 'min', 'max'])
        meas_agg.columns = [f'{c}_{stat}' for c, stat in meas_agg.columns]
        meas_agg = meas_agg.reset_index()
        base_df = base_df.merge(meas_agg, on='patch_id', how='left')

    # Join and aggregate tables with key 'survey_unit_id'
    if 'survey_unit_id' in base_df.columns:
        survey_files = ['county_soil_atlas.csv', 'nrcs_soil_map.csv', 'usfs_soil_survey.csv', 'sensor_calibration.csv']
        for f in survey_files:
            fpath = os.path.join(input_dir, f)
            prefix = f.replace('.csv', '')
            sub_df = _aggregate_and_prep(fpath, 'survey_unit_id', prefix)
            if sub_df is not None:
                dup_cols = [c for c in sub_df.columns if c != 'survey_unit_id' and c in base_df.columns]
                if dup_cols:
                    sub_df = sub_df.drop(columns=dup_cols)
                base_df = base_df.merge(sub_df, on='survey_unit_id', how='left')

    # Join and aggregate tables with key 'parcel_id'
    if 'parcel_id' in base_df.columns:
        parcel_files = ['parcel_land_status.csv', 'parcel_soil_addendum.csv', 'parcel_soil_records.csv', 'plot_notes.csv']
        for f in parcel_files:
            fpath = os.path.join(input_dir, f)
            prefix = f.replace('.csv', '')
            sub_df = _aggregate_and_prep(fpath, 'parcel_id', prefix)
            if sub_df is not None:
                dup_cols = [c for c in sub_df.columns if c != 'parcel_id' and c in base_df.columns]
                if dup_cols:
                    sub_df = sub_df.drop(columns=dup_cols)
                base_df = base_df.merge(sub_df, on='parcel_id', how='left')

    # Join and aggregate tables with key 'stand_id'
    if 'stand_id' in base_df.columns:
        stand_files = ['stand_land_status.csv', 'stand_soil_atlas.csv', 'stand_soil_records.csv']
        for f in stand_files:
            fpath = os.path.join(input_dir, f)
            prefix = f.replace('.csv', '')
            sub_df = _aggregate_and_prep(fpath, 'stand_id', prefix)
            if sub_df is not None:
                dup_cols = [c for c in sub_df.columns if c != 'stand_id' and c in base_df.columns]
                if dup_cols:
                    sub_df = sub_df.drop(columns=dup_cols)
                base_df = base_df.merge(sub_df, on='stand_id', how='left')

    # Construct domain interaction features crossing terrain topography with soil & land status
    topo_keywords = ['elev', 'slope', 'hydro', 'water', 'topo', 'aspect', 'alt']
    soil_keywords = ['soil', 'clay', 'sand', 'silt', 'ph', 'organic', 'depth', 'matter', 'status']

    topo_cols = [c for c in base_df.columns if any(k in c.lower() for k in topo_keywords) and pd.api.types.is_numeric_dtype(base_df[c])]
    soil_cols = [c for c in base_df.columns if any(k in c.lower() for k in soil_keywords) and pd.api.types.is_numeric_dtype(base_df[c])]

    for t_col in topo_cols[:5]:
        for s_col in soil_cols[:5]:
            if t_col != s_col:
                mult_name = f"inter_{t_col}_x_{s_col}"
                div_name = f"inter_{t_col}_div_{s_col}"
                if mult_name not in base_df.columns:
                    base_df[mult_name] = base_df[t_col] * base_df[s_col]
                if div_name not in base_df.columns:
                    base_df[div_name] = base_df[t_col] / (base_df[s_col].abs() + 1e-5)

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

    eps = 1e-6

    if h_dist_col and v_dist_col:
        base_df['euclidean_distance_to_hydrology'] = np.sqrt(base_df[h_dist_col]**2 + base_df[v_dist_col]**2)
        base_df['hydrology_slope_gradient'] = np.arctan2(base_df[v_dist_col], base_df[h_dist_col])

    if elev_col and v_dist_col:
        base_df['hydrology_water_elevation'] = base_df[elev_col] - base_df[v_dist_col]
        base_df['hydrology_elevation_diff'] = base_df[elev_col] - base_df[v_dist_col]
        base_df['hydrology_elevation_sum'] = base_df[elev_col] + base_df[v_dist_col]

    if am_col and noon_col and pm_col:
        hs_cols = [am_col, noon_col, pm_col]
        base_df['hillshade_mean'] = base_df[hs_cols].mean(axis=1)
        base_df['hillshade_range'] = base_df[hs_cols].max(axis=1) - base_df[hs_cols].min(axis=1)
        base_df['hillshade_std'] = base_df[hs_cols].std(axis=1)
        base_df['hillshade_norm_diff_pm_am'] = (base_df[pm_col] - base_df[am_col]) / (base_df[pm_col] + base_df[am_col] + eps)

    if noon_col and am_col:
        base_df['hillshade_norm_diff_noon_9am'] = (base_df[noon_col] - base_df[am_col]) / (base_df[noon_col] + base_df[am_col] + eps)
        base_df['hillshade_diff_noon_9am'] = base_df[noon_col] - base_df[am_col]

    if noon_col and pm_col:
        base_df['hillshade_norm_diff_3pm_noon'] = (base_df[pm_col] - base_df[noon_col]) / (base_df[pm_col] + base_df[noon_col] + eps)
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
            base_df['elevation_slope_interaction'] = base_df[elev_col] * np.sin(np.radians(base_df[slope_col].fillna(0)))

    if aspect_col and slope_col:
        base_df['northness_slope'] = base_df['northness'] * base_df['slope_sin']
        base_df['eastness_slope'] = base_df['eastness'] * base_df['slope_sin']
        base_df['aspect_slope_sin_vec'] = base_df['aspect_sin'] * base_df['slope_sin']
        base_df['aspect_slope_cos_vec'] = base_df['aspect_cos'] * base_df['slope_sin']

    if road_col and fire_col:
        base_df['road_fire_dist_sum'] = base_df[road_col] + base_df[fire_col]
        base_df['road_fire_dist_diff'] = base_df[road_col] - base_df[fire_col]
        base_df['road_fire_norm_diff'] = (base_df[road_col] - base_df[fire_col]) / (base_df[road_col] + base_df[fire_col] + eps)

    if road_col and h_dist_col:
        base_df['road_hydro_dist_diff'] = base_df[road_col] - base_df[h_dist_col]
        base_df['road_hydro_norm_diff'] = (base_df[road_col] - base_df[h_dist_col]) / (base_df[road_col] + base_df[h_dist_col] + eps)

    if fire_col and h_dist_col:
        base_df['fire_hydro_norm_diff'] = (base_df[fire_col] - base_df[h_dist_col]) / (base_df[fire_col] + base_df[h_dist_col] + eps)

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

    # Prepare datasets for CatBoost, LightGBM, and XGBoost
    X_cb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].fillna('missing').astype(str)

    X_lgb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype('category')

    X_xgb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].fillna('missing').astype('category').cat.codes

    # Safety checks
    assert len(X_cb) == 423680, f"row count changed: {len(X_cb)}"

    # 3-Fold Stratified Group Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    cb_oof = np.zeros(len(base_df))
    lgb_oof = np.zeros(len(base_df))
    xgb_oof = np.zeros(len(base_df))

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

    xgb_params = {
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_cb, y, groups=groups)):
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost
        X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(
            X_tr_cb, y_tr,
            eval_set=(X_va_cb, y_va),
            cat_features=cat_cols if len(cat_cols) > 0 else None,
            early_stopping_rounds=50,
            verbose=False
        )
        cb_oof[val_idx] = cb_model.predict_proba(X_va_cb)[:, 1]

        # LightGBM
        X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
        lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
        lgb_model.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_oof[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]

        # XGBoost
        X_tr_xgb, X_va_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr_xgb, y_tr,
            eval_set=[(X_va_xgb, y_va)],
            verbose=False
        )
        xgb_oof[val_idx] = xgb_model.predict_proba(X_va_xgb)[:, 1]

    # Evaluate Individual Model Performances
    cb_auc = roc_auc_score(y, cb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    xgb_auc = roc_auc_score(y, xgb_oof)

    print(f"CatBoost OOF AUC: {cb_auc:.5f}")
    print(f"LightGBM OOF AUC: {lgb_auc:.5f}")
    print(f"XGBoost OOF AUC:  {xgb_auc:.5f}")

    # Raw probability matrix
    raw_matrix = np.column_stack([cb_oof, lgb_oof, xgb_oof])
    raw_weights = optimize_weights(raw_matrix, y)
    raw_blend = np.dot(raw_matrix, raw_weights)
    raw_blend_auc = roc_auc_score(y, raw_blend)

    # Rank-transformed probability matrix
    cb_rank = rankdata(cb_oof) / len(cb_oof)
    lgb_rank = rankdata(lgb_oof) / len(lgb_oof)
    xgb_rank = rankdata(xgb_oof) / len(xgb_oof)

    rank_matrix = np.column_stack([cb_rank, lgb_rank, xgb_rank])
    rank_weights = optimize_weights(rank_matrix, y)
    rank_blend = np.dot(rank_matrix, rank_weights)
    rank_blend_auc = roc_auc_score(y, rank_blend)

    # Choose best performing strategy
    if rank_blend_auc > raw_blend_auc:
        final_auc = rank_blend_auc
        best_weights = rank_weights
        mode = "Rank Average Stacking"
    else:
        final_auc = raw_blend_auc
        best_weights = raw_weights
        mode = "Probability Stacking"

    print(f"Ensemble Strategy: {mode}")
    print(f"Optimized Model Weights (CatBoost, LightGBM, XGBoost): {np.round(best_weights, 4)}")
    print(f"Final Validation Performance: {final_auc:.5f}")


if __name__ == "__main__":
    main()
