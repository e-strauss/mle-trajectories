
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize


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

    # Prepare datasets for different model families
    X_cb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].fillna('missing').astype(str)

    X_lgb = base_df[feature_cols].copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype('category')

    X_encoded = base_df[feature_cols].copy()
    for c in cat_cols:
        X_encoded[c] = X_encoded[c].astype('category').cat.codes

    # Safety checks
    assert len(X_cb) == 423680, f"row count changed: {len(X_cb)}"

    # 3-Fold Stratified Group Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    cb_oof = np.zeros(len(base_df))
    lgb_oof = np.zeros(len(base_df))
    xgb_oof = np.zeros(len(base_df))
    hgb_oof = np.zeros(len(base_df))

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
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }

    hgb_params = {
        'max_iter': 400,
        'learning_rate': 0.05,
        'max_depth': 8,
        'random_state': 42,
        'early_stopping': True,
        'n_iter_no_change': 30,
        'scoring': 'roc_auc'
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_cb, y, groups=groups)):
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # 1. CatBoost Classifier
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

        # 2. LightGBM Classifier
        X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
        lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
        lgb_model.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_oof[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]

        # 3. XGBoost Classifier
        X_tr_enc, X_va_enc = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_va_enc, y_va)],
            verbose=False
        )
        xgb_oof[val_idx] = xgb_model.predict_proba(X_va_enc)[:, 1]

        # 4. HistGradientBoosting Classifier
        hgb_model = HistGradientBoostingClassifier(**hgb_params)
        hgb_model.fit(X_tr_enc, y_tr)
        hgb_oof[val_idx] = hgb_model.predict_proba(X_va_enc)[:, 1]

    oof_matrix = np.column_stack([cb_oof, lgb_oof, xgb_oof, hgb_oof])

    # Convert OOF predictions to normalized ranks for rank-based strategies
    oof_ranks = np.zeros_like(oof_matrix)
    for i in range(oof_matrix.shape[1]):
        oof_ranks[:, i] = rankdata(oof_matrix[:, i]) / len(oof_matrix)

    # Context-Aware Meta-Learner (Level-2 Stacking)
    top_context_cols = [c for c in [elev_col, slope_col, h_dist_col, v_dist_col, road_col, fire_col] if c is not None]
    context_data = base_df[top_context_cols].fillna(0).values if top_context_cols else np.empty((len(base_df), 0))

    if context_data.shape[1] > 0:
        scaler = StandardScaler()
        context_scaled = scaler.fit_transform(context_data)
        Z_stack = np.column_stack([oof_matrix, context_scaled])
    else:
        Z_stack = oof_matrix

    meta_oof = np.zeros(len(base_df))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr_i, va_i in skf.split(Z_stack, y):
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta_lr.fit(Z_stack[tr_i], y.iloc[tr_i])
        meta_oof[va_i] = meta_lr.predict_proba(Z_stack[va_i])[:, 1]

    auc_meta = roc_auc_score(y, meta_oof)

    # Rank Soft-Voting Optimization (SLSQP / Nelder-Mead)
    def rank_voting_obj(w_raw):
        w = np.exp(w_raw) / np.sum(np.exp(w_raw))
        pred = np.dot(oof_ranks, w)
        return -roc_auc_score(y, pred)

    res_voting = minimize(rank_voting_obj, x0=np.zeros(4), method='Nelder-Mead', options={'maxiter': 300})
    w_voting = np.exp(res_voting.x) / np.sum(np.exp(res_voting.x))
    oof_rank_voting = np.dot(oof_ranks, w_voting)
    auc_rank_voting = roc_auc_score(y, oof_rank_voting)

    # Rank-Power Non-Linear Blending Optimization
    def rank_power_obj(params):
        w_raw = params[:4]
        p_raw = params[4:]
        w = np.exp(w_raw) / np.sum(np.exp(w_raw))
        p = np.exp(p_raw)
        powered_ranks = np.power(oof_ranks, p)
        pred = np.dot(powered_ranks, w)
        return -roc_auc_score(y, pred)

    res_power = minimize(rank_power_obj, x0=np.zeros(8), method='Nelder-Mead', options={'maxiter': 500})
    w_power = np.exp(res_power.x[:4]) / np.sum(np.exp(res_power.x[:4]))
    p_power = np.exp(res_power.x[4:])
    oof_rank_power = np.dot(np.power(oof_ranks, p_power), w_power)
    auc_rank_power = roc_auc_score(y, oof_rank_power)

    # Dynamic Strategy Selection
    scores = {
        'Context-Aware Logistic Meta-Learner': auc_meta,
        'SLSQP Rank Soft-Voting': auc_rank_voting,
        'Rank-Power Non-Linear Blend': auc_rank_power
    }

    best_strategy, final_auc = max(scores.items(), key=lambda item: item[1])
    print(f"Final Validation Performance: {final_auc:.5f}")


if __name__ == "__main__":
    main()
