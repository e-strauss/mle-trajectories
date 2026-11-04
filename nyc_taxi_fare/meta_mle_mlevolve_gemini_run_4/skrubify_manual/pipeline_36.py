"""Hand skrubify of pipeline_36 (expansion_f4c61b33e057_variant_1, the run's best
candidate) into a plan whose structure an optimizer can read.

The original is two opaque nodes: a stateless TaxiFeatureTransformer and a
TriEnsembleResidualRegressor that hides the residual target, a StandardScaler, a
hand-written torch training loop, LightGBM, XGBoost, the blend and the clip. Here:

- features: every column is a recorded DataOp, grouped in named blocks
  (raw_coords, trip_geometry, airports, boroughs, crossings, landmarks,
  pickup_time, fare_rules, harmonics) and reordered to the original's column
  order (column order feeds LightGBM/XGBoost colsample).
- model: residual_target = y - statutory_fare_estimate; resnet, lgbm and xgb are
  separate .skb.apply nodes; the blend and clip are one eval-mode-gated function
  (blend_predictions), the only plan-defined function in the graph.
- resnet: a plain skorch NeuralNetRegressor, no overridden loop methods.
  in_features is len(FEATURE_ORDER), fixed by the plan (the original read it from
  the data; a DataOp such as scaled.shape[1] works in skrub, which evaluates
  DataOps in estimator params, but not in stratum's scheduler). The per-batch
  cosine schedule (T_max = epochs x batches of this fold) and the head-bias start
  at mean(y) are callbacks reading the fit's own data; TorchSeed seeds torch
  before the module is built, as the original did at the top of fit (same
  initial weights and batch order).

Dropped: the XGBoost CPU fallback on a failed GPU fit (the device is chosen once),
the no-op np.random.seed and the second nan_to_num inside the ResNet wrapper.
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
N_JOBS = 8
W_RESNET, W_LGB, W_XGB = 0.34, 0.33, 0.33
FARE_FLOOR, FARE_CAP = 2.5, 500.0

ROT_RAD = 29.0 * np.pi / 180.0
COS_ROT = np.cos(ROT_RAD)
SIN_ROT = np.sin(ROT_RAD)
MEAN_LAT = np.radians(40.75)
WAVELENGTHS = (1.0, 2.5, 5.0, 10.0)

FEATURE_ORDER = (
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
    'dlon', 'dlat', 'abs_dlon', 'abs_dlat', 'haversine_km', 'manhattan_km', 'euclidean_km',
    'rot_dx', 'rot_dy', 'rot_pick_x', 'rot_pick_y', 'rot_drop_x', 'rot_drop_y',
    'center_rot_x', 'center_rot_y', 'center_dist_midtown',
    'avenue_dist_frac', 'crosstown_dist_frac', 'tortuosity', 'grid_bearing',
    'is_cp_crossing', 'cp_crossing_dist',
    'manhattan_crosstown_km', 'manhattan_avenue_km', 'manhattan_rush_crosstown', 'manhattan_midday_crosstown',
    'non_jfk_manhattan_km', 'non_jfk_haversine_km', 'post_2012_dist', 'post_2012_hav',
    'sqrt_manhattan_km', 'log_manhattan_km', 'sqrt_haversine_km', 'log_haversine_km',
    'rush_hour_dist', 'morning_rush_dist', 'midday_dist', 'overnight_dist', 'suburban_dist', 'post_2012_rush',
    'jfk_toll', 'jfk_flat_rate', 'ewr_surcharge', 'statutory_surcharge', 'statutory_fare_estimate',
    'jfk_p_dist', 'jfk_d_dist', 'is_jfk', 'is_jfk_manhattan',
    'lga_p_dist', 'lga_d_dist', 'is_lga',
    'ewr_p_dist', 'ewr_d_dist', 'is_ewr', 'is_ewr_manhattan',
    'p_manhattan', 'd_manhattan', 'is_both_manhattan', 'is_cross_borough',
    'is_manhattan_brooklyn', 'is_manhattan_queens', 'is_manhattan_bronx',
    'mid_midtown_tunnel', 'mid_rfk', 'mid_queensboro', 'mid_battery_tunnel', 'mid_brooklyn_bridge',
    'mid_lincoln', 'mid_holland', 'mid_gwb',
    'tolled_midtown_tunnel', 'tolled_rfk', 'tolled_battery_tunnel', 'tolled_hudson_tunnel',
    'free_queensboro', 'free_brooklyn',
    'midtown_p_dist', 'midtown_d_dist', 'fidi_p_dist', 'fidi_d_dist', 'gc_p_dist', 'gc_d_dist',
    'penn_p_dist', 'penn_d_dist', 'ues_p_dist', 'ues_d_dist', 'uws_p_dist', 'uws_d_dist',
    'bk_downtown_p_dist', 'bk_downtown_d_dist', 'lic_p_dist', 'lic_d_dist',
    'is_zero_dist', 'is_westchester', 'is_nassau', 'is_suburban',
    'year', 'month', 'day', 'hour', 'weekday', 'hour_float',
    'sin_diurnal', 'cos_diurnal', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'is_weekend', 'is_rush_hour', 'is_morning_rush', 'is_midday', 'is_overnight',
    'is_post_nov_2009', 'is_post_2012', 'is_post_2015',
) + tuple(f'{fn}_{end}_{axis}_{str(wl).replace(".", "_")}'
          for wl in WAVELENGTHS for end in ('pick', 'drop') for axis in ('x', 'y') for fn in ('sin', 'cos'))


def as_float(mask):
    return mask.astype('float64')


def in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
    return as_float((lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max))


def degree_dist(lon, lat, lon0, lat0):
    return ((lon - lon0) ** 2 + (lat - lat0) ** 2).skb.apply_func(np.sqrt)


def km_dist(lat, lon, lat0, lon0):
    return (((lat - lat0) * 111.13) ** 2 + ((lon - lon0) * 84.18) ** 2).skb.apply_func(np.sqrt)


def block(X, name, **columns):
    return X[[]].assign(**columns).skb.set_name(name)


def taxi_features(X):
    """The original TaxiFeatureTransformer.transform, one recorded op per step.
    Returns the feature frame and the statutory fare estimate (the residual baseline)."""
    p_lon = X['pickup_longitude'].astype('float64')
    p_lat = X['pickup_latitude'].astype('float64')
    d_lon = X['dropoff_longitude'].astype('float64')
    d_lat = X['dropoff_latitude'].astype('float64')
    passengers = X['passenger_count'].astype('float64')
    raw_coords = block(X, 'raw_coords', pickup_longitude=p_lon, pickup_latitude=p_lat,
                       dropoff_longitude=d_lon, dropoff_latitude=d_lat, passenger_count=passengers)

    # trip geometry: haversine, the 29-degree rotated Manhattan grid, bearing
    dlon = d_lon - p_lon
    dlat = d_lat - p_lat
    lat1, lon1 = p_lat.skb.apply_func(np.radians), p_lon.skb.apply_func(np.radians)
    lat2, lon2 = d_lat.skb.apply_func(np.radians), d_lon.skb.apply_func(np.radians)
    a = ((lat2 - lat1) / 2.0).skb.apply_func(np.sin) ** 2 + lat1.skb.apply_func(np.cos) * lat2.skb.apply_func(
        np.cos) * ((lon2 - lon1) / 2.0).skb.apply_func(np.sin) ** 2
    haversine_km = 6371.0 * 2.0 * a.clip(0.0, 1.0).skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)

    dx_km = dlon * 111.32 * np.cos(MEAN_LAT)
    dy_km = dlat * 110.57
    dx_rot = dx_km * COS_ROT - dy_km * SIN_ROT
    dy_rot = dx_km * SIN_ROT + dy_km * COS_ROT
    rot_dx = dx_rot.abs()
    rot_dy = dy_rot.abs()
    manhattan_km = rot_dx + rot_dy
    euclidean_km = (dx_km ** 2 + dy_km ** 2).skb.apply_func(np.sqrt)

    pick_x_km = (p_lon - -74.0) * 84.18
    pick_y_km = (p_lat - 40.7) * 111.13
    drop_x_km = (d_lon - -74.0) * 84.18
    drop_y_km = (d_lat - 40.7) * 111.13
    rot_pick_x = pick_x_km * COS_ROT - pick_y_km * SIN_ROT
    rot_pick_y = pick_x_km * SIN_ROT + pick_y_km * COS_ROT
    rot_drop_x = drop_x_km * COS_ROT - drop_y_km * SIN_ROT
    rot_drop_y = drop_x_km * SIN_ROT + drop_y_km * COS_ROT
    is_zero_dist = as_float(manhattan_km < 0.05)
    trip_geometry = block(
        X, 'trip_geometry', dlon=dlon, dlat=dlat, abs_dlon=dlon.abs(), abs_dlat=dlat.abs(),
        haversine_km=haversine_km, manhattan_km=manhattan_km, euclidean_km=euclidean_km,
        rot_dx=rot_dx, rot_dy=rot_dy, rot_pick_x=rot_pick_x, rot_pick_y=rot_pick_y,
        rot_drop_x=rot_drop_x, rot_drop_y=rot_drop_y,
        center_rot_x=(rot_pick_x + rot_drop_x) / 2.0, center_rot_y=(rot_pick_y + rot_drop_y) / 2.0,
        avenue_dist_frac=rot_dy / (manhattan_km + 0.05), crosstown_dist_frac=rot_dx / (manhattan_km + 0.05),
        tortuosity=manhattan_km / (haversine_km + 0.05),
        grid_bearing=dy_rot.skb.apply_func(np.arctan2, dx_rot), is_zero_dist=is_zero_dist)

    # boroughs (bounding boxes) and the Central Park transverse crossing
    p_manhattan = in_box(p_lon, p_lat, -74.02, -73.93, 40.7, 40.87)
    d_manhattan = in_box(d_lon, d_lat, -74.02, -73.93, 40.7, 40.87)
    is_both_manhattan = p_manhattan * d_manhattan
    p_brooklyn = in_box(p_lon, p_lat, -74.04, -73.85, 40.57, 40.73)
    d_brooklyn = in_box(d_lon, d_lat, -74.04, -73.85, 40.57, 40.73)
    p_queens = in_box(p_lon, p_lat, -73.96, -73.7, 40.54, 40.8)
    d_queens = in_box(d_lon, d_lat, -73.96, -73.7, 40.54, 40.8)
    p_bronx = in_box(p_lon, p_lat, -73.93, -73.76, 40.8, 40.91)
    d_bronx = in_box(d_lon, d_lat, -73.93, -73.76, 40.8, 40.91)
    is_manhattan_brooklyn = as_float((p_manhattan > 0) & (d_brooklyn > 0) | (d_manhattan > 0) & (p_brooklyn > 0))
    is_manhattan_queens = as_float((p_manhattan > 0) & (d_queens > 0) | (d_manhattan > 0) & (p_queens > 0))
    is_manhattan_bronx = as_float((p_manhattan > 0) & (d_bronx > 0) | (d_manhattan > 0) & (p_bronx > 0))
    p_ues = in_box(p_lon, p_lat, -73.965, -73.945, 40.765, 40.800)
    d_ues = in_box(d_lon, d_lat, -73.965, -73.945, 40.765, 40.800)
    p_uws = in_box(p_lon, p_lat, -73.988, -73.968, 40.765, 40.800)
    d_uws = in_box(d_lon, d_lat, -73.988, -73.968, 40.765, 40.800)
    is_cp_crossing = as_float(((p_ues > 0) & (d_uws > 0)) | ((p_uws > 0) & (d_ues > 0)))
    is_westchester = as_float(d_lat > 40.915)
    is_nassau = as_float(d_lon > -73.7)
    is_suburban = as_float((is_westchester > 0) | (is_nassau > 0))
    boroughs = block(
        X, 'boroughs', is_cp_crossing=is_cp_crossing, cp_crossing_dist=is_cp_crossing * rot_dx,
        p_manhattan=p_manhattan, d_manhattan=d_manhattan, is_both_manhattan=is_both_manhattan,
        is_cross_borough=as_float(p_manhattan != d_manhattan),
        is_manhattan_brooklyn=is_manhattan_brooklyn, is_manhattan_queens=is_manhattan_queens,
        is_manhattan_bronx=is_manhattan_bronx,
        is_westchester=is_westchester, is_nassau=is_nassau, is_suburban=is_suburban)

    # airports (degree distances, 0.03 radius)
    jfk_p_dist = degree_dist(p_lon, p_lat, -73.7781, 40.6413)
    jfk_d_dist = degree_dist(d_lon, d_lat, -73.7781, 40.6413)
    lga_p_dist = degree_dist(p_lon, p_lat, -73.874, 40.7769)
    lga_d_dist = degree_dist(d_lon, d_lat, -73.874, 40.7769)
    ewr_p_dist = degree_dist(p_lon, p_lat, -74.1745, 40.6895)
    ewr_d_dist = degree_dist(d_lon, d_lat, -74.1745, 40.6895)
    is_jfk_manhattan = as_float((jfk_p_dist < 0.03) & (d_manhattan > 0) | (jfk_d_dist < 0.03) & (p_manhattan > 0))
    is_ewr_manhattan = as_float((ewr_p_dist < 0.03) & (d_manhattan > 0) | (ewr_d_dist < 0.03) & (p_manhattan > 0))
    airports = block(
        X, 'airports', jfk_p_dist=jfk_p_dist, jfk_d_dist=jfk_d_dist,
        is_jfk=as_float((jfk_p_dist < 0.03) | (jfk_d_dist < 0.03)), is_jfk_manhattan=is_jfk_manhattan,
        lga_p_dist=lga_p_dist, lga_d_dist=lga_d_dist, is_lga=as_float((lga_p_dist < 0.03) | (lga_d_dist < 0.03)),
        ewr_p_dist=ewr_p_dist, ewr_d_dist=ewr_d_dist,
        is_ewr=as_float((ewr_p_dist < 0.03) | (ewr_d_dist < 0.03)), is_ewr_manhattan=is_ewr_manhattan)

    # bridges and tunnels: trip-midpoint distances, tolled / free crossings
    center_lat = (p_lat + d_lat) / 2.0
    center_lon = (p_lon + d_lon) / 2.0
    mid_midtown_tunnel = km_dist(center_lat, center_lon, 40.744, -73.968)
    mid_rfk = km_dist(center_lat, center_lon, 40.78, -73.924)
    mid_queensboro = km_dist(center_lat, center_lon, 40.757, -73.955)
    mid_battery_tunnel = km_dist(center_lat, center_lon, 40.697, -74.015)
    mid_brooklyn_bridge = km_dist(center_lat, center_lon, 40.706, -73.997)
    crossings = block(
        X, 'crossings', mid_midtown_tunnel=mid_midtown_tunnel, mid_rfk=mid_rfk, mid_queensboro=mid_queensboro,
        mid_battery_tunnel=mid_battery_tunnel, mid_brooklyn_bridge=mid_brooklyn_bridge,
        mid_lincoln=km_dist(center_lat, center_lon, 40.76, -74.004),
        mid_holland=km_dist(center_lat, center_lon, 40.726, -74.016),
        mid_gwb=km_dist(center_lat, center_lon, 40.852, -73.953),
        tolled_midtown_tunnel=as_float((is_manhattan_queens > 0) & (mid_midtown_tunnel < 3.0)),
        tolled_rfk=as_float(((is_manhattan_queens > 0) | (is_manhattan_bronx > 0)) & (mid_rfk < 3.0)),
        tolled_battery_tunnel=as_float((is_manhattan_brooklyn > 0) & (mid_battery_tunnel < 2.5)),
        tolled_hudson_tunnel=as_float((is_ewr_manhattan > 0) | (d_lon < -74.03)),
        free_queensboro=as_float((is_manhattan_queens > 0) & (mid_queensboro < 3.0)),
        free_brooklyn=as_float((is_manhattan_brooklyn > 0) & (mid_brooklyn_bridge < 2.5)))

    # landmark distances (km) for pickup and dropoff
    landmarks = {'midtown': (40.755, -73.985), 'fidi': (40.7128, -74.006), 'gc': (40.7527, -73.9772),
                 'penn': (40.7505, -73.9935), 'ues': (40.7736, -73.9566), 'uws': (40.787, -73.9754),
                 'bk_downtown': (40.689, -73.985), 'lic': (40.7447, -73.9485)}
    landmark_cols = {'center_dist_midtown': km_dist(center_lat, center_lon, 40.755, -73.985)}
    for name, (lat0, lon0) in landmarks.items():
        landmark_cols[f'{name}_p_dist'] = km_dist(p_lat, p_lon, lat0, lon0)
        landmark_cols[f'{name}_d_dist'] = km_dist(d_lat, d_lon, lat0, lon0)
    landmark_block = block(X, 'landmarks', **landmark_cols)

    # pickup time: calendar, cycles, rush periods, fare-regime dates
    dt = X['pickup_datetime'].astype(str).str.slice(0, 19).skb.apply_func(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
    year = dt.dt.year.astype('float64')
    month = dt.dt.month.astype('float64')
    day = dt.dt.day.astype('float64')
    hour = dt.dt.hour.astype('float64')
    weekday = dt.dt.weekday.astype('float64')
    hour_float = hour + dt.dt.minute.astype('float64') / 60.0
    is_rush_hour = as_float((weekday < 5) & (hour >= 16) & (hour < 20))
    is_morning_rush = as_float((weekday < 5) & (hour >= 7) & (hour < 10))
    is_midday = as_float((weekday < 5) & (hour >= 10) & (hour < 16))
    is_overnight = as_float((hour >= 20) | (hour < 6))
    is_post_nov_2009 = as_float((year > 2009) | (year == 2009) & (month >= 11))
    is_post_2012 = as_float((year > 2012) | (year == 2012) & ((month > 9) | (month == 9) & (day >= 4)))
    is_post_2015 = as_float(year >= 2015)
    pickup_time = block(
        X, 'pickup_time', year=year, month=month, day=day, hour=hour, weekday=weekday, hour_float=hour_float,
        sin_diurnal=(2.0 * np.pi * hour_float / 24.0).skb.apply_func(np.sin),
        cos_diurnal=(2.0 * np.pi * hour_float / 24.0).skb.apply_func(np.cos),
        sin_hour=(2.0 * np.pi * hour / 24.0).skb.apply_func(np.sin),
        cos_hour=(2.0 * np.pi * hour / 24.0).skb.apply_func(np.cos),
        sin_month=(2.0 * np.pi * month / 12.0).skb.apply_func(np.sin),
        cos_month=(2.0 * np.pi * month / 12.0).skb.apply_func(np.cos),
        is_weekend=as_float(weekday >= 5), is_rush_hour=is_rush_hour, is_morning_rush=is_morning_rush,
        is_midday=is_midday, is_overnight=is_overnight,
        is_post_nov_2009=is_post_nov_2009, is_post_2012=is_post_2012, is_post_2015=is_post_2015)

    # fare rules: statutory surcharges, JFK/EWR flat rates, metered estimate, interactions
    non_jfk_manhattan_km = (1.0 - is_jfk_manhattan) * manhattan_km
    non_jfk_haversine_km = (1.0 - is_jfk_manhattan) * haversine_km
    jfk_toll = is_jfk_manhattan * (5.33 * is_post_2012 + 4.50 * (1.0 - is_post_2012))
    jfk_flat_rate = is_jfk_manhattan * (52.0 * is_post_2012 + 45.0 * (1.0 - is_post_2012)) + jfk_toll
    ewr_surcharge = is_ewr_manhattan * (17.5 * is_post_2012 + 15.0 * (1.0 - is_post_2012))
    statutory_surcharge = 0.5 * is_post_nov_2009 + 0.5 * is_overnight + 1.0 * is_rush_hour + 0.3 * is_post_2015
    rate_per_km = 1.2427 * (1.0 - is_post_2012) + 1.5534 * is_post_2012
    effective_dist_km = is_both_manhattan * manhattan_km + (1.0 - is_both_manhattan) * (
        0.6 * manhattan_km + 0.4 * haversine_km)
    base_fare = 2.5 + effective_dist_km * rate_per_km
    statutory_fare_estimate = ((1.0 - is_jfk_manhattan) * base_fare + jfk_flat_rate + statutory_surcharge
                               + ewr_surcharge)
    fare_rules = block(
        X, 'fare_rules',
        manhattan_crosstown_km=is_both_manhattan * rot_dx, manhattan_avenue_km=is_both_manhattan * rot_dy,
        manhattan_rush_crosstown=is_both_manhattan * is_rush_hour * rot_dx,
        manhattan_midday_crosstown=is_both_manhattan * is_midday * rot_dx,
        non_jfk_manhattan_km=non_jfk_manhattan_km, non_jfk_haversine_km=non_jfk_haversine_km,
        post_2012_dist=is_post_2012 * non_jfk_manhattan_km, post_2012_hav=is_post_2012 * non_jfk_haversine_km,
        sqrt_manhattan_km=non_jfk_manhattan_km.skb.apply_func(np.sqrt),
        log_manhattan_km=non_jfk_manhattan_km.skb.apply_func(np.log1p),
        sqrt_haversine_km=non_jfk_haversine_km.skb.apply_func(np.sqrt),
        log_haversine_km=non_jfk_haversine_km.skb.apply_func(np.log1p),
        rush_hour_dist=is_rush_hour * non_jfk_manhattan_km, morning_rush_dist=is_morning_rush * non_jfk_manhattan_km,
        midday_dist=is_midday * non_jfk_manhattan_km, overnight_dist=is_overnight * non_jfk_manhattan_km,
        suburban_dist=is_suburban * non_jfk_manhattan_km, post_2012_rush=is_post_2012 * is_rush_hour,
        jfk_toll=jfk_toll, jfk_flat_rate=jfk_flat_rate, ewr_surcharge=ewr_surcharge,
        statutory_surcharge=statutory_surcharge, statutory_fare_estimate=statutory_fare_estimate)

    # multi-scale harmonic encodings of the rotated pickup/dropoff coordinates
    harmonic_cols = {}
    for wl in WAVELENGTHS:
        freq = 2.0 * np.pi / wl
        wl_str = str(wl).replace('.', '_')
        for end, (cx, cy) in {'pick': (rot_pick_x, rot_pick_y), 'drop': (rot_drop_x, rot_drop_y)}.items():
            for axis, coord in (('x', cx), ('y', cy)):
                harmonic_cols[f'sin_{end}_{axis}_{wl_str}'] = (coord * freq).skb.apply_func(np.sin)
                harmonic_cols[f'cos_{end}_{axis}_{wl_str}'] = (coord * freq).skb.apply_func(np.cos)
    harmonics = block(X, 'harmonics', **harmonic_cols)

    features = raw_coords.skb.concat(
        [trip_geometry, boroughs, airports, crossings, landmark_block, pickup_time, fare_rules, harmonics],
        axis=1)[list(FEATURE_ORDER)]
    return features.skb.set_name('features'), statutory_fare_estimate


class ResBlock(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class WideAndDeepTabularResNet(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int = 256, num_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.wide = nn.Linear(in_features, 1, bias=False)
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = self.gelu(h)
        deep_out = self.head(h)
        return wide_out + deep_out


class TorchSeed(Callback):
    """Seeds torch when the net initializes, before skorch builds the module."""

    def __init__(self, seed=0):
        self.seed = seed

    def initialize(self):
        torch.manual_seed(self.seed)
        return super().initialize()


class HeadBiasFromTarget(Callback):
    """Starts the regression head's bias at the mean training target."""

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        with torch.no_grad():
            net.module_.head.bias.fill_(float(np.asarray(y, dtype=np.float32).ravel().mean()))


class CosineOverFit(Callback):
    """CosineAnnealingLR stepped after every training batch, over the whole fit
    (T_max = max_epochs x batches per epoch of this fit's rows)."""

    def __init__(self, eta_min=0.0):
        self.eta_min = eta_min

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        steps = net.max_epochs * math.ceil(len(X) / net.batch_size)
        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(net.optimizer_, T_max=steps,
                                                                     eta_min=self.eta_min)

    def on_batch_end(self, net, training=False, **kwargs):
        if training:
            self.scheduler_.step()


def blend_predictions(resnet, lgbm, xgb, baseline, mode):
    """Weighted residual blend on the statutory baseline, clipped. While fitting, the
    learner nodes evaluate to fitted estimators, not predictions: return nothing then.
    (eval_mode().skb.match would keep this as recorded ops, but stratum's scheduler
    takes the 'fit' branch of such a match when predicting.)"""
    if mode == 'fit':
        return None
    blend = baseline + (W_RESNET * resnet.ravel() + W_LGB * lgbm + W_XGB * xgb)
    return np.clip(blend, FARE_FLOOR, FARE_CAP)


def fare_model(features, statutory_fare_estimate, y):
    """The original TriEnsembleResidualRegressor as graph nodes: residual target,
    ResNet (skorch) + LightGBM + XGBoost, weighted blend on the baseline, clip."""
    residual = (y - statutory_fare_estimate).skb.set_name('residual_target')
    matrix = features.astype('float32').skb.apply_func(
        np.nan_to_num, nan=0.0, posinf=0.0, neginf=0.0).skb.set_name('feature_matrix')

    scaled = matrix.skb.apply(StandardScaler())
    resnet = NeuralNetRegressor(
        module=WideAndDeepTabularResNet,
        module__in_features=len(FEATURE_ORDER),
        module__hidden_dim=256,
        module__num_blocks=3,
        module__dropout=0.1,
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        lr=0.003,
        optimizer__weight_decay=0.0001,
        max_epochs=25,
        batch_size=2048,
        iterator_train__shuffle=True,
        iterator_valid__batch_size=4096,
        train_split=None,
        callbacks=[('seed', TorchSeed(SEED)), ('head_bias', HeadBiasFromTarget()),
                   ('cosine', CosineOverFit(eta_min=1e-05))],
        device=DEVICE,
        verbose=0,
    )
    resnet_pred = scaled.skb.apply(
        resnet, y=residual.to_numpy(dtype='float32').reshape(-1, 1)).skb.set_name('resnet')

    lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.04, num_leaves=63, min_child_samples=50,
                         reg_alpha=1.0, reg_lambda=5.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                         random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    lgbm_pred = features.skb.apply(lgbm, y=residual).skb.set_name('lgbm')

    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.04, max_depth=8, reg_alpha=1.0, reg_lambda=5.0,
                       colsample_bytree=0.8, subsample=0.8, random_state=SEED, n_jobs=N_JOBS,
                       tree_method='hist', device=DEVICE)
    xgb_pred = matrix.skb.apply(xgb, y=residual).skb.set_name('xgb')

    return resnet_pred.skb.apply_func(
        blend_predictions, lgbm_pred, xgb_pred, statutory_fare_estimate.to_numpy(dtype='float64'),
        skrub.eval_mode()).skb.set_name('model')


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    features, statutory_fare_estimate = taxi_features(X)
    pred = fare_model(features, statutory_fare_estimate, y)

    result = {'pred': pred, 'scoring': scoring}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result
