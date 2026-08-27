"""pipeline_07 -- richer feature set, same high-capacity LightGBM as pipeline_06.

Model config is IDENTICAL to pipeline_06 (400 trees, lr=0.05, 255 leaves) so
this isolates the effect of additional features:
  - soil_count / wild_count: the soil (0-5) and wilderness (0-2) indicators are
    multi-hot in this synthetic data (data_exploration_2), so their active-count
    carries real signal a single argmax code would destroy.
  - Hillshade_9am_minus_3pm: morning-vs-afternoon illumination asymmetry.
  - abs_VDH: magnitude of vertical distance to hydrology (sign already kept).
plus the geometric features from pipeline_05/06.
"""
import numpy as np
from skrub import selectors as s
from lightgbm import LGBMClassifier

from common import load_xy, SEED

X, y = load_xy(target="Cover_Type")
X = X.skb.drop(cols="Id")

hdh = X["Horizontal_Distance_To_Hydrology"]
vdh = X["Vertical_Distance_To_Hydrology"]
road = X["Horizontal_Distance_To_Roadways"]
fire = X["Horizontal_Distance_To_Fire_Points"]
elev = X["Elevation"]
aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)

X = X.assign(
    Euclidean_Hydro=(hdh**2 + vdh**2).skb.apply_func(np.sqrt),
    Elev_minus_VDH=elev - vdh,
    Elev_minus_HDH=elev - hdh * 0.2,
    Hydro_plus_Fire=hdh + fire,
    Hydro_minus_Fire=hdh - fire,
    Hydro_plus_Road=hdh + road,
    Hydro_minus_Road=hdh - road,
    Fire_plus_Road=fire + road,
    Fire_minus_Road=fire - road,
    Aspect_sin=aspect_rad.skb.apply_func(np.sin),
    Aspect_cos=aspect_rad.skb.apply_func(np.cos),
    Hillshade_mean=(X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]) / 3.0,
    # --- new in pipeline_07 ---
    soil_count=X.skb.select(s.glob("Soil_Type*")).sum(axis=1),
    wild_count=X.skb.select(s.glob("Wilderness_Area*")).sum(axis=1),
    Hillshade_9am_minus_3pm=X["Hillshade_9am"] - X["Hillshade_3pm"],
    abs_VDH=vdh.skb.apply_func(np.abs),
)

pred = X.skb.apply(
    LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=255,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    ),
    y=y,
)

DESCRIPTION = "LightGBM (pipeline_06 config) + richer features (soil/wild counts, hillshade spread, |VDH|)"
PARENT = "pipeline_06"
