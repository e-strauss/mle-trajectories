"""pipeline_06 -- higher-capacity LightGBM on the engineered features.

4M training rows can support a much larger model than the defaults
(num_leaves=31, n_estimators=100). Bump capacity and add more, slower-learning
trees. Single config (no grid) to read the ceiling before spending on a sweep.
PARENT set to the engineered-feature pipeline (05) assuming features helped;
revisit if not.
"""
from lightgbm import LGBMClassifier

from common import load_xy, SEED
import numpy as np

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

DESCRIPTION = "LightGBM higher capacity (400 trees, lr=0.05, 255 leaves) + engineered features"
PARENT = "pipeline_05"
