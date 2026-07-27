"""pipeline_08 -- final capacity push: more trees, deeper, slower learning.

Same (winning) feature set as pipeline_07; only the LightGBM budget grows:
n_estimators 400->700, num_leaves 255->511, learning_rate 0.05->0.035. 4M rows
comfortably support this, and capacity has been the strongest lever so far
(pipeline_05 -> 06). This is the strongest-combination finale.
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
    soil_count=X.skb.select(s.glob("Soil_Type*")).sum(axis=1),
    wild_count=X.skb.select(s.glob("Wilderness_Area*")).sum(axis=1),
    Hillshade_9am_minus_3pm=X["Hillshade_9am"] - X["Hillshade_3pm"],
    abs_VDH=vdh.skb.apply_func(np.abs),
)

pred = X.skb.apply(
    LGBMClassifier(
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=511,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    ),
    y=y,
)

DESCRIPTION = "LightGBM max capacity (700 trees, lr=0.035, 511 leaves) + richer features"
PARENT = "pipeline_07"
