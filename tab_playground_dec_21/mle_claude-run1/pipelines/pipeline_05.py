"""pipeline_05 -- LightGBM + classic Forest-Cover engineered features.

Adds the geometric interactions known to help on this dataset, as fine-grained
recorded ops on top of pipeline_03 (LightGBM, default params):
  - Euclidean distance to hydrology from its H/V components
  - Elevation adjusted by vertical distance to hydrology (proxy for water-table
    elevation), which is very informative for cover type
  - pairwise sums/diffs of the three "distance to X" features
  - Aspect (degrees, circular) as sin/cos
  - Hillshade mean
Raw columns are kept alongside the new ones; trees pick what helps.
"""
import numpy as np

from common import load_xy, SEED
from lightgbm import LGBMClassifier

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
    LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
    y=y,
)

DESCRIPTION = "LightGBM + engineered geometric features (hydro/elev/distance/aspect)"
PARENT = "pipeline_03"
