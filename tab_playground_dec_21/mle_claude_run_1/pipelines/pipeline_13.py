"""pipeline_13 -- direction D: hillshade dynamics & slope-adjusted illumination.

base_geo + shade statistics across the three hillshade columns (min/max/range/
std) and physics-flavoured features (cos of slope, noon illumination adjusted by
slope). Same fast model.
"""
from common import load_xy
from features import base_geo, add_hillshade_physics, fast_model

X, y = load_xy(target="Cover_Type")
X = add_hillshade_physics(base_geo(X.skb.drop(cols="Id")))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "D: base_geo + hillshade dynamics & slope-adjusted illumination"
PARENT = "pipeline_09"
