"""pipeline_10 -- direction A: multiplicative interaction features.

base_geo + products a tree can't form on its own (Elevation x Slope/Aspect,
Slope x Hillshade, Road x Fire, Elevation x Hydro). Same fast model as the anchor.
"""
from common import load_xy
from features import base_geo, add_interactions, fast_model

X, y = load_xy(target="Cover_Type")
X = add_interactions(base_geo(X.skb.drop(cols="Id")))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "A: base_geo + multiplicative interactions (elev x slope/aspect, etc.)"
PARENT = "pipeline_09"
