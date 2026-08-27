"""pipeline_14 -- MERGE: the two feature themes that beat the anchor.

09-13 held the model fixed (fast_model) and varied one feature theme each. Only
two themes helped over the anchor (09=0.94791): soil descriptors (12=+0.00286,
the best) and distance ratios (11=+0.00116). Interactions (10) were flat and
hillshade physics (13) hurt, so both are excluded. This unions the two winners
on top of base_geo, same fast model, to see if their gains stack.
"""
from common import load_xy
from features import base_geo, add_soil_descriptors, add_ratios, fast_model

X, y = load_xy(target="Cover_Type")
X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "MERGE: base_geo + soil descriptors (12) + distance ratios (11)"
PARENT = "pipeline_12"
