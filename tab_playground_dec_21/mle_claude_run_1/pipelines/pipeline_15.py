"""pipeline_15 -- IMPROVE: enrich the merge by crossing soil with geo.

Builds on the pipeline_14 merge (base_geo + soil descriptors + ratios) and
extends the strongest theme: crosses the soil descriptor aggregates with the
dominant base signals (Cryic/Stoniness/RockOutcrop x Elevation/Slope). Cryic
soils track the cold, high-elevation regime that separates cover types, so these
products give the tree ready-made high-elevation splits the soil aggregates alone
can't express. Same fast model.
"""
from common import load_xy
from features import (
    base_geo, add_soil_descriptors, add_ratios, add_soil_geo_cross, fast_model,
)

X, y = load_xy(target="Cover_Type")
X = add_soil_geo_cross(add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id")))))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "IMPROVE: pipeline_14 merge + soil x geo crosses (cryic/stoniness/rock x elev/slope)"
PARENT = "pipeline_14"
