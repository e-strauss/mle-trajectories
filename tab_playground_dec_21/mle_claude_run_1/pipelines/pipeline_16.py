"""pipeline_16 -- FINAL: the winning pipeline_14 feature set at full capacity.

The feature exploration (09-15) found the best feature set at the fast model:
base_geo + soil descriptors + distance ratios (pipeline_14 = 0.95192). This runs
that exact feature set at the big_model config (700 trees, lr=0.035, 511 leaves),
the same capacity that carried pipeline_08 to the overall best (0.95589) on the
OLDER pipeline_07 feature set. Since 14 already edges out 07 at fixed capacity,
combining the best features with the best capacity should push past 0.956.
"""
from common import load_xy
from features import base_geo, add_soil_descriptors, add_ratios, big_model

X, y = load_xy(target="Cover_Type")
X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
pred = X.skb.apply(big_model(), y=y)

DESCRIPTION = "FINAL: pipeline_14 feature set (base_geo + soil + ratios) at big_model (700/511/lr0.035)"
PARENT = "pipeline_14"