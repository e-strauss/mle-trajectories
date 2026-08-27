"""pipeline_19 -- MEDIUM config on the best (base14) feature set.

The fast model is exhausted at ~0.95192: features are ablated to the pipeline_14
set and pipeline_17 showed lr=0.1/200 trees can't use more leaves. The only lever
left is capacity done right -- lower learning rate + more rounds + more leaves.
medium_model (400 trees, lr=0.05, 255 leaves, ~800s) is that trade at ~3x fast
cost but still ~3x cheaper than big_model. Tests whether the cheap-model ceiling
lifts, and gives a stronger big_model proxy for future feature work.
"""
from common import load_xy
from features import base_geo, add_soil_descriptors, add_ratios, medium_model

X, y = load_xy(target="Cover_Type")
X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
pred = X.skb.apply(medium_model(), y=y)

DESCRIPTION = "MEDIUM config (400 trees/lr0.05/255 leaves) on base14 features"
PARENT = "pipeline_14"
