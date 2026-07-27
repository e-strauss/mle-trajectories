"""pipeline_09 -- ANCHOR: pipeline_07 feature set at the fast model config.

Re-establishes the current best feature set (geometric + indicator counts) under
the compute-friendly model (200 trees, lr=0.1, 127 leaves) so the exploratory
pipelines 10-13, which add one theme each on top of these features at the SAME
config, have a clean same-config reference for their deltas.
"""
from common import load_xy
from features import base_geo, fast_model

X, y = load_xy(target="Cover_Type")
X = base_geo(X.skb.drop(cols="Id"))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "ANCHOR: base_geo features, fast model (200 trees/127 leaves/lr0.1)"
PARENT = "pipeline_07"
