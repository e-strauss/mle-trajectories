"""pipeline_11 -- direction B: distance ratios & aggregates.

base_geo + total/mean of the three distances and pairwise ratios
(hydro/road, fire/road, hydro/fire, VDH/HDH, elevation/hydro). Same fast model.
"""
from common import load_xy
from features import base_geo, add_ratios, fast_model

X, y = load_xy(target="Cover_Type")
X = add_ratios(base_geo(X.skb.drop(cols="Id")))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "B: base_geo + distance ratios & aggregates"
PARENT = "pipeline_09"
