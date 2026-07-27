"""pipeline_12 -- direction C: soil-descriptor domain features.

base_geo + row aggregates of soil-type descriptor flags parsed from the data
dictionary: stoniness intensity, cryic (cold/high-elevation) soils, rock
outcrop / rock land / rubbly, and leighcan family. Aggregated over the multi-hot
soil block. The cryic flag in particular tracks elevation -> cover type.
Same fast model.
"""
from common import load_xy
from features import base_geo, add_soil_descriptors, fast_model

X, y = load_xy(target="Cover_Type")
X = add_soil_descriptors(base_geo(X.skb.drop(cols="Id")))
pred = X.skb.apply(fast_model(), y=y)

DESCRIPTION = "C: base_geo + soil-descriptor domain features (stoniness, cryic, rock)"
PARENT = "pipeline_09"
