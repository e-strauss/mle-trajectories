"""pipeline_18 -- ABLATION (single run): does extra soil domain knowledge help?

pipeline_17 confirmed the current fast config (200 trees, lr=0.1, 127 leaves,
min_child_samples=20) is already best-at-budget -- more leaves or more leaf
regularization both hurt at lr=0.1/200 trees. So we hold that config FIXED and
ablate FEATURES in one run via skrub.choose_from over prediction sub-graphs. The
harness scores every variant, records the best to the board, and saves the full
per-variant table in results.json -> extra.grid:

  base14          : base_geo + soil descriptors + ratios         (control = pipeline_14)
  base14+soilx    : + add_soil_extra (warm/aquic/till/bouldery)  (does new soil info add?)
  soilx_no_ratios : base_geo + soil + soil_extra (no ratios)     (re-ablate ratios)
"""
import skrub

from common import load_xy
from features import (
    base_geo, add_soil_descriptors, add_ratios, add_soil_extra, fast_model,
)

X, y = load_xy(target="Cover_Type")
Xd = X.skb.drop(cols="Id")
base14 = add_ratios(add_soil_descriptors(base_geo(Xd)))

variants = {
    "base14": base14.skb.apply(fast_model(), y=y),
    "base14+soilx": add_soil_extra(base14).skb.apply(fast_model(), y=y),
    "soilx_no_ratios": add_soil_extra(add_soil_descriptors(base_geo(Xd))).skb.apply(fast_model(), y=y),
}
pred = skrub.choose_from(variants, name="featureset").as_data_op()

DESCRIPTION = "ABLATION: soil_extra on/off x ratios on/off (fast config, pipeline_14 base)"
PARENT = "pipeline_14"
